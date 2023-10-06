from src.egnn import GCL, GaussianSmearing
import torch.nn as nn
import torch
from src.egnn import coord2diff
from torch_scatter import scatter_mean

class FragSizeGNN(nn.Module):
    def __init__(self,
                 lig_nf,
                 pocket_nf,
                 joint_nf,
                 hidden_nf,
                 out_node_nf, # number of classes (fragment sizes)
                 n_layers,
                 normalization=True,
                 attention=True,
                 normalization_factor=100,
                 aggregation_method='sum',
                 edge_cutoff_ligand=None,
                 edge_cutoff_pocket=5,
                 edge_cutoff_interaction=5,
                 dataset_type='CrossDock',
                 gaussian_expansion=True,
                 num_gaussians=16):
        super(FragSizeGNN, self).__init__()

        self.dataset_type = dataset_type
        if self.dataset_type == 'CrossDock':
            context_node_nf = 3 # mask on the pocket atoms and anchor points

        if gaussian_expansion:
            self.gauss_exp = GaussianSmearing(start=0., stop=5., num_gaussians=num_gaussians)
            in_edge_nf = num_gaussians

        self.hidden_nf = hidden_nf
        self.out_node_nf = out_node_nf
        self.n_layers = n_layers
        self.normalization = normalization
        self.attention = attention
        self.normalization_factor = normalization_factor
        self.gaussian_expansion = gaussian_expansion
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction

        self.mol_encoder = nn.Sequential(
            nn.Linear(lig_nf, joint_nf),
        )

        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_nf, joint_nf),
        )

        self.embed_both = nn.Linear(joint_nf+context_node_nf, hidden_nf) # concatenate the context features to joint space

        self.gcl1 = GCL(
            input_nf=self.hidden_nf,
            output_nf=self.hidden_nf,
            hidden_nf=self.hidden_nf, 
            normalization_factor=normalization_factor, 
            aggregation_method=aggregation_method,
            edges_in_d=in_edge_nf,
            activation=nn.ReLU(),
            attention=attention,
            normalization=normalization 
        )

        layers = []
        for i in range(n_layers - 1):
            layer = GCL(
                input_nf=self.hidden_nf,
                output_nf=self.hidden_nf,
                hidden_nf=self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf,
                activation=nn.ReLU(),
                attention=attention,
                normalization=normalization 
            )
            layers.append(layer)
        
        self.gcl_layers = nn.ModuleList(layers)
        self.embedding_out = nn.Linear(self.hidden_nf, self.out_node_nf)
        self.act = nn.ReLU()

        self.edge_cache = {}
        #self.lin_out = nn.Linear(self.out_node_nf, 1)

    def forward(self, mol_x, mol_h, node_mask, pocket_x, pocket_h, pocket_mask, anchors, pocket_anchors):
        """
        mol_x: [B, N, 3] positions of scaffold atoms
        mol_h: [B, N, nf] onehot of scaffold atoms
        node_mask: [B, N] only for scaffold-based
        pocket_x: [B, N, 3] positions of pocket atoms
        pocket_h: [B, N, nf] onehot of pocket atoms
        anchors: [B, N, 3] positions of anchor points
        pocket_anchors: [B, N, 3] positions of anchor points
        """
        bs, n_nodes_lig = mol_x.shape[0], mol_x.shape[1]
        n_nodes_pocket = pocket_x.shape[1]
        node_mask = node_mask.squeeze()

        N = n_nodes_lig + n_nodes_pocket
        mol_x = mol_x[node_mask.bool()] # [N_l, 3]
        mol_h = mol_h[node_mask.bool()] # [N_l, nf]

        pocket_x = pocket_x[pocket_mask.bool()] # [N_p, 3]
        pocket_h = pocket_h[pocket_mask.bool()] # [N_p, nf]

        mol_h = self.mol_encoder(mol_h) # [N_l, joint_nf]
        pocket_h = self.pocket_encoder(pocket_h)

        h = torch.cat([mol_h, pocket_h], dim=0) # [N, joint_nf]

        batch_mask_ligand = self.get_batch_mask(node_mask, device=mol_x.device) # [N_l]
        batch_mask_pocket = self.get_batch_mask(pocket_mask, device=mol_x.device) # [N_p]
        new_anchor_mask = torch.cat([anchors[node_mask.bool()], pocket_anchors[pocket_mask.bool()]], dim=0).unsqueeze(-1)
        new_scaffold_mask = torch.cat([torch.ones_like(batch_mask_ligand, device=mol_x.device), torch.zeros_like(batch_mask_pocket)], dim=0).unsqueeze(-1)
        new_pocket_mask = torch.cat([torch.zeros_like(batch_mask_ligand), torch.ones_like(batch_mask_pocket)], dim=0).unsqueeze(-1)

        h = torch.cat([h, new_anchor_mask, new_scaffold_mask, new_pocket_mask], dim=1) # [N, joint_nf+2]
        x = torch.cat([mol_x, pocket_x], dim=0) # [N, 3]

        mask = torch.cat([batch_mask_ligand, batch_mask_pocket], dim=0) # [N]
        device = mol_x.device

        h = self.embed_both(h)
        edges = self.get_edges_cutoff(batch_mask_ligand, batch_mask_pocket, mol_x, pocket_x) # [2, E]
        
        # selected only edges based on a 7A distance (all protein and scaffold atoms considered)
        distances, _ = coord2diff(x, edges) # TODO: consider adding more edge info such as the type of bond
        if self.gaussian_expansion:
            distances = self.gauss_exp(distances)

        for gcl in self.gcl_layers:
            h, _ = gcl(h, edges, edge_attr=distances, node_mask=None, edge_mask=None)

        h_final = self.act(self.embedding_out(h)) # [N, out_node_nf]

        # convert to batch
        #out = scatter_mean(h_final, mask, dim=0, dim_size=bs) # [B, out_node_nf]
        num_atoms = node_mask.sum(dim=1).int() + pocket_mask.sum(dim=1).int()
        reshaped_out = torch.zeros(bs, N, h_final.shape[-1], dtype=h.dtype, device=h.device)
        positions = torch.zeros_like(mask).to(h.device)
        for idx in range(bs):
            positions[mask == idx] = torch.arange(num_atoms[idx], device=h.device)
        reshaped_out[mask, positions] = h_final
        return reshaped_out # [B, N, out_node_nf]

    def get_edges_cutoff(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):

        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)
        
        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)
        
        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)
        
        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                        torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj), dim=0)
        return edges
    
    @staticmethod
    def get_batch_mask(mask, device):
        n_nodes = mask.float().sum(dim=1).int()
        batch_size = mask.shape[0]
        batch_mask = torch.cat([torch.ones(n_nodes[i]) * i for i in range(batch_size)]).long().to(device)
        return batch_mask