from src.egnn import GCL, GaussianSmearing
import torch.nn as nn
import torch
from src.egnn import coord2diff

class MaskedBCEWithLogitsLoss2(torch.nn.Module):
    def __init__(self):
        super(MaskedBCEWithLogitsLoss2, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, input, target, scaffold_mask, pocket_mask, is_first_frag_mask):
        # TODO:
        """
        if_first_frag_mask -> mask for the first fragment (if the fragment is the first the mask is 1)
        """
        masked_loss = self.loss(input, target)
        masked_loss_1 = masked_loss * (~is_first_frag_mask.bool()) # only for parts that are not the first fragment
        masked_loss_1 = masked_loss_1 * scaffold_mask.float() # only for the scaffold atoms

        masked_loss_2 = masked_loss * is_first_frag_mask.bool() # only for parts that are the first fragment
        masked_loss_2 = masked_loss_2 * pocket_mask.float() # only for the pocket atoms

        total_masked_loss = (masked_loss_1.sum() / scaffold_mask.sum().float()) + (masked_loss_2.sum() / pocket_mask.sum().float())
        return total_masked_loss

class MaskedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target, mask):
        masked_loss = self.loss(input, target)
        masked_loss = masked_loss * mask.float()
        return masked_loss.sum() / mask.sum().float()

class AnchorGNNPocket(nn.Module):
    def __init__(self,
                 lig_nf, # ligand node features
                 pocket_nf, # pocket node features
                 joint_nf, # joint number of features
                 hidden_nf, 
                 out_node_nf,
                 n_layers,
                 normalization,
                 attention=True,
                 normalization_factor=100,
                 aggregation_method='sum',
                 dist_cutoff=7,
                 gaussian_expansion=False,
                 num_gaussians=16,
                 edge_cutoff_ligand=None,
                 edge_cutoff_pocket=4.5,
                 edge_cutoff_interaction=4.5
                 ):
        
        super(AnchorGNNPocket, self).__init__()

        #in_node_nf = in_node_nf + context_node_nf # adding the context pocket
        if gaussian_expansion:
            self.gauss_exp = GaussianSmearing(start=0., stop=7., num_gaussians=16)
            in_edge_nf = num_gaussians
        else:
            in_edge_nf = 1

        self.hidden_nf = hidden_nf
        self.out_node_nf = out_node_nf
        self.n_layers = n_layers
        self.normalization = normalization
        self.attention = attention
        self.dist_cutoff = dist_cutoff
        self.normalization_factor = normalization_factor
        self.gaussian_expansion = gaussian_expansion
        self.num_gaussians = num_gaussians
        self.joint_nf = joint_nf
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction

        self.mol_encoder = nn.Sequential(
            nn.Linear(lig_nf, joint_nf),
            nn.SiLU()
        )

        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_nf, joint_nf),
            nn.SiLU()
        )

        self.embed_both = nn.Linear(joint_nf, self.hidden_nf)

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
        layers.append(self.gcl1)
        for i in range(n_layers - 1):
            layer = GCL(
                input_nf=self.hidden_nf,
                output_nf=self.hidden_nf,
                hidden_nf=self.hidden_nf,
                normalization_factor=normalization_factor, 
                aggregation_method='sum',
                edges_in_d=in_edge_nf, 
                activation=nn.ReLU(),
                attention=attention, 
                normalization=normalization
            )
            layers.append(layer)
        
        self.gcl_layers = nn.ModuleList(layers)
        self.embedding_out = nn.Linear(self.hidden_nf, self.out_node_nf)
        self.lin_out = nn.Linear(self.out_node_nf, 1)
        self.act = nn.ReLU()
        #self.bce_loss = MaskedBCEWithLogitsLoss()
    
    def forward(self, mol_x, mol_h, node_mask, pocket_x, pocket_h, pocket_mask):
        """
        input:
            mol_x: [B, Ns, 3] coordinates of scaffold
            mol_h: [B, Ns, nf] onehot of scaffold
            node_mask: [B, Ns]  masking on the scaffold
            pocket_x: [B, Np] coordinates of pocket
            pocket_h: [B, NP, nf_p] onehot of pocket
            pocket_mask: [B, Np] masking on pocket atoms
        output:
            h_out: [B, Ns, 1] logits for the scaffold atoms
        """
        bs, n_lig_nodes = mol_x.shape[0], mol_x.shape[1]
        n_pocket_nodes = pocket_x.shape[1]
        node_mask = node_mask.squeeze()

        N = n_lig_nodes + n_pocket_nodes
        mol_x = mol_x[node_mask.bool()] # [N_l, 3]
        mol_h = mol_h[node_mask.bool()] # [N_l, nf]

        pocket_x = pocket_x[pocket_mask.bool()] # [N_p, 3]
        pocket_h = pocket_h[pocket_mask.bool()] # [N_p, nf]

        mol_h = self.mol_encoder(mol_h) # [N_l, joint_nf]
        pocket_h = self.pocket_encoder(pocket_h) # [N_p, joint_nf]

        h = torch.cat([mol_h, pocket_h], dim=0) # [N_l+N_p, joint_nf]
        x = torch.cat([mol_x, pocket_x], dim=0) # [N_l+N_p, 3]
        
        batch_mask_ligand = self.get_batch_mask(node_mask.bool(), device=x.device) # [N_l]
        batch_mask_pocket = self.get_batch_mask(pocket_mask.bool(), device=x.device) # [N_p]
        
        edges = self.get_edges_cutoff(batch_mask_ligand, batch_mask_pocket, mol_x, pocket_x) # [2, num_edges]
        
        h = self.embed_both(h) # [N_l+N_p, hidden_nf]

        distances, _ = coord2diff(x, edges) 
        if self.gaussian_expansion:
            distances = self.gauss_exp(distances)

        for gcl in self.gcl_layers:
            h, _ = gcl(h, edges, edge_attr=distances, node_mask=None, edge_mask=None) # [N_l+N_p, hidden_nf]
        
        h_atoms = h[:len(batch_mask_ligand)] # [N_l, hidden_nf]
        h_atoms = self.act(self.embedding_out(h_atoms)) # [N_l, out_node_nf]
        h_out = self.lin_out(h_atoms) # [N_l, 1]

        # convert to batch
        num_atoms = node_mask.sum(dim=1).int() # [B]
        reshaped_h_out = torch.zeros(bs, n_lig_nodes, 1, dtype=h_out.dtype).to(h_out.device)
        positions = torch.zeros_like(batch_mask_ligand).to(h_out.device)
        for idx in range(bs):
            positions[batch_mask_ligand == idx] = torch.arange(num_atoms[idx]).to(x.device)
        reshaped_h_out[batch_mask_ligand, positions] = h_out # [B, n_lig_nodes, 1]

        return reshaped_h_out 
    
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