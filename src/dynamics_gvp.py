# Following DiffHopp implementation of GVP: https://github.com/jostorge/diffusion-hopping/tree/main

import torch.nn as nn
import torch
import numpy as np
from src.gvp_model import GVPNetwork

class DynamicsWithPockets(nn.Module):
    def __init__(
            self, n_dims, lig_nf, pocket_nf, context_node_nf=3, joint_nf=32, hidden_nf=128, activation=nn.SiLU(),
            n_layers=4, attention=False, condition_time=True, tanh=False, normalization_factor=100, model='gvp',
            centering=False, edge_cutoff=7, edge_cutoff_interaction=4.5, edge_cutoff_pocket=4.5, edge_cutoff_ligand=None
    ):
        super().__init__()
        
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction

        self.atom_encoder = nn.Sequential(
            nn.Linear(lig_nf, joint_nf),
        )

        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_nf, joint_nf),
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, lig_nf),
        )
 
        if condition_time:
            dynamics_node_nf = joint_nf + 1
        else:
            print('Warning: dynamics moddel is _not_ conditioned on time')
            dynamics_node_nf = joint_nf
            
        self.dynamics = GVPNetwork(
            in_dims=(dynamics_node_nf + context_node_nf, 0), # (scalar_features, vector_features)
            out_dims=(joint_nf, 1),
            hidden_dims=(hidden_nf, hidden_nf//2),
            vector_gate=True,
            num_layers=n_layers,
            attention=attention,
            normalization_factor=normalization_factor,
        ) # other parameters are default 

        self.n_dims = n_dims
        self.condition_time = condition_time
        self.centering = centering
        self.context_node_nf = context_node_nf
        self.edge_cutoff = edge_cutoff
        self.model = model

    def forward(self, t, xh, pocket_xh, extension_mask, scaffold_mask, anchors, pocket_anchors, pocket_mask):
        """ 
        input:
            t: timestep: [B]
            xh: ligand atoms (noised) [B, N_l, h_l+3] 
            pocket_xh: pocket atoms (no noised added) [B, N_p, h_p + 3] 
            extension_masks: mask on fragment extension atoms [B, N] 
            scaffold_masks: mask on scaffold atoms [B, N] 
            anchor_masks: mask on anchor atoms [B, N] 
            pocket_masks: masking on all the pocket atoms [B, N_p] 
        output: 
            (x_out,h_out)  for ligand
        """
        bs, n_lig_nodes = xh.shape[0], xh.shape[1]
        n_pocket_nodes = pocket_xh.shape[1]

        N = n_lig_nodes + n_pocket_nodes
        
        node_mask = (scaffold_mask.bool() | extension_mask.bool()) # [B, N_l]
        xh = xh[node_mask] # [N_l, h_l+3]
        pocket_xh = pocket_xh[pocket_mask.bool()] # [N_p, h_p+3]

        x_atoms = xh[:, :self.n_dims].clone() # [N_l,3]
        h_atoms = xh[:, self.n_dims:].clone() # [N_l,nf]

        x_pocket = pocket_xh[:, :self.n_dims].clone() # [N_p, 3]
        h_pocket = pocket_xh[:, self.n_dims:].clone() # [N_p, hp]
         
        h_atoms = self.atom_encoder(h_atoms) # [N_l, joint_nf]
        h_pocket = self.pocket_encoder(h_pocket) # [N_p, joint_nf]

        x = torch.cat((x_atoms, x_pocket), dim=0) # [N_l+N_p, 3]
        h = torch.cat((h_atoms, h_pocket), dim=0) # [N_l+N_p, joint_nf]
        
        batch_mask_ligand = self.get_batch_mask(node_mask, device=x.device) # [N_l]
        batch_mask_pocket = self.get_batch_mask(pocket_mask, device=x.device) # [N_p]
        mask = torch.cat([batch_mask_ligand, batch_mask_pocket], dim=0) # [N_l+N_p]

        new_anchor_mask = torch.cat([anchors[node_mask], pocket_anchors[pocket_mask.bool()]], dim=0).unsqueeze(-1)
        new_scaffold_msak = torch.cat([scaffold_mask[node_mask], torch.zeros_like(batch_mask_pocket, device=xh.device)], dim=0).unsqueeze(-1)
        new_pocket_mask = torch.cat([torch.zeros_like(batch_mask_ligand, device=xh.device), torch.ones_like(batch_mask_pocket)], dim=0).unsqueeze(-1)
        
        h = torch.cat([h, new_anchor_mask, new_scaffold_msak, new_pocket_mask], dim=1) # [N_l+N_p, joint_nf+3]

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        edges = self.get_edges_cutoff(batch_mask_ligand, batch_mask_pocket, x_atoms, x_pocket) # [2, num_edges]
        assert torch.all(mask[edges[0]] == mask[edges[1]]) 

        #  --------------- apply the GVP dynamics ----------
        h_final, pos_out = self.dynamics(h, x, edges) # [N_l+N_p, joint_nf], [N_l+N_p, 3]
        pos_out = pos_out.reshape(-1,3) # [N_l+N_p, 3]
        
        # decode atoms
        h_final_atoms = self.atom_decoder(h_final[:len(batch_mask_ligand)]) # [N_l, h_l]

        vel_ligand = pos_out[:len(batch_mask_ligand)] # [N_l, 3]
        vel_h_ligand = torch.cat([vel_ligand, h_final_atoms], dim=1) # [N_l, h_l+3]

        # convert to batch 
        num_atoms = node_mask.sum(dim=1).int() # [B]
        reshaped_vel_h = torch.zeros(bs, n_lig_nodes, vel_h_ligand.shape[-1]).to(xh.device) 
        positions = torch.zeros_like(batch_mask_ligand).to(xh.device) 
        for idx in range(bs):
            positions[batch_mask_ligand == idx] = torch.arange(num_atoms[idx]).to(xh.device)
        reshaped_vel_h[batch_mask_ligand, positions] = vel_h_ligand
        
        return reshaped_vel_h # [B, N_l, h_l+3]

    @staticmethod
    def get_dist_edges(x, node_mask, batch_mask):
        node_mask = node_mask.squeeze().bool()
        batch_adj = (batch_mask[:, None] == batch_mask[None, :])
        nodes_adj = (node_mask[:, None] & node_mask[None, :])
        dists_adj = (torch.cdist(x, x) <= 7)
        rm_self_loops = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        adj = batch_adj & nodes_adj & dists_adj & rm_self_loops
        edges = torch.stack(torch.where(adj))
        return edges
    
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