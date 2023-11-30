import numpy as np
import pandas as pd
import torch

from utils.templates import create_template_for_pocket_anchor_prediction, create_template_for_pocket_mol
from utils.sample_frag_size import sample_fragment_size, fragsize_prob_df, bounds
#from sampling.rejection_sampling import compute_number_of_clashes

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}
CROSSDOCK_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'B':5, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket
vdws = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'B': 1.92, 'Br': 1.85, 'Cl': 1.75, 'P': 1.8, 'I': 1.98, 'F': 1.47}

def extend_scaffold(n_samples,
                    num_frags,
                    x, # scaffold coords
                    h, # scaffold onehots 
                    pocket_coords,
                    pocket_onehot,
                    anchor_model,
                    diff_model,
                    device,
                    return_all=False,
                    max_mol_sizes=None,
                    prot_path=None,
                    custom_anchors=None,
                    all_grids=None,
                    lj_guidance=True,
                    prot_mol_lj_rm=None,
                    mol_mol_lj_rm=None,
                    all_H_coords=None,
                    guidance_weights=None,
                    ): 
    """ 
    pocket_corods: [Np, 3]
    pocket_oneoht: [Np, hp]
    x: [B, Ns, 3]
    h: [B, Ns, hf]
    """
    center_of_mass = 'anchors'
    all_frag_sizes = np.zeros((n_samples, num_frags+1), dtype=np.int32) # num_frags+1 because we have the first scaffold

    all_x = []
    all_h = []
    all_extension_masks = []
    all_scaffold_masks = []
    all_pocket_masks = []
    total_grids = []
    all_pocket_x = []
    scaff_size = x.shape[1]
    pocket_size = pocket_coords.shape[0]

    extension_masks = torch.ones(scaff_size, dtype=torch.float32, device=device) 
    scaffold_masks = torch.tensor(scaff_size, dtype=torch.float32, device=device)
    
    extension_masks = extension_masks.unsqueeze(0).repeat(n_samples, 1) 
    scaffold_masks = scaffold_masks.unsqueeze(0).repeat(n_samples, 1) 
    pocket_x = pocket_coords.unsqueeze(0).repeat(n_samples, 1, 1).to(device)
    pocket_h = pocket_onehot.unsqueeze(0).repeat(n_samples, 1, 1).to(device) 

    all_x.append(x)
    all_h.append(h)
    all_scaffold_masks.append(scaffold_masks)
    all_extension_masks.append(extension_masks)
    all_pocket_x.append(pocket_x)

    generation_steps = np.ones(n_samples, dtype=np.int32)
    steps = np.ones(n_samples, dtype=np.int32) # we start from 1 since we already have the scaffold
    scaffold_sizes = np.ones(n_samples, dtype=np.int32) * scaff_size # size of scaffold molecule
    prev_ext_sizes = scaffold_sizes
    all_frag_sizes[:,0] = scaffold_sizes #
    # first need to concatenate pocket coords/atomtypes with those of the scaffold
    # NOTE: x and h are already concatenated -> [B, pocket_atoms + scaffold_atoms, ...]
    if custom_anchors is not None:
        n_custom_anchors = len(custom_anchors) # number of custom anchors
        print('number of custom anchors:', n_custom_anchors)

    for i in range(1, num_frags+1):
        print(f'generating fragment at step {i}')
        batch_anchor = create_template_for_pocket_anchor_prediction(n_samples=n_samples,
                                                                    steps=steps,
                                                                    prev_ext_sizes=prev_ext_sizes,
                                                                    prev_mol_sizes=all_frag_sizes.cumsum(axis=1)[:,i-1],
                                                                    pocket_size=pocket_size,
                                                                    prev_x=x,
                                                                    prev_h=h,
                                                                    pocket_x=pocket_x,
                                                                    pocket_h=pocket_onehot.to(device)
                                                                    )
        
        scaffold_masks = batch_anchor['scaffold_masks'].to(device)
        scaff_x = batch_anchor['scaff_x'].to(device)
        scaff_h = batch_anchor['scaff_h'].to(device)
        pocket_masks = batch_anchor['pocket_masks'].to(device)
        pocket_x = batch_anchor['pocket_x'].to(device)
        pocket_h = batch_anchor['pocket_h'].to(device)
        
        anchor_out = anchor_model.anchor_predictor.forward(mol_x=scaff_x,
                                                            mol_h=scaff_h,
                                                            pocket_x=pocket_x,
                                                            pocket_h=pocket_h,
                                                            node_mask=scaffold_masks,
                                                            pocket_mask=pocket_masks)
        
        anchor_prob = torch.sigmoid(anchor_out).squeeze()
        anchor_prob = anchor_prob * scaffold_masks

        all_num_grids = torch.zeros_like(anchor_prob, dtype=torch.int32)
        for l in range(n_samples):
            mol_mask = scaffold_masks[l].cpu().bool()
            mol_coords = scaff_x[l].cpu()
            mol_dists = torch.cdist(all_grids[l].float(), mol_coords)
            num_grids = (mol_dists < 3).sum(dim=0) # within a 3A distance of anchor point
            all_num_grids[l] = num_grids

        if np.random.rand() < 0.5:
            new_anchor_prob = (anchor_prob * all_num_grids.to(device))
            anchors_pred = torch.argmax(new_anchor_prob, dim=1).cpu().numpy()
        else:
            anchors_pred = torch.argmax(anchor_prob, dim=1).cpu().numpy()
        
        grid_points_anchor = all_num_grids[np.arange(n_samples), anchors_pred] # number of grid points around each anchor point

        # ---------------------------------------------------------------------------------
        # --------------------------- Predict fragment size -------------------------------

        frag_sizes = np.zeros((n_samples), dtype=np.int32)      
        for l in range(n_samples):
            frag_sizes[l] = sample_fragment_size(int(grid_points_anchor[l]), bounds, fragsize_prob_df)

        all_frag_sizes[:, i] = frag_sizes
        print('fragment sizes: ', frag_sizes)
        mol_n_atoms = np.sum(all_frag_sizes, axis=1)
        if max_mol_sizes is not None:
            for idx, n in enumerate(max_mol_sizes):
                if mol_n_atoms[idx] > n+5: # +5 to account for smaller fragments
                    all_frag_sizes[idx, i] = 0
        print('mol sizes:', mol_n_atoms)
        # ------------------------------------------------------------------------------------------
        # --------------------------------- generating the fragments -------------------------------
        batch = create_template_for_pocket_mol(pocket_x=pocket_x,
                                                pocket_h=pocket_onehot,
                                                n_samples=n_samples,
                                                steps=steps,
                                                frag_sizes=all_frag_sizes[:,i],
                                                prev_mol_sizes=all_frag_sizes.cumsum(axis=1)[:,i-1],
                                                prev_ext_sizes=prev_ext_sizes,
                                                prev_x=x,
                                                prev_h=h,
                                                anchors_scaffold=anchors_pred,
                                                anchors_pocket=None)
        
        x = batch['positions'].to(device)
        h = batch['one_hot'].to(device)
        pocket_x = batch['pocket_coords'].to(device)
        pocket_h = batch['pocket_onehot'].to(device)

        B = x.shape[0]
        Ns = x.shape[1]
        Np = pocket_x.shape[1]
        N = Ns + Np
        node_masks = batch['node_masks'].to(device).view(B,Ns)
        extension_masks = batch['extension_masks'].to(device).view(B,Ns)
        scaffold_masks = batch['scaffold_masks'].to(device).view(B,Ns)
        anchors = batch['anchors'].to(device).view(B,Ns)
        pocket_anchors = batch['pocket_anchors'].to(device)
        pocket_masks = batch['pocket_masks'].to(device)

        if center_of_mass == 'anchors':
            anchor_pos = torch.zeros((B, 3), device=x.device)
            row1, col1 = torch.where(pocket_anchors) # no need (all zeros) but just in case
            anchor_pos[row1] = pocket_x[row1, col1]  

            row2, col2 = torch.where(anchors)
            anchor_pos[row2] = x[row2, col2]
            x = x - anchor_pos.unsqueeze(1) * node_masks.unsqueeze(-1)
            pocket_x = pocket_x - anchor_pos.unsqueeze(1) * pocket_masks.unsqueeze(-1) 

            # shift the grid points according to anchor
        for l in range(n_samples):
            all_grids[l] = all_grids[l] - anchor_pos[l].cpu()
            all_H_coords[l] = all_H_coords[l] - anchor_pos[l]

        x, h, chain_0 = diff_model.edm.sample_chain_single_fragment(x=x,
                                                                    h=h,
                                                                    pocket_x=pocket_x,
                                                                    pocket_h=pocket_h,
                                                                    scaffold_mask=scaffold_masks, # first step scaffold mask is pocket-mask
                                                                    extension_mask=extension_masks,
                                                                    pocket_mask=pocket_masks,
                                                                    anchors=anchors,
                                                                    pocket_anchors=pocket_anchors,
                                                                    keep_frames=1,
                                                                    lj_guidance=lj_guidance,
                                                                    prot_mol_lj_rm=prot_mol_lj_rm,
                                                                    all_H_coords=all_H_coords,
                                                                    guidance_weights=guidance_weights,)
                                                                    

        prev_ext_sizes = all_frag_sizes[:,i]  
        steps += 1  
        generation_steps += 1

        for l in range(n_samples):
            mol_mask = node_masks[l].cpu().bool()
            mol_coords = x[l].cpu()[mol_mask]
            mol_dists = torch.cdist(all_grids[l].float(), mol_coords)
            mask_grids = (mol_dists < 2).any(dim=1) # mask based on 2 A distance
            all_grids[l] = all_grids[l][~mask_grids]

        # ------------------------------------------------------------------------------------------------
        all_x.append(x)
        all_h.append(h)
        all_extension_masks.append(extension_masks)
        all_scaffold_masks.append(scaffold_masks)
        all_pocket_x.append(pocket_x)
        total_grids.append(all_grids)

    if return_all:
        return all_x, all_h, all_extension_masks, all_scaffold_masks, all_pocket_masks
    else:
        mol_masks = (extension_masks.bool() | scaffold_masks.bool()).int().to(device)
        # translate molecule to original pocket coords
        n_atoms = x.shape[1]
        diff = (pocket_x - pocket_coords.to(device))[:,0,:].unsqueeze(1).repeat(1, n_atoms,1) # [B, 3]
        x_tr = x - diff # [B, N, 3]
        return x_tr, h, mol_masks
