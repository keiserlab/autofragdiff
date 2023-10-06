import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm 
from itertools import combinations
import argparse

from rdkit import Chem
import torch
import time
import shutil
from pathlib import Path
import torch.nn.functional as F

from utils.volume_sampling import sample_discrete_number, bin_edges, prob_dist_df
from utils.volume_sampling import remove_output_files, run_fpocket, extract_values
from utils.templates import get_one_hot, get_pocket
from utils.templates import create_template_for_pocket_anchor_prediction, create_template_for_pocket_mol,get_anchors_pocket
from utils.sample_frag_size import sample_fragment_size, fragsize_prob_df, bounds
from sampling.rejection_sampling import compute_number_of_clashes

from src.lightning_anchor_gnn import AnchorGNN_pl
from src.lightning import AR_DDPM
from scipy.spatial import distance

from analysis.reconstruct_mol import reconstruct_from_generated
from analysis.vina_docking import VinaDockingTask

from rdkit.Chem import rdmolfiles

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}
CROSSDOCK_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'B':5, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket
vdws = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'B': 1.92, 'Br': 1.85, 'Cl': 1.75, 'P': 1.8, 'I': 1.98, 'F': 1.47}

def generate_mols_for_pocket(n_samples, 
                             num_frags, 
                             pocket_size, 
                             pocket_coords, 
                             pocket_onehot, 
                             anchor_model, 
                             diff_model, 
                             device, 
                             lig_coords=None,
                             return_all=False,
                             max_mol_sizes=None,
                             all_grids=None,
                             rejection_sampling=False,
                             pocket_anchors=None,
                             prot_path=None,
                             rejection_criteria='clash'): # check if the generated fragment improves the vina score or not
    
    # TODO: consider experimenting with random anchors and random number of atoms in fragments
    all_frag_sizes = np.zeros((n_samples, num_frags), dtype=np.int32)
    anchors_context = True
    center_of_mass = 'anchors'
    pocket_inds = pocket_onehot[:,:4].argmax(1).cpu().numpy()
    pocket_atom_types = [idx2atom[a] for a in pocket_inds]

    all_x = []
    all_h = []
    total_grids = []
    all_extension_masks = []
    all_scaffold_masks = []
    all_pocket_masks = []
    all_pocket_x = []
    generation_steps = np.zeros(n_samples, dtype=np.int32)
    steps = np.zeros(n_samples, dtype=np.int32)

    for i in range(num_frags):
        if i == 0:
            print(f'generating fragment {i+1}')
            if lig_coords is not None:
                anchors_pocket = get_anchors_pocket(n_samples=n_samples, pocket_coords=pocket_coords, mol_coords=lig_coords) # using crystallographic ligand for pocket anchor
            else:
                anchors_pocket = pocket_anchors
            # creating a template for computing fragment sizes
            pocket_coords = pocket_coords.repeat(n_samples, 1, 1) # [B, Np, 3]
            # sampler larger fragments for the first fragment to build
            frag_sizes = np.random.choice([10,11,12,13,14,15,16], p=[0.32, 0.21, 0.11, 0.10, 0.12, 0.08, 0.06], size=n_samples)
            print('fragment sizes:', frag_sizes)
            all_frag_sizes[:,i] = frag_sizes
            prev_ext_sizes = frag_sizes
            # -------------------------- generate the fragment ---------------------------
            # making the template for diffusing the first fragment
            #pocket_x = pocket_coords.repeat(n_samples, 1, 1) # [B, Np, 3]
            batch = create_template_for_pocket_mol(pocket_x=pocket_coords,
                                                   pocket_h=pocket_onehot,
                                                   n_samples=n_samples,
                                                   steps=steps, # generation step
                                                   frag_sizes=all_frag_sizes[:,i],
                                                   anchors_scaffold=None,
                                                   anchors_pocket=anchors_pocket)
            
            x = batch['positions'].to(device) # # [B, N, 3]
            h = batch['one_hot'].to(device) # [B, N, 10]
            pocket_x = batch['pocket_coords'].to(device) # [B, Np, 3]
            pocket_h = batch['pocket_onehot'].to(device) # [B, Np, 25]

            B = x.shape[0]
            Ns = x.shape[1]
            Np = pocket_x.shape[1]

            node_masks = batch['node_masks'].to(device)
            extension_masks = batch['extension_masks'].to(device)
            scaffold_masks = batch['scaffold_masks'].to(device)
            anchors = batch['anchors'].to(device)
            pocket_anchors = batch['pocket_anchors'].to(device)
            pocket_masks = batch['pocket_masks'].to(device)

            if center_of_mass == 'anchors':
                anchor_pos = torch.zeros((B, 3), device=x.device)
                row1, col1 = torch.where(pocket_anchors)
                anchor_pos[row1] = pocket_x[row1, col1] # 

                row2, col2 = torch.where(anchors) # these must be just zeros but just in case
                anchor_pos[row2] = x[row2, col2]
                x = x - anchor_pos.unsqueeze(1) * node_masks.unsqueeze(-1)
                pocket_x = pocket_x - anchor_pos.unsqueeze(1) * pocket_masks.unsqueeze(-1)
            
            for l in range(n_samples): # remove the COM for the grids at step 0
                all_grids[l] = all_grids[l] - anchor_pos[l].cpu()

            x, h, chain_0 = diff_model.edm.sample_chain_single_fragment(x=x,
                                                                        h=h,
                                                                        pocket_x=pocket_x,
                                                                        pocket_h=pocket_h,
                                                                        scaffold_mask=scaffold_masks, # first step scaffold mask is pocket-mask
                                                                        extension_mask=extension_masks,
                                                                        anchors=anchors,
                                                                        pocket_mask=pocket_masks,
                                                                        pocket_anchors=pocket_anchors,
                                                                        keep_frames=1,
                                                                        )

            # mask the grids that have been occupied by the generated molecule
            for l in range(n_samples):
                mol_mask = node_masks[l].cpu().bool()
                mol_coords = x[l].cpu()[mol_mask]
                mol_dists = torch.cdist(all_grids[l].float(), mol_coords)
                mask_grids = (mol_dists < 2).any(dim=1) # mask based on 2 A distance
                all_grids[l] = all_grids[l][~mask_grids]

            all_x.append(x)
            all_h.append(h)
            all_extension_masks.append(extension_masks)
            all_scaffold_masks.append(scaffold_masks)
            all_pocket_masks.append(pocket_masks)
            all_pocket_x.append(pocket_x)
            total_grids.append(all_grids)
            steps += 1

        else:
            # -------------------- anchors predictor ----------------------------
            print(f'generating fragment at step {i+1}')
            batch_anchor = create_template_for_pocket_anchor_prediction(n_samples=n_samples,
                                                                        steps=steps,
                                                                        prev_ext_sizes=prev_ext_sizes, # size of the previous fragment 
                                                                        prev_mol_sizes=all_frag_sizes.cumsum(axis=1)[:,i-1], # size of previous molecule (scaffold)
                                                                        pocket_size=pocket_size, # pocket size
                                                                        prev_x=x, # coordinates of previous molecule
                                                                        prev_h=h, # atom types of previous molecule
                                                                        pocket_x=pocket_x, # translated 
                                                                        pocket_h=pocket_onehot.to(device))
            
            scaffold_masks = batch_anchor['scaffold_masks'].to(device)
            scaff_x = batch_anchor['scaff_x'].to(device)
            scaff_h = batch_anchor['scaff_h'].to(device)
            pocket_masks = batch_anchor['pocket_masks'].to(device)
            pocket_x = batch_anchor['pocket_x'].to(device)
            pocket_h = batch_anchor['pocket_h'].to(device)

            B, N_s = scaff_x.shape[0], scaff_x.shape[1]
            anchor_out  = anchor_model.anchor_predictor.forward(mol_x=scaff_x,
                                                                mol_h=scaff_h,
                                                                pocket_x=pocket_x,
                                                                pocket_h=pocket_h,
                                                                node_mask=scaffold_masks,
                                                                pocket_mask=pocket_masks,
                                                                )
            
            anchor_prob = torch.sigmoid(anchor_out).squeeze()
            anchor_prob = anchor_prob * scaffold_masks

            # NOTE: computing the number of grids points within 3A distance for each scaffold atom
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
            
            #anchor_prob_normalized = torch.nn.functional.softmax(new_anchor_prob, dim=1) 
            #anchors_pred = torch.multinomial(anchor_prob_normalized, 1).squeeze().cpu().numpy()
            grid_points_anchor = all_num_grids[np.arange(n_samples), anchors_pred] # number of grid points around each anchor point

            # ---------------------------------------------------------------------------------
            # --------------------------- Sample fragment size -------------------------------
 
            frag_sizes = np.zeros((n_samples), dtype=np.int32)      
            for l in range(n_samples):
                frag_sizes[l] = sample_fragment_size(int(grid_points_anchor[l]), bounds, fragsize_prob_df)
        
            print('Sampled fragsizes', frag_sizes)
            all_frag_sizes[:, i] = frag_sizes
            print('fragment sizes: ', frag_sizes)
            mol_n_atoms = np.sum(all_frag_sizes, axis=1)
            if max_mol_sizes is not None:
                for idx, n in enumerate(max_mol_sizes):
                    if mol_n_atoms[idx] > n+6: # add 6 to account for smaller fragment sizes
                        all_frag_sizes[idx, i] = 0
            print('mol sizes:', mol_n_atoms)
            # -------------------------------------------------------------------------------------------------------
            # -------------------------------- generating fragment --------------------------------------------------
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
                                                   anchors_pocket=None # not needed it for this step but just in case 
                                                )
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
                                                                        )

            ## mask the grids that have been occupied by the generated molecule
            for l in range(n_samples):
                mol_mask = node_masks[l].cpu().bool()
                mol_coords = x[l].cpu()[mol_mask]
                mol_dists = torch.cdist(all_grids[l].float(), mol_coords)
                mask_grids = (mol_dists < 2).any(dim=1) # mask based on 2 A distance
                all_grids[l] = all_grids[l][~mask_grids]  

            if rejection_sampling:        
                prev_ext_sizes = []
                for s in range(len(x)):
                    mask = extension_masks[s].bool()
                    h_mol = h[s].clone()
                    x_mol = x[s].clone()[mask].cpu()

                    atom_inds = h_mol[mask].cpu().numpy().argmax(axis=1)
                    atom_types = [idx2atom[a] for a in atom_inds]
                    atomic_nums = [CROSSDOCK_CHARGES[a] for a in atom_types]
                    pocket_pos = pocket_x[s].clone().cpu()
                    num_atoms = x_mol.shape[0]
                    # TODO: compute number of clashes (only for the new fragment)
                    _, _, _, num_clashes = compute_number_of_clashes(pocket_pos, x_mol, pocket_atom_types, atom_types) 
                    print(num_clashes)

                    reverted = False
                    if num_clashes > 0: # if the vina score has increased 
                        if (num_clashes / num_atoms) < 0.5: 
                            rand_num = np.random.rand(1)[0]
                            if rand_num < 0.5: # accept the new fragment
                                # Revert back to the previous step (previous scaffold)
                                if len(all_x[i-1][s]) <= x.shape[1]: # TODO: going back to previous step for the 
                                    print(f'rejecting the fragment {i} for molecule {s}')
                                    x[s, :len(all_x[i-1][s]), :] = all_x[i-1][s] # this was reverting all x and h
                                    h[s, :len(all_h[i-1][s]), :] = all_h[i-1][s]
                                    pocket_x[s] = all_pocket_x[i-1][s]
                                    all_grids[s] = total_grids[i-1][s]
                                    

                                    x[s, len(all_x[i-1][s]):, :] = 0
                                    h[s, len(all_h[i-1][s]):, :] = 0
                                    all_frag_sizes[s, i] = 0 # set to zero so that when you compute the molecule size, you don't add this fragment

                                    extension_masks[s, :len(all_extension_masks[i-1][s])] = all_extension_masks[i-1][s]
                                    scaffold_masks[s, :len(all_scaffold_masks[i-1][s])] = all_scaffold_masks[i-1][s]

                                    extension_masks[s, len(all_extension_masks[i-1][s]):] = 0
                                    scaffold_masks[s, len(all_scaffold_masks[i-1][s]):] = 0

                                    reverted = True  
                                else:
                                    print('previous fragment had more atoms...')
                                    diff = len(all_x[i-1][s])-x.shape[1] 
                                    x = F.pad(x, (0,0,0, diff))
                                    h = F.pad(h, (0,0,0, diff))
                                    pocket_x[s] = all_pocket_x[i-1][s]
                                    all_grids[s] = total_grids[i-1][s]
                                    all_frag_sizes[s,i] = 0
                                    extension_masks = F.pad(extension_masks, (0, diff))
                                    scaffold_masks = F.pad(scaffold_masks, (0, diff))
                                    #print('extension_masks', extension_masks.shape) 
                                    #print('previous ext mask', all_extension_masks[i-1].shape)
                                    x[s, :len(all_x[i-1][s]), :] = all_x[i-1][s] # this was reverting all x and h
                                    h[s, :len(all_h[i-1][s]), :] = all_h[i-1][s]
                                    extension_masks[s, :len(all_extension_masks[i-1][s])] = all_extension_masks[i-1][s]
                                    scaffold_masks[s, :len(all_scaffold_masks[i-1][s])] = all_scaffold_masks[i-1][s]
                                    reverted = True
                                    
                        else:
                            if len(all_x[i-1][s]) <= x.shape[1]: # TODO: going back to previous step for other case needs padding with zeros (rarely happens)
                                print(f'rejecting the fragment {i} for molecule {s}')
                                x[s, :len(all_x[i-1][s]), :] = all_x[i-1][s]
                                h[s, :len(all_h[i-1][s]), :] = all_h[i-1][s]
                                pocket_x[s] = all_pocket_x[i-1][s]
                                all_grids[s] = total_grids[i-1][s]
                                x[s, len(all_x[i-1][s]):, :] = 0
                                h[s, len(all_h[i-1][s]):, :] = 0
                                all_frag_sizes[s,i] = 0

                                extension_masks[s, :len(all_extension_masks[i-1][s])] = all_extension_masks[i-1][s]
                                scaffold_masks[s, :len(all_scaffold_masks[i-1][s])] = all_scaffold_masks[i-1][s]

                                extension_masks[s, len(all_extension_masks[i-1][s]):] = 0
                                scaffold_masks[s, len(all_scaffold_masks[i-1][s]):] = 0
                                reverted = True
                            else:
                                print('previous fragment had more atoms ... ') 
                                diff = len(all_x[i-1][s])-x.shape[1] 
                                x = F.pad(x, (0,0,0, diff))
                                h = F.pad(h, (0,0,0, diff))
                                pocket_x[s] = all_pocket_x[i-1][s]
                                all_grids[s] = total_grids[i-1][s]
                                all_frag_sizes[s,i] = 0
                                extension_masks = F.pad(extension_masks, (0, diff))
                                scaffold_masks = F.pad(scaffold_masks, (0, diff))
                                x[s, :len(all_x[i-1][s]), :] = all_x[i-1][s] # this was reverting all x and h
                                h[s, :len(all_h[i-1][s]), :] = all_h[i-1][s]
                                extension_masks[s, :len(all_extension_masks[i-1][s])] = all_extension_masks[i-1][s]
                                scaffold_masks[s, :len(all_scaffold_masks[i-1][s])] = all_scaffold_masks[i-1][s]
                                reverted = True
                    
                    if not reverted:
                        generation_steps[s] = i
                        steps[s] += 1

                    prev_ext_sizes.append(all_frag_sizes[s, generation_steps[s]])
            else:
                prev_ext_sizes = all_frag_sizes[:,i]
                steps += 1
              
            # -----------------------------------------------------------------------------------------------------------------
            all_x.append(x)
            all_h.append(h)
            all_extension_masks.append(extension_masks)
            all_scaffold_masks.append(scaffold_masks)
            all_pocket_masks.append(pocket_masks)
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