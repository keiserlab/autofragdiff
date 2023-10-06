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

from src.lightning_anchor_gnn import AnchorGNN_pl
from src.lightning import AR_DDPM
from scipy.spatial import distance

from analysis.reconstruct_mol import reconstruct_from_generated
from analysis.vina_docking import VinaDockingTask

from rdkit.Chem import rdmolfiles
from sampling.sample_mols import generate_mols_for_pocket

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}
CROSSDOCK_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'B':5, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket

vdws = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'B': 1.92, 'Br': 1.85, 'Cl': 1.75, 'P': 1.8, 'I': 1.98, 'F': 1.47}

parser = argparse.ArgumentParser()
parser.add_argument('--results-path', type=str, default='results',
                    help='path to save the results ')
parser.add_argument('--data-path', action='store', type=str, default='/srv/home/mahdi.ghorbani/FragDiff/crossdock',
                        help='path to the test data for generating molecules')
parser.add_argument('--use-anchor-model', action='store_true', default=False,
                    help='Whether to use an anchor prediction model')
parser.add_argument('--anchor-model', type=str, default='anchor_model.ckpt',
                    help='path to the anchor model. Note that for guidance, the anchor model should incorporate the conditionals')
parser.add_argument('--n-samples', type=int, default=20,
                       help='total number of ligands to generate per pocket')
parser.add_argument('--exp-name', type=str, default='exp-1',
                    help='name of the generation experiment')
parser.add_argument('--diff-model', type=str, default='diff-model.ckpt',
                    help='path to the diffusion model checkpoint')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--rejection-sampling', action='store_true', default=False, help='enable rejection sampling')

if __name__ == '__main__':
    args = parser.parse_args()
    torch_device = args.device
    anchor_checkpoint = args.anchor_model
    data_path = args.data_path
    diff_model_checkpoint = args.diff_model

    model = AR_DDPM.load_from_checkpoint(diff_model_checkpoint, device=torch_device) # load diffusion model
    model = model.to(torch_device)

    if args.use_anchor_model is not None:
        anchor_model = AnchorGNN_pl.load_from_checkpoint(anchor_checkpoint, device=torch_device)
        anchor_model = anchor_model.to(torch_device)
    else:
        anchor_model = None # TODO: implement random anchor selection
    
    split = torch.load(data_path + '/' + 'split_by_name.pt')
    prefix = data_path + '/crossdocked_pocket10/'

    if not os.path.exists(args.results_path):
        print('creating results directory')
    
    save_dir = args.results_path + '/' + args.exp_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for n in range(100):
        prot_name = prefix + split['test'][n][0]
        lig_name = prefix + split['test'][n][1]

        pocket_onehot, pocket_coords, lig_coords, _ = get_pocket(prot_name, lig_name, atom_dict, pocket_atom_dict=pocket_atom_dict, dist_cutoff=7)

        # ---------------  make a grid box around the pocket ----------------
        min_coords = pocket_coords.min(axis=0) - 2.5 # 
        max_coords = pocket_coords.max(axis=0) + 2.5

        x_range = slice(min_coords[0], max_coords[0] + 1, 1.5) # spheres of radius 1.2 (vdw radius of H)
        y_range = slice(min_coords[1], max_coords[1] + 1, 1.5)
        z_range = slice(min_coords[2], max_coords[2] + 1, 1.5)

        grid = np.mgrid[x_range, y_range, z_range]
        grid_points = grid.reshape(3, -1).T  # This transposes the grid to a list of coordinates

        # remove grids points not in 3.5A neighborhood of original ligand
        distances_mol = distance.cdist(grid_points, lig_coords)
        mask_mol = (distances_mol < 3.5).any(axis=1)
        filtered_mol_points = grid_points[mask_mol]

        # remove grid points that are close to the pocket
        pocket_distances = distance.cdist(filtered_mol_points, pocket_coords)
        mask_pocket = (pocket_distances < 2).any(axis=1)
        grids = filtered_mol_points[~mask_pocket]

        n_samples = args.n_samples
        max_mol_sizes = []

        fpocket_out = prot_name[:-4] + '_out'
        shutil.rmtree(fpocket_out, ignore_errors=True)

        #print('running fpocket!')
        #try:
        #    run_fpocket(prot_name)
        #except:
        #    print('Error in running fpocket! using random sizes')

        # NOTE: using original molecule coordinates for making the grid

        grids = torch.tensor(grids)
        all_grids = [] # list of grids
        for i in range(n_samples):
            all_grids.append(grids) 

        pocket_vol = len(grids)
        #if os.path.exists(fpocket_out):
        #    filename = prot_name[:-4] + '_out/pockets/pocket1_atm.pdb'
        #    score, drug_score, pocket_volume = extract_values(filename)
        #else:
        #    print('running fpocket!')
        #    run_fpocket(prot_name)
        #    filename = prot_name[:-4] + '_out/pockets/pocket1_atm.pdb'
        #    score, drug_score, pocket_volume = extract_values(filename)

        #print('pocket_volume', pocket_volume)
        
        for i in range(n_samples):
            max_mol_sizes.append(sample_discrete_number(pocket_vol))

        pocket_onehot = torch.tensor(pocket_onehot).float()
        pocket_coords = torch.tensor(pocket_coords).float()
        lig_coords = torch.tensor(lig_coords).float()
        pocket_size = len(pocket_coords)

        t1 = time.time()

        max_mol_sizes = np.array(max_mol_sizes)
        print('maximum sizes for molecules', max_mol_sizes)
        x, h, mol_masks  = generate_mols_for_pocket(n_samples=n_samples, 
                                                    num_frags=8, 
                                                    pocket_size=pocket_size, 
                                                    pocket_coords=pocket_coords, 
                                                    pocket_onehot=pocket_onehot, 
                                                    lig_coords=lig_coords, 
                                                    anchor_model=anchor_model, 
                                                    diff_model=model, 
                                                    device=torch_device,
                                                    return_all=False,
                                                    prot_path=prot_name,
                                                    max_mol_sizes=max_mol_sizes,
                                                    all_grids=all_grids,
                                                    rejection_sampling=args.rejection_sampling,
                                                    rejection_criteria='clash')
        
        x = x.cpu().numpy()
        h = h.cpu().numpy()
        mol_masks = mol_masks.cpu().cpu().numpy()

        # convert to SDF
        all_mols = []
        for k in range(len(x)):
            mask = mol_masks[k]
            h_mol = h[k]
            x_mol = x[k][mask.astype(np.bool_)]

            atom_inds = h_mol[mask.astype(np.bool_)].argmax(axis=1)
            atom_types = [idx2atom[x] for x in atom_inds]
            atomic_nums = [CROSSDOCK_CHARGES[i] for i in atom_types]

            try:
                mol_rec = reconstruct_from_generated(x_mol.tolist(), atomic_nums)
                all_mols.append(mol_rec)
            except:
                continue

        t2 = time.time()
        print('time to generate one is: ', (t2-t1)/n_samples)
        save_path = save_dir + '/' + 'pocket_' + str(n)
        
        # write sdf file of molecules
        with rdmolfiles.SDWriter(save_path + '_mols.sdf') as writer:
            for mol in all_mols:
                if mol:
                    writer.write(mol)
            
        np.save(save_path + '_coords.npy', x)
        np.save(save_path + '_onehot.npy', h)
        np.save(save_path + '_mol_masks.npy', mol_masks)
