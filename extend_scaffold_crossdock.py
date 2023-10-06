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

from utils.volume_sampling import sample_discrete_number
from utils.volume_sampling import remove_output_files, run_fpocket, extract_values
from utils.templates import get_one_hot, get_pocket
from utils.templates import create_template_for_pocket_anchor_prediction, create_template_for_pocket_mol, \
                          get_anchors_pocket, random_sample_anchors 
from utils.sample_frag_size import sample_fragment_size
from sampling.rejection_sampling import compute_number_of_clashes

from src.lightning_anchor_gnn import AnchorGNN_pl
from src.lightning import AR_DDPM
from scipy.spatial import distance

from analysis.reconstruct_mol import reconstruct_from_generated
from sampling.scaffold_extension import extend_scaffold

from scipy.spatial import distance
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdmolfiles

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}
CROSSDOCK_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'B':5, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket
vdws = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'B': 1.92, 'Br': 1.85, 'Cl': 1.75, 'P': 1.8, 'I': 1.98, 'F': 1.47}

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', action='store', type=str, default='/srv/home/mahdi.ghorbani/FragDiff/crossdock',
                        help='path to the test data for generating molecules')
parser.add_argument('--results-path', type=str, default='results_scaffold',
                    help='path to save the scaffold based optimization')
parser.add_argument('--use-anchor-model', action='store_true', default=False,
                    help='Whether to use an anchor prediction model')
parser.add_argument('--anchor-model', type=str, default='anchor_model.ckpt',
                    help='path to the anchor model. Note that for guidance, the anchor model should incorporate the conditionals')
parser.add_argument('--n-samples', type=int, default=20,
                       help='total number of ligands to generate per pocket')
parser.add_argument('--exp-name', type=str, default='scaff-ext-1',
                    help='name of the generation experiment')
parser.add_argument('--diff-model', type=str, default='diff-model.ckpt',
                    help='path to the diffusion model checkpoint')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--rejection-sampling', action='store_true', default=False, 
                    help='if activated, at each step, it computes the dock score of the molecule. \
                        If the new fragment improves the dock score, it accepts the new fragment, \
                        If the new fragment has a higher dock score, we accept the fragment with 50% probablity')
parser.add_argument('--custom-anchors', nargs='+', type=int,
                    help='custom anchors, e.g 1 2')

if __name__ == '__main__':
    args = parser.parse_args()
    torch_device = args.device
    anchor_checkpoint = args.anchor_model
    data_path = args.data_path
    diff_model_checkpoint = args.diff_model

    # custom anchors
    if args.custom_anchors is not None:
        custom_anchors = args.custom_anchors # list of ints
        custom_anchors = np.array(custom_anchors)
    else:
        custom_anchors = None

    rejection_sampling = args.rejection_sampling
    
    n_samples = args.n_samples

    if rejection_sampling:
        print('Generating molecules based on rejection sampling ...')
    
    model = AR_DDPM.load_from_checkpoint(diff_model_checkpoint, device=torch_device) # load diffusion model
    model = model.to(torch_device)

    if args.use_anchor_model:
        anchor_model = AnchorGNN_pl.load_from_checkpoint(anchor_checkpoint, device=torch_device)
        anchor_model = anchor_model.to(torch_device)
    else:
        anchor_model = None # TODO: implement random anchor selection

    split = torch.load(data_path + '/' + 'split_by_name.pt')
    prefix = data_path + '/crossdocked_pocket10/'

    if not os.path.exists(args.results_path):
        print('creating results directory...')
    
    save_dir = args.results_path + '/' + args.exp_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for n in range(100):
        prot_name = prefix + split['test'][n][0]
        lig_name = prefix + split['test'][n][1]
        mol = Chem.SDMolSupplier(lig_name)[0]
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold.GetNumAtoms() == 0:
            print('No Scaffold found')
            continue
        pocket_onehot, pocket_coords, lig_coords, lig_onehot = get_pocket(prot_name, lig_name, atom_dict, pocket_atom_dict=pocket_atom_dict, dist_cutoff=7)
        
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

        # NOTE: removing grids points around 2A of scaffold
        scaff_pos = scaffold.GetConformer().GetPositions()
        scaff_onehot = []
        for atom in scaffold.GetAtoms():
            atom_type = atom.GetSymbol().capitalize()
            scaff_onehot.append(get_one_hot(atom_type, atom_dict))
        scaff_onehot = np.array(scaff_onehot)
        
        dist_scaff = distance.cdist(grids, scaff_pos)
        mask_scaff = (dist_scaff < 2).any(axis=1)
        grids = grids[~mask_scaff] # remove grid points that are close to the scaffold

        n_samples = 20
        max_mol_sizes = []
        grids = torch.tensor(grids)
        all_grids = [] # list of grids
        for i in range(n_samples):
            all_grids.append(grids) 

        pocket_vol = len(grids)
        max_mol_sizes = []
        for i in range(n_samples):
            max_mol_sizes.append(sample_discrete_number(pocket_vol))

        pocket_onehot = torch.tensor(pocket_onehot).float()
        pocket_coords = torch.tensor(pocket_coords).float()
        x = torch.tensor(scaff_pos).float().unsqueeze(0).repeat(n_samples, 1, 1)
        h = torch.tensor(scaff_onehot).float().unsqueeze(0).repeat(n_samples, 1, 1)
        pocket_size = len(pocket_coords)

        print('custom anchors: ', custom_anchors)
        x, h, mol_masks = extend_scaffold(n_samples=n_samples,
                                            num_frags=4,
                                            x=x,
                                            h=h,
                                            pocket_coords=pocket_coords,
                                            pocket_onehot=pocket_onehot,
                                            anchor_model=anchor_model,
                                            diff_model=model,
                                            device=torch_device,
                                            return_all=False,
                                            prot_path=prot_name, # path to pdb file NOTE: the directory must also contains the pdbqt file of receptor
                                            max_mol_sizes=max_mol_sizes,
                                            custom_anchors=None,
                                            all_grids=all_grids)
                                                

        x = x.cpu().numpy()
        h = h.cpu().numpy()
        mol_masks = mol_masks.cpu().numpy()
        
        t2 = time.time()
        save_path = save_dir + '/' + 'pocket_' + str(n) 

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
        
        with rdmolfiles.SDWriter(save_path + '_mols.sdf') as writer:
            for mol in all_mols:
                if mol:
                    writer.write(mol)
        np.save(save_path + '_coords.npy', x)
        np.save(save_path + '_onehot.npy', h)
        np.save(save_path + '_mol_masks.npy', mol_masks)