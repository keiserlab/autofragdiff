import numpy as np
import os
import argparse

import torch
import time

from utils.volume_sampling import sample_discrete_number, bin_edges, prob_dist_df
from utils.volume_sampling import remove_output_files, run_fpocket, extract_values
from utils.templates import get_one_hot, get_pocket
from utils.templates import add_hydrogens, extract_hydrogen_coordinates

from src.lightning_anchor_gnn import AnchorGNN_pl
from src.lightning import AR_DDPM
from src.noise import cosine_beta_schedule
from scipy.spatial import distance
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
from Bio.PDB.Polypeptide import is_aa

from analysis.reconstruct_mol import reconstruct_from_generated
from src.const import prot_mol_lj_rm, CROSSDOCK_LJ_RM

from rdkit.Chem import rdmolfiles
from sampling.sample_mols import generate_mols_for_pocket

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}
CROSSDOCK_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'B':5, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket
amino_acid_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
vdws = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'B': 1.92, 'Br': 1.85, 'Cl': 1.75, 'P': 1.8, 'I': 1.98, 'F': 1.47}

parser = argparse.ArgumentParser()
parser.add_argument('--results-path', type=str, default='results',
                    help='path to save the results ')
parser.add_argument('--pdb', type=str, help='path to the pdb file')
parser.add_argument('--data-path', action='store', type=str, default='/srv/home/mahdi.ghorbani/FragDiff/crossdock',
                        help='path to the test data for generating molecules')
parser.add_argument('--anchor-model', type=str, default='anchor_model.ckpt',
                    help='path to the anchor model. Note that for guidance, the anchor model should incorporate the conditionals')
parser.add_argument('--n-samples', type=int, default=10,
                       help='total number of ligands to generate per pocket')
parser.add_argument('--exp-name', type=str, default='exp-1',
                    help='name of the generation experiment')
parser.add_argument('--diff-model', type=str, default='diff-model.ckpt',
                    help='path to the diffusion model checkpoint')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--clash-guidance', action='store_true', default=False, help='enable clash guidance')
parser.add_argument('--rejection-sampling', action='store_true', default=False, help='enable rejection sampling')
parser.add_argument('--pocket-number', type=int, default=1, help='pocket number for fpocket')


def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def extract_alpha_spheres_coords(pqr_file):
    """
    Extract alpha sphere coordinates from an fpocket .pqr file.

    Args:
    - pqr_file (str): Path to the .pqr file.

    Returns:
    - List[Tuple[float, float, float]]: List of alpha sphere coordinates.
    """
    coordinates = []

    with open(pqr_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Extract x, y, and z coordinates from the line
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coordinates.append((x, y, z))

    return coordinates

def get_pocket(pdbfile, pocket_atom_dict, remove_H=True, ca_only=False):

    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    # find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True):
            pocket_residues.append(residue)

    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]
        
    if ca_only:
        try:
            pocket_one_hot = []
            pocket_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == 'CA':
                        pocket_one_hot.append(np.eye(1, len(amino_acid_dict),
                        amino_acid_dict[three_to_one(res.get_resname())]).squeeze())
                    pocket_coords.append(atom.coord)
            pocket_one_hot = np.stack(pocket_one_hot)
            pocket_coords = np.stack(pocket_coords)
        except KeyError as e:
            raise KeyError(f'{e} not in amino acid dict ({pdbfile})')
    else: 
        full_atoms = np.concatenate([np.array([atom.element for atom in res.get_atoms()]) for res in pocket_residues], axis=0)
        full_coords = np.concatenate([np.array([atom.coord for atom in res.get_atoms()]) for res in pocket_residues], axis=0)
        full_atoms_names = np.concatenate([np.array([atom.get_id() for atom in res.get_atoms()]) for res in pocket_residues], axis=0)
        pocket_AA = np.concatenate([([three_to_one(atom.get_parent().get_resname()) for atom in res.get_atoms()]) for res in pocket_residues], axis=0)
        
        # removing Hs if present
        if remove_H:
            h_mask = full_atoms == 'H'
            full_atoms = full_atoms[~h_mask]
            pocket_coords = full_coords[~h_mask]
            full_atoms_names = full_atoms_names[~h_mask]
            pocket_AA = pocket_AA[~h_mask]
        try:
            pocket_one_hot = []
            for i in range(len(full_atoms)):
                a = full_atoms[i]
                aa = pocket_AA[i]
                atom_onehot = np.eye(1, len(pocket_atom_dict), pocket_atom_dict[a.capitalize()]).squeeze()
                amino_onehot = np.eye(1, len(amino_acid_dict), amino_acid_dict[aa.capitalize()]).squeeze()
                is_backbone = 1 if full_atoms_names[i].capitalize() in ['N','CA','C','O'] else 0
                pocket_one_hot.append(np.concatenate([atom_onehot, amino_onehot, (is_backbone,)]))
                
  
            pocket_one_hot = np.stack(pocket_one_hot)
        except KeyError as e:
            raise KeyError(
            f'{e} not in atom dict ({pdbfile})')

    pocket_one_hot = np.array(pocket_one_hot)
    return pocket_one_hot, pocket_coords

if __name__ == '__main__':
    args = parser.parse_args()
    torch_device = args.device
    anchor_checkpoint = args.anchor_model
    data_path = args.data_path
    diff_model_checkpoint = args.diff_model
    pdb = args.pdb
    n_samples = args.n_samples

    model = AR_DDPM.load_from_checkpoint(diff_model_checkpoint, device=torch_device) # load diffusion model
    model = model.to(torch_device)

    anchor_model = AnchorGNN_pl.load_from_checkpoint(anchor_checkpoint, device=torch_device)
    anchor_model = anchor_model.to(torch_device)

    if not os.path.exists(args.results_path):
        print('creating results directory')
    
    save_dir = args.results_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # pdb of pocket only (no ligand and only maximum 4A around the pocket)
    pocket_onehot, pocket_coords = get_pocket(pdb, pocket_atom_dict, remove_H=True, ca_only=False)

    # use fpocket to identify the protein pocket
    # NOTE: --------------------------
    # fpocket can sometimes give you the wrong pocket, make sure to check the output and visualize the pocket
    
    try:
        if not os.path.exists(pdb[:-4] + '_out'):
            print('running fpocket...')
            run_fpocket(pdb)
        pqr_dir = pdb[:-4] + '_out/pockets/pocket' + str(args.pocket_number) + '_vert.pqr'
        alpha_spheres = np.array(extract_alpha_spheres_coords(pqr_dir))

    except:
        raise ValueError('fpocket failed!')
        #exit()

    add_H = True

    if add_H:
        add_hydrogens(pdb)
        prot_name_with_H = pdb[:-4] + '_H.pdb'

        H_coords = extract_hydrogen_coordinates(prot_name_with_H)
        H_coords = torch.tensor(H_coords).float().to(torch_device)

    # ---------------  make a grid box around the pocket ----------------
    min_coords = pocket_coords.min(axis=0) - 2.5 #
    max_coords = pocket_coords.max(axis=0) + 2.5

    x_range = slice(min_coords[0], max_coords[0] + 1, 1.5) # spheres of radius 1.5 (vdw radius of C)
    y_range = slice(min_coords[1], max_coords[1] + 1, 1.5)
    z_range = slice(min_coords[2], max_coords[2] + 1, 1.5)

    grid = np.mgrid[x_range, y_range, z_range]
    grid_points = grid.reshape(3, -1).T  # This transposes the grid to a list of coordinates

    # make sure the pocket-number is correct and you identified the correct pocket
    pqr_file = pdb[:-4] + '_out/pockets/pocket' + str(args.pocket_number) + '_vert.pqr'
    alpha_spheres = np.array(extract_alpha_spheres_coords(pqr_file))


    distances_spheres = distance.cdist(grid_points, alpha_spheres)
    mask_spheres = (distances_spheres < 3).any(axis=1)
    filtered_alpha_points = grid_points[mask_spheres]

    # remove grid points that are close to the pocket
    pocket_distances = distance.cdist(filtered_alpha_points, pocket_coords)
    mask_pocket = (pocket_distances < 2).any(axis=1)
    grids = filtered_alpha_points[~mask_pocket]

    grids = torch.tensor(grids)
    all_grids = [] # list of grids
    all_H_coords = []
    for i in range(n_samples):
        all_grids.append(grids) 
        all_H_coords.append(H_coords)
    
    pocket_vol = len(grids)
    max_mol_sizes = []
    for i in range(n_samples):
        max_mol_sizes.append(sample_discrete_number(pocket_vol))

    pocket_onehot = torch.tensor(pocket_onehot).float()
    pocket_coords = torch.tensor(pocket_coords).float()
    pocket_size = len(pocket_coords)

    max_mol_sizes = np.array(max_mol_sizes)
    print('maximum molecule sizes', max_mol_sizes)
    
    # NOTE: choose pocket anchors from the pocket atoms that are close to alpha sphere
    alpha_spheres_pocket_distances = distance.cdist(pocket_coords, alpha_spheres)
    possible_pocket_anchors = np.argsort((alpha_spheres_pocket_distances < 4.5).sum(1))[::-1][:7]

    pocket_anchors = np.random.choice(possible_pocket_anchors, size=n_samples, replace=True)
   
    prot_mol_lj_rm = torch.tensor(prot_mol_lj_rm).to(torch_device)
    mol_mol_lj_rm = torch.tensor(CROSSDOCK_LJ_RM).to(torch_device) / 100

    lj_weight_scheduler = cosine_beta_schedule(500, s=0.01, raise_to_power=2)
    weights = 1 - lj_weight_scheduler 
    weights = np.clip(weights, a_min=0.1, a_max=1.)
    x, h, mol_masks = generate_mols_for_pocket(n_samples=n_samples,
                                               num_frags=8,
                                               pocket_size=pocket_size,
                                               pocket_coords=pocket_coords,
                                               pocket_onehot=pocket_onehot,
                                               lig_coords=None,
                                               anchor_model=anchor_model,
                                               diff_model=model,
                                               device=torch_device,
                                               return_all=False,
                                               max_mol_sizes=max_mol_sizes,
                                               all_grids=all_grids,
                                               rejection_sampling=args.rejection_sampling,
                                               pocket_anchors=pocket_anchors,
                                               lj_guidance=args.clash_guidance,
                                               prot_mol_lj_rm=prot_mol_lj_rm,
                                               mol_mol_lj_rm=mol_mol_lj_rm,
                                               all_H_coords=all_H_coords,
                                               guidance_weights=weights,)


    x = x.cpu().numpy()
    h = h.cpu().numpy()
    mol_masks = mol_masks.cpu().cpu().numpy()

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
    save_path = save_dir + '/' + 'pocket_'
    
    # write sdf file of molecules
    with rdmolfiles.SDWriter(save_path + '_mols.sdf') as writer:
        for mol in all_mols:
            if mol:
                writer.write(mol)
        
    np.save(save_path + '_coords.npy', x)
    np.save(save_path + '_onehot.npy', h)
    np.save(save_path + '_mol_masks.npy', mol_masks)