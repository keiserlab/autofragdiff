import numpy as np
import pandas as pd
import os

from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

import torch
from src import const

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
from rdkit import Chem

amino_acid_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket
atom_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}

def add_hydrogens(pdf_file):
    """
    Add hydrogens to a PDB file using reduce.
    """
    #print('adding hydrogens')
    out_pdb = pdf_file[:-4] + '_H.pdb'
    os.system(f'reduce -Quiet -NOFLIP {pdf_file} > {out_pdb}')

def extract_hydrogen_coordinates(pdb_file):
    coordinates = []

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name.startswith('H'):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coordinates.append([x, y, z])

    return np.array(coordinates)

def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def get_pocket(pdbfile, sdffile, atom_dict, pocket_atom_dict, dist_cutoff, remove_H=True, ca_only=False):

    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')
    
    # remove H atom if not in atom_dict, other atom types taht aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms()
                 if (a.GetSymbol().capitalize() in atom_dict or a.element !='H')]
    lig_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx))
                           for idx in range(ligand.GetNumAtoms())])

    # find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1)**0.5).min() < dist_cutoff:
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
            raise KeyError(f'{e} not in amino acid dict ({pdbfile}, {sdffile})')
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

    lig_onehot = []
    for atom in lig_atoms:
        lig_onehot.append(get_one_hot(atom, atom_dict))

    pocket_one_hot = np.array(pocket_one_hot)
    return pocket_one_hot, pocket_coords, lig_coords, lig_onehot

def create_template_for_pocket_anchor_prediction(n_samples, # batch size 
                                                 steps, # the fragment generation step 
                                                 prev_ext_sizes=None, # size of previous extension
                                                 pocket_size=None, # size of pocket
                                                 prev_mol_sizes=None,  # size of the molecule generated so far
                                                 prev_x=None,  # previous molecule coords
                                                 prev_h=None,  # previous molecule onehot
                                                 pocket_x=None,  # pocket coords (Tensor) [B, Np, 3]
                                                 pocket_h=None, # pocket onehot (Tensor) [B, Np, hp]
                                                ):

    decoupled_data = []
    for i in range(n_samples):

        data_dict = {}
        scaff_size = prev_mol_sizes[i] # the total number of atoms in the previous molecule
        prev_ext_size = prev_ext_sizes[i] # number of atoms in the previous extension (fragment)
        prev_scaff_size = scaff_size - prev_ext_size 
        scaffold_masks = np.ones(scaff_size)
        pocket_masks = np.ones(pocket_size)
        extension_masks = np.zeros(scaff_size) # just put it here for collation not using it

        x = np.zeros((scaff_size, 3))
        h = np.zeros((scaff_size, 10))
        
        if steps[i] == 1:
            # for the previous extension
            x = prev_x[i, :scaff_size, :].cpu().numpy()
            h = prev_h[i, :scaff_size, :].cpu().numpy() 
        else:
            # prev scaff
            x[:prev_scaff_size] = prev_x[i, :prev_scaff_size, :].cpu().numpy()
            h[:prev_scaff_size] = prev_h[i, :prev_scaff_size, :].cpu().numpy()

            # prev ext
            x[prev_scaff_size:prev_scaff_size+prev_ext_size] = prev_x[i, prev_scaff_size:prev_scaff_size+prev_ext_size, :].cpu().numpy()
            h[prev_scaff_size:prev_scaff_size+prev_ext_size] = prev_h[i, prev_scaff_size:prev_scaff_size+prev_ext_size, :].cpu().numpy()
            
        data_dict['scaff_x'] = torch.tensor(x, dtype=torch.float32)
        data_dict['scaff_h'] = torch.tensor(h, dtype=torch.float32)
        data_dict['pocket_x'] = pocket_x[i]
        data_dict['pocket_h'] = pocket_h
        data_dict['extension_masks'] = torch.tensor(extension_masks, dtype=torch.float32)
        data_dict['scaffold_masks'] = torch.tensor(scaffold_masks, dtype=torch.float32)
        data_dict['pocket_masks'] = torch.tensor(pocket_masks, dtype=torch.float32)
        data_dict['pocket_size'] = pocket_size
        
        decoupled_data.append(data_dict)
        
    return collate_template(decoupled_data)

def create_template_for_pocket_mol(pocket_x,  # pocket coords [B, Np, 3]
                                   pocket_h,  # pocket onehot [B, Np, hp]
                                   n_samples, # batch size
                                   steps, # step at which to generate the fragment
                                   frag_sizes,  # size of fragments
                                   prev_ext_sizes=None,  # previous extension sizes
                                   prev_mol_sizes=None,  # previous mol sizes
                                   prev_x=None,  # previous coords
                                   prev_h=None,  # previous onehot
                                   anchors_scaffold=None,  # anchors on the scaffold
                                   anchors_pocket=None, # anchors on the pocket atoms (for step=0)
                                    ):

    decoupled_data = []
    pocket_size = pocket_x.shape[1]

    for i in range(n_samples):
        if steps[i] == 0:
            data_dict = {}
            num_atoms = frag_sizes[i] # number of atoms for the fragment to generate at current step
            
            extension_masks = np.ones(num_atoms)
            scaffold_masks = np.zeros(num_atoms)
            pocket_masks = np.ones(pocket_size)
            
            scaffold_anchors = np.zeros_like(extension_masks)
            pocket_anchors = np.zeros_like(pocket_masks)

            pocket_anchors[anchors_pocket[i]] = 1 
            
            positions = np.zeros((num_atoms, 3))
            one_hot = np.zeros((num_atoms, 10))
            
            data_dict['positions'] = torch.tensor(positions, dtype=torch.float32)
            data_dict['one_hot'] = torch.tensor(one_hot, dtype=torch.float32)
            data_dict['extension_masks'] = torch.tensor(extension_masks, dtype=torch.float32)
            data_dict['scaffold_masks'] = torch.tensor(scaffold_masks, dtype=torch.float32)
            data_dict['anchors'] = torch.tensor(scaffold_anchors, dtype=torch.float32)
            data_dict['pocket_masks'] = torch.tensor(pocket_masks, dtype=torch.float32)
            data_dict['pocket_size'] = torch.tensor(len(pocket_x), dtype=torch.int32)
            data_dict['pocket_coords'] = pocket_x[i]
            data_dict['pocket_onehot'] = pocket_h
            data_dict['pocket_anchors'] = torch.tensor(pocket_anchors, dtype=torch.float32)
            decoupled_data.append(data_dict)
        else:

            data_dict = {}
            num_atoms = frag_sizes[i] # extension number of atoms at current step
            scaff_size = prev_mol_sizes[i] # the total number of atoms in the previous molecule
            prev_ext_size = prev_ext_sizes[i]
            prev_scaff_size = scaff_size - prev_ext_size
            
            extension_masks = np.concatenate([np.zeros(scaff_size), np.ones(num_atoms)])
            scaffold_masks = np.concatenate([np.ones(scaff_size), np.zeros(num_atoms)])
            pocket_masks = np.ones(pocket_size)
 
            anchor_mask = np.zeros_like(extension_masks)
            anchor_mask[anchors_scaffold[i]] = 1

            pocket_anchors = np.zeros_like(pocket_masks) # these are zero for i!=0
            
            positions = np.zeros((num_atoms + scaff_size, 3))
            one_hot = np.zeros((num_atoms + scaff_size, 10))

            if steps[i] == 1:
                # for the previous extension 
                positions[:scaff_size] = prev_x[i, :scaff_size, :].cpu().numpy()
                one_hot[:scaff_size] = prev_h[i, :scaff_size, :].cpu().numpy() 
             
            else:
                # prev scaff
                positions[:prev_scaff_size] = prev_x[i, :prev_scaff_size, :].cpu().numpy()
                one_hot[:prev_scaff_size] = prev_h[i, :prev_scaff_size, :].cpu().numpy()
                
                # prev ext
                positions[prev_scaff_size:prev_scaff_size+prev_ext_size] = prev_x[i, prev_scaff_size:prev_scaff_size+prev_ext_size, :].cpu().numpy()
                one_hot[prev_scaff_size:prev_scaff_size+prev_ext_size] = prev_h[i, prev_scaff_size:prev_scaff_size+prev_ext_size, :].cpu().numpy()
                
            data_dict['positions'] = torch.tensor(positions, dtype=torch.float32)
            data_dict['one_hot'] = torch.tensor(one_hot, dtype=torch.float32)
            data_dict['extension_masks'] = torch.tensor(extension_masks, dtype=torch.float32)
            data_dict['scaffold_masks'] = torch.tensor(scaffold_masks, dtype=torch.float32)
            data_dict['anchors'] = torch.tensor(anchor_mask, dtype=torch.float32)
            data_dict['pocket_anchors'] = torch.tensor(pocket_anchors, dtype=torch.float32)
            data_dict['pocket_coords'] = pocket_x[i]
            data_dict['pocket_onehot'] = pocket_h
            data_dict['pocket_masks'] = torch.tensor(pocket_masks, dtype=torch.float32)
            data_dict['pocket_size'] = pocket_size
            
            decoupled_data.append(data_dict)
        
    return collate_template(decoupled_data)
    
def collate_template(batch):
    out = {}
    
    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)
    
    for key, value in out.items():
        if key != 'pocket_size':
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
    
    atom_mask = (out['scaffold_masks'].bool() | out['extension_masks'].bool()).to(torch.int32)
    out['node_masks'] = atom_mask

    batch_size, n_nodes = atom_mask.size()
    pocket_size = out['pocket_masks'].shape[1]
    #------------------------ note----------------------
    batch_mask = torch.cat([
        torch.ones(n_nodes + pocket_size, dtype=const.TORCH_INT) * i
        for i in range(batch_size)
    ]).to(atom_mask.device)
    out['batch_masks'] = batch_mask
    # -----------------------------------------------
    return out


def pairwise_distances(x, y):
    # Calculate pairwise distances between two sets of coordinates
    dists = torch.cdist(x, y, p=2)
    return dists

def top_k_lowest_indices(pocket_coords, lig_coords, k=7):
    # Find the minimum distance for each coordinate in the first matrix
    dists = ((pocket_coords - lig_coords.mean(0))**2).sum(dim=1)
    
    sorted_dists, sorted_inds = torch.sort(dists)
    return sorted_inds[:k]

def get_anchors_pocket(n_samples, pocket_coords, mol_coords):
    anchors = []
    #dists = pairwise_distances(pocket_coords, mol_coords)
    top_k_inds = top_k_lowest_indices(pocket_coords, mol_coords, k=5).cpu().numpy() # top 5 closest atoms

    for i in range(n_samples):   
        anchor = np.random.choice(top_k_inds)
        anchors.append(anchor)
    return np.array(anchors)

def random_sample_anchors(n_samples, num_atoms_scaffold, step):
    # randomly samples anchors from the scaffold
    anchors = []
    for i in range(n_samples):
        possible_anchors = list(np.arange(num_atoms_scaffold[i, step]))
        anchor_id = np.random.choice(possible_anchors, replace=False)
        anchors.append(anchor_id)
    return np.array(anchors)