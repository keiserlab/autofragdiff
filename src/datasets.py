
import numpy as np
import torch
import pandas as pd
import os

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from src import const

def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

class HierCrossDockDataset(Dataset):
    """
    atom features: [atom types (10), hybridization (3), aromaticity (1)]
    pocket features [atom types (4), amino acid type (20), backbone or not (1)]
    """
    def __init__(self, data_path, prefix, device, dataframe_path, max_num_steps=8):
        dataset_path = os.path.join(data_path, f'{prefix}.pt')
        self.max_num_steps = max_num_steps
        
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = self.preprocess(dataframe_path, device)
            torch.save(self.data, dataset_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
    
    def preprocess(self, dataframe_path, device):
        data = []
        
        table = pd.read_csv(dataframe_path)
        generator = tqdm(table.iterrows(), total=len(table))
        for n, row in generator:  

            mol_positions = np.load(row['mol_conf'])
            mol_one_hot = np.load(row['mol_onehot'])
            anchors = np.load(row['mol_anchorIds'], allow_pickle=True)
            hier_atom_ids = np.load(row['mol_scaffoldIds'], allow_pickle=True)
            charges = np.load(row['mol_charges'])
            extension_ids = np.load(row['mol_extensionIds'], allow_pickle=True)
            vina_scores = np.load(row['vina'], allow_pickle=True)
    
            print(len(vina_scores))
            
            # TODO: all hier_atom_ids must be collated to the same length (maximum length of 8) and excessive elements could be replaced with something

            pocket_coords = np.load(row['pocket_coords'])
            pocket_onehot = np.load(row['pocket_onehot'])
            is_single_frag = row['is_singe_frag']
            
            if is_single_frag == True:

                hier_masks = np.zeros((1, len(mol_positions), self.max_num_steps))
                extension_masks = np.zeros((1, len(mol_positions), self.max_num_steps))
                anchor_ids = np.zeros((1, len(mol_positions), self.max_num_steps))
                extension_masks[0, :, 0] = np.ones(len(mol_positions))
                n_frags = 1

            elif is_single_frag == False:

                if len(hier_atom_ids) == 0:
                    continue

                if hier_atom_ids.shape[1] > self.max_num_steps:
                    print('more than maximum fragments')
                    continue

                n_frags = np.array(hier_atom_ids.shape[1])
                hier_masks = np.zeros((len(hier_atom_ids), self.max_num_steps, len(mol_positions)))
                extension_masks = np.zeros((len(hier_atom_ids), self.max_num_steps, len(mol_positions)))
                anchors_ids = np.zeros((len(hier_atom_ids), self.max_num_steps, len(mol_positions)))
                vina_scores = vina_scores.reshape((n_frags*2, n_frags))
                anchors = anchors.astype(np.int32)

                for k in range(len(hier_atom_ids)):
                    hier_ids = hier_atom_ids[k]
                    ext_ids = extension_ids[k]
                    for i in range(self.max_num_steps):
                        if i < len(hier_atom_ids[k]):
                            if i > 0:
                                this_row = list(hier_ids[i-1])
                                hier_masks[k][i][this_row] = 1.
                            this_row = list(ext_ids[i])
                            extension_masks[k][i][this_row] = 1.

                            if i == 0:
                                continue
                            else:
                                anchors_ids[k][i][anchors[k][i-1]] = 1.
                        else:
                            hier_masks[k][i] = 1.
                            extension_masks[k][i] = 0.

                hier_masks = hier_masks.swapaxes(2,1)
                extension_masks = extension_masks.swapaxes(2,1)
                anchors_ids = anchors_ids.swapaxes(2,1)

            data.append({
                'positions': torch.tensor(mol_positions, dtype=torch.float32),
                'one_hot': torch.tensor(mol_one_hot, dtype=torch.torch.float32),
                'charges': torch.tensor(charges, dtype=torch.float32),
                'scaffold_masks': torch.tensor(hier_masks, dtype=torch.float32),
                'extension_masks': torch.tensor(extension_masks, dtype=torch.float32),
                'anchors': torch.tensor(anchors_ids, dtype=torch.float32),
                'n_frags': torch.tensor(n_frags, dtype=torch.int32),
                'pocket_coords': torch.tensor(pocket_coords, dtype=torch.float32),
                'pocket_onehot': torch.tensor(pocket_onehot, dtype=torch.float32),
                'vina_scores': torch.tensor(vina_scores, dtype=torch.float32)
            })
        return data

def collate_pocket(batch):
    """ collate function handles sampling from permutation order and fragment step 
    This gets the vina score for each fragment in the molecule and guidance is based on each fragment not the whole molecule
    #NOTE: I have experimented on adding extra features to ligand but they actually lowered the quality of molecules
         there was disconnected and weired molecules as a result
    """
    out = {}
    # for the main data 
    for key in ['scaffold_masks', 'extension_masks', 'positions', 'one_hot', 'pocket_coords', 'pocket_onehot', 'anchors', 'pocket_anchors', 'pocket_masks', 'atom_mask', 'edge_mask', 'n_frags', 'first_frag', 'frag_sizes', 'rand_perm', 'rand_scaff_n', 'mol_sizes', 'pocket_sizes']:
        out.setdefault(key, [])
        
    max_hier_steps = 8
    for i, data in enumerate(batch):
        n_frags = int(data['n_frags'])
        n_perms = len(data['scaffold_masks'])
        rand_perm = torch.randint(0, n_perms, (1,)).squeeze() # sampling one permutation order
        low = 0
        high = n_frags - 1
        if n_frags == 1:
            rand_scaff_n = 0
        else:
            rand_scaff_n = torch.randint(low, high, (1,)).squeeze() # sample scaffold and extension (more than 50% of fragments are scaffold)
        
        scaff_mask = data['scaffold_masks'][rand_perm][:,rand_scaff_n]
        ext_mask = data['extension_masks'][rand_perm][:,rand_scaff_n]

        frag_size = int(ext_mask.sum())

        x = data['positions'] # molecule only
        h = data['one_hot'][:,:10] # only use the atom types
        
        pocket_x = data['pocket_coords']
        pocket_h = data['pocket_onehot']
        
        #if n_frags != 1:
        anchor_old = data['anchors'][rand_perm][:,rand_scaff_n]
        anchor_id = torch.where(anchor_old)[0]
        
        x_ext = x[ext_mask.bool()]
        h_ext = h[ext_mask.bool()]
        x_scaff = x[scaff_mask.bool()]
        h_scaff = h[scaff_mask.bool()]

        scaff_mask_new = torch.concat([torch.ones(len(x_scaff)), torch.zeros(len(x_ext))])
        ext_mask_new = torch.concat([torch.zeros(len(x_scaff)), torch.ones(len(x_ext))])
        
        pocket_mask = torch.ones(len(pocket_x))
        pocket_anchors = torch.zeros_like(pocket_mask)
        
        anchor = torch.zeros_like(scaff_mask_new, device=x.device)
        
        if n_frags != 1:
            new_anchor_id = torch.cumsum(scaff_mask - anchor_old, dim=0)[anchor_id]
        
        h_new = torch.concat([h_scaff, h_ext], dim=0)
        x_new = torch.concat([x_scaff, x_ext], dim=0)

        mol_size = len(x_scaff) + len(x_ext)
        pocket_size = len(pocket_x)
        
        if rand_scaff_n == 0:
            # we have to choose from pocket atoms during the training 
            out['first_frag'].append(1)
            dists = pairwise_distances(pocket_x, x_ext.mean(dim=0).reshape(1,3))
            top_k_inds = top_k_lowest_indices(dists, k=1)
            pocket_anchors[top_k_inds] = 1
        else:
            out['first_frag'].append(0)
            if len(new_anchor_id) > 0:
                anchor[int(new_anchor_id)] = 1
        # --------------------------------------------------------------
        out['scaffold_masks'].append(scaff_mask_new)
        out['extension_masks'].append(ext_mask_new)
        out['positions'].append(x_new)
        out['one_hot'].append(h_new)
        out['anchors'].append(anchor)
        out['frag_sizes'].append(frag_size)
        out['n_frags'].append(n_frags)
        out['pocket_masks'].append(pocket_mask)
        out['pocket_anchors'].append(pocket_anchors)
        out['pocket_coords'].append(pocket_x)
        out['pocket_onehot'].append(pocket_h)
        out['rand_perm'].append(rand_perm)
        out['rand_scaff_n'].append(rand_scaff_n)
        out['mol_sizes'].append(mol_size)
        out['pocket_sizes'].append(pocket_size)

    for key, value in out.items():
        
        if key in ['anchors', 'pocket_anchors', 'extension_masks', 'scaffold_masks', 'positions', 'one_hot', 'pocket_coords', 'pocket_onehot',  'pocket_masks']:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)

    atom_mask = (out['scaffold_masks'].bool() | out['extension_masks'].bool()).to(torch.int32)
    out['atom_mask'] = atom_mask
    
    batch_size, n_nodes = atom_mask.size()
    pocket_nodes = out['pocket_masks'].shape[1]
    
    #------------------------ batch-mask on pocket-ligand atoms for each batch ----------------------
    batch_mask = torch.cat([
        torch.ones(n_nodes+pocket_nodes, dtype=const.TORCH_INT) * i
        for i in range(batch_size)
    ]).to(atom_mask.device)
    out['batch_mask'] = batch_mask
    # --------------------------------------------------------------------------------
    out['n_frags'] = torch.tensor(out['n_frags'])
    out['first_frag'] = torch.tensor(out['first_frag'])
    out['rand_perm'] = torch.tensor(out['rand_perm'])
    out['rand_scaff_n'] = torch.tensor(out['rand_scaff_n'])
    out['mol_sizes'] = torch.from_numpy(np.array(out['mol_sizes']))
    out['pocket_sizes'] = torch.from_numpy(np.array(out['pocket_sizes']))
    return out

def collate_pocket_aux(batch):
    """ collate function handles sampling from permutation order and fragment step"""
    out = {} 
    # for the anchor predictor
    for key in ['position_aux', 'onehot_aux', 'scaffold_masks_aux', 'anchors_aux', 'scaffold_masks_aux', 'edge_mask_an', 'pocket_mask_aux', 'n_frags', 'first_frag', 'frag_sizes', 'pocket_anchors_aux', 'pocket_coords_aux', 'pocket_onehot_aux', 'rand_perm', 'rand_scaff_n']:
        out.setdefault(key, [])
    
    max_hier_steps = 8
    for i, data in enumerate(batch):
        n_frags = int(data['n_frags'])
        n_perms = len(data['scaffold_masks'])
        rand_perm = torch.randint(0, n_perms, (1,)).squeeze() # sampling one permutation order
        low = 0
        high = n_frags - 1
        if n_frags == 1:
            rand_scaff_n = 0
        else:
            rand_scaff_n = torch.randint(low, high, (1,)).squeeze() # sample scaffold and extension (more than 50% of fragments are scaffold)

        scaff_mask = data['scaffold_masks'][rand_perm][:,rand_scaff_n]
        ext_mask = data['extension_masks'][rand_perm][:,rand_scaff_n]

        frag_size = int(ext_mask.sum())
        x = data['positions'] # molecule only
        h = data['one_hot'][:,:10] # molecule only
        
        pocket_x = data['pocket_coords'] # pocket x
        pocket_h = data['pocket_onehot'] # pocket oenhot
        
        #if n_frags != 1:
        anchor_old = data['anchors'][rand_perm][:,rand_scaff_n] # anchor of the scaffold
        anchor_id = torch.where(anchor_old)[0] # id of scaffold
        
        x_ext = x[ext_mask.bool()] # x of extension
        x_scaff = x[scaff_mask.bool()] # x of scaffold
        h_scaff = h[scaff_mask.bool()] # h of scaffold

        scaffold_mask_aux = torch.ones(len(x_scaff)) # only scaffold atoms
        pocket_mask_aux =  torch.ones(len(pocket_x)) # only pocket atoms
        pocket_anchors = torch.zeros_like(pocket_mask_aux) # only pocket anchors
        node_mask = torch.ones(len(x))

        if n_frags != 1:
            new_anchor_id = torch.cumsum(scaff_mask - anchor_old, dim=0)[anchor_id]
        
        #if n_frags != 1:
        scaffold_anchor = torch.zeros(len(x_scaff))
            
        if rand_scaff_n == 0:
            out['first_frag'].append(1)
            dists = pairwise_distances(pocket_x, x_ext.mean(dim=0).reshape(1,3))
            top_k_inds = top_k_lowest_indices(dists, k=1)
            pocket_anchors[top_k_inds] = 1 # predicting the pocket atom as the anchor for scaffold (anchor prediction model)
            #anchor[top_k_inds] = 1
        else:
            out['first_frag'].append(0)
            if len(new_anchor_id) > 0:
                #anchor[int(new_anchor_id)] = 1
                scaffold_anchor[int(new_anchor_id)] = 1
        # --------------------------------------------------------------
        out['frag_sizes'].append(frag_size)
        out['n_frags'].append(n_frags)

        # ---------------- for anchor and size prediction ----------------
        out['position_aux'].append(x_scaff)
        out['onehot_aux'].append(h_scaff)
        out['scaffold_masks_aux'].append(scaffold_mask_aux)
        out['pocket_mask_aux'].append(pocket_mask_aux)
        out['anchors_aux'].append(scaffold_anchor)
        out['pocket_anchors_aux'].append(pocket_anchors)
        out['pocket_coords_aux'].append(pocket_x)
        out['pocket_onehot_aux'].append(pocket_h)
        out['rand_perm'].append(rand_perm)
        out['rand_scaff_n'].append(rand_scaff_n)
        # -----------------------------------------------------------------

    for key, value in out.items():
        
        if key in ['position_aux', 'onehot_aux', 'scaffold_masks_aux', 'anchors_aux', 'pocket_mask_aux', 'pocket_anchors_aux', 'pocket_coords_aux', 'pocket_onehot_aux']:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
    
    out['n_frags'] = torch.tensor(out['n_frags'])
    out['first_frag'] = torch.tensor(out['first_frag'])
    out['frag_sizes'] = torch.tensor(out['frag_sizes'])
    out['rand_perm'] = torch.tensor(out['rand_perm'])
    out['rand_scaff_n'] = torch.tensor(out['rand_scaff_n'])

    scaff_node_mask = out['scaffold_masks_aux']
    batch_size_scaff, n_nodes_scaff = scaff_node_mask.size()
    pocket_nodes = out['pocket_mask_aux'].shape[1]

    batch_mask_scaff = torch.cat([
        torch.ones(n_nodes_scaff + pocket_nodes, dtype=const.TORCH_INT) * i
        for i in range(batch_size_scaff)
    ]).to(scaff_node_mask.device)
    out['batch_mask_aux'] = batch_mask_scaff
    return out

def get_dataloader(dataset, batch_size, collate_fn=collate_pocket, num_workers=0, shuffle=False):
    return DataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=shuffle, drop_last=True)

def pairwise_distances(x, y):
    # Calculate pairwise distances between two sets of coordinates
    dists = torch.cdist(x, y, p=2)
    return dists

def top_k_lowest_indices(dists, k=5):
    # Find the minimum distance for each coordinate in the first matrix
    min_dists, _ = torch.min(dists, dim=1)
    # Find the indices of the top k elements with the lowest distance
    top_k_indices = torch.topk(min_dists, k, largest=False, sorted=True).indices
    return top_k_indices