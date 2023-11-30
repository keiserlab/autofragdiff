import numpy as np
import os
from rdkit import Chem
import torch
from pathlib import Path
from analysis.docking import calculate_qvina2_score

from analysis.reconstruct_mol import reconstruct_from_generated
from analysis.metrics import is_connected

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}
CROSSDOCK_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'B':5, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket
vdws = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'B': 1.92, 'Br': 1.85, 'Cl': 1.75, 'P': 1.8, 'I': 1.98, 'F': 1.47}

def compute_number_of_clashes(lig_x, lig_h, pocket_x, pocket_h, pocket_H_coords=None, tolerace=0.5, prot_mol_lj_rm=None):
    """ 
    lig_x and lig_h [n_atoms, 3] and [n_atoms] coordinates and atom types of the ligand (only extension atoms)
    pocket_x, pocket_h => [N_pocket, 3 or hp]
    pocket_H_coords -> [N_pocket, 3] coordinates of the pocket H atoms
    """

    dists = torch.cdist(lig_x, pocket_x, p=2) # [n_lig_atoms, n_pocket_atoms]
    dists = torch.where(dists==0, 1e-5, dists)
    inds_lig = torch.argmax(lig_h, dim=1) # [n_lig_atoms]

    inds_pocket = torch.argmax(pocket_h, dim=1).long() # [n_pocket_atoms]
    rm = prot_mol_lj_rm[inds_lig][:, inds_pocket] # [n_lig_atoms, n_pocket_atoms]
    clashes = ((dists + tolerace ) < rm).sum().item()

    dists_h = torch.cdist(lig_x, pocket_H_coords, p=2)
    inds_h = torch.ones(len(pocket_H_coords), device=lig_x.device).long() * 10
    rm_h = prot_mol_lj_rm[inds_lig][:, inds_h] # [n_lig_atoms, n_pocket_atoms]
    clashes_h = ((dists_h + tolerace ) < rm_h).sum().item()

    total_clashes = clashes + clashes_h
    return total_clashes


def reject_sample(x, h, pocket_x, pocket_h, prot_path=None, rejection_criteria='clashes'):
    # NOTE: x and pocket_x must already be translated to COM
    #  x :torch.Tensor -> [n_atoms, 3] coordiantes of a single molecule
    #  h :list-> [n_atoms] atom types (eg. 'C', 'N') of a single molecule
    atomic_nums = [CROSSDOCK_CHARGES[a] for a in h]
    if rejection_criteria == 'qvina':
        try:
            mol_rec = reconstruct_from_generated(x.tolist(), atomic_nums)
            Chem.SanitizeMol(mol_rec)

            if not is_connected(mol_rec):
                m_frags = Chem.GetMolFrags(mol_rec, asMols=True, sanitizeFrags=False)
                mol_rec = max(m_frags, key=lambda x: x.GetNumAtoms())
            
                prot_pdbqt_file = prot_path[:-4] + '.pdbqt'
                out_sdf_file =  'mol.sdf'
                with Chem.SDWriter(out_sdf_file) as writer:
                    writer.write(mol_rec)
                sdf_file = Path(out_sdf_file)
                if not os.path.exists('qvina-path'):
                    os.mkdir('qvina-path')
                score_result = calculate_qvina2_score(prot_pdbqt_file, sdf_file, out_dir='qvina-path', return_rdmol=False, score_only=True)
                print('qvina score: ', score_result)
                files = os.listdir('qvina-path')
                for file in files:
                    if file.endswith('.sdf') or file.endswith('.pdbqt'):
                        os.remove(os.path.join('qvina-path', file))

        except:
            score_result = 100

        return score_result
    
    elif rejection_criteria == 'clashes':
        # pocket_x -> [n_atoms, 3] coordiantes of a single pocket
        # pocket_h -> [n_atoms] atom types (eg. 'C', 'N') of a single pocket
        # x -> [n_atoms, 3] coordiantes of a single molecule
        clashes, clashed_ids, clashed_pocket_ids, n_clashes = compute_number_of_clashes(pocket_x, x, pocket_h, h)
        return n_clashes

def compute_lj(lig_x, lig_h, extension_mask, scaffold_mask, pocket_x, pocket_h, pocket_mask, prot_mol_lj_rm, all_H_coords, mol_mol_lj_rm=None):
    """ compute the LJ between protein and ligand 
    lig_x: [B, N, 3]
    lig_h: [B, N, hf]
    """

    num_atoms = extension_mask.sum()
    
    #  ------------- ligand - ligand LJ ----------
    mol_mask = (scaffold_mask.bool() | extension_mask.bool())
    N = mol_mask.sum()

    x_mol = lig_x[mol_mask] # [N_mol, 3]
    h_mol = lig_h[mol_mask] # [N_mol, hf]

    x = lig_x[extension_mask.bool()]  
    h = lig_x[extension_mask.bool()] 

    dists_mol = torch.cdist(x, x_mol, p=2) # [N_ext, N_mol]
    
    inds_mol = torch.argmax(h_mol, dim=1) # [N_mol]
    inds_ext = torch.argmax(h, dim=1) # [N_ext]
    rm_mol = mol_mol_lj_rm[inds_ext][:, inds_mol] # [N_ext, N_mol]
    

    dists_mol = torch.where(dists_mol==0.0, 1, dists_mol)
    rm_mol = torch.where(rm_mol==0.0, 1, rm_mol)


    dists_mol = torch.where(dists_mol < 0.5, 0.5, dists_mol) # clamp the distance to 0.1
    lj_mol = ((rm_mol / dists_mol) ** 12 - (rm_mol / dists_mol) ** 6) # [N_mol, N_mol]

    lj_lig_lig = lj_mol.sum() / num_atoms

    # --------------- compute the LJ between protein and ligand --------------


    pocket_x = pocket_x[pocket_mask.bool()] # [N_p, 3]
    pocket_h = pocket_h[pocket_mask.bool()][:, :4] # [N_p, hf]
    h_coords = all_H_coords # [N_p, 3]

    # --------------- compute the LJ between protein and ligand --------------
    dists = torch.cdist(x, pocket_x, p=2)
    inds_lig = torch.argmax(h, dim=1) # [N_l]
    inds_pocket = torch.argmax(pocket_h, dim=1).long() # [N_p]

    rm = prot_mol_lj_rm[inds_lig][:, inds_pocket] # [N_l, N_p] 
    lj = ((rm / dists) ** 12 - (rm / dists) ** 6) # [N_l, N_p]
    lj[torch.isnan(lj)] = 0

    # -------------  compute the loss for h atoms ----------------
    dists_h = torch.cdist(x, h_coords, p=2)
    #dists_h = torch.where(dists_h<0.5, 0.5, dists_h)
    inds_H = torch.ones(len(h_coords), device=x.device).long() * 10 # index of H is 10 in the table
    rm_h = prot_mol_lj_rm[inds_lig][:, inds_H]
    lj_h = ((rm_h / dists_h) ** 12 - (rm_h / dists_h) ** 6) # [N_l, N_p]
    
    lj_h[torch.isnan(lj_h)] = 0 # remove nan values

    lj = lj.sum() 
    lj_h = lj_h.sum()

    lj_prot_lig = (lj + lj_h) / num_atoms
    return lj_prot_lig, lj_lig_lig 