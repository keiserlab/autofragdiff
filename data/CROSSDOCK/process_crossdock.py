
from process_pockets import process_pocket
from process_ligands import get_one_hot, process_ligand

from pathlib import Path
from time import time
import argparse
import shutil
import random 
import matplotlib.pyplot as plt
import seaborn as sns 

from tqdm import tqdm 
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import os
import pandas as pd
from rdkit import RDLogger
import re
import glob
import time
import json
import shutil

from joblib import Parallel, delayed

affinity_pattern = r"Affinity:\s+(-?\d+\.\d+)\s+\(kcal/mol\)"

def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'obabel {sdf_file} -O {pdbqt_outfile} -f {0} -l {mol_id} -m').read()
    return pdbqt_outfile

def get_vina_score(receptor_pdbqt_file, ligand_pdbqt_file, cx, cy, cz, size):
    out = os.popen(
    f'qvina2.1 --score_only --receptor {receptor_pdbqt_file} '
    f'--ligand {ligand_pdbqt_file} '
    f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
    f'--size_x {size} --size_y {size} --size_z {size} ' 
    ).read()
    match = re.search(affinity_pattern, out)
    affinity_value = float(match.group(1))
    return affinity_value

def process_vina_iteration(n, mol_frags_sdf_path, receptor_pdbqt_file, mol_pos, size):
    pdbqt_file = mol_frags_sdf_path[:-4] + '_' + str(n+1) + '.pdbqt'
    cx, cy, cz = mol_pos.mean(0)
    affinity_value = get_vina_score(receptor_pdbqt_file, pdbqt_file, cx, cy, cz, size)
    return affinity_value

RDLogger.DisableLog('rdApp.*')                                                                                                                                                           

amino_acid_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}

pocket_atom_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, required=True, help='/srv/home/mahdi.ghorbani/FragDiff/crossdock/') # base
    parser.add_argument('--outdir', type=str, required=True, help='output directory path')
    parser.add_argument('--no_H', action='store_true')
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--dist_cutoff', type=float, default=7.)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--max-num-frags', type=int, default=12, help='maximum number of fragments')
    parser.add_argument('--split', type=str, default='test', help='split of the data for making the dataset')
    parser.add_argument('--max-atoms-single-fragment', type=int, default=24, help='maximum number of atoms allowed in a fragment')
    parser.add_argument('--add-Vina-score', action='store_true', default=False, help='Whether to add Vina scores for each fragment')
    parser.add_argument('--add-QED-score', action='store_true', default=False, help='adding QED scores')
    parser.add_argument('--add-SA-score', action='store_true', default=False, help='adding SA scores')
    parser.add_argument('--n-cores', type=int, default=32, help='number of cores for multiprocessing')
    parser.add_argument('--get-pocket-info', action='store_true', default=False, help='uses fpocket to get pocket info (volume, SASA, pocket score, ...')
    args = parser.parse_args()

    dist_cutoff = args.dist_cutoff
    ca_only = args.ca_only
    split = args.split
    n_cores = args.n_cores

    add_vina_score = True if args.add_Vina_score else False
    add_QED_score = True if args.add_QED_score else False
    add_SA_score = True if args.add_SA_score else False
    get_pocket_info = True if args.get_pocket_info else False

    print('add Vina', add_vina_score)

    # TODO: add ca_only to suffix
    suffix = split
    
    # Read data split
    split_path = Path(args.rootdir, 'split_by_name.pt')
    data_split = torch.load(split_path)

    # There is no validation set, copy 300 training examples (the validation set
    # is not very important in this application)
    # Note: before we had a data leak but it should not matter too much as most
    # metrics monitored during training are independent of the pockets
    if split == 'val':
        #data_split['val'] = random.sample(data_split['train'], 1000)
        data_split['val'] = data_split['train'][:500]
        
    #data_split['val'] = data_split['train'][:1000]
    all_pocket_coords_paths = []
    all_pocket_onehot_paths = []
    all_mol_conf_paths = []
    all_mol_onehot_paths = []
    all_mol_scaffold_id_paths = []
    all_mol_anchor_id_paths = []
    all_mol_charges_paths = []
    all_mol_extension_ids_paths = []
    all_pocket_info_paths = []

    all_single_frags = []
    all_qed_paths = []
    all_sa_paths = []
    all_vina_paths = []
    frag_smiles_dict = dict()
    all_ids = []
    
    for i, (pocket_name, ligand_name) in enumerate(tqdm(data_split[split])):
        try:
            file_name = args.outdir + '/' + suffix + '/' + 'data_' + str(i).zfill(5)
            if not os.path.exists(file_name):
                os.makedirs(file_name)

            pdbfile = args.rootdir + '/' + 'crossdocked_pocket10/' + pocket_name
            sdffile = args.rootdir + '/' + 'crossdocked_pocket10/' + ligand_name

            if add_vina_score:
                receptor_pdbqt_file = pdbfile[:-3] + 'pdbqt'
                if not os.path.exists(receptor_pdbqt_file):
                    raise FileNotFoundError('pdbqt file could not be found! generate pdbqt from pdb again!')
            else:
                receptor_pdbqt_file = None
            
            pocket_one_hot, pocket_coords, _ = process_pocket(pdbfile, sdffile, atom_dict, pocket_atom_dict, dist_cutoff, remove_H=True,  ca_only=False) 

            pocket_coords_path = file_name + '/pocket_coords.npy'
            pocket_onehot_path = file_name + '/pocket_onehot.npy'

            mol_conf_path = file_name + '/mol_conf.npy'
            mol_onehot_path = file_name + '/mol_onehot.npy'
            mol_charges_path = file_name + '/mol_charges.npy'
            mol_scaffolld_path = file_name + '/scaffold_ids.npy'
            mol_extensionids_path = file_name + '/extension_ids.npy'
            mol_anchorids_path = file_name + '/anchor_ids.npy'
            mol_frags_sdf_path = file_name + '/all_frags.sdf'
            mol_qed_path = file_name + '/mol_QED.npy'
            mol_sa_path = file_name + '/mol_SA.npy'
            mol_vina_score_path = file_name + '/mol_vina_score.npy'
            mol_pocket_info_path = file_name +'/pocket_info.json'
            
            mol_pos, mol_onehot, mol_charges, \
            mol_atom_ids, mol_extension_ids, mol_anchor_ids, is_single_frag, frag_smiles, frag_n_atoms, mol_QED_scores, mol_SA_scores, all_sub_mols = process_ligand(sdffile, max_num_frags=args.max_num_frags, num_atoms_cutoff=args.max_atoms_single_fragment, add_QED=add_QED_score, add_SA=add_SA_score)
            
            print('processed molecule ')
            sdf_writer = Chem.SDWriter(mol_frags_sdf_path) 
            for frag in all_sub_mols:
                sdf_writer.write(frag) # write to a SDF file
            sdf_writer.close()

            if get_pocket_info:
                pocket_info_dict = {}
                command = f'fpocket -f {pdbfile}' # creates a directory that will be deleted later
                out = os.popen(command).read()

                with open(os.path.join(pdbfile[:-4] +'_out', pdbfile[:-4].split('/')[-1] + '_info.txt'), 'r') as fp:
                    lines = fp.readlines()

                    pocket_info_started = False
                    for line in lines:
                        line = line.strip()
                        if line == 'Pocket 1 :':
                            pocket_info_started = True
                            continue
                        if pocket_info_started:
                            if line == "":
                                break
                            key, value = line.split(':')
                            pocket_info_dict[key.strip()] = float(value.strip())
                
                #os.rmdir(pdbfile[:-4] + '_out')
                shutil.rmtree(pdbfile[:-4] + '_out')

            if add_vina_score:
                ligand_pdbqt_file = mol_frags_sdf_path[:-4] + '_' + '.pdbqt'
                sdfdir = Path(mol_frags_sdf_path) 
                sdf_to_pdbqt(sdfdir, ligand_pdbqt_file, len(all_sub_mols))
                size = 25
                
                affinity_results = Parallel(n_jobs=n_cores)(
                    delayed(process_vina_iteration)(n, mol_frags_sdf_path, receptor_pdbqt_file, mol_pos, size)
                    for n in range(len(all_sub_mols))
                )

            if len(affinity_pattern) == 0:
                print('----------------- Error in vina score ---------------')
            #print(all_affinity)
            # remove all pdbqt and sdf files to save space
            pdbqt_files = glob.glob(file_name + '/*.pdbqt')
            sdf_files = glob.glob(file_name + '/*.sdf')

            all_files = pdbqt_files + sdf_files
            for file in all_files:
                if os.path.isfile(file):
                    os.remove(file)
            # -----------------------------------------------------------------------------
            if get_pocket_info:
                with open(mol_pocket_info_path, 'w') as f:
                    json.dump(pocket_info_dict, f)

            np.save(pocket_coords_path, pocket_coords)
            np.save(pocket_onehot_path, pocket_one_hot)
            all_pocket_coords_paths.append(pocket_coords_path)
            all_pocket_onehot_paths.append(pocket_onehot_path)

            np.save(mol_conf_path, mol_pos)
            np.save(mol_onehot_path, mol_onehot)
            np.save(mol_charges_path, mol_charges)

            if add_QED_score:
                np.save(mol_qed_path, mol_QED_scores)
            if add_SA_score:
                np.save(mol_sa_path, mol_SA_scores)
            if add_vina_score:
                np.save(mol_vina_score_path, affinity_results, allow_pickle=True)

            all_mol_conf_paths.append(mol_conf_path)
            all_mol_charges_paths.append(mol_charges_path)
            all_mol_onehot_paths.append(mol_onehot_path)
            all_single_frags.append(is_single_frag)
            all_qed_paths.append(mol_qed_path)
            all_sa_paths.append(mol_sa_path)
            all_vina_paths.append(mol_vina_score_path)
            all_pocket_info_paths.append(mol_pocket_info_path)

            np.save(mol_scaffolld_path, mol_atom_ids, allow_pickle=True)
            np.save(mol_extensionids_path, mol_extension_ids, allow_pickle=True)
            np.save(mol_anchorids_path, mol_anchor_ids)

            all_mol_scaffold_id_paths.append(mol_scaffolld_path)
            all_mol_extension_ids_paths.append(mol_extensionids_path)
            all_mol_anchor_id_paths.append(mol_anchorids_path)
            
            all_ids.append(i)
            for j, f in enumerate(frag_smiles):
                if f not in frag_smiles_dict.keys():
                    frag_smiles_dict[f] = [1, frag_n_atoms[j]]
                else:
                    frag_smiles_dict[f][0] += 1  
        
        except Exception as e:
            print(f'------------------------ Error ----------------------- \n {e}')

    path_mol = pd.DataFrame()
    path_mol['mol_conf'] = all_mol_conf_paths
    path_mol['mol_onehot'] = all_mol_onehot_paths
    path_mol['mol_scaffoldIds'] = all_mol_scaffold_id_paths
    path_mol['mol_anchorIds'] = all_mol_anchor_id_paths
    path_mol['mol_charges'] = all_mol_charges_paths
    path_mol['mol_extensionIds'] = all_mol_extension_ids_paths
    path_mol['pocket_coords'] = all_pocket_coords_paths
    path_mol['pocket_onehot'] = all_pocket_onehot_paths
    path_mol['is_singe_frag'] = all_single_frags
    
    if get_pocket_info:
        path_mol['pocket_info'] = all_pocket_coords_paths
    if add_QED_score:
        path_mol['QED'] = all_qed_paths
    if add_SA_score:
        path_mol['SA'] = all_sa_paths
    if add_vina_score:
        path_mol['vina'] = all_vina_paths
    path_mol['id'] = all_ids
    path_mol.to_csv(args.outdir + '/paths_' + split + '.csv')
    
    all_frag_smiles = list(frag_smiles_dict.keys())
    count_n_atoms = list(frag_smiles_dict.values())

    frag_smiles_data = [(all_frag_smiles[i], count_n_atoms[i][0], count_n_atoms[i][1]) for i in range(len(all_frag_smiles))]
    frag_smiles_library = pd.DataFrame(frag_smiles_data, columns=['smiles', 'counts', 'n_atoms'])
    frag_smiles_library.to_csv(args.outdir + '/fragment_library_' + split + '.csv')