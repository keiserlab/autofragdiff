import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm 
import argparse
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from openbabel import openbabel

from analysis import eval_bond_length
from analysis.reconstruct_mol import reconstruct_from_generated, MolReconsError, make_mol_openbabel
from analysis.metrics import is_connected, is_valid, get_chem
from analysis.eval_bond_angles import get_distribution, eval_angle_dist_profile, find_angle_dist
from analysis.vina_docking import VinaDockingTask
from joblib import Parallel, delayed
from analysis.bond_angle_config import frag1_angles_bins_CROSSDOCK, frag1_dihedral_bins_CROSSDOCK, \
                                       frag2_angles_bins_CROSSDOCK, frag2_dihedral_bins_CROSSDOCK, \
                                       frag3_angles_bins_CROSSDOCK, frag3_dihedral_bins_CROSSDOCK, \
                                       frag4_angles_bins_CROSSDOCK, frag4_dihedral_bins_CROSSDOCK, \
                                       frag5_angles_bins_CROSSDOCK, frag5_dihedral_bins_CROSSDOCK

from analysis.docking import calculate_qvina2_score, sdf_to_pdbqt
from src.utils import get_logger
import collections
import torch

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}

CROSSDOCK_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'B':5, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:4f}')
        else:
            logger.info(f'{k}\tNone')

def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')

frag1 = 'c1ccccc1' # benzene ring
frag2 = 'C1CCOC1' # 
frag3 = 'c1ccncc1' # 
frag4 = 'C1CCNCC1' # 
frag5 = 'C1CCCCC1' # 

frag1 = Chem.MolFromSmiles(frag1)
frag2 = Chem.MolFromSmiles(frag2)
frag3 = Chem.MolFromSmiles(frag3)
frag4 = Chem.MolFromSmiles(frag4)
frag5 = Chem.MolFromSmiles(frag5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', type=str, help='path to generated molecules')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'None'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--n-mols-per-file', type=int, default=20, help='number of molecules per each file')
    parser.add_argument('--crossdock-dir', type=str, default='/srv/home/mahdi.ghorbani/FragDiff/crossdock')

    args = parser.parse_args()
    results_path = args.results_path 
    n_mols_per_file = args.n_mols_per_file
    eval_path = os.path.join(results_path, 'eval_results')
    root_dir = args.crossdock_dir
    split = torch.load(os.path.join(root_dir, 'split_by_name.pt'))
    split = split['test']
    
    os.makedirs(eval_path, exist_ok=True)   
    logger = get_logger('evaluate', log_dir=eval_path)

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    valid_mols = 0
    connected_mols = 0
    all_pair_dist = []
    all_bond_dist = []
    all_validity_results = [] 
    success_pair_dist = []

    all_frag1_angles = []
    all_frag2_angles = []
    all_frag3_angles = []
    all_frag4_angles = []
    all_frag5_angles = []

    all_frag1_dihedrals = []
    all_frag2_dihedrals = []
    all_frag3_dihedrals = []
    all_frag4_dihedrals = []
    all_frag5_dihedrals = []

    results = []

    n_files = 0
    n_samples = 0
    all_results = []
    for n in tqdm(range(100), desc='Eval'):
        prot_filename = split[n][0]
        prot_path = root_dir + '/crossdocked_pocket10/' + prot_filename
        mols_vina_scores = []
        if os.path.exists(results_path + 'pocket_' + str(n) + '_coords.npy'):

            n_files += 1
            x = np.load(results_path + 'pocket_' + str(n) + '_coords.npy')
            h = np.load(results_path + 'pocket_' + str(n) + '_onehot.npy')
            mol_masks = np.load(results_path + 'pocket_' + str(n) + '_mol_masks.npy')

            mols_generated_per_pocket = []
            mols_smiles = []
            results_pocket = []
            for k in range(len(x)):

                mask = mol_masks[k]
                h_mol = h[k]
                x_mol = x[k][mask.astype(np.bool_)]

                atom_inds = h_mol[mask.astype(np.bool_)].argmax(axis=1)
                atom_types = [idx2atom[x] for x in atom_inds]
                atomic_nums = [CROSSDOCK_CHARGES[i] for i in atom_types]

                pair_dist = eval_bond_length.pair_distance_from_pos_v(x_mol, atomic_nums) # computes the pair distribution from all atoms
                #validity_results = check_stability(x_mol, atomic_nums)
                #all_validity_results.append(validity_results)
                n_samples += 1
                try:
                    mol_rec = reconstruct_from_generated(x_mol.tolist(), atomic_nums)
                    #mol_rec = make_mol_openbabel(x_mol, atom_types, CROSSDOCK_CHARGES)
                    smiles = Chem.MolToSmiles(mol_rec)
                    Chem.SanitizeMol(mol_rec)
                
                except Exception as e:
                    print(e)
                    continue
                valid_mols += 1

                if is_connected(mol_rec):
                    connected_mols += 1
                else:
                    # if the molecule is not connected, then take the largest fragment
                    m_frags = Chem.GetMolFrags(mol_rec, asMols=True, sanitizeFrags=False)
                    mol_rec = max(m_frags, default=mol_rec, key=lambda m: m.GetNumAtoms())
                    Chem.SanitizeMol(mol_rec)

                bond_dist = eval_bond_length.bond_distance_from_mol(mol_rec)
                all_pair_dist += pair_dist
                all_bond_dist += bond_dist
                    
                mols_generated_per_pocket.append(mol_rec)
                mols_smiles.append(Chem.MolToSmiles(mol_rec))
                chem_results = get_chem(mol_rec) # a dictionary with qed, sa, logp, lipinski, ring_size

                out_sdf_file = eval_path + '/mol.sdf'
                with Chem.SDWriter(out_sdf_file) as writer:
                    writer.write(mol_rec)

                if args.docking_mode == 'qvina':
                    # --------------------------- Getting Vina Docking results ---------------------------
                    prot_pdbqt_file = prot_path[:-4] + '.pdbqt'
                    #try:
                    sdf_file = Path(out_sdf_file)
                    qvina_score, docked_mol = calculate_qvina2_score(prot_pdbqt_file, sdf_file, out_dir=eval_path, return_rdmol=True)
                    #print(f'Vina scores for {n}: {qvina_scores}')
                    
                    files = os.listdir(eval_path)
                    for file in files:
                        if file.endswith('.sdf') or file.endswith('.pdbqt'):
                            os.remove(os.path.join(eval_path, file))
                    mols_vina_scores.append(qvina_score[0])
                else:
                    qvina_score = [None]
                    docked_mol = [None]

                result = {'mol': mol_rec,
                            'smiles': Chem.MolToSmiles(mol_rec),
                            'QED': chem_results['qed'],
                            'SA': chem_results['sa'],
                            'logP': chem_results['logp'],
                            'lipinski': chem_results['lipinski'],
                            'ring_size': chem_results['ring_size'],
                            'qvina': qvina_score[0],
                            'docked_mol': docked_mol[0]}
                
                results_pocket.append(result)

                success_pair_dist += pair_dist

                try:
                    angles_frag1, dihedrals_frag1 = find_angle_dist(mol_rec, frag1)
                    angles_frag2, dihedrals_frag2 = find_angle_dist(mol_rec, frag2)
                    angles_frag3, dihedrals_frag3 = find_angle_dist(mol_rec, frag3)
                    angles_frag4, dihedrals_frag4 = find_angle_dist(mol_rec, frag4)
                    angles_frag5, dihedrals_frag5 = find_angle_dist(mol_rec, frag5)

                    all_frag1_angles += angles_frag1
                    all_frag2_angles += angles_frag2
                    all_frag3_angles += angles_frag3
                    all_frag4_angles += angles_frag4
                    all_frag5_angles += angles_frag5

                    all_frag1_dihedrals += dihedrals_frag1
                    all_frag2_dihedrals += dihedrals_frag2
                    all_frag3_dihedrals += dihedrals_frag3
                    all_frag4_dihedrals += dihedrals_frag4
                    all_frag5_dihedrals += dihedrals_frag5
                except:
                    continue
        
        if args.docking_mode == 'qvina':
            print(mols_vina_scores)
        all_results.append(results_pocket)

    logger.info(f'Evaluation is done! {n_samples}  samples in total')

    fraction_valid = valid_mols / n_samples
    fraction_connected = connected_mols / n_samples
    
    print('fraction_connected is: ', fraction_connected)
    print('fraction_valid is :' , fraction_valid)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist,)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile, data_type='CrossDock')
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    print('success mols JS metrics: ')
    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile, data_type='CrossDock')
    print_dict(success_js_metrics, logger)

    eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                        metrics=success_js_metrics,
                                        data_type='CrossDock',
                                        save_path=os.path.join(eval_path, f'pair_dist_hist.png'))
    
    # ------ ANGLE distribution ------
    # --------------------------------------------------------------------------
    # get the angle and dihedral profiles and JSD
    angle_profile_frag1 = get_distribution(all_frag1_angles, frag1_angles_bins_CROSSDOCK)
    dihedral_profile_frag1 = get_distribution(all_frag1_dihedrals, frag1_dihedral_bins_CROSSDOCK)
    
    angle_profile_frag2 = get_distribution(all_frag2_angles, frag2_angles_bins_CROSSDOCK)
    dihedral_profile_frag2 = get_distribution(all_frag2_dihedrals, frag2_dihedral_bins_CROSSDOCK)
    
    angle_profile_frag3 = get_distribution(all_frag3_angles, frag3_angles_bins_CROSSDOCK)
    dihedral_profile_frag3 = get_distribution(all_frag3_dihedrals, frag3_dihedral_bins_CROSSDOCK)
    
    angle_profile_frag4 = get_distribution(all_frag4_angles, frag4_angles_bins_CROSSDOCK)
    dihedral_profile_frag4 = get_distribution(all_frag4_dihedrals, frag4_dihedral_bins_CROSSDOCK)
    
    angle_profile_frag5 = get_distribution(all_frag5_angles, frag5_angles_bins_CROSSDOCK)
    dihedral_profile_frag5 = get_distribution(all_frag5_dihedrals, frag5_dihedral_bins_CROSSDOCK)
    
    eval_frag1 = eval_angle_dist_profile(angle_profile_frag1, dihedral_profile_frag1, Chem.MolToSmiles(frag1), data_type='CrossDock')
    eval_frag2 = eval_angle_dist_profile(angle_profile_frag2, dihedral_profile_frag2, Chem.MolToSmiles(frag2), data_type='CrossDock')
    eval_frag3 = eval_angle_dist_profile(angle_profile_frag3, dihedral_profile_frag3, Chem.MolToSmiles(frag3), data_type='CrossDock')
    eval_frag4 = eval_angle_dist_profile(angle_profile_frag4, dihedral_profile_frag4, Chem.MolToSmiles(frag4), data_type='CrossDock')
    eval_frag5 = eval_angle_dist_profile(angle_profile_frag5, dihedral_profile_frag5, Chem.MolToSmiles(frag5), data_type='CrossDock')

    print('JS of angles for fragment 1:')
    print_dict(eval_frag1, logger)

    print('JS of angles for fragment 2:')
    print_dict(eval_frag2, logger)

    print('JS of angles for fragment 3:')
    print_dict(eval_frag3, logger)

    print('JS of angles for fragment 4:')
    print_dict(eval_frag4, logger)

    print('JS of angles for fragment 5:')
    print_dict(eval_frag5, logger)  

    torch.save({
        'fraction_connected': fraction_connected,
        'fraction_valid': fraction_valid,
        'bond_length': all_bond_dist,
        'all_results': all_results,
        'success_JS': success_js_metrics,
        'bond_length_JS': c_bond_length_dict,
        'frag1_JS': eval_frag1,
        'frag2_JS': eval_frag2, 
        'frag3_JS': eval_frag3,
        'frag4_JS': eval_frag4,
        'frag5_JS': eval_frag5
    }, os.path.join(eval_path, 'metrics2.pt'))

    all_qed = [results['all_results'][j][i]['QED'] for j in range(n_files) for i in range(len(results['all_results'][j]))]
    all_sa = [results['all_results'][j][i]['SA'] for j in range(n_files) for i in range(len(results['all_results'][j]))]

    logger.info('QED: Mean: %.3f Median: %.3f std: %.3f' % (np.mean(all_qed), np.median(all_qed), np.std(all_qed)))
    logger.info('SA: Mean: %.3f Median: %.3f std: %.3f' % (np.mean(all_sa), np.median(all_sa), np.std(all_sa)))

    if args.docking_mode == 'qvina':
        vina_scores = [results['all_results'][j][i]['qvina'] for j in range(n_files) for i in range(len(results['all_results'][j]))]
    logger.info('Vina: Mean: %.3f Median: %.3f Std: %.3f' %(np.mean(vina_scores), np.median(vina_scores), np.std(vina_scores)))

    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)