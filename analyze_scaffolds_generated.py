import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm 
import argparse
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from openbabel import openbabel

from analysis import eval_bond_length
from analysis.reconstruct_mol import reconstruct_from_generated, MolReconsError
from analysis.metrics import is_connected, is_valid, get_chem
from analysis.eval_bond_angles import get_distribution, eval_angle_dist_profile, find_angle_dist
from analysis.vina_docking import VinaDockingTask
from joblib import Parallel, delayed

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', type=str, default='results_scaffold',
                        help='path to save the scaffold based optimization')
    parser.add_argument('--scaffold-path', type=str, default='scaffolds/1a2g_scaff.sdf',
                        help='path to sdf of scaffold')
    parser.add_argument('--original-path', type=str, default='scaffolds/1a2g_orig.sdf',
                        help='path to original molecule')
    parser.add_argument('--receptor-path', type=str, default='scaffolds/1a2g.pdb',
                        help='path to pdb file of receptor')
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
    
    scaffold_path = args.scaffold_path # sdf file of scaffold
    receptor_path = args.receptor_path # pdb file of receptor

    os.makedirs(eval_path, exist_ok=True)   
    logger = get_logger('evaluate', log_dir=eval_path)

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    valid_mols = 0
    connected_mols = 0
    results = []

    n_files = 0
    n_samples = 0

    scaff_mol = Chem.SDMolSupplier(scaffold_path)[0]
    orig_mol = Chem.SDMolSupplier(args.original_path)[0]

    # compute vina score for the scaffold
    vina_task = VinaDockingTask.from_generated_mol(orig_mol, protein_path=receptor_path)
    score_result = vina_task.run(mode='score_only', exhaustiveness=16)
    scaffold_score = score_result[0]['affinity'] 
    print('------> Vina score for original molecule is :  ', scaffold_score)

    # compute vina score for the scaffold
    vina_task = VinaDockingTask.from_generated_mol(scaff_mol, protein_path=receptor_path)
    score_result = vina_task.run(mode='score_only', exhaustiveness=16)
    scaffold_score = score_result[0]['affinity'] 
    print('------> Vina score for scaffold is : ', scaffold_score)
    

    for n in tqdm(range(10), desc='Eval'):
        prot_path = receptor_path
        if os.path.exists(results_path + 'pocket_' + str(n) + '_coords.npy'):

            n_files += 1
            x = np.load(results_path + 'pocket_' + str(n) + '_coords.npy')
            h = np.load(results_path + 'pocket_' + str(n) + '_onehot.npy')
            mol_masks = np.load(results_path + 'pocket_' + str(n) + '_mol_masks.npy')

            all_mols = []
            for k in range(len(x)):

                mask = mol_masks[k]
                h_mol = h[k]
                x_mol = x[k][mask.astype(np.bool_)]

                atom_inds = h_mol[mask.astype(np.bool_)].argmax(axis=1)
                atom_types = [idx2atom[x] for x in atom_inds]
                atomic_nums = [CROSSDOCK_CHARGES[i] for i in atom_types]

                #all_validity_results.append(validity_results)
                n_samples += 1
                try:
                    mol_rec = reconstruct_from_generated(x_mol.tolist(), atomic_nums, aromatic=None, basic_mode=True)
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
            
                chem_results = get_chem(mol_rec) # a dictionary with qed, sa, logp, lipinski, ring_size

                # --------------------------- Getting Vina Docking results ---------------------------
                try:
                    if args.docking_mode == 'qvina':
                        pass # TODO: add the qvina like in TargetDiff
                    elif args.docking_mode in ['vina_score', 'vina_dock']:
                        vina_task = VinaDockingTask.from_generated_mol(mol_rec, protein_path=prot_path)
                        score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                        minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                        print('score_only: ', score_only_results[0]['affinity'])
                        print('minimized score: ', minimize_results[0]['affinity'])
                        vina_results = {
                            'score_only': score_only_results,
                            'minimize': minimize_results
                        }
                        if args.docking_mode == 'vina_dock':
                            docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                            vina_results['dock'] = docking_results
                            print('vina dock: ', docking_results[0]['affinity'])
                    else:
                        vina_results = None

                except:
                    if args.verbose:
                        logger.warning(f'Docking failed for pocket {n} and molecule {k}')
                    continue
                
                results.append({
                    'mol': mol_rec,
                    'smiles': smiles,
                    'chem_results': chem_results,
                    'vina' :vina_results
                })
    
    logger.info(f'Evaluation is done! {n_samples}  samples in total')

    fraction_valid = valid_mols / n_samples
    fraction_connected = connected_mols / n_samples
    
    print('fraction_connected is: ', fraction_connected)
    print('fraction_valid is :' , fraction_valid)   

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED: Mean: %.3f Median: %.3f std: %.3f' % (np.mean(qed), np.median(qed), np.std(qed)))
    logger.info('SA: Mean: %.3f Median: %.3f std: %.3f' % (np.mean(qed), np.median(sa), np.std(sa)))

    if args.docking_mode == 'qvina':
        vina = [r['vina'[0]]['affinity'] for r in results]
        logger.info('Vina: Mean: %.3f Median: %.3f Std: %.3f' %(np.mean(vina), np.median(vina), np.std(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score     : Mean %.3f Median: %.3f Std: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only), np.std(vina_score_only)))
        logger.info('Vina minimized : Mean %.3f Median: %.3f Std: %.3f' % (np.mean(vina_min), np.median(vina_min), np.std(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock : Mean: %.3f Median: %.3f Std: %.3f' % (np.mean(vina_dock), np.median(vina_dock), np.std(vina_dock)))

    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    torch.save({
        'fraction_connected': fraction_connected,
        'fraction_valid': fraction_valid,
        'all_results': results,
    }, os.path.join(eval_path, 'metrics.pt'))