from joblib import Parallel, delayed
import os
from pathlib import Path
import random
import shutil
import re
import glob

from tqdm import tqdm
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pandas as pd
import torch

affinity_pattern = r"Affinity:\s+(-?\d+\.\d+)\s+\(kcal/mol\)"
RDLogger.DisableLog('rdApp.*')  

def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'obabel {sdf_file} -O {pdbqt_outfile} -f {0} -l {mol_id} -m').read()
    return pdbqt_outfile

def get_vina_dock_score(receptor_pdbqt_file, ligand_pdbqt_file, cx, cy, cz, size):
    # Vina docking and getting the vina score
    out = os.popen(
    f'qvina2.1 --receptor {receptor_pdbqt_file} '
    f'--ligand {ligand_pdbqt_file} '
    f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
    f'--size_x {size} --size_y {size} --size_z {size} --exhaustiveness 16' 
    ).read()
    out_split = out.splitlines()
    best_idx = out_split.index('-----+------------+----------+----------') + 1
    best_line = out_split[best_idx].split()
    print('\n best Affinity:', float(best_line[1]))
    return float(best_line[1])

def get_vina_score(receptor_pdbqt_file, ligand_pdbqt_file, cx, cy, cz, size):
    # TODO: using QVina to get vina scores gives weird results. Use Vina
    # scores the generated poses without docking them
    out = os.popen(
    f'qvina2.1 --score_only --receptor {receptor_pdbqt_file} '
    f'--ligand {ligand_pdbqt_file} '
    f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
    f'--size_x {size} --size_y {size} --size_z {size}' 
    ).read()
    match = re.search(affinity_pattern, out)
    affinity_value = float(match.group(1))
    print('vina score is:', affinity_value)
    return affinity_value

def process_vina_iteration(n, save_file, receptor_pdbqt_file, mol_pos, size, result_type='vina_score'):
    pdbqt_file = save_file + 'all_mols_' + str(n) + '.pdbqt'
    cx, cy, cz = mol_pos
    if result_type == 'vina_score':
        affinity_value = get_vina_score(receptor_pdbqt_file, pdbqt_file, cx, cy, cz, size)
    elif result_type == 'dock_score':
        affinity_value = get_vina_dock_score(receptor_pdbqt_file, pdbqt_file, cx, cy, cz, size)
    return affinity_value


