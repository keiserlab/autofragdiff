import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm 
from itertools import combinations
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel
import tempfile
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from torch.utils.data import Dataset, DataLoader
import torch
from src.datasets import HierCrossDockDataset, pairwise_distances
from src import anchor_gnn
from src import const
from src.lightning_anchor_gnn import AnchorGNN_pl
from src.const import EXTENSION_SIZE_DIST_CROSSDOCK, NUM_FRAGS_DIST_CROSSDOCK
from src import utils
from src.extension_size import DistributionNodes
from src.lightning import AR_DDPM
from scipy.spatial import distance
import time
import shutil
from pathlib import Path
from analysis.docking import calculate_qvina2_score, sdf_to_pdbqt

import torch.nn.functional as F

from utils.volume_sampling import sample_discrete_number, bin_edges, prob_dist_df
from utils.volume_sampling import remove_output_files, run_fpocket, extract_values

from utils.templates import get_one_hot, get_pocket
from utils.templates import create_template_for_pocket_anchor_prediction, create_template_for_pocket_mol, \
                          get_anchors_pocket

from analysis.reconstruct_mol import reconstruct_from_generated
from analysis.metrics import is_connected
from analysis.vina_docking import VinaDockingTask

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}
CROSSDOCK_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'B':5, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket
vdws = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'B': 1.92, 'Br': 1.85, 'Cl': 1.75, 'P': 1.8, 'I': 1.98, 'F': 1.47}

def compute_number_of_clashes(pocket_coords, lig_coords, pocket_types, lig_types, tolerace=0.25):
    # pocket_coords [Np, 3]
    # lig_coords [Nl, 3]
    # pocket_types [Np] index of type
    # lig_types [Nl] index of type
    dists = torch.cdist(lig_coords, pocket_coords)
    lig_ids, pocket_ids = torch.where(dists < 4)

    num_clashes = 0
    clashed_atom_ids = set()
    clashed_pocket_ids = set()
    for i in range(len(lig_ids)):
        lig_atom, pocket_atom = lig_types[lig_ids[i]], pocket_types[pocket_ids[i]]
        allowed_radii = vdws[lig_atom] + vdws[pocket_atom]

        if (dists[lig_ids[i], pocket_ids[i]] - allowed_radii + tolerace) < 0:
            clashed_atom_ids.add(int(lig_ids[i]))
            clashed_pocket_ids.add(int(pocket_ids[i]))
            num_clashes += 1
    return len(clashed_atom_ids), clashed_atom_ids, clashed_pocket_ids, num_clashes

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
    