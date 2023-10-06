import numpy as np

import collections
from typing import Tuple, Sequence, Dict, Optional

from scipy import spatial as sci_spatial
from .bond_angle_config import ANGLE_DIST_CROSSDOCK, DIHED_DIST_CROSSDOCK
from rdkit.Chem.rdMolTransforms import GetAngleRad, GetDihedralRad, GetAngleDeg, GetDihedralDeg

def get_distribution(angles, bins):

    bin_counts = collections.Counter(np.searchsorted(bins, angles))
    bin_counts = [bin_counts[i] if i in bin_counts else 0 for i in range(len(bins))]
    bin_counts = np.array(bin_counts) / np.sum(bin_counts)
    return bin_counts

def eval_angle_dist_profile(bond_angle_profile, dihedral_angle_profile, frag):
    
    # frag is the smiles of fragment
    # bond_angle_profile -> a dictionary with keys the smiles of fragmenst and values the distribution of angles/dihedrals
    metrics = {}
    gt_distribution = ANGLE_DIST_CROSSDOCK[frag]
    metrics[f'Angle-JSD_{frag}'] = sci_spatial.distance.jensenshannon(gt_distribution,
                                                                        bond_angle_profile) 

    gt_distribution = DIHED_DIST_CROSSDOCK[frag]
    metrics[f'Dihedral-JSD_{frag}'] = sci_spatial.distance.jensenshannon(gt_distribution,
                                                                            dihedral_angle_profile)   
    return metrics


def find_angle_dist(mol, frag):
    all_frag_angles = []
    all_frag_dihedrals = []

    conf = mol.GetConformer()
    
    matches_frag = mol.GetSubstructMatches(frag)
    for match in matches_frag:
        match_angles = []
        match_dih = []
        match_set = set(match)

        for atom_index in match:
            atom = mol.GetAtomWithIdx(atom_index)
            neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetIdx() in match_set]
            for i in range(len(neighbors)-1):
                for j in range(i+1, len(neighbors)):
                    angle_deg = GetAngleDeg(conf, neighbors[i], atom_index, neighbors[j])

                    if angle_deg < 0:
                        angle_deg += 360
                    match_angles.append(angle_deg)

            for neighbor in neighbors:
                next_neighbors = [next_neighbor.GetIdx() for next_neighbor in mol.GetAtomWithIdx(neighbor).GetNeighbors() if next_neighbor.GetIdx() in match_set]
                for next_neighbor in next_neighbors:
                    if next_neighbor != atom_index: # don't want to go to original atom
                        # calculate and print dihedral angle
                        dihedral_deg = GetDihedralDeg(conf, neighbor, atom_index, next_neighbor, neighbors[(neighbors.index(neighbor)+1) % len(neighbors)])
                        if dihedral_deg < 0:
                            dihedral_deg += 360
                        match_dih.append(dihedral_deg)

        all_frag_angles += match_angles
        all_frag_dihedrals += match_dih
    
    return all_frag_angles, all_frag_dihedrals