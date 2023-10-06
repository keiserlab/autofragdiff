import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os

import collections
from typing import Tuple, Sequence, Dict, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetAngleRad, GetDihedralRad, GetAngleDeg, GetDihedralDeg
from math import degrees

frag1_dihedral_bins = np.arange(150, 210, 0.5)
frag2_dihedral_bins = np.arange(150, 210, 0.5)
frag3_dihedral_bins = np.arange(80, 280, 1)
frag4_dihedral_bins = np.arange(80, 280, 1)
frag5_dihedral_bins = np.arange(170, 190, 0.5)

frag1_angles_bins = np.arange(100, 140, 0.5)
frag2_angles_bins = np.arange(100, 140, 0.5)
frag3_angles_bins = np.arange(80, 140, 0.5)
frag4_angles_bins = np.arange(90, 140, 0.5)
frag5_angles_bins = np.arange(100, 120, 0.5)

if __name__ == '__main__':

    supplier = list(Chem.SDMolSupplier('/srv/ds/set-1/user/mahdi.ghorbani/FragDiff/datasets/geom_conformers.sdf'))
    
    frag1 = 'c1ccccc1' # benzene ring
    frag2 = 'c1ccncc1' # pyridine
    frag3 = 'C1CCNCC1' # Piperidine
    frag4 = 'C1CNCCN1' # Morpholine
    frag5 = 'c1ccoc1' # Furan

    frag1 = Chem.MolFromSmiles(frag1)
    frag2 = Chem.MolFromSmiles(frag2)
    frag3 = Chem.MolFromSmiles(frag3)
    frag4 = Chem.MolFromSmiles(frag4)
    frag5 = Chem.MolFromSmiles(frag5)

    all_frag1_angles = []
    all_frag1_dihedrals = []

    all_frag2_angles = []
    all_frag2_dihedrals = []

    all_frag3_angles = []
    all_frag3_dihedrals = []

    all_frag4_angles = []
    all_frag4_dihedrals = []

    all_frag5_angles = []
    all_frag5_dihedrals = []

    for mol_id, mol in tqdm(enumerate(supplier)):
        
        try:
            conf = mol.GetConformer()

            matches_frag1 = mol.GetSubstructMatches(frag1)
            matches_frag2 = mol.GetSubstructMatches(frag2)
            matches_frag3 = mol.GetSubstructMatches(frag3)
            matches_frag4 = mol.GetSubstructMatches(frag4)
            matches_frag5 = mol.GetSubstructMatches(frag5)

            for match in matches_frag1:
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

                all_frag1_angles += match_angles
                all_frag1_dihedrals += match_dih

            # -----------------------------------------------------------------------------------------------------------
            for match in matches_frag2:
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

                all_frag2_angles += match_angles
                all_frag2_dihedrals += match_dih

            # -----------------------------------------------------------------------------------------------------------
            for match in matches_frag3:
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

                all_frag3_angles += match_angles
                all_frag3_dihedrals += match_dih

            # -----------------------------------------------------------------------------------------------------------
            for match in matches_frag4:
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

                all_frag4_angles += match_angles
                all_frag4_dihedrals += match_dih

            # -----------------------------------------------------------------------------------------------------------
            for match in matches_frag5:
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

                all_frag5_angles += match_angles
                all_frag5_dihedrals += match_dih
        except:
            print(f'error for {mol_id}')