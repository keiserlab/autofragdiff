
import json
import json
import numpy as np
import pandas as pd
import re

from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.BRICS import FindBRICSBonds
import scaffoldgraph as sg
from itertools import combinations
import matplotlib.pyplot as plt
from rdkit.Chem import BRICS
from copy import deepcopy

import numpy as np
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import BRICS
import networkx as nx

REGEX = re.compile('\[\d*\*\]')
def find_single_ring_fragments(mol):
    """ find all fragments in the molecule with a single ring using Murcko scaffold fragmentation"""
    all_frags = sg.get_all_murcko_fragments(mol, break_fused_rings=False) # find fragments without breaking fused rings
    
    frags_one_ring = []
    for frag in all_frags:
        if len(sg.get_all_murcko_fragments(frag, break_fused_rings=False)) == 1:
            frags_one_ring.append(frag)
    return frags_one_ring


def set_atomindex_draw(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx()))
    return mol

def find_atom_ids(frag):
    """ gives the atom ids in a fragment"""
    frag_atoms = set()
    for atom in frag.GetAtoms():
        if atom.HasProp('ID'):
            frag_atoms.add(int(atom.GetProp('ID')))
    return frag_atoms

def find_atom2frag(all_frag_atom_ids):
    """ finds a mapping between atoms and their corresponding fragments"""
    atom2frag = {}
    for frag_idx, row in enumerate(all_frag_atom_ids):
        for idx in row:
            atom2frag[idx] = frag_idx

    return atom2frag

def find_neighboring_frags(all_frags, atom2frag, bonds_broken_frags):
    # find neighboring fragments
    neighbors = np.zeros((len(all_frags), len(all_frags)))
    for atom1, atom2 in bonds_broken_frags:
        neighbors[atom2frag[atom1], atom2frag[atom2]] = 1
        neighbors[atom2frag[atom2], atom2frag[atom1]] = 1
    
    return neighbors

def find_single_ring_fragments(mol):
    """ find all fragments in the molecule with a single ring using Murcko scaffold fragmentation"""
    all_frags = sg.get_all_murcko_fragments(mol, break_fused_rings=False) # find fragments without breaking fused rings
    
    frags_one_ring = []
    for frag in all_frags:
        if len(sg.get_all_murcko_fragments(frag, break_fused_rings=False)) == 1:
            frags_one_ring.append(frag)
    return frags_one_ring


def set_atomindex_draw(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx()))
    return mol

def find_atom_ids(frag):
    """ gives the atom ids in a fragment"""
    frag_atoms = set()
    for atom in frag.GetAtoms():
        if atom.HasProp('ID'):
            frag_atoms.add(int(atom.GetProp('ID')))
    return frag_atoms

def find_atom2frag(all_frag_atom_ids):
    """ finds a mapping between atoms and their corresponding fragments"""
    atom2frag = {}
    for frag_idx, row in enumerate(all_frag_atom_ids):
        for idx in row:
            atom2frag[idx] = frag_idx

    return atom2frag

def find_neighboring_frags(all_frags, atom2frag, bonds_broken_frags):
    # find neighboring fragments
    neighbors = np.zeros((len(all_frags), len(all_frags)))
    for atom1, atom2 in bonds_broken_frags:
        neighbors[atom2frag[atom1], atom2frag[atom2]] = 1
        neighbors[atom2frag[atom2], atom2frag[atom1]] = 1
    
    return neighbors


def find_bonds_broken_frags_ids(atom2frag, bonds_broken_frags):
    """ find the fragments ids in each bond broken"""
    bonds_broken_fragment_ids = []
    for bond in bonds_broken_frags:
        frag1 = atom2frag[bond[0]]
        frag2 = atom2frag[bond[1]]
        bonds_broken_fragment_ids.append((frag1, frag2))
    return bonds_broken_fragment_ids


def find_single_ring_fragments(mol):
    """ find all fragments in the molecule with a single ring using Murcko scaffold fragmentation"""
    all_frags = sg.get_all_murcko_fragments(mol, break_fused_rings=False) # find fragments without breaking fused rings
    
    frags_one_ring = []
    for frag in all_frags:
        if len(sg.get_all_murcko_fragments(frag, break_fused_rings=False)) == 1:
            frags_one_ring.append(frag)
    return frags_one_ring


def set_atomindex_draw(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx()))
    return mol

def find_atom_ids(frag):
    """ gives the atom ids in a fragment"""
    frag_atoms = set()
    for atom in frag.GetAtoms():
        if atom.HasProp('ID'):
            frag_atoms.add(int(atom.GetProp('ID')))
    return frag_atoms

def find_atom2frag(all_frag_atom_ids):
    """ finds a mapping between atoms and their corresponding fragments"""
    atom2frag = {}
    for frag_idx, row in enumerate(all_frag_atom_ids):
        for idx in row:
            atom2frag[idx] = frag_idx

    return atom2frag

def find_neighboring_frags(all_frags, atom2frag, bonds_broken_frags):
    # find neighboring fragments
    neighbors = np.zeros((len(all_frags), len(all_frags)))
    for atom1, atom2 in bonds_broken_frags:
        neighbors[atom2frag[atom1], atom2frag[atom2]] = 1
        neighbors[atom2frag[atom2], atom2frag[atom1]] = 1
    
    return neighbors


def find_bonds_broken_frags_ids(atom2frag, bonds_broken_frags):
    """ find the fragments ids in each bond broken"""
    bonds_broken_fragment_ids = []
    for bond in bonds_broken_frags:
        frag1 = atom2frag[bond[0]]
        frag2 = atom2frag[bond[1]]
        bonds_broken_fragment_ids.append((frag1, frag2))
    return bonds_broken_fragment_ids

def find_single_ring_fragments(mol):
    """ find all fragments in the molecule with a single ring using Murcko scaffold fragmentation"""
    all_frags = sg.get_all_murcko_fragments(mol, break_fused_rings=False) # find fragments without breaking fused rings
    
    frags_one_ring = []
    for frag in all_frags:
        if len(sg.get_all_murcko_fragments(frag, break_fused_rings=False)) == 1:
            frags_one_ring.append(frag)
    return frags_one_ring


def set_atomindex_draw(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx()))
    return mol

def find_atom_ids(frag):
    """ gives the atom ids in a fragment"""
    frag_atoms = set()
    for atom in frag.GetAtoms():
        if atom.HasProp('ID'):
            frag_atoms.add(int(atom.GetProp('ID')))
    return frag_atoms

def find_atom2frag(all_frag_atom_ids):
    """ finds a mapping between atoms and their corresponding fragments"""
    atom2frag = {}
    for frag_idx, row in enumerate(all_frag_atom_ids):
        for idx in row:
            atom2frag[idx] = frag_idx

    return atom2frag

def find_neighboring_frags(all_frags, atom2frag, bonds_broken_frags):
    # find neighboring fragments
    neighbors = np.zeros((len(all_frags), len(all_frags)))
    for atom1, atom2 in bonds_broken_frags:
        neighbors[atom2frag[atom1], atom2frag[atom2]] = 1
        neighbors[atom2frag[atom2], atom2frag[atom1]] = 1
    
    return neighbors


def find_bonds_broken_frags_ids(atom2frag, bonds_broken_frags):
    """ find the fragments ids in each bond broken"""
    bonds_broken_fragment_ids = []
    for bond in bonds_broken_frags:
        frag1 = atom2frag[bond[0]]
        frag2 = atom2frag[bond[1]]
        bonds_broken_fragment_ids.append((frag1, frag2))
    return bonds_broken_fragment_ids


def find_bonds_broken_with_frags(mol, frags, max_num_frags=12, max_num_atoms_single_frag=24):
    """ 
    Fragments the molecule based on the scaffold into multiple fragments.
    In case, there is no murcko scaffold, it uses BRICS to break bonds.

    IF the number of fragments is higher than max_num_frags, it removes (combines) fragments that are bonded to the scaffold by 
    first sorting them by their number of atoms and then combining those with the murcko fragemnts until the max_num_frags is reached. 
    This process is similar for the BRICS fragmentation
    """
    for atom in mol.GetAtoms(): # set id property to all atom in molecules
        atom.SetProp('ID', str(atom.GetIdx()))
    
    if frags[0].GetNumAtoms() == 0: # in this case use BRICS for fragmentation
        print('Using BRICS')
        bonds_broken_frags = [bond[0] for bond in FindBRICSBonds(mol)]
        if len(bonds_broken_frags) == 0:
            # just a single fragment:
            num_atoms = mol.GetNumAtoms()
            if num_atoms < max_num_atoms_single_frag:
                return [mol], None, None, None, None,  None 

        bonds_broken_indices = []
        for bond in bonds_broken_frags:
            bonds_broken_indices.append(mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx())  # all bond indices broken

        all_frags = Chem.FragmentOnBonds(mol, bonds_broken_indices, addDummies=True, dummyLabels=bonds_broken_frags)
        all_frags = Chem.GetMolFrags(all_frags, asMols=True)

        all_frag_atom_ids = []
        for frag in all_frags:
            all_frag_atom_ids.append(find_atom_ids(frag))
        
        atom2frag = find_atom2frag(all_frag_atom_ids) # dictionary mapping atoms to corresponding fragments
        neighbors = find_neighboring_frags(all_frags, atom2frag, bonds_broken_frags) # adjacency of neighboring fragments

        bonds_broken_frag_ids = find_bonds_broken_frags_ids(atom2frag, bonds_broken_frags)

        if len(all_frags) > max_num_frags: # checking the total number of fragments
            single_conn_frag_ids = np.argwhere(neighbors.sum(axis=0) == 1.).flatten() # fragments with only one connection

            frag_candidates_to_del = []
            frag_candidates_to_del_numatoms = []
            for f_id in single_conn_frag_ids:
                frag = all_frags[f_id]
                n_atoms = frag.GetNumAtoms()
                frag_candidates_to_del.append(f_id)
                frag_candidates_to_del_numatoms.append(n_atoms)
            
            # sort the candidates to delete from fragments based on their number of atoms
            sorted_candidates = np.array(frag_candidates_to_del)[np.argsort(frag_candidates_to_del_numatoms)]
            candidate_bond_to_del = []
            candidate_bond_atomids_to_del = []
            n_frags_to_del = len(all_frags) - max_num_frags
            for i, bond in enumerate(bonds_broken_frag_ids):
                if bond[0] in sorted_candidates[:n_frags_to_del] or bond[1] in sorted_candidates[:n_frags_to_del]:

                    candidate_bond_to_del.append(bonds_broken_indices[i])
                    candidate_bond_atomids_to_del.append(bonds_broken_frags[i])

            bonds_broken_frags = list(set(bonds_broken_frags) - set(candidate_bond_atomids_to_del))
            bonds_broken_indices = [] # bond indices broken
            for bond in bonds_broken_frags:
                bonds_broken_indices.append(mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx())  # all bond indices broken

            # fragment the molecule again with the new bonds 
            all_frags = Chem.FragmentOnBonds(mol, bonds_broken_indices, addDummies=True, dummyLabels=bonds_broken_frags)
            all_frags = Chem.GetMolFrags(all_frags, asMols=True)

            all_frag_atom_ids = [] # find the ids of the new fragments
            for frag in all_frags:
                all_frag_atom_ids.append(find_atom_ids(frag))
            
            atom2frag = find_atom2frag(all_frag_atom_ids) # find the new atom2frag
            bonds_broken_frag_ids = find_bonds_broken_frags_ids(atom2frag, bonds_broken_frags)


        return all_frags, bonds_broken_frags, bonds_broken_indices, bonds_broken_frag_ids, all_frag_atom_ids,  atom2frag 

    all_matches_frags = [] # list of tuples of all matcesh btween fragmenst and molecule
    for frag in frags:
        all_matches = mol.GetSubstructMatches(frag)
        for m in all_matches:
            all_matches_frags.append(set(m))

    bonds_broken_frags = []
    for i, frag in enumerate(frags): 
        matches_frag = mol.GetSubstructMatches(frag) # there could be multiple matches for each molecule

        for match in matches_frag:

            # check the match is not corresponding to other fragment
            # -----------------------------------------------------
            frag_match_check = True
            if len(matches_frag) > 1: # more than one fragment matches
                for other_match in all_matches_frags:
                    if other_match.issuperset(set(match))  and len(other_match) != len(match):
                        frag_match_check = False
                        break

            if frag_match_check == False:
                continue
            # -----------------------------------------------------
            bonds_frag_mol = np.where(Chem.GetAdjacencyMatrix(mol)[np.array(match)]) # find the bonds fragment makes the the molecule
            
            for i, atom in enumerate(bonds_frag_mol[1]): # enumerate through all the bonds (atom is target atom from fragment)
                if atom not in match:
                    if (int(match[bonds_frag_mol[0][i]]), int(atom)) not in bonds_broken_frags and (int(atom), int(match[bonds_frag_mol[0][i]])) not in bonds_broken_frags:
                        bonds_broken_frags.append((int(match[bonds_frag_mol[0][i]]), int(atom)))
                # tuple of atoms of bonds broken

    bonds_broken_indices = [] # bond indices broken
    for bond in bonds_broken_frags:
        bonds_broken_indices.append(mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx())  # all bond indices broken

    # NOTE: later consider single fragments
    if len(bonds_broken_frags) == 0:
        # just a single fragment:
        num_atoms = mol.GetNumAtoms()
        if num_atoms < max_num_atoms_single_frag:
            return [mol], None, None, None, None,  None
        else:
            print('no fragment found')
        
    
    all_frags = Chem.FragmentOnBonds(mol, bonds_broken_indices, addDummies=True, dummyLabels=bonds_broken_frags)
    all_frags = Chem.GetMolFrags(all_frags, asMols=True)

    all_frag_atom_ids = []
    for frag in all_frags:
        all_frag_atom_ids.append(find_atom_ids(frag))
    
    atom2frag = find_atom2frag(all_frag_atom_ids)
    neighbors = find_neighboring_frags(all_frags, atom2frag, bonds_broken_frags)

    bonds_broken_frag_ids = find_bonds_broken_frags_ids(atom2frag, bonds_broken_frags)
    
    # get number of atoms in each fragment
    n_atoms_frags = [all_frags[i].GetNumAtoms() for i in range(len(all_frags))]
    
    bonds_broken_frags_new = [bond[0] for bond in FindBRICSBonds(mol)]
    
    
    if len(bonds_broken_frags_new) != 0:
        use_brics = True
        print('using brics +')
    else:
        use_brics = False
    
    
    # -------------------------------------------- BRICS ----------------------------------
    if use_brics:

        bonds_broken_frags_new = [bond[0] for bond in FindBRICSBonds(mol)]
        bonds_broken_frags_temp = list(set(bonds_broken_frags).union(set(bonds_broken_frags_new)))

        bonds_broken_frags= []
        for i, pair in enumerate(bonds_broken_frags_temp):
            if pair not in bonds_broken_frags and pair[::-1] not in bonds_broken_frags:
                bonds_broken_frags.append(pair)

        if len(bonds_broken_frags) == 0:
            return


        bonds_broken_indices = []
        for bond in bonds_broken_frags:
            bonds_broken_indices.append(mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx())  # all bond indices broken


        all_frags = Chem.FragmentOnBonds(mol, bonds_broken_indices, addDummies=True, dummyLabels=bonds_broken_frags)
        all_frags = Chem.GetMolFrags(all_frags, asMols=True)

        all_frag_atom_ids = []
        for frag in all_frags:
            all_frag_atom_ids.append(find_atom_ids(frag))

        atom2frag = find_atom2frag(all_frag_atom_ids)
        neighbors = find_neighboring_frags(all_frags, atom2frag, bonds_broken_frags)

        bonds_broken_frag_ids = find_bonds_broken_frags_ids(atom2frag, bonds_broken_frags)

        # get number of atoms in each fragment
        n_atoms_frags = [all_frags[i].GetNumAtoms() for i in range(len(all_frags))]

        # get number of atoms in each fragment

        for i in range(len(n_atoms_frags)):
            if n_atoms_frags[i] > max_num_atoms_single_frag: # the number of atoms is larger than the maximum
                print('large fragment detected again!')
                return 
        # --------------------------- checking the total number of fragments and iteratively removing them -----------------
        if len(all_frags) > max_num_frags:
            print('removing fragments for maximum number of fragments')
            single_conn_frag_ids = np.argwhere(neighbors.sum(axis=0) == 1.).flatten() # fragments with only one connection


            frag_candidates_to_del = []
            frag_candidates_to_del_numatoms = []

            for f_id in single_conn_frag_ids:

                # first check the fragis not part of the scaffold
                frag = all_frags[f_id]
                rN = frag.GetRingInfo().AtomRings()
                if len(rN) > 0:
                    continue

                n_atoms = frag.GetNumAtoms()
                frag_candidates_to_del.append(f_id)
                frag_candidates_to_del_numatoms.append(n_atoms)


            # sort the candidates to delete from fragments based on their number of atoms
            sorted_candidates = np.array(frag_candidates_to_del)[np.argsort(frag_candidates_to_del_numatoms)]
            candidate_bond_to_del = []
            candidate_bond_atomids_to_del = []
            n_frags_to_del = len(all_frags) - max_num_frags

            for i, bond in enumerate(bonds_broken_frag_ids):
                if bond[0] in sorted_candidates[:n_frags_to_del] or bond[1] in sorted_candidates[:n_frags_to_del]:

                    candidate_bond_to_del.append(bonds_broken_indices[i])
                    candidate_bond_atomids_to_del.append(bonds_broken_frags[i])



            bonds_broken_frags = list(set(bonds_broken_frags) - set(candidate_bond_atomids_to_del))
            bonds_broken_indices = [] # bond indices broken
            for bond in bonds_broken_frags:
                bonds_broken_indices.append(mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx())  # all bond indices broken

            all_frags = Chem.FragmentOnBonds(mol, bonds_broken_indices, addDummies=True, dummyLabels=bonds_broken_frags)
            all_frags = Chem.GetMolFrags(all_frags, asMols=True)



        if len(all_frags) > max_num_frags:
            print('maxmum number of fragments exceeded!')
            return None, None, None, None, None,  None
            # -------------------------------
        all_frag_atom_ids = [] # find the ids of the new fragments
        for frag in all_frags:
            all_frag_atom_ids.append(find_atom_ids(frag))

        atom2frag = find_atom2frag(all_frag_atom_ids) # find the new atom2frag
        bonds_broken_frag_ids = find_bonds_broken_frags_ids(atom2frag, bonds_broken_frags)

    return all_frags, bonds_broken_frags, bonds_broken_indices, bonds_broken_frag_ids, all_frag_atom_ids,  atom2frag