import numpy as np
import pandas as pd
import re
import argparse

from rdkit import Chem
from tqdm import tqdm 

from fragment_hierarchy import *
from prepare_fragments import *
from sascorer import calculateScore
from rdkit.Chem import rdchem

import os
from pathlib import Path

atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
atom_charges = {'C':6, 'N': 7, 'O': 8, 'S': 16, 'B':5, 'Br':35, 'Cl':17, 'P':15, 'I':53, 'F':9}
hybrid_to_onehot = {'SP':0, 'SP2': 1, 'SP3': 2}

def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def process_ligand(sdffile, max_num_frags=12, num_atoms_cutoff=22, add_QED=True, add_SA=True):
    try:
        mol = Chem.SDMolSupplier(str(sdffile))[0]

        add_Hs= True
        
        # get positions and ont-hot
        mol_pos = mol.GetConformer().GetPositions()
        all_symbols = []
        mol_onehot = []
        mol_onehot = []
        for atom in mol.GetAtoms():
            #all_symbols.append(atom.GetSymbol())
            atom_symb_onehot = get_one_hot(atom.GetSymbol(), atom_dict)
            hyb_onehot = np.eye(1,len(hybrid_to_onehot), hybrid_to_onehot[str(atom.GetHybridization())]).squeeze()
            aromatic_onehot = float(atom.GetIsAromatic())
            mol_onehot.append(np.concatenate([atom_symb_onehot, hyb_onehot, (aromatic_onehot,)]))

        # NOTE: adding extra node features (aromaticity and hybridization) see if these help
        mol_onehot = np.array(mol_onehot)

        mol_charges = []
        for atom in all_symbols:
            mol_charges.append(atom_charges[atom])

        # get charges
        mol_charges = np.array(mol_charges)

        output = find_bonds_broken_with_frags(mol, find_single_ring_fragments(mol), max_num_frags=max_num_frags, max_num_atoms_single_frag=num_atoms_cutoff)
        if output is not None:
            all_frags, bonds_broken_frags, bonds_broken_indices, \
            bonds_broken_frag_ids, all_frag_atom_ids,  atom2frag = output

            # -------------- get the smiles of fragments for making a fragment library
            du = Chem.MolFromSmiles('*')
            frag_smiles_temp = [Chem.MolFromSmiles(Chem.MolToSmiles(all_frags[i])) for i in range(len(all_frags))]

            frag_smiles = []
            frag_n_atoms = []
            for i in range(len(all_frags)):
                frag=AllChem.ReplaceSubstructs(frag_smiles_temp[i],du,Chem.MolFromSmiles('[H]'),True)[0]
                frag = Chem.RemoveAllHs(frag)
                frag_n_atoms.append(frag.GetNumAtoms())
                frag_smiles.append(Chem.MolToSmiles(frag))
            # --------------------------------------------------------------------

            if len(all_frags) > 1: # more than 1 fragment exists in the molecule
                adjacency = find_neighboring_frags(all_frags, atom2frag, bonds_broken_frags)
                neigh_frags_to_neigh_atoms = find_neigh_frags_to_neigh_atoms(bonds_broken_frags, bonds_broken_frag_ids)

                frag_to_id_dict = {}
                for i,frag in enumerate(all_frags):
                    frag_to_id_dict[frag] = i

                g = FragmentGraph(mol,
                            all_frags, 
                            adjacency,
                            frag_atom_ids=all_frag_atom_ids, 
                            frag_to_id_dict=frag_to_id_dict,
                            neigh_frags_to_neigh_atoms=neigh_frags_to_neigh_atoms)

                n_frags = len(all_frags)

                assert n_frags <= max_num_frags

                mol_atom_ids = []
                mol_extension_ids = []
                mol_anchor_ids = []
                mol_QED_scores = []
                mol_SA_scores = []
                all_sub_mols = []

                for order in ['BFS', 'DFS']:
                    for j in range(n_frags): # 5 different ways to reconstruct the molecule in total
                        hier, perm_atom_ids, perm_extensions_atom_ids, _, _, perm_anchor_ids = g.hierarchical_reconstruct(edge_order=order, starting_point=j)
                        # save this hierarchy     

                        assert len(perm_atom_ids) != 0
                        assert len(perm_anchor_ids) != 0
                        assert len(perm_extensions_atom_ids) != 0

                        assert len(perm_extensions_atom_ids) == len(perm_atom_ids)
                        assert (len(perm_extensions_atom_ids) == len(perm_anchor_ids) + 1)

                        num_atoms = [len(perm_extensions_atom_ids[i]) for i in range(len(perm_extensions_atom_ids))] 
                        max_num_atoms = max(num_atoms)
                        if max_num_atoms > num_atoms_cutoff:
                            break
                        mol_atom_ids.append(perm_atom_ids)
                        mol_extension_ids.append(perm_extensions_atom_ids)
                        mol_anchor_ids.append(perm_anchor_ids)

                        QED_scores = []
                        SA_scores = []
                        if add_QED:
                            for i in range(len(all_frags)):
                                atom_indices = list(perm_atom_ids[i])
                                sub_mol = rdchem.EditableMol(Chem.Mol())
                                atom_map = {}
                                for atom_idx in atom_indices:
                                    atom = mol.GetAtomWithIdx(atom_idx)
                                    new_idx = sub_mol.AddAtom(atom)
                                    atom_map[atom_idx] = new_idx

                                for bond in mol.GetBonds():
                                    begin_idx = bond.GetBeginAtomIdx()
                                    end_idx = bond.GetEndAtomIdx()
                                    if begin_idx in atom_indices and end_idx in atom_indices:
                                        bond_type = bond.GetBondType()
                                        sub_mol.AddBond(atom_indices.index(begin_idx), atom_indices.index(end_idx), bond_type)


                                sub_mol = sub_mol.GetMol()
                                # Adding 3d Coordinates to the fragments
                                try:
                                    Chem.SanitizeMol(sub_mol)
                                    conf = Chem.Conformer(sub_mol.GetNumAtoms())
                                    for atom_idx, new_atom_idx in atom_map.items():
                                        conf.SetAtomPosition(new_atom_idx, mol.GetConformer().GetAtomPosition(atom_idx))
                                    sub_mol.AddConformer(conf)
                                except:
                                    print('sanitization failed! using smarts instead!')
                                    sub_mol = Chem.MolFromSmarts(Chem.MolToSmarts(sub_mol))
                                    Chem.SanitizeMol(sub_mol)
                                    conf = Chem.Conformer(sub_mol.GetNumAtoms())
                                    for atom_idx, new_atom_idx in atom_map.items():
                                        conf.SetAtomPosition(new_atom_idx, mol.GetConformer().GetAtomPosition(atom_idx))
                                    sub_mol.AddConformer(conf)

                                #sub_mol = Chem.MolFromSmarts(Chem.MolToSmarts(sub_mol))
                                #Chem.SanitizeMol(sub_mol)
                                if add_Hs:
                                    sub_mol_h = Chem.AddHs(sub_mol, addCoords=True)

                                all_sub_mols.append(sub_mol_h)

                                QED_scores.append(Chem.QED.qed(sub_mol))  
                                if add_SA:
                                    sa = calculateScore(sub_mol)
                                    sa_as_pocket2mol = round((10-sa)/9, 2) # from pocket2mol
                                    SA_scores.append(sa_as_pocket2mol)
                                

                        mol_QED_scores.append(QED_scores)
                        mol_SA_scores.append(SA_scores) 

                mol_atom_ids = np.array(mol_atom_ids, dtype=object)
                mol_extension_ids = np.array(mol_extension_ids, dtype=object)
                mol_anchor_ids = np.array(mol_anchor_ids, dtype=object)
                
                is_single_frag = False

                return mol_pos, mol_onehot, mol_charges, mol_atom_ids, mol_extension_ids, mol_anchor_ids, is_single_frag, frag_smiles, frag_n_atoms, mol_QED_scores, mol_SA_scores, all_sub_mols
            else:
                print('using single fragment')
                is_single_frag = True
                mol_H = Chem.AddHs(mol, addCoords=True)
                all_sub_mols = [mol_H]
                mol_QED_score = [Chem.QED.qed(mol)]
                sa = calculateScore(mol)
                mol_SA_score = [round((10-sa)/9, 2)]
                return mol_pos, mol_onehot, mol_charges, None, None, None, is_single_frag, frag_smiles, frag_n_atoms, mol_QED_score, mol_SA_score, all_sub_mols

    except Exception as e:
        print(f'Error {e} for sdffile {sdffile}')
        return 