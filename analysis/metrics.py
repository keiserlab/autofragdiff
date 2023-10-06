import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from rdkit import Chem, DataStructs

from collections import Counter
from copy import deepcopy

from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem.QED import qed 

from analysis.SA_Score.sascorer import compute_sa_score


def is_connected(mol):
    try:
        mol_frags = Chem.GetMolFrags(mol, asMols=True)
    except Chem.rdchem.AtomValenceException:
        return False
    if len(mol_frags) != 1:
        return False
    return True

def is_valid(mol):
    try:
        Chem.SanitizeMol(mol)
    except:
        return False
    return True

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight

def get_rdkit_rmsd(mol, n_conf=20, random_seed=42, mode='energy'):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the n_conf rdkit conformers
    """
    
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    
    mol_smiles = Chem.MolToSmiles(mol)
    mol_smiles = Chem.MolFromSmiles(mol_smiles)
    mol3d = Chem.AddHs(mol)

    rmsd_list = []
    conf_energies = []
    # predict 3d
    try:
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
        for confId in confIds:
            AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
            rmsd = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(mol3d), refId=confId)
            rmsd_list.append(rmsd)
            #conf_energies.append(get_conformer_energies(mol3d))
        
        mol_energy = get_conformer_energies(Chem.AddHs(mol, addCoords=True))
        conf_energies = get_conformer_energies(mol3d)
        rmsd_list = np.array(rmsd_list)
        conf_lowest_en = np.argmin(conf_energies)
        
        mol = Chem.AddHs(mol)
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        conf_ids = [conf.GetId() for conf in mol3d.GetConformers()]
        conf = mol3d.GetConformer(conf_ids[conf_lowest_en])
        new_mol.AddConformer(conf, assignId=True)
        
        return rmsd_list[conf_lowest_en], new_mol, conf_energies, mol_energy
    except:
        return np.nan, np.nan, np.nan, np.nan

def get_logp(mol):
    return Crippen.MolLogP(mol)

def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = get_logp(mol)
    lipinski_score = obey_lipinski(mol)
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])

    return {
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski_score,
        'ring_size': ring_size
    }

def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """
    Get a force field for a molecule.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    force_field : str, optional
        Force Field name.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff

def get_conformer_energies(mol, force_field='mmff'):
    """
    Calculate conformer energies.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    force_field : str, optional
        Force Field name.
    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
        ff = get_molecule_force_field(mol, conf_id=conf.GetId(), force_field=force_field)
        ff.Minimize()
        energy = ff.CalcEnergy()
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies
