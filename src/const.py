import torch
from rdkit import Chem

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int8

# #################################################################################### #
# ####################################### GEOM ####################################### #
# #################################################################################### #

# Atom idx for one-hot encoding
GEOM_ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
GEOM_IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P'}

GEOM_LJ_RM = [[120, 113, 116, 135, 160, 177, 194, 214, 184],
            [113, 121, 121, 142, 151, 164, 172, 194, 150],
            [116, 121, 110, 136, 168, 175, 214, 222, 177],
            [135, 142, 136, 142, 158, 166, 178, 187, 156],
            [182, 151, 168, 158, 204, 207, 225, 234, 186],
            [177, 164, 175, 166, 207, 199, 214, 0.0, 203],
            [194, 172, 214, 178, 225, 214, 228, 0.0, 222],
            [214, 194, 222, 187, 234, 0.0, 0.0, 266, 0.0],
            [184, 150, 177, 156, 186, 203, 222, 0.0, 221]]


# Atomic numbers (Z)
GEOM_CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}

# One-hot atom types
GEOM_NUMBER_OF_ATOM_TYPES = len(GEOM_ATOM2IDX)

# dataset keys
DATA_LIST_ATTRS = {
    'uuid', 'name', 'scaffold_smi', 'extension_smi', 'num_atoms'
}

DATA_ATTRS_TO_PAD = {
    'positions', 'one_hot', 'charges', 'anchors', 'scaffold_masks', 'extension_masks', 'pocket_mask', 'fragment_only_mask'
}

DATA_ATTRS_TO_ADD_LAST_DIM = {
    'charges', 'anchors', 'scaffold_mask', 'extension_mask', 'pocket_mask', 'fragment_only_mask'
}

# Distribution of extension sizes in data
GEOM_FRAG_SIZE_DIST = {  # TODO: find the distirbutions
1: 558549,
2: 266096,
3: 179173,
4: 61410,
5: 121410,
6: 382299,
7: 55336,
8: 22766,
9: 43852,
10: 34354,
11: 24275,
12: 7400, 
13: 7227,
14: 5397,
15: 3147,
16: 1994,
17: 978,
18: 895,
19: 436,
20: 107,
21: 81,
22: 6
}

GEOM_NUM_FRAGS_DIST = { # TODO:
    2: 1986,
    3: 8769,
    4: 22158,
    5: 39761,
    6: 51277,
    7: 53615,
    8: 97062,
}


GEOM_FRAGMENT_ID2SIZE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
GEOM_FRAGMENT_SIZE2ID = {
    size: idx
    for idx, size in enumerate(GEOM_FRAGMENT_ID2SIZE)
}

GEOM_FRAGMENT_SIZE_WEIGHTS = [0.0016, 0.0023, 0.0028, 0.0048, 
                              0.0034, 0.0019, 0.0050, 0.0078,
                               0.0056,  0.0064, 0.0076, 0.0137, 
                               0.0139, 0.0160, 0.0210, 0.0264, 
                               0.0377, 0.0394, 0.0565, 0.1140, 
                               0.1310, 0.4813]   

#  weights on sizes (inverse square root of class frequencies)

# #################################################################################### #
# ####################################### CrossDock ####################################### #
# #################################################################################### #

crossdock_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
crossdock_idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}

CROSSDOCK_NUMBER_OF_ATOMS = 10

EXTENSION_SIZE_DIST_CROSSDOCK = {  # TODO: find the distirbutions
1: 156918,
2: 35498,
3: 38461,
4: 24061,
5: 47018,
6: 101555,
7: 15199,
8: 11268,
9: 27994,
10: 20969,
11: 9285,
12: 3454, 
13: 3550,
14: 5180,
15: 1096,
16: 1238,
17: 1262,
18: 1254,
19: 557,
20: 527,
21: 397,
22: 91,
23: 67,
24: 13 
}

NUM_FRAGS_DIST_CROSSDOCK = { # TODO:
    1: 4399,
    2: 6799,
    3: 11423,
    4: 13097,
    5: 17413,
    6: 19485,
    7: 11301,
    8: 14897
}

CROSSDOCK_FRAGMENT_ID2SIZE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
CROSSDOCK_FRAGMENT_SIZE2ID = {
    size: idx
    for idx, size in enumerate(CROSSDOCK_FRAGMENT_ID2SIZE)
}

CROSSDOCK_FRAGMENT_SIZE_WEIGHTS = [0.0029, 0.0061, 0.0059, 0.0074, 0.0053, 0.0036, 0.0093, 0.0108, 0.0069,
                                   0.0079, 0.0119, 0.0195, 0.0193, 0.0160, 0.0347, 0.0326, 0.0323, 0.0324,
                                   0.0486, 0.0500, 0.0576, 0.1203, 0.1402, 0.3184]    #  TODO: weights on sizes (inverse square root of class frequencies)

CROSSDOCK_LJ_RM = [[120.0, 116.0, 113.0, 160.0, 133.0, 194.0, 177.0, 184.0, 214.0, 135.0], 
                   [116.0, 110.0, 121.0, 168.0, 127.0, 214.0, 175.0, 177.0, 222.0, 136.0], 
                   [113.0, 121.0, 121.0, 151.0, 126.0, 172.0, 164.0, 150.0, 194.0, 142.0], 
                   [160.0, 168.0, 151.0, 204.0, 167.0, 225.0, 207.0, 186.0, 234.0, 158.0], 
                   [133.0, 127.0, 126.0, 167.0, 146.0, 182.0, 175.0, 167.0, 198.0, 126.0], 
                   [194.0, 214.0, 172.0, 225.0, 182.0, 228.0, 214.0, 222.0, 234.0, 178.0], 
                   [177.0, 175.0, 164.0, 207.0, 175.0, 214.0, 199.0, 203.0, 218.0, 166.0], 
                   [184.0, 177.0, 150.0, 186.0, 167.0, 222.0, 203.0, 221.0, 219.0, 156.0], 
                   [214.0, 222.0, 194.0, 234.0, 198.0, 234.0, 218.0, 219.0, 266.0, 187.0], 
                   [135.0, 136.0, 142.0, 158.0, 126.0, 178.0, 166.0, 156.0, 187.0, 142.0]]

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
BONDS_1 = {
    'H': {
        'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
        'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
        'Cl': 127, 'Br': 141, 'I': 161
    },
    'C': {
        'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
        'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
        'I': 214
    },
    'N': {
        'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
        'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177
    },
    'O': {
        'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
        'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
        'I': 194
    },
    'F': {
        'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
        'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
        'I': 187
    },
    'B': {
        'H':  119, 'Cl': 175
    },
    'Si': {
        'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
        'F': 160, 'Cl': 202, 'Br': 215, 'I': 243,
    },
    'Cl': {
        'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
        'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
        'Br': 214
    },
    'S': {
        'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
        'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
        'I': 234
    },
    'Br': {
        'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
        'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222
    },
    'P': {
        'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
        'S': 210, 'F': 156, 'N': 177, 'Br': 222
    },
    'I': {
        'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
        'S': 234, 'F': 187, 'I': 266
    },
    'As': {
        'H': 152
    }
}

BONDS_2 = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}

BONDS_3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}

BOND_DICT = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND2IDX = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

ALLOWED_BONDS = {
    'H': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'B': 3,
    'Al': 3,
    'Si': 4,
    'P': [3, 5],
    'S': 4,
    'Cl': 1,
    'As': 3,
    'Br': 1,
    'I': 1,
    'Hg': [1, 2],
    'Bi': [3, 5]
}

# ----------------------- getting all lj potentials ---------------------
all_lj = []
for atom1 in GEOM_ATOM2IDX.keys():
    lj_row = []
    for atom2 in GEOM_ATOM2IDX.keys():
        
        if atom1 in BONDS_1.keys():
            if atom2 in BONDS_1[atom1]:
                lj_row.append(BONDS_1[atom1][atom2])
            else:
                lj_row.append(0.)
        
        if atom1 in BONDS_2.keys():
            if atom2 in BONDS_2[atom1]:
                lj_row[-1] = BONDS_2[atom1][atom2]
        
        if atom1 in BONDS_3.keys():
            if atom2 in BONDS_3[atom1]:
                lj_row[-1] = BONDS_3[atom1][atom2] 
                
         
        
    all_lj.append(lj_row)
# ----------------------------------------------------------------------

MARGINS_EDM = [10, 5, 2]

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
# RADII = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
RADII = [0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77]

ZINC_TRAIN_LINKER_ID2SIZE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ZINC_TRAIN_LINKER_SIZE2ID = {
    size: idx
    for idx, size in enumerate(ZINC_TRAIN_LINKER_ID2SIZE)}

