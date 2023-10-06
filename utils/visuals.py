import py3Dmol
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Geometry import Point3D
from openbabel import openbabel
import numpy as np
from openbabel import openbabel
import tempfile


atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
idx2atom = {0:'C', 1:'N', 2:'O', 3:'S', 4:'B', 5:'Br', 6:'Cl', 7:'P', 8:'I', 9:'F'}

def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)

def visualize_molecules_grid(mols, grid_size=(3, 3), spacing=5.0, spin=True):
    viewer = py3Dmol.view(width=900, height=900)
    
    for i, mol in enumerate(mols):
        try:
            Chem.SanitizeMol(mol)
        except:
            print('couldnt sanitize')
        #AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
        #AllChem.MMFFOptimizeMolecule(mol, maxIters=500)  # Optimize the geometry using MMFF94 force field

        # Calculate the grid position
        grid_x = i % grid_size[0]
        grid_y = i // grid_size[0]

        # Translate the molecule according to its position in the grid
        conf = mol.GetConformer()
        translation_vector = Point3D((grid_x * spacing) + (spacing / 2), (grid_y * spacing) + (spacing / 2), 0.0)
        for atom_idx in range(mol.GetNumAtoms()):
            atom_position = conf.GetAtomPosition(atom_idx)
            atom_position += translation_vector
            conf.SetAtomPosition(atom_idx, atom_position)

        mb = Chem.MolToMolBlock(mol)
        viewer.addModel(mb, 'sdf')
        
        #if spin:
        #    viewer.spin({'x': 0, 'y': 1, 'z': 0})

    # Draw separating lines
    for i in range(grid_size[0] - 1):
        x = (i + 1) * spacing
        viewer.addLine({'start': {'x': x, 'y': 0, 'z': 0},
                        'end': {'x': x, 'y': grid_size[1] * spacing, 'z': 0},
                        'color': 'gray'})
    for i in range(grid_size[1] - 1):
        y = (i + 1) * spacing
        viewer.addLine({'start': {'x': 0, 'y': y, 'z': 0},
                        'end': {'x': grid_size[0] * spacing, 'y': y, 'z': 0},
                        'color': 'gray'})
    
    #viewer.spin({'x': 0, 'y': 1, 'z': 0}, origin=(grid_size[0] * spacing / 2, grid_size[1] * spacing / 2, 0))
    viewer.setStyle({}, {'stick': {'colorscheme': ['silverCarbon', 'redOxygen', 'blueNitrogen'], 'radius': 0.15, 'opacity': 1},
                         'sphere': {'colorscheme': ['silverCarbon', 'redOxygen', 'blueNitrogen'], 'radius': 0.35, 'opacity': 1}})
    viewer.zoomTo()
    viewer.show()

def get_pocket_mol(pocket_coords, pocket_onehot):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        
        atom_inds= pocket_onehot.argmax(1)
        atom_types = [idx2atom[x] for x in atom_inds]
        # write xyz file
        write_xyz_file(pocket_coords, atom_types, tmp_file)
    
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats('xyz', 'sdf')
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)
    
        obConversion.WriteFile(ob_mol, tmp_file)
        pocket_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0] 

    return pocket_mol

def visualize_3d_pocket_molecule(pocket_mol, mol=None, spin=False, optimize_coords=False, sphere_positions1=None, sphere_positions2=None, rotate=None):
    viewer = py3Dmol.view()

    pocket_mol = Chem.RemoveHs(pocket_mol)
    pocket_mb = Chem.MolToMolBlock(pocket_mol)
    viewer.addModel(pocket_mb, 'sdf')
    viewer.setStyle({'model': -1}, {"sphere": {'color': 'grey', 'opacity': 0.8, 'radius':0.9}})
    #viewer.setStyle({'model': 0}, {'stick': {'colorscheme': ['whiteCarbon', 'redOxygen', 'blueNitrogen'], 'radius': 0.2, 'opacity': 1},
    #                               'sphere': {'colorscheme': ['whiteCarbon', 'redOxygen', 'blueNitrogen'], 'radius': 0.3, 'opacity': 1}})
    
    viewer.zoomTo()
    #viewer.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}}) # Updated style for cartoon representation
    #viewer.addSurface(py3Dmol.SAS, {'opacity': 0.9, 'radius': 0.5})

    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except:
            print('Problem with the molecule')
            return
    
        mol = Chem.RemoveHs(mol)
        mol_mb = Chem.MolToMolBlock(mol)
        viewer.addModel(mol_mb, 'sdf')
        viewer.setStyle({'model': 1}, {'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.15, 'opacity': 1},
                                       'sphere': {'colorscheme': 'cyanCarbon', 'radius': 0.35, 'opacity': 1}})
    
    if sphere_positions1 is not None:
        for pos in sphere_positions1:
            sphere_spec = {'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])}, 'radius': 1, 'color': 'green', 'opacity': 0.75}
            viewer.addSphere(sphere_spec)
    
    if sphere_positions2 is not None:
        for pos in sphere_positions2:
            sphere_spec = {'center': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])}, 'radius': 0.3, 'color': 'yellow', 'opacity': 0.75}
            viewer.addSphere(sphere_spec)


    if spin:
        viewer.spin({'x': 0, 'y': 1, 'z': 0})
    
    if rotate:
        viewer.rotate(rotate,'y',1);
    return viewer
