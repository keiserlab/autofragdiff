import os
import re
import torch
from pathlib import Path
import argparse

import pandas as pd
from rdkit import Chem
from tqdm import tqdm 

affinity_pattern = r"Affinity:\s+(-?\d+\.\d+)\s+\(kcal/mol\)"
def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f'smina.static -l {sdf_file} -r {pdb_file} '
                   f'--score_only').read()
    matches = re.findall(
        r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]

def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'obabel {sdf_file} -O {pdbqt_outfile} -f {mol_id + 1} -l {mol_id + 1}').read()
    return pdbqt_outfile

def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=20, 
                            exhaustiveness=16, return_rdmol=False, score_only=False):
    """ 
    receptor_file: pdbqt file for receptor
    sdf_file: sdf file for ligand
    out_dir: output directory

    returns:
        scores: list of scores for each ligand
        rdmols: list of qvina docked ligands
    """
    
    receptor_pdbqt_file = Path(receptor_file)
    sdf_file = Path(sdf_file)
 
    scores = []
    rdmols = [] # for if return rdmols
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)

    for i, mol in enumerate(suppl): # sdf file may contain several ligands
        ligand_name = f'{sdf_file.stem}_{i}'
        # prepare ligand
        ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
        out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')

        if out_sdf_file.exists():
            with open(out_sdf_file, 'r') as f:
                scores.append(min([float(x.split()[2]) for x in f.readlines() 
                                   if x.startswith(' VINA RESULT:')]))
        else:
            sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

            # center box at ligand's center of mass
            cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

            # run QuckVina2
            # run QuickVina 2
            if not score_only:

                out = os.popen(
                    f'qvina2.1 --receptor {receptor_pdbqt_file} '
                    f'--ligand {ligand_pdbqt_file} '
                    f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                    f'--size_x {size} --size_y {size} --size_z {size} '
                    f'--exhaustiveness {exhaustiveness}'
                ).read()
                out_split = out.splitlines()
                best_ids = out_split.index('-----+------------+----------+----------') + 1
                best_line = out_split[best_ids].split()
                assert best_line[0] == '1'
                scores.append(float(best_line[1]))

                out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
                if out_pdbqt_file.exists():
                    os.popen(f'obabel {out_pdbqt_file} -O {out_sdf_file}').read()

                if return_rdmol:
                    rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
                    rdmols.append(rdmol)
            
            else:
                out = os.popen(
                    f'qvina2.1 --score_only --receptor {receptor_pdbqt_file} '
                    f'--ligand {ligand_pdbqt_file} '
                    f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                    f'--size_x {size} --size_y {size} --size_z {size} '
                ).read() 
                match = re.search(affinity_pattern, out)
                scores = float(match.group(1)) 

    if return_rdmol:
        return scores, rdmols
    else:
        return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser('QuickVina evaulation')
    parser.add_argument('--pdbqt_dir', type=Path,
                        help='Receptor files in pdbqt format')
    parser.add_argument('--sdf_dir', type=Path, default=None,
                        help='Ligand files in sdf format')
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('--write_dict', action='store_true')
    parser.add_argument('--dataset', type=str, default='CROSSDOCK')
    args = parser.parse_args()

    assert (args.sdf_dir is not None)

    results = {'receptor': [], 'ligand': [], 'scores':[]}
    results_dict = {}

    sdf_files = list(os.listdir(args.sdf_dir))
    pbar = tqdm(sdf_files)

    for sdf_file in pbar:
        pbar.set_description(f'Processing {sdf_file}')

        if args.dataset == 'CROSSDOCK':
            receptor_name = sdf_file.split('_')[0] + '_pocket'
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')

            sdf_path = Path(str(args.sdf_dir) + '/' + sdf_file)
        try:
            scores, rdmols = calculate_qvina2_score(receptor_file, sdf_path, args.out_dir, return_rdmol=True)
        except (ValueError, AttributeError) as e:
            print(e)
            continue
        results['receptor'].append(str(receptor_file))
        results['ligand'].append(str(sdf_file))
        results['scores'].append(scores)

        if args.write_dict:
            results_dict[receptor_name] = [scores, rdmols]
    
    if args.write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(args.out_dir, 'qvina2_scores.csv'))
    
    if args.write_dict:
        torch.save(results_dict, Path(args.out_dir, 'qvina2_scores.pt'))