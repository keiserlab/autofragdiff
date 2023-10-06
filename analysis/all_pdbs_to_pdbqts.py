import os
import sys


if __name__ == '__main__':
    with open('/srv/home/mahdi.ghorbani/FragDiff/pdb_paths.txt', 'r') as f:
        all_files = [line.strip() for line in f.readlines()]
    root_dir = '/srv/home/mahdi.ghorbani/FragDiff/crossdock/crossdocked_pocket10/'
    for i, file in enumerate(all_files):
        if i % 100 == 0:
            print(i)
        prot_name = root_dir + file
        pdbqt_name = prot_name[:-3] + 'pdbqt'
        if not os.path.exists(pdbqt_name):
            os.system('prepare_receptor4.py -r {} -o {}'.format(prot_name, pdbqt_name))
