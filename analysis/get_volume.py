import subprocess
import tempfile
import numpy as np
import os

info_dict = {}
root_dir = '/Users/mahdimac/Science/Keiser_lab/diffusion/AutoFragDiff/scaffolds'
with tempfile.TemporaryDirectory() as tmp_dir:
    command = f"fpocket -f {root_dir}/1a2g.pdb"
    os.chdir(tmp_dir)

    #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = os.popen(command).read()
    #stdout, stderr = process.communicate()

    with open(os.path.join('1a2g_out', '1a2g_info.txt'), 'r') as fp:
        #file_content = fp.read()

        lines = fp.readlines()
        pocket_info_started = False

        for line in lines:
            line = line.strip()
            if line == "Pocket 1 :":
                pocket_info_started = True
                continue
            if pocket_info_started:
                if line == "":
                    break
                key, value = line.split(":")
                info_dict[key.strip()] = float(value.strip())

print(info_dict)