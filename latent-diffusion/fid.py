import subprocess
import os
import re
import glob


def compute_fid():
    folders = 'logs/8cell/samples/*' 
    folders = glob.glob(folders)

    # datasets = ['2cell_2'] #, '4cell', '8cell']
    with open('logs/fid_results.csv', 'a') as file:
        for folder in folders:
            print(folder)
            command = f"python -m pytorch_fid data/embryo/8cell {folder}"
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            fid_value = re.findall(r"FID:\s+(\d+\.\d+)", result.stdout)
            if fid_value:
                rounded_value = round(float(fid_value[0]))
                file.write(f"{rounded_value}\n")
            else:
                file.write("FID value not found\n")


def generate_images():
    dataset_folder = '/global/D1/homes/oriana/LDM/logs/4cell/' 
    ckpt_files = glob.glob(os.path.join(dataset_folder, '*.ckpt'))

    for ckpt_file in ckpt_files:
        command = f"python sample_diffusion.py --resume {ckpt_file} --n_samples 1136"
        subprocess.run(command, shell=True)

# python sample_diffusion.py --resume global/D1/homes/oriana/LDM/logs/2cell/my_ckpt_2cell/epoch=149-step=13200 --n_samples 5000
def rename_files():
    folder = "training-runs/2cell_2"
    for filename in os.listdir(folder):
        if filename.endswith(".pkl") and 'network-snapshot-' in filename:
            number = int(filename.split('-')[-1].split('.')[0])
            rounded_number = (int(number) // 10) * 10
            new_filename = f"{rounded_number}.pkl"
            os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))


compute_fid()