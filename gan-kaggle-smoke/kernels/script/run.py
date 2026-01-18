import os, sys

print("Inputs:", os.listdir("/kaggle/input"))

CODE_DIR = "/kaggle/input/ganlab-lib"
sys.path.append(CODE_DIR)

from ganlab import train

if __name__ == "__main__":
    # długi profil
    train("train")
