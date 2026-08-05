# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:09:03 2024

@author: Alex-PC
"""

import os
from tqdm import tqdm

os.chdir("G:/UNIVERSIDAD/4ºCurso/TFG/ENTORNO-ML/Datasets/")

def split_data(filename, train_filename, val_filename, split_ratio=0.8):
    with open(filename, 'r', encoding="utf-8") as f, \
         open(train_filename, 'w', encoding="utf-8") as train_file, \
         open(val_filename, 'w', encoding="utf-8") as val_file:
        
        for i, line in tqdm(enumerate(f)):
            if i % (1/split_ratio) < 1:
                # Write to train file
                train_file.write(line)
            else:
                # Write to validation file
                val_file.write(line)

# Usage
split_data('Dataset_opewebtext_full.txt', 'train.txt', 'val.txt', split_ratio=0.8)