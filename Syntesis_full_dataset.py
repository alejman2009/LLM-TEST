# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:32:20 2024

@author: Alex-PC
"""

import os
from tqdm import tqdm
from datasets import load_dataset

os.chdir("G:/UNIVERSIDAD/4ºCurso/TFG/ENTORNO-ML/Datasets")
dataset = load_dataset("Skylion007/openwebtext", 
                       split="train",
                       cache_dir="G:/UNIVERSIDAD/4ºCurso/TFG/ENTORNO-ML/Datasets/cache", 
                       trust_remote_code=True)

print("Start")

with open("Dataset_opewebtext_full.txt", 'w',encoding='utf-8') as f:
    for data in tqdm(dataset):
        f.write(data['text'] + '\n')
print("End.")