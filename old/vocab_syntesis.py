# -*- coding: utf-8 -*-

from tqdm import tqdm
import os

os.chdir("G:/UNIVERSIDAD/4ºCurso/TFG/ENTORNO-ML/Datasets/")

#Funcion para crear el vocab
def create_vocab(file_path, vocab_path, chunk_size=1024*1024*512):
    num_chunk=0;
    written_chars = set()
    with open(file_path, 'r',encoding="utf-8") as f, open(vocab_path, 'w', encoding="utf-8") as v:
        while True:
            num_chunk =+ 1
            print("Chunk Iteration: " + num_chunk)
            chunk = f.read(chunk_size)
            if not chunk:  # end of file
                break
            for char in tqdm(chunk):
                if char not in written_chars:
                    #print("\rActual:"+ char ,end="",flush=True)
                    v.write(char + '\n')
                    written_chars.add(char)

# usage
create_vocab('Dataset_opewebtext_full.txt', 'vocab.txt')