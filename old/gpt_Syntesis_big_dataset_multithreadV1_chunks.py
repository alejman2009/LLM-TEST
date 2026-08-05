## Preparing the dataset

import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_data(data):
    return data['text']


os.chdir("G:/UNIVERSIDAD/4ºCurso/TFG/ENTORNO-ML/Datasets")

from datasets import load_dataset
dataset = load_dataset("Skylion007/openwebtext", 
                       split="train",
                       cache_dir="G:/UNIVERSIDAD/4ºCurso/TFG/ENTORNO-ML/Datasets/cache", 
                       trust_remote_code=True)
print("Done...\n")

#Procesado del dataset

text = ""

text_chunks = []  # List to store processed text chunks

chunk_size = 1000  # Number of samples to process in each chunk

with ThreadPoolExecutor() as executor:
    for chunk_start in tqdm(range(0, len(dataset), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        chunk = dataset[chunk_start:chunk_end]  # Get a chunk of the dataset
        results = list(executor.map(process_data, chunk))  # Process the chunk
        text_chunks.extend(results)  # Store the processed text chunk


text = "".join(results)

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)

#Almacenamiento del dataset como un formato .pkl

import pickle
dataset_pkl_file = "LLM_Dataset_opewebtext_full.pkl"
with open(dataset_pkl_file, 'wb') as file:
    pickle.dump(text, file)