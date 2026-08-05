import os
import tempfile
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

# Define a function to write text chunk to temporary file
def write_to_temp_file(text_chunk):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        temp_file.write(text_chunk)
        return temp_file.name

# Process the dataset and write text chunks to temporary files
temp_files = []
with ThreadPoolExecutor() as executor:
    for text_chunk in tqdm(executor.map(process_data, dataset), total=len(dataset)):
        temp_file_path = write_to_temp_file(text_chunk)
        temp_files.append(temp_file_path)

# Concatenate temporary files into a single file
merged_text_file = "merged_text.txt"
with open(merged_text_file, 'w', encoding='utf-8') as output_file:
    for temp_file_path in temp_files:
        with open(temp_file_path, 'r', encoding='utf-8') as temp_file:
            print("Writing chunk to merged text file:", text_chunk[:50])
            output_file.write(temp_file.read())
        os.remove(temp_file_path)  # Delete temporary file after reading

# Load merged text file and pickle it
with open(merged_text_file, 'r', encoding='utf-8') as file:
    text = file.read()

# Almacenamiento del dataset como un formato .pkl
import pickle
dataset_pkl_file = "LLM_Dataset_opewebtext_full.pkl"
with open(dataset_pkl_file, 'wb') as file:
    pickle.dump(text, file)