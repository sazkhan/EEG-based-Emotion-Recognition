#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def convert_dict_to_spectrogram(dictionary):
    plt.specgram(dictionary)
    plt.xlabel('Time')
    plt.ylabel('Frequency')

# Directory containing the .mat files
directory_path = "/content/drive/MyDrive/SEED_EEG/Preprocessed_EEG/"

# Output directory to save the spectrograms
output_directory = "/content/drive/MyDrive/675"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get the list of .mat files
mat_files = [file for file in os.listdir(directory_path) if file.endswith(".mat")]

# Sort the .mat files
mat_files.sort()

# Iterate over the .mat files and their dictionaries
for mat_file in mat_files:
    mat_file_path = os.path.join(directory_path, mat_file)
    data = loadmat(mat_file_path)

    print(f"File: {mat_file}")
    dict_names = [key for key in data.keys() if isinstance(data[key], np.ndarray)]
    
    for i, dict_name in enumerate(dict_names, 1):
        spectrogram = data[dict_name]

        # Create a new figure for each spectrogram
        plt.figure()
        convert_dict_to_spectrogram(spectrogram)
        plt.title(f"Dictionary {i} in {mat_file}")
        plt.tight_layout()

        # Save the spectrogram to the output directory
        output_filename = f"spectrogram_{os.path.splitext(mat_file)[0]}_{dict_name}.png"
        output_file = os.path.join(output_directory, output_filename)
        plt.savefig(output_file)
        plt.close()  # Close the figure to free memory

        print(f"Saved spectrogram: {output_filename}")

    print()  # Print a new line between files

