import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import cwt, ricker

def plot_scalogram(data, time, scales, cmap='jet'):
    scalogram = cwt(data, ricker, scales)
    plt.imshow(np.abs(scalogram), aspect='auto', cmap=cmap, extent=[time[0], time[-1], scales[-1], scales[0]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Scale')

# Directory containing the .mat files
directory_path = "D:/MSEE21/Thesis/EEG_DataSets/SEED_EEG/SEED_EEG/Preprocessed_EEG"

# Output directory to save the scalograms
output_directory = "D:/MSEE21/raww"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get the list of .mat files
mat_files = [file for file in os.listdir(directory_path) if file.endswith(".mat")]

# Sort the .mat files
mat_files.sort()

# Define the scales for the CWT (you can adjust these based on your data)
scales = np.arange(1, 128)

# Iterate over the .mat files and their dictionaries
for mat_file in mat_files:
    mat_file_path = os.path.join(directory_path, mat_file)
    data = loadmat(mat_file_path)

    print(f"File: {mat_file}")
    dict_names = [key for key in data.keys() if isinstance(data[key], np.ndarray)]
    
    for i, dict_name in enumerate(dict_names, 1):
        signal = data[dict_name][0]  # Assuming the signal is stored as a 1D array in the dictionary
        time = np.linspace(0, len(signal) / 256, len(signal))  # Assuming a sampling rate of 256 Hz

        # Create a new figure for each scalogram
        plt.figure()
        plot_scalogram(signal, time, scales)
        plt.title(f"Dictionary {i} in {mat_file}")
        plt.tight_layout()

        # Save the scalogram to the output directory
        output_filename = f"scalogram_{os.path.splitext(mat_file)[0]}_{dict_name}.png"
        output_file = os.path.join(output_directory, output_filename)
        plt.savefig(output_file)
        plt.close()  # Close the figure to free memory

        print(f"Saved scalogram: {output_filename}")

    print()  # Print a new line between files
