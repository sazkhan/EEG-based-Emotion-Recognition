#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install librosa


# In[18]:


import os
import scipy.io as sio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

folder_path = "D:/MSEE21/Thesis/EEG_DataSets/SEED_EEG/SEED_EEG/Preprocessed_EEG"  # Replace with the folder path containing the .mat files
output_folder = "E:/Thesis/Melspectrograms" # Replace with the folder path to save the mel spectrograms

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all .mat files in the folder
for mat_file in os.listdir(folder_path):
    if mat_file.endswith(".mat"):
        file_path = os.path.join(folder_path, mat_file)
        data = sio.loadmat(file_path)

        # Iterate over all dictionaries in the .mat file
        for i, (key, value) in enumerate(data.items()):
            # Skip the dictionary with the file header information
            if key != '__header__':
                # Check if the value is an array-like object (e.g., ndarray)
                if isinstance(value, np.ndarray):
                    # Convert the EEG signal data to a NumPy array
                    eeg_signal = np.array(value, dtype=np.float32)

                    # Compute the mel spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(y=eeg_signal, sr=200)

                    # Resize the mel spectrogram to have shape (128, time_steps)
                    mel_spectrogram = np.resize(mel_spectrogram, (128, mel_spectrogram.shape[1]))

                    # Plot and save the mel spectrogram
                    plt.figure()
                    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max),
                                             sr=200, hop_length=512, x_axis='time', y_axis='mel')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(f"Mel Spectrogram of Dictionary {i} in {mat_file}")

                    # Save the mel spectrogram as an image file
                    output_file = f"{mat_file}_dict{i}_mel.png"
                    output_path = os.path.join(output_folder, output_file)
                    plt.savefig(output_path)
                    plt.close()


# In[1]:


import os
import csv

folder_path = "E:/Thesis/STFT spectrogarams"  # Replace with the folder path containing the images
output_dir = "E:/Thesis"  # Replace with the directory path to save the output file
output_file = os.path.join(output_dir, "image_names_STFT.csv")  # Full path of the output CSV file

# Create a list to store the image names
image_names = []

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_names.append(file_name)

# Write the image names to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image Name'])  # Write the header
    writer.writerows([[name] for name in image_names])

print("Image names saved to", output_file)


# In[ ]:




