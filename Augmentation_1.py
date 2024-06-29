#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np

# Define the paths
input_folder = "E:/Mel spectrograms_nobar/224x224"
output_folder = "E:/Mel spectrograms_nobar/Augmented_Mel_nobar"
csv_file = "E:/Mel spectrograms_nobar/image_names_MEL.csv"

# Create output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_file)

# Initialize the image augmenter
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Affine(rotate=(-10, 10)),  # rotate images within the range of -10 to 10 degrees
])

# Create a list to store augmented images and labels
augmented_data = []

# Iterate over the images
for i, row in df.iterrows():
    image_path = os.path.join(input_folder, row['images'])
    label = row['label']

    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)  # Convert PIL image to NumPy array

    # Perform augmentation
    augmented_images = []
    
    # Add original image to the list
    augmented_images.append(image_array)
    
    # Add flipped images to the list
    augmented_images.extend(seq.augment_images([image_array] * 3))
    
    # Add rotated images to the list
    for angle in range(-10, 11, 2):
        augmented_images.append(seq.augment_images([iaa.Affine(rotate=angle).augment_image(image_array)])[0])
    
    # Save augmented images
    for j, augmented_image in enumerate(augmented_images):
        output_path = os.path.join(output_folder, f"augmented_{i}_{j}.png")
        augmented_image_pil = Image.fromarray(augmented_image)  # Convert NumPy array back to PIL image
        augmented_image_pil.save(output_path)

        # Append the augmented image and label to the list
        augmented_data.append({'images': f"augmented_{i}_{j}.png", 'label': label})

# Convert the augmented data list to a DataFrame
augmented_df = pd.DataFrame(augmented_data)

# Concatenate the original DataFrame with the augmented DataFrame
combined_df = pd.concat([df, augmented_df], ignore_index=True)

# Save the modified DataFrame as a CSV file
combined_csv = os.path.join(output_folder, 'augmented_labels.csv')
combined_df.to_csv(combined_csv, index=False)