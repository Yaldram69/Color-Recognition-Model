import numpy as np
from PIL import Image
import random
import os

# Define the colors and their RGB values
color_dict = {
    'Black': (0, 0, 0),
    'Blue': (0, 0, 255),
    'Green': (0, 255, 0),
    'Gray': (128, 128, 128),
    'White': (255, 255, 255),
    'Yellow': (255, 255, 0),
    'Purple': (128, 0, 128),
    'Orange': (255, 165, 0),
    'Red': (255, 0, 0)
}

def generate_image_with_shade(filename, color, width=150, height=150):
    """
    Generate an image of a color with a random shade and optional noise.
    """
    # Create a blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply a random shade to the color
    shade_factor = random.uniform(0.6, 1.4)  # Range for more diverse shades
    shaded_color = [int(c * shade_factor) for c in color]
    shaded_color = [min(max(0, c), 255) for c in shaded_color]  # Ensure valid color range

    # Add random noise to color for more diversity
    noise = [random.randint(-15, 15) for _ in range(3)]  # Increased noise range
    shaded_color = [min(max(0, shaded_color[i] + noise[i]), 255) for i in range(3)]

    # Fill the image with the shaded color
    image[:] = shaded_color

    # Save the image
    img = Image.fromarray(image)
    img.save(filename)

# Directory for dataset
base_dataset_dir = "C:/Pycharm Projects/Color Recognition Model/Dataset"
os.makedirs(base_dataset_dir, exist_ok=True)

# Set the number of images per color
num_images_per_color = 1000  # Increased number of images for more data

# Generate the dataset
for color_name, color_rgb in color_dict.items():
    color_folder = os.path.join(base_dataset_dir, color_name)
    os.makedirs(color_folder, exist_ok=True)

    for i in range(num_images_per_color):
        filename = os.path.join(color_folder, f"{i}.png")
        generate_image_with_shade(filename, color_rgb)

print("Dataset generation complete!")
