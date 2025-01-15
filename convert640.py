from PIL import Image
import os

# Define input and output directories
input_dir = "/home/oatgz/pcb/test_full_size/images"
output_dir = "/home/oatgz/pcb/test_640"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Resize images to 640x640
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Open and resize the image
        with Image.open(input_path) as img:
            img = img.resize((640, 640), Image.ANTIALIAS)
            img.save(output_path)
            print(f"Resized and saved: {output_path}")
