import os
from PIL import Image
import pillow_heif

def convert_all_heic_to_jpg(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".heic"):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.splitext(input_path)[0] + ".jpg"
            
            try:
                heif_file = pillow_heif.read_heif(input_path)
                image = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data
                )
                image.save(output_path, "JPEG")
                print(f"✅ Converted: {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"❌ Failed to convert {filename}: {e}")

# Your actual path (use raw string to avoid \ issues)
folder_path = r"I:\Clients\Eco Blooms\Project 1 25-05-2025\Meera shoot\Meera shoot\HEIC"
convert_all_heic_to_jpg(folder_path)
