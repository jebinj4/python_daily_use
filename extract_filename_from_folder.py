import os
import pandas as pd
import re

# Get the current working directory
directory_path = os.getcwd()  # Uses the script's running directory

# Function to extract numerical values for sorting
def extract_number(file_name):
    match = re.search(r'Widget\s*(\d+)(?:\s*-\s*Part\s*(\d+))?', file_name, re.IGNORECASE)
    if match:
        widget_num = int(match.group(1))  # Extract main widget number
        part_num = int(match.group(2)) if match.group(2) else 0  # Extract part number if exists
        return widget_num, part_num
    return float('inf'), float('inf')  # If no match, push to the end

# Function to properly format title case, handling hyphens and spaces
def to_proper_case(text):
    return ' '.join(word.capitalize() for word in re.split(r'[-\s]', text) if word)  # Capitalizes each word after spaces or hyphens

# Get a list of all MP4 files in the current directory (without extension)
file_names = [os.path.splitext(f)[0] for f in os.listdir(directory_path) if f.endswith(".mp4")]

# Sort files based on extracted numbers
sorted_files = sorted(file_names, key=extract_number)

# Convert filenames to Proper Title Case, ensuring hyphens & spaces are handled correctly
formatted_files = [
    re.sub(r'widget\s*(\d+)(.*)', lambda m: f'Widget {m.group(1)}-{to_proper_case(m.group(2))}', name, flags=re.IGNORECASE)
    for name in sorted_files
]

# Convert to DataFrame
df = pd.DataFrame(formatted_files, columns=["File Name"])

# Save to Excel
output_file = os.path.join(directory_path, "formatted_mp4_file_names.xlsx")
df.to_excel(output_file, index=False)

print(f"Formatted and sorted MP4 file names saved to {output_file}")
