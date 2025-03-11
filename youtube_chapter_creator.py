import os
import pandas as pd
import re
import cv2
from datetime import timedelta

# Get the current working directory
directory_path = os.getcwd()

# Function to extract numerical values for sorting
def extract_number(file_name):
    """
    Extracts the widget number and part number for sorting.
    Ensures "01.Introduction" is always first.
    """
    if file_name.startswith("01.Introduction"):  # Ensure Introduction always comes first
        return (-1, -1)
    
    match = re.search(r'Widget\s*(\d+)(?:\s*-\s*Part\s*(\d+))?', file_name, re.IGNORECASE)
    if match:
        widget_num = int(match.group(1))
        part_num = int(match.group(2)) if match.group(2) else 0
        return (widget_num, part_num)
    
    return (float('inf'), float('inf'))  # If no match, push to the end

# Function to format title case
def to_proper_case(text):
    return ' '.join(word.capitalize() for word in re.split(r'[-\s]', text) if word)

# Function to get video duration
def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0  # If file can't be read, return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_seconds = total_frames / fps if fps > 0 else 0
    cap.release()
    return duration_seconds

# Get all MP4 files in the directory
mp4_files = [f for f in os.listdir(directory_path) if f.endswith(".mp4")]

# Sort files based on extracted numbers
sorted_files = sorted(mp4_files, key=extract_number)

# Initialize variables
cumulative_time = 0
video_data = []

# Process each video file
for file in sorted_files:
    video_path = os.path.join(directory_path, file)
    duration = get_video_duration(video_path)

    # Extract and format title
    file_name_no_ext = os.path.splitext(file)[0]
    formatted_title = re.sub(r'widget\s*(\d+)(.*)', lambda m: f'Widget {m.group(1)}-{to_proper_case(m.group(2))}', file_name_no_ext, flags=re.IGNORECASE)

    # Generate timestamp
    timestamp = str(timedelta(seconds=int(cumulative_time)))  # Convert seconds to HH:MM:SS format
    video_data.append([file, formatted_title, timestamp, duration])

    # Update cumulative time
    cumulative_time += duration

# Convert to DataFrame
df = pd.DataFrame(video_data, columns=["File Name", "Formatted Title", "Timestamp", "Duration (sec)"])

# Save to Excel
output_excel = os.path.join(directory_path, "youtube_chapters.xlsx")
df.to_excel(output_excel, index=False)

# Generate YouTube-friendly format
youtube_chapters = "\n".join([f"{row[2]} - {row[1]}" for row in video_data])

# Save YouTube chapters to text file
output_txt = os.path.join(directory_path, "youtube_chapters.txt")
with open(output_txt, "w") as f:
    f.write(youtube_chapters)

# Print final result
print("\n🎬 YouTube Chapters Generated:\n")
print(youtube_chapters)
print(f"\n✅ Chapters saved to: {output_txt}")
print(f"📊 Excel file saved to: {output_excel}")
print(f"🕒 Total Duration: {str(timedelta(seconds=int(cumulative_time)))}")
