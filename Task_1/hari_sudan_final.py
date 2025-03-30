# Hari Sudan 
# Final Project
# Fish Tracking and Analysis

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from skimage import io, color
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
import pandas as pd

#  Setup 
path = 'Archive'
os.chdir(path)

image_files = sorted([i for i in os.listdir() if i.endswith(".png")])
seq_frames = sorted([[int(f.replace('tframe', '').replace('.png', '')), f] for f in image_files])
img3c_list = [io.imread(f[1]) for f in seq_frames]
img_list = [color.rgb2gray(img) for img in img3c_list]

#  Normalize and subtract median 
def normframe(frame):
    return (frame - frame.min()) / (frame.max() - frame.min())

class img_seq_n:
    def __init__(self, pics):
        self.gray = np.asarray([normframe(f) for f in pics])
        self._median = np.median(pics, axis=0)
        self.medr = np.asarray([normframe(f - self._median) for f in pics])

pics = img_seq_n(img_list)

#  Smart Fish Detection and Save 
output_folder = 'fish_marked_smart'
os.makedirs(output_folder, exist_ok=True)

fish_positions = []
prev_img = None

def draw_and_save(img, region, fname, frame_number):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    minr, minc, maxr, maxc = region.bbox
    rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                     edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)

    cy, cx = region.centroid
    label_text = f"({int(cx)}, {int(cy)}) @ Frame {frame_number}"
    ax.text(minc, minr - 5, label_text, color='yellow', fontsize=8, weight='bold', backgroundcolor='black')

    ax.axis('off')
    plt.savefig(os.path.join(output_folder, fname), bbox_inches='tight', pad_inches=0)
    plt.close()
    
# -------------------------------------------------------

for i, img in enumerate(pics.medr):
    region = None
    fname = f"frame_{i:03}.png"

    # Fixed threshold
    bw = np.where(img > 0.4, 0, 1)
    props = regionprops(label(bw))
    region = next((r for r in props if 40 < r.area < 200), None)

    # Gaussian blur
    if not region:
        blurred = gaussian(img, sigma=1)
        bw = np.where(blurred > 0.4, 0, 1)
        props = regionprops(label(bw))
        region = next((r for r in props if 40 < r.area < 200), None)

    # Otsu threshold
    if not region:
        try:
            thresh_val = threshold_otsu(img)
            bw = np.where(img > thresh_val, 0, 1)
            props = regionprops(label(bw))
            region = next((r for r in props if 40 < r.area < 200), None)
        except:
            pass

    # Frame differencing
    if not region and prev_img is not None:
        diff = np.abs(img - prev_img)
        bw = np.where(diff > 0.1, 1, 0)
        props = regionprops(label(bw))
        region = next((r for r in props if 30 < r.area < 400), None)

    if region:
        cy, cx = region.centroid
        fish_positions.append({"frame": i, "x_pos": round(cx, 2), "y_pos": round(cy, 2)})
        draw_and_save(img, region, fname, i)
    else:
        fish_positions.append({"frame": i, "x_pos": None, "y_pos": None})
        print(f"[✖] No fish detected in {fname}")

    prev_img = img
# -------------------------------------------------------

#  Save CSV 
df = pd.DataFrame(fish_positions).dropna()
df.to_csv("fish_positions.csv", index=False)

#  Plot Vertical Movement Over Time 
plt.figure(figsize=(10, 5))
plt.plot(df["frame"], df["y_pos"], marker='o', linestyle='-', color='blue')
plt.xlabel("Frame Number")
plt.ylabel("Fish Vertical Position (Y)")
plt.title("Fish Movement Over Time")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("fish_movement_plot.png", dpi=300)
plt.show()

#  Plot X vs Y Trajectory (Colored by Frame) 
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["x_pos"], df["y_pos"], c=df["frame"], cmap='plasma', s=50)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Fish 2D Trajectory (X vs Y)")
plt.gca().invert_yaxis()
plt.colorbar(scatter, label='Frame Index')
plt.grid(True)
plt.tight_layout()
plt.savefig("fish_trajectory_path.png", dpi=300)
plt.show()

#  GUI Slider Viewer 
i0_frame = 0
fig_gui, ax_gui = plt.subplots(figsize=(8, 10))
img_display = ax_gui.imshow(pics.medr[i0_frame], cmap=plt.cm.Greys_r)
label_display = ax_gui.text(10, 20, "", color='yellow', fontsize=10, backgroundcolor='black')

sFID = Slider(plt.axes([0.1, 0.02, 0.8, 0.03]), "Frame ID", 0, len(pics.medr) - 1,
              valinit=i0_frame, valfmt="%i")

def update(val):
    idx = int(sFID.val)
    img_display.set_data(pics.medr[idx])
    point = next((item for item in fish_positions if item['frame'] == idx and item['x_pos'] is not None), None)
    if point:
        label_display.set_text(f"({int(point['x_pos'])}, {int(point['y_pos'])}) @ Frame {idx}")
    else:
        label_display.set_text("No fish")
    plt.draw()

sFID.on_changed(update)
plt.show()