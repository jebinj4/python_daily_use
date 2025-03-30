#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 10 17:10:52 2025

@author: michaelwinklhofer
"""

import os
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Rectangle

import skimage as ski  # scimage= scikitimage
from skimage import io  # scimage= scikitimage
from skimage.measure import label, regionprops, regionprops_table

path='Archive'  
os.chdir(path)

image_files = [i for i in os.listdir() if i.endswith(".png") ]


"""

# pick random file
frame= io.imread(image_files[14] )

ny,nx,ncolors =frame.shape # e.g., (362,152) pixels (high, wide)

colmap=[ plt.cm.Reds,plt.cm.Greens, plt.cm.Blues ]


fig, ax=plt.subplots(ncols=3)
for i in range(3):
    ax[i].imshow(img[:,:,i],cmap=colmap[i]) 
        
# CONVERT TO GRAY SCALE (0. .. 1.)
g=ski.color.rgb2gray(img)

"""

# image_files 
# list needs to be sorted according to actual order in the sequence 
# we extract the number of the frame in the sequence

""" 
file0=image_files[0]
file0.split('tframe')[-1].split('.png')[0]
file0.replace('tframe','').replace('.png','')
"""

seq_frames=sorted([ [ int(frame.replace('tframe','').replace('.png','')), frame] for frame in image_files ])

def read_img(seq): 
    for img in seq: yield img

img3c_list=[ io.imread(img[1]) for img in read_img(seq_frames) ] 

# monochromatic
img_list=[ ski.color.rgb2gray(img) for img in img3c_list ] 


# colorplot 
fig,ax=plt.subplots(10,30)
ax=ax.ravel()
for i,axis in enumerate(ax):
    axis.imshow(img3c_list[i]) 
    axis.set_xticks([],None)    
    axis.set_yticks([],None)    

#%% identify light-on phases automatically
# where was the light on? Mean hue value of the picture should be close to 1 

hsv_list=np.asarray([ ski.color.rgb2hsv(pic.mean(axis=(0,1))) for pic in img3c_list ])

fig,ax=plt.subplots(1,3)
for axis,idx in zip(ax,range(3)):
    axis.plot(hsv_list[:,idx])

dark= hsv_list[:,0]<0.75
x=np.arange(len(hsv_list[:,0]))
ax[0].scatter(x[~dark],hsv_list[~dark,0],c='m')


#%%  Slider to move through Frame ID

i0_frame=0

mov, axm=plt.subplots(figsize=(7,15))
axh=axm.imshow(img3c_list[i0_frame]) 

sFID= Slider( ax = plt.axes([0.2,0.05,0.6,0.03]), label="Frame ID", 
                    valmin=i0_frame,valmax=len(image_files)-1,
                    valinit=i0_frame,valfmt="%i")

def update(val):
    axh.set_data( img3c_list[int(val)])
    plt.draw()
    
    
sFID.on_changed(update) 

#%%   

# 3channel images no longer needed
# del(img3c_list)

#%%

# Image "Registration"
#
# define a container for the images to switch between original, median removed and diffimage

class img_seq:
    def __init__(self,pics):
        self.gray    = np.asarray(pics)
        self._median = np.median(self.gray,axis=0)
        self.medr    = self.gray-self._median
    def show_median(self):
        fig,ax=plt.subplots()
        ax.imshow(self._median,cmap=plt.cm.Greys_r)
        ax.set_title("Median Image")
        return fig,ax
        
pics=img_seq(img_list)
pics.show_median()


#%%

i0_frame=0
mov, axm=plt.subplots(figsize=(7,10))
axh=axm.imshow(pics.medr[i0_frame],cmap=plt.cm.Greys_r) 


sFID= Slider( ax = plt.axes([0.4,0.02,0.2,0.03]), label="Frame ID", 
                    valmin=i0_frame,valmax=len(image_files)-1,
                    valinit=i0_frame,valfmt="%i")

def update(*args):
    axh.set_data(pics.medr[int(sFID.val)])        
     
sFID.on_changed(update) 



#%%

def normframe(frame: np.array):
     return (frame-frame.min())/(frame.max()-frame.min())
def normseq(seq):
     return [ normframe(frame) for frame in seq ]
 

class img_seq_n:
    def __init__(self,pics):
        self.gray    = np.asarray(self.__normseq(pics))
        self._median = np.median(pics,axis=0)
        self.medr    = np.asarray(self.__normseq(pics-self._median))

    @staticmethod
    def __normseq(seq):
        return [ normframe(frame) for frame in seq ]


pics=img_seq_n(img_list)

#%%

mov, axm=plt.subplots(figsize=(7,10))
axh=axm.imshow(pics.medr[i0_frame],cmap=plt.cm.Greys_r) 


sFID= Slider( ax = plt.axes([0.1,0.02,0.8,0.03]), label="Frame ID", 
                    valmin=i0_frame,valmax=len(image_files)-1,
                    valinit=i0_frame,valfmt="%i")


attributes= [ d for d in dir(pics) if not d.startswith("_") ]

rmode= RadioButtons( plt.axes([0.05,0.1,0.1,0.1]), 
                    tuple(attributes))

bmode= RadioButtons( plt.axes([0.05,0.2,0.1,0.1]), 
                    ('bin','nonbin'))

thresh= Slider( plt.axes([0.05,0.3,0.1,0.05]), 
              label="bin threshold", 
                                  valmin=-1,valmax=1,
                                  valinit=0,valfmt="%5f.2")

def update(*args):
    
    if bmode.value_selected == 'bin':
        axh.set_data(np.where(getattr(pics,rmode.value_selected)[int(sFID.val)] > thresh.val,0,1))    
    else:
        axh.set_data(getattr(pics,rmode.value_selected)[int(sFID.val)])        
    plt.draw()
    
    
thresh.on_changed(update) 
sFID.on_changed(update) 
bmode.on_clicked(update)
rmode.on_clicked(update)
plt.show()

#%% labelling 

d={}


bw=np.where(pics.medr[265]>0.4,0,1) # fish is a single, contiguous object

bw=np.where(pics.medr[260]>0.4,0,1) # fish disperses into two objects

bw=np.where(pics.medr[92]>0.4,0,1) # fish plus smaller speckles

bw_label=label(bw)

r_props = regionprops_table(bw_label)

props = regionprops_table(
    bw_label,
    properties=('area','centroid', 'orientation', 
                'axis_major_length', 'axis_minor_length',
                'eccentricity',),
)

props['aratio']=np.sqrt(1-props['eccentricity']**2)  #props['axis_minor_length']/props['axis_major_length']
props['label']=[ str(el) for el in (1+np.arange(len(props['aratio']))) ]



"""

if you want to convert the dict into a data frame right now
import pandas as pd
dfr=pd.DataFrame(props)

good=(dfr.area > 40) & (dfr.area < 200) & (dfr.aratio < 0.7 )


d[92]=dfr[good]

"""

#%%
f,a=plt.subplots()
a.imshow(bw,cmap=plt.cm.Greys)

for i in range(len(props['label'])):
    if 40 < props['area'][i] < 200:
        # bbox=> (min_row, min_col, max_row, max_col) 
        xy=(r_props["bbox-1"][i],r_props["bbox-0"][i]) # min_col, min_row
        dx=r_props["bbox-3"][i]-r_props["bbox-1"][i] # max_col-min_col
        dy=r_props["bbox-2"][i]-r_props["bbox-0"][i] # max_row-min_row
    
        rect=plt.Rectangle(xy,dx,dy,alpha=0.4)
        a.add_patch(rect)


# eccentricity e: e=np.sqrt(1- (axis_minor_length/axis_major_length)**2)  ) 
a.scatter(props["centroid-1"],props["centroid-0"],c='r',marker="o")



#%% 

# your task
# 
# write a loop over all images in the sequence 
# where you isolate the putative fish in each frame.
# save the selected region_prop in a list of dictionaries
# and plot the fish position (up-down-coordinate) as a function of sequence number

#%%#%%
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
import pandas as pd

# === Load specific frame (tframe18) and detect fish ===
img_path = "tframe18.png"  # make sure you're in the Archive directory
img = imread(img_path)
gray = rgb2gray(img)

# Use interactive settings: median-removed frame, non-binarized view, threshold = 0.2
bw = gray < 0.4  # original thresholding for grayscale view
lbl = label(bw)
props = regionprops(lbl)

best_region = None
for region in props:
    area = region.area
    ecc = region.eccentricity
    cy, cx = region.centroid

    if 30 < area < 500 and ecc > 0.85 and cy < 300:
        if best_region is None or region.area > best_region.area:
            best_region = region

# === Show overlay image with detected fish point ===
if best_region:
    y_pos = best_region.centroid[0]
    x_pos = best_region.centroid[1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(gray, cmap='gray')  # background image
    ax.plot(x_pos, y_pos, 'ro', markersize=10, label="Fish Position", alpha=0.8)
    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Fish Position on Frame tframe18.png")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig("fish_overlay_tframe18.png", dpi=300)
    plt.show()
else:
    print("❌ Fish not detected in tframe18.png")

#%%

# === Loop through all frames and track fish movement ===
from skimage.measure import label, regionprops

fish_positions = []

# pics.medr should already be available from your main script
for idx, frame in enumerate(pics.medr):
    bw = np.where(frame > 0.2, 0, 1)  # match interactive threshold
    lbl = label(bw)
    props = regionprops(lbl)

    best_region = None
    for region in props:
        area = region.area
        ecc = region.eccentricity
        cy, cx = region.centroid

        if 30 < area < 500 and ecc > 0.85:
            if best_region is None or region.area > best_region.area:
                best_region = region

    if best_region:
        fish_positions.append({"frame": idx, "y_pos": best_region.centroid[0]})
    else:
        fish_positions.append({"frame": idx, "y_pos": None})

# Save results to CSV
df = pd.DataFrame(fish_positions).dropna()
df.to_csv("fish_positions.csv", index=False)

# === Plot fish vertical trajectory ===
plt.figure(figsize=(10, 5))
plt.plot(df["frame"], df["y_pos"], marker='o', linestyle='-', color='blue')
plt.xlabel("Frame Number")
plt.ylabel("Fish Vertical Position (y-coordinate)")
plt.title("Fish Movement Over Time (medr, threshold=0.2)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig("./fish_movement_plot.png", dpi=300)
plt.show()