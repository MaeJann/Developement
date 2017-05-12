#%% Load Modules

import cv2
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd 

#%% Load Data

# --- Load maximum intensity images:
max_ppl = cv2.imread('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Images/Processed/PPL_RGB.jpg')
max_ppl = cv2.cvtColor(max_ppl, cv2.COLOR_BGR2HSV) # transform image to HSV color space

max_xpl = cv2.imread('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Images/Processed/MaxInt_XPL_RGB.jpg')
max_xpl = cv2.cvtColor(max_xpl, cv2.COLOR_BGR2HSV) # transform image to HSV color space

                      
                      

# --- Load coordinates and labels 
data = pd.DataFrame(pd.read_csv('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Extracted/All_Minerals.csv', header=None, sep = ';' ))  
data = data.as_matrix()
x,y = data[:, 1], data[:, 2]
labels = data[:,3]

# --- Define quadratic slide around coordinates. If slice would cross image edge, value for slide edge set to image edge.
extend = 10

x_min = x - extend 
x_min[x_min < 0] = 0

x_max = x + extend
x_max[x_max > max_ppl.shape[1]] = max_ppl.shape[1]
     
y_min = y - extend
x_min[y_min< 0] = 0
     
y_max = y + extend
y_max[y_max > max_ppl.shape[0]] = max_ppl.shape[0]


#%% Functions for feature extraction

# Extract average color value for each channel around the coordinates:
def extract_col_val(image, x_min,x_max, y_min, y_max):
    col_vals = np.zeros((len(x_min), 3))
    for i in range(len(x_min)):
        col_vals[i, 0:3] = np.array([np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i],0]), np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 1]), np.mean(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 2])])   
    return col_vals

def comp_std(image, x_min,x_max, y_min, y_max):
    std_vals = np.zeros((len(x_min), 3))
    for i in range(len(x_min)):
        std_vals[i, 0:3] = np.array([np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i],0]), np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 1]), np.std(image[y_min[i]:y_max[i], x_min[i]:x_max[i], 2])])
    return std_vals


def ext_text(image, x_min,x_max, y_min, y_max, channel):
    
    texture_props = np.zeros((len(x_min), 6))
    
    for i in range(len(x_min)):
        window = image[y_min[i]:y_max[i], x_min[i]:x_max[i], channel]
    
        GLCM = greycomatrix(window, [5], [0, np.pi/4, np.pi/2, (3*np.pi)/4], symmetric=True, normed=True)  # "angular" mean for 0°, 45°, 90°, 135° as defined in Albregtsen
    
        texture_props[i,0] = np.mean(greycoprops(GLCM, 'contrast'))
        texture_props[i,1] = np.mean(greycoprops(GLCM, 'dissimilarity'))
        texture_props[i,2] = np.mean(greycoprops(GLCM, 'homogeneity'))
        texture_props[i,3] = np.mean(greycoprops(GLCM, 'ASM'))
        texture_props[i,4] = np.mean(greycoprops(GLCM, 'energy'))
        texture_props[i,5] = np.mean(greycoprops(GLCM, 'correlation'))
    
    return texture_props


#%% extract individual features

PPL_Color = extract_col_val(max_ppl, x_min,x_max, y_min, y_max)
XPL_Color = extract_col_val(max_xpl, x_min,x_max, y_min, y_max)

PPL_std = comp_std(max_ppl, x_min,x_max, y_min, y_max)
XPL_std = comp_std(max_xpl, x_min,x_max, y_min, y_max)

#PPL_textures = ext_text(max_ppl, x_min,x_max, y_min, y_max, 2)
#PPL_textures = normalize(PPL_textures)

Feature_Names = ["PPL-H", "PPL-I", "PPL-V", "XPL-H", "XPL-I", "XPL-V", "PPL-H-STD", "PPL-I-STD", "PPL-V-STD", "XPL-H-STD", "XPL-I-STD", "XPL-V-STD" ]
#%% Combine extracted features to feature matrix

Features = np.hstack((PPL_Color, XPL_Color, PPL_std, XPL_std))
Features = pd.DataFrame(Features)
Features.to_csv('Featurematrix.csv', sep = ';', header = Feature_Names)


#%% Displaying windows 

import matplotlib.pyplot as plt
i = 60

test = max_xpl[y_min[i]:y_max[i], x_min[i]:x_max[i],2]
plt.imshow(test, cmap = "gray")
