import os
import re
import PIL
import numpy as np

# --- Define Path to image folder:
mypath = 'C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/Images/Maximum Intensity Image - Raw Data/Val/interleaving/8.0/0.0'


# --- Extract all filenames:
f_names = []
for (dirpath, dirnames, filenames) in os.walk(mypath):
    f_names.extend(filenames)
    break



# --- Sort file name list in alphanumeric fashion: (https://arcpy.wordpress.com/2012/05/11/sorting-alphanumeric-strings-in-python/)
def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

f_names = sorted_nicely(f_names)


# --- Generate list with all (sorted) file paths:    
f_paths = []
for i in range(len(f_names)):
    f_paths.append(mypath + "/" + f_names[i])


# --- Extract position parameters from file names:
f_parameters = []
for name in f_names:
   f_parameters.append((re.findall("[-+]?\d+[\]?\d*[eE]?[-+]?\d*", name)))
   
f_parameters = np.array(f_parameters, dtype = 'int')


# --- Computing the strip id and number of images per strip:
strip_id, count = np.unique(f_parameters[:,0], return_counts=True)


# --- Merge first images into vertical strips and then merge all strips horizontally:
all_strips = []
for i in range(len(strip_id)):
    strip_start = count[0]*i
    strip_end = count[0]*i + count[0]
    strip_imagepaths = f_paths[strip_start : strip_end]
    strip_images  = [PIL.Image.open(i) for i in strip_imagepaths]
    all_strips.append(np.vstack(strip_images))
    
horizontal_merge = np.hstack(all_strips[i] for i in range(len(all_strips))) 


# --- Save fully merged image:
imgs_comb = PIL.Image.fromarray(horizontal_merge)
imgs_comb.save( 'Naxos_Boudin_8_xpl_hue.jpg' )
