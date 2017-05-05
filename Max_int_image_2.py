# Code to generate composite Maximum Intensity Image (RGB) from max hue, sat and val 
# values extracted from cont. interpolations with tile viewer!
#%% Load Modules
import numpy as np
import cv2

#%%  Generate Maximum intensity image

# --- Load extracted grey scale images for hue, sat and val:

hue = cv2.imread("C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data\Merged Data/Naxos_Boudin_8/Images/MaxIntIm Processed/Naxos_Boudin_8_xpl_hue.jpg")
sat = cv2.imread("C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data\Merged Data/Naxos_Boudin_8/Images/MaxIntIm Processed/Naxos_Boudin_8_xpl_sat.jpg")
val = cv2.imread("C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data\Merged Data/Naxos_Boudin_8/Images/MaxIntIm Processed/Naxos_Boudin_8_xpl_val.jpg")

# --- Generate composite HSV matrix
composite = np.zeros(hue.shape, dtype = "uint8")
composite[:,:,0] = (hue[:,:,0]) / 250 * 180
composite[:,:,1] = sat[:,:,0]
composite[:,:,2] = val[:,:,0]

# --- Convert HSV image to RGB image
compositeRGB = cv2.cvtColor(composite, cv2.COLOR_HSV2BGR)



# --- Display image on screen
#cv2.imshow("Over the Clouds - gray", compositeRGB)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite("Composite_RGB.jpg", compositeRGB)


