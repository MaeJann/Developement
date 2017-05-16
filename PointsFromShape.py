import shapefile
import numpy as np
import pandas as pd 


SF = shapefile.Reader('C:/Users/j-mae/Desktop/Master Thesis/Image Data/Test Data/Merged Data/Naxos_Boudin_8/QGIS Naxos_Boudin_8/Mineral Phases/Biotite/testarea2.shp')
shapes = SF.shapes()
  
coordinates = []  # extract coordinates; key = dictionairy index = feature index
for i in range(len(shapes)):
    coordinates.append( shapes[i].points)



coordinates = np.abs(np.array(coordinates, dtype = 'int'))
coordinates = np.reshape(coordinates, (4,2))
print(coordinates.shape)

df = pd.DataFrame(coordinates)
df.to_csv("testarea22.csv", header=None, sep = ';' )




