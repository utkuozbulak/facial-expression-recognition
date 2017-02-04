import pandas as pd
import numpy as np
from keras.utils import np_utils 

#Import data to pandas data frame
raw_data = pd.read_csv('../data/fer2013.csv')

x_vectoral_data = []
y_data = []
for row in raw_data.iterrows():
    single_vector = []
    pixels = row[1]['pixels']
    single_pixel = ''
    for index,pixel in enumerate(pixels):
        if pixel != ' ':
            single_pixel = single_pixel + str(pixel)
        if pixel == ' ':
            single_vector.append(int(single_pixel))
            single_pixel = ''
        if index == len(pixels)-1: #To get the last pixel
            single_vector.append(int(single_pixel))
    x_vectoral_data.append(single_vector)#Append a single vector to data matrix
    y_data.append(row[1]['emotion'])#Append Y
    
#X data separation begins
x_train_vectoral = np.array(x_vectoral_data[0:28708])
x_public_test_vectoral  = np.array(x_vectoral_data[28709:32297])
x_private_test_vectoral = np.array(x_vectoral_data[32297:])
#X data sep. ends

#Y data separation begins
y_data_categorical = np_utils.to_categorical(y_data, 7)#Divide y data into 7 columns instead of 1
y_train = np.array(y_data_categorical[0:28708])
y_public_test = np.array(y_data_categorical[28709:32297])
y_private_test = np.array(y_data_categorical[32297:])
#Y data sep. ends

#TODO: Create X data in matrix form instead of vectoral so that we can run convolutional nets

del raw_data
del y_data
del y_data_categorical
del x_vectoral_data
del index
del pixel
del pixels
del row
del single_pixel
del single_vector
