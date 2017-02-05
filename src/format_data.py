from keras.utils import np_utils 
import numpy as np
import pandas as pd

raw_data_csv_file_name = '../../data/fer2013.csv'

if __name__ == "__main__":
    raw_data = pd.read_csv(raw_data_csv_file_name)#Import data to pandas data frame
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
            if index == len(pixels)-1:#To get the last pixel
                single_vector.append(int(single_pixel))
        x_vectoral_data.append(single_vector)#Append a single vector to data matrix
        y_data.append(row[1]['emotion'])#Append Y
    #X data separation begins
    x_train_vectoral = np.array(x_vectoral_data[0:28708])#28708:Train set last element
    x_public_test_vectoral  = np.array(x_vectoral_data[28709:32297])#28709:Public test first element - 32297:Public test last element
    x_private_test_vectoral = np.array(x_vectoral_data[32297:])#32297:Private test first element
    #X data sep. ends
    #Y data separation begins
    y_data_categorical = np_utils.to_categorical(y_data, 7)#Divide y data into 7 columns instead of 1
    y_train = np.array(y_data_categorical[0:28708])
    y_public_test = np.array(y_data_categorical[28709:32297])
    y_private_test = np.array(y_data_categorical[32297:])
    #Y data sep. ends
    #TODO: Create X data in matrix form instead of vectoral so that we can run convolutional nets
    