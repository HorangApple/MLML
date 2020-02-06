import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self):
        file_path = './train.csv'
        dfl = pd.read_csv(file_path)
        input_file = dfl.to_numpy()
        x = input_file[:,0]
        y = input_file[:,1:]
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

Data()