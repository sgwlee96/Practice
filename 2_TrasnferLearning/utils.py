import os
import csv
from keras import datasets
from sklearn.model_selection import train_test_split

def load_data():
    (train_img, )