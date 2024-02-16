# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:48:11 2024

@author: romas
"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow.keras.models as tkm

import sklearn.model_selection as skms

import os

import functions_for_plotting as ffp

import model_work_functions as mwf



elements_to_plot = 3, 3

try:
    
    models_built_amount = len(os.listdir('./saved_models'))
    
except FileNotFoundError:
    
    models_built_amount = 0

filepath = f'./saved_models/model_{models_built_amount}'




# print('TakaoGothic' in [f.name for f in matplotlib.font_manager.fontManager.ttflist])
# print(matplotlib.get_cachedir())

plt.rcParams['font.family'] = 'TakaoGothic';

imgs = np.load('kmnist-train-imgs.npz')['arr_0']

imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2], 1)

labels = np.load('kmnist-train-labels.npz')['arr_0']

classmap = pd.read_csv('kmnist_classmap.csv')

indexes = ffp.plot_chars(imgs, labels, classmap, elements_to_plot)

try:
    
    os.listdir('./saved_models/')
    
    model = tkm.load_model(f'./saved_models/model_{0}', compile = True)
    
    rebuild_model = str(input('Build a new model? '))

except (FileNotFoundError, OSError):
        
    rebuild_model = 'Y'

if rebuild_model == 'Yes' or rebuild_model == 'True' or rebuild_model == 'Y':
    
    X_train, X_test, Y_train, Y_test = skms.train_test_split(imgs, labels, test_size=0.20, random_state=42)
    
    model = mwf.build_model(X_train, Y_train, X_test, Y_test, filepath)
    
else:
    
    model = mwf.choose_and_load_model(models_built_amount)

    imgs_to_predict = imgs[indexes].reshape(
        imgs[indexes].shape[0]*imgs[indexes].shape[1], 
        imgs[indexes].shape[2], imgs[indexes].shape[3], 1)
    
    prediction = model.predict(imgs_to_predict)
    
    classes = np.argmax(prediction, axis = 1)
    
    print("Plotting predictions")
    
    ffp.plot_chars(imgs, labels, classmap, elements_to_plot, indexes = indexes, 
               prediction = classmap.char[classes].values.reshape(elements_to_plot))