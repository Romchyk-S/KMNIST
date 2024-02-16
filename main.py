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


# put into graphic interface.

elements_to_plot = 3, 3

kernel_size = 3, 3

pool_size= 2, 2

codes_for_rebuilding_model = ['Yes', 'Y', 'True', 'T']

try:
    
    models_built_amount = len(os.listdir('./saved_models'))
    
except FileNotFoundError:
    
    models_built_amount = 0

filepath = f'./saved_models/model_{models_built_amount}'


# print('TakaoGothic' in [f.name for f in matplotlib.font_manager.fontManager.ttflist])
# print(matplotlib.get_cachedir())

datasets = ['kmnist', 'k-49', 'kkanji']

# put into graphic interface.

dataset_chosen = datasets[0]

# dataset_chosen = datasets[1]

# dataset_chosen = datasets[2]



plt.rcParams['font.family'] = 'TakaoGothic'

if dataset_chosen == 'kmnist' or 'k-49':

    imgs = np.load(f'{dataset_chosen}-train-imgs.npz')['arr_0']
    
    imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2], 1)
    
    labels = np.load(f'{dataset_chosen}-train-labels.npz')['arr_0']
    
    classmap = pd.read_csv(f'{dataset_chosen}_classmap.csv')

indexes = ffp.plot_chars(imgs, labels, classmap, elements_to_plot)

try:
    
    os.listdir('./saved_models/')
    
    model = tkm.load_model(f'./saved_models/model_{0}', compile = True)
    
    rebuild_model = str(input('Build a new model? '))

except (FileNotFoundError, OSError):
        
    rebuild_model = 'Y'
    

if rebuild_model in codes_for_rebuilding_model:
    
    X_train, X_test, Y_train, Y_test = skms.train_test_split(imgs, labels, test_size=0.20, random_state=42)
    
    model = mwf.build_model(X_train, Y_train, X_test, Y_test, filepath, kernel_size, pool_size, len(set(labels)))
    
    do_a_prediction = str(input('Make a new prediction? '))
    
    if do_a_prediction in codes_for_rebuilding_model:
        
        mwf.make_and_plot_prediction(imgs, indexes, model, labels, classmap, elements_to_plot)
    
else:
    
    model = mwf.choose_and_load_model(models_built_amount)
    
    print(model.summary)

    mwf.make_and_plot_prediction(imgs, indexes, model, labels, classmap, elements_to_plot)