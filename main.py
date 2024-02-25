# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:48:11 2024

@author: romas
"""

# import numpy as np

import matplotlib.pyplot as plt

# import pandas as pd

import tensorflow.keras.models as tkm

# import sklearn.model_selection as skms

import os

import load_plot_data as lpd

import model_work_functions as mwf

# put into graphic interface.

elements_to_plot = 3, 3

kernel_size = 3, 3

pool_size = 2, 2

epochs = 10

batch_size = 5

codes_for_rebuilding_model = ['Yes', 'Y', 'True', 'T']

print()

try:
    
    models_built_amount_keras = len(os.listdir('./saved_models_keras'))
    
except FileNotFoundError:
    
    models_built_amount_keras = 0
    

try:
    
    models_built_amount_pytorch = len(os.listdir('./saved_models_pytorch'))
    
except FileNotFoundError:
    
    models_built_amount_pytorch = 0
    

filepath_keras = f'./saved_models_keras/model_{models_built_amount_keras}'

filepath_pytorch = f'./saved_models_pytorch/model_{models_built_amount_pytorch}'


# print('TakaoGothic' in [f.name for f in matplotlib.font_manager.fontManager.ttflist])
# print(matplotlib.get_cachedir())

datasets = ['kmnist', 'k-49', 'kkanji']

# put into graphic interface.

dataset_chosen = datasets[0]

# dataset_chosen = datasets[1]

# dataset_chosen = datasets[2]


plt.rcParams['font.family'] = 'TakaoGothic'

X_train, Y_train, X_test, Y_test, classmap = lpd.data_loading(dataset_chosen)

classes_amount = len(set(Y_train))

indexes = lpd.plot_chars(X_train, Y_train, classmap, elements_to_plot)

try:
    
    os.listdir('./saved_models_keras/')
    
    tkm.load_model('./saved_models_keras/model_0', compile = True)
    
    rebuild_model = str(input('Build a new model? '))

except (FileNotFoundError, OSError):
        
    rebuild_model = 'Y'
    

if rebuild_model in codes_for_rebuilding_model:
    
    print("Keras training")
    
    model_keras = mwf.build_keras_model(X_train, Y_train, X_test, Y_test, filepath_keras, epochs, kernel_size, 
                                pool_size, classes_amount, batch_size)
    
    
    print("Pytorch training")
    
    model_pytorch = mwf.build_torch_model(X_train, Y_train, X_test, Y_test, filepath_pytorch, epochs, kernel_size, 
                                          pool_size, classes_amount, batch_size)
    
    # print(model.layers[0].weight)
    
    do_a_prediction = str(input('Make a new prediction? '))
    
    if do_a_prediction in codes_for_rebuilding_model:
        
        print("Keras prediction")
    
        mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_keras, classmap, elements_to_plot, f'Keras model_{models_built_amount_keras}')
        
        print("Pytorch prediction")
        
        mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_pytorch, classmap, elements_to_plot, f'Pytorch model_{models_built_amount_pytorch}')
    
else:
    
    model_keras, chosen_model = mwf.choose_and_load_model(models_built_amount_keras)
    
    filepath_keras = f'./saved_models_keras/model_{chosen_model}'
    
    try:
        
        open(f'{filepath_keras}/model_summary.txt', 'r')
        
    except FileNotFoundError:
        
        print('Adding uncreated summary')
        
        with open(f'{filepath_keras}/model_summary.txt', 'w') as f:
        
            model_keras.summary(print_fn=lambda x: f.write(x + '\n'))
    
    model_keras.summary()

    mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_keras, classmap, elements_to_plot)