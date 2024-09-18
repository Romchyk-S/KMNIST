# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:48:11 2024

@author: romas
"""

import torch as torch
# import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import tensorflow as tf
import tensorflow.keras.models as tkm
# import sklearn.model_selection as skms
import os
import load_plot_data as lpd
import model_work_functions as mwf
import graphic_interface as gi

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")

language_versions = ['Українська', 'English']
chosen_language = gi.choose_language("Записати вибір/Write selection", language_versions)
try:
    with open(f'{chosen_language}.txt', 'r', encoding='utf-8') as f:
        text_for_labels = f.readlines()
except FileNotFoundError:
    with open('Українська.txt', 'r', encoding='utf-8') as f:
        text_for_labels = f.readlines()
    
subfolders = next(os.walk('.'))[1]
datasets = list(filter(lambda x: x[0] == 'k', subfolders))
plt.rcParams['font.family'] = 'TakaoGothic'

elements_to_plot = 3, 3
kernel_size = 3, 3
pool_size = 2, 2

models_built_amount_keras = mwf.find_model_amount("keras")
models_built_amount_pytorch = mwf.find_model_amount("pytorch")
build_new_model = 'N'

if models_built_amount_keras == 0 or models_built_amount_pytorch == 0:
    build_new_model = 'Y'
    
save_filepath_keras = './saved_models_keras'
save_filepath_pytorch = f'./saved_models_pytorch/model_{models_built_amount_pytorch}'

parms = gi.choose_dataset_build_new_model(text_for_labels, datasets, build_new_model)
dataset_chosen = parms.get("dataset_chosen", "kmnist")
rebuild_model = parms.get("rebuild_model", "Y")

X_train, Y_train, X_test, Y_test, classmap = lpd.data_loading(dataset_chosen)
classes_amount = len(set(Y_train))
indexes = lpd.plot_chars(X_train, Y_train, classmap, elements_to_plot)

if rebuild_model:
    parms_learning = gi.new_learning_parameters(text_for_labels[2:])
    epochs = parms_learning.get("epochs", 10)
    batch_size = parms_learning.get("batch_size", 8)
    
    if epochs < 1:
        epochs = 10
        print(f"Epochs changed by default to {epochs}")
        
    if batch_size < 1:
        batch_size = 8
        print(f"Batch size changed by default to {batch_size}")
        
    print("Keras training")
    model_keras = mwf.build_keras_model(X_train, Y_train, X_test, Y_test, save_filepath_keras,
                                        models_built_amount_keras, epochs, kernel_size, 
                                        pool_size, classes_amount, batch_size)
    print("Pytorch training")
    
    model_pytorch = mwf.build_torch_model(X_train, Y_train, X_test, Y_test, save_filepath_pytorch, epochs, kernel_size, 
                                          pool_size, classes_amount, batch_size, device_torch)  
    make_a_prediction = gi.choose_making_a_prediction(text_for_labels[4:])
    
    if make_a_prediction: 
        print("Keras prediction")
        mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_keras, classmap, elements_to_plot, 
                                      f'Keras model_{models_built_amount_keras}', dataset_chosen)
        
        print("Pytorch prediction")
        mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_pytorch, classmap, elements_to_plot, 
                                      f'Pytorch model_{models_built_amount_pytorch}', dataset_chosen)
    
else:
    
    chosen_models = gi.choose_models_to_load(text_for_labels[5:])
    model_keras = chosen_models.get("model_keras")
    model_pytorch = chosen_models.get("model_pytorch")
    filepath_keras = f'./saved_models_keras/{model_keras}'
    filepath_pytorch = f'./saved_models_pytorch/{model_pytorch}'
    model_keras = tkm.load_model(filepath_keras, compile = True)
    
    # how to summarize and load pytorch model?
    try:
        # rewrite for new keras saving model process
        open(f'{filepath_keras}/model_summary.txt', 'r')
        
    except FileNotFoundError:
        print('Adding uncreated summary')
        with open(f'{filepath_keras}/model_summary.txt', 'w') as f:
            model_keras.summary(print_fn=lambda x: f.write(x + '\n'))
    
    model_keras.summary()
    mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_keras, classmap, elements_to_plot)