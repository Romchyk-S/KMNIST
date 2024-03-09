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

import graphic_interface as gi


ua_text = ['Оберіть дані', 'Будувати нову модель?', 'Введіть кількість епох навчання', 
           'Введіть розмір партії навчання', 'Оберіть модель keras', 'Оберіть модель pytorch', 'Записати вибір']

en_text = ['Choose data', 'Build a new model?', 'Enter training epochs', 
           'Enter batch size', 'Choose keras model', 'Choose pytorch model', 'Write selection']

language_versions = {'ua': ua_text, 'en': en_text}

# put in graphic interface
chosen_language = 'ua'

text_for_labels = language_versions.get(chosen_language, 'ua')


elements_to_plot = 3, 3

kernel_size = 3, 3

pool_size = 2, 2

print()

try:
    
    models_built_amount_keras = len(os.listdir('./saved_models_keras'))
    
except FileNotFoundError:
    
    models_built_amount_keras = 0
    

try:
    
    models_built_amount_pytorch = len(os.listdir('./saved_models_pytorch'))
    
except FileNotFoundError:
    
    models_built_amount_pytorch = 0
    

save_filepath_keras = f'./saved_models_keras/model_{models_built_amount_keras}'

save_filepath_pytorch = f'./saved_models_pytorch/model_{models_built_amount_pytorch}'

# print('TakaoGothic' in [f.name for f in matplotlib.font_manager.fontManager.ttflist])
# print(matplotlib.get_cachedir())

datasets = ['kmnist', 'k49', 'kkanji']

plt.rcParams['font.family'] = 'TakaoGothic'

try:
    
    os.listdir('./saved_models_keras/')
    
    tkm.load_model('./saved_models_keras/model_0', compile = True)
    
    rebuild_model = 0

except (FileNotFoundError, OSError):
        
    rebuild_model = 'Y'
    

parms = gi.main_graphing(text_for_labels, datasets, rebuild_model)

dataset_chosen = parms.get("dataset_chosen", "kmnist")

rebuild_model = parms.get("rebuild_model", "Y")

X_train, Y_train, X_test, Y_test, classmap = lpd.data_loading(dataset_chosen)

classes_amount = len(set(Y_train))

indexes = lpd.plot_chars(X_train, Y_train, classmap, elements_to_plot)


if rebuild_model:
    
    parms_learning = gi.new_learning_parameters(text_for_labels[2:])
    
    epochs = parms_learning.get("epochs", 10)

    batch_size = parms_learning.get("batch_size", 5)

    
    # print("Keras training")
    
    # model_keras = mwf.build_keras_model(X_train, Y_train, X_test, Y_test, save_filepath_keras, epochs, kernel_size, 
    #                             pool_size, classes_amount, batch_size)
    
    
    # print("Pytorch training")
    
    # model_pytorch = mwf.build_torch_model(X_train, Y_train, X_test, Y_test, save_filepath_pytorch, epochs, kernel_size, 
    #                                       pool_size, classes_amount, batch_size)
    
    # print(model.layers[0].weight)
    
    # put in graphic interface
    # do_a_prediction = str(input('Make a new prediction? '))
    
    # if do_a_prediction in codes_for_rebuilding_model:
        
    #     print("Keras prediction")
    
    #     mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_keras, classmap, elements_to_plot, 
    #                                   f'Keras model_{models_built_amount_keras}', dataset_chosen)
        
    #     print("Pytorch prediction")
        
    #     mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_pytorch, classmap, elements_to_plot, 
    #                                   f'Pytorch model_{models_built_amount_pytorch}', dataset_chosen)
    
else:
    
    chosen_models = gi.choose_models_to_load(text_for_labels[4:])
    
    model_keras = chosen_models.get("model_keras")
    
    model_pytorch = chosen_models.get("model_pytorch")
    
    filepath_keras = f'./saved_models_keras/{model_keras}'
    
    filepath_pytorch = f'./saved_models_pytorch/{model_pytorch}'
    
    model_keras = tkm.load_model(filepath_keras, compile = True)
    
    # how to summarize and load pytorch model?
    
    try:
        
        open(f'{filepath_keras}/model_summary.txt', 'r')
        
    except FileNotFoundError:
        
        print('Adding uncreated summary')
        
        with open(f'{filepath_keras}/model_summary.txt', 'w') as f:
        
            model_keras.summary(print_fn=lambda x: f.write(x + '\n'))
    
    model_keras.summary()

#     mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_keras, classmap, elements_to_plot)