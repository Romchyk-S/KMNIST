# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:48:11 2024

@author: romas
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# import numpy as np

import matplotlib.pyplot as plt

# import pandas as pd

import keras as keras

import tensorflow.keras.models as tkm

# import torch as torch

# import torchinfo as info

# import sklearn.model_selection as skms

import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models as tkm
import sklearn.model_selection as skms
import os
import load_plot_data as lpd
import model_work_functions as mwf
import graphic_interface as gi
from sklearn.metrics import f1_score

print()

print(tf.config.list_physical_devices("GPU"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

print()

language_versions = {'Українська': ua_text, 'English': en_text}
chosen_language = gi.choose_language("Записати вибір/Write selection", list(language_versions.keys()))

text_for_labels = language_versions.get(chosen_language, 'ua')

datasets = [name for name in os.listdir() if os.path.isdir(os.path.join(name)) and name[0] == 'k']

plt.rcParams['font.family'] = 'TakaoGothic'

# into graphic interface
kernel_size = 3, 3

# into graphic interface
pool_size = 2, 2

models_built_amount_keras = mwf.find_model_amount("keras")

models_built_amount_pytorch = mwf.find_model_amount("pytorch")

text_for_labels = language_versions.get(chosen_language, language_versions.get('Українська'))
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

rebuild_model = parms.get("build_new_model", "Y")

elements_to_plot = (parms.get('elements_to_plot_0'), parms.get('elements_to_plot_1'))

rebuild_model = parms.get("rebuild_model", "Y")

X_train, Y_train, X_test, Y_test, classmap = lpd.data_loading(dataset_chosen)
classes_amount = len(set(Y_train))
indexes = lpd.plot_chars(X_train, Y_train, classmap, elements_to_plot)

if rebuild_model:
    
    parms_learning = gi.new_learning_parameters(text_for_labels[3:])
    
    # probably add stride and padding
    
    epochs = parms_learning.get("epochs", 10)
    batch_size = parms_learning.get("batch_size", 8)

    kernel_size = parms_learning.get("kernel_size").split('x')
    pool_size = parms_learning.get("pool_size").split('x')
    
    kernel_size = tuple(int(num) for num in kernel_size)
    pool_size = tuple(int(num) for num in pool_size)

    print("Keras training")
    model_keras = mwf.build_keras_model(X_train, Y_train, X_test, Y_test, save_filepath_keras,
                                        models_built_amount_keras, epochs, kernel_size, 
                                        pool_size, classes_amount, batch_size)
    print("Pytorch training")
    
    model_pytorch = mwf.build_torch_model(X_train, Y_train, X_test, Y_test, save_filepath_pytorch, epochs, kernel_size, 
                                          pool_size, classes_amount, batch_size)  
    make_a_prediction = gi.choose_making_a_prediction(text_for_labels[4:])
    
    if make_a_prediction: 
        print("Keras prediction")
        mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_keras, classmap, elements_to_plot, 
                                      f'Keras model_{models_built_amount_keras}', dataset_chosen)
        
        print("Pytorch prediction")
        mwf.make_and_plot_prediction(X_train, Y_train, indexes, model_pytorch, classmap, elements_to_plot, 
                                      f'Pytorch model_{models_built_amount_pytorch}', dataset_chosen)
    
else:
    
    chosen_models = gi.choose_models_to_load(text_for_labels[8:])

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