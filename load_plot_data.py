# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:17:09 2024

@author: romas
"""

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os as os


def data_loading(dataset_chosen: str):
    
    if dataset_chosen == 'kmnist' or 'k-49':
    
        X_train = np.load(f'./{dataset_chosen}/{dataset_chosen}-train-imgs.npz')['arr_0']
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        
        Y_train = np.load(f'./{dataset_chosen}/{dataset_chosen}-train-labels.npz')['arr_0']
        
        X_test = np.load(f'./{dataset_chosen}/{dataset_chosen}-test-imgs.npz')['arr_0']
        
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        
        Y_test = np.load(f'./{dataset_chosen}/{dataset_chosen}-test-labels.npz')['arr_0']
        
        classmap = pd.read_csv(f'./{dataset_chosen}/{dataset_chosen}_classmap.csv')
        
        return X_train, Y_train, X_test, Y_test, classmap

def plot_chars(imgs, labels, classmap, plotted_elements: tuple, *args,  **kwargs):
    
    indexes = kwargs.get("indexes", [])
    
    dataset = kwargs.get("dataset", "")
    
    model_name = kwargs.get("model_name", "")
    
    true_classes = classmap.char[labels[indexes]].values
    
    indexes_given = len(indexes) > 0

    fig, axes_list = plt.subplots(plotted_elements[0], plotted_elements[1], figsize=(plotted_elements[0]+1, plotted_elements[0]+1))
    
    
    
    if indexes_given: 
        
        fig.suptitle(dataset + ' ' + model_name)
        
        plot_folder = './plots'
    
        image_num = len(os.listdir(plot_folder))
        
        prediction = kwargs.get('prediction')
        
        for index, ax in enumerate(axes_list.flat):
                
                ax.axis('off')
                
                predicted_class = prediction[index]
                
                letter = imgs[index]
                
                ax.imshow(letter)
                
                ax.set_title(f"p: {predicted_class}, t: {true_classes[index]}")
                
        plt.savefig(f'./plots/image_{image_num}.png')
        
    else:
        
        fig.suptitle(dataset)

        for ax in axes_list.flat:
        
            ax.axis('off')
            
            index = np.random.randint(0, len(imgs))
            
            indexes.append(index)
            
            letter = imgs[index]
        
            ax.imshow(letter)
            
            ax.set_title(classmap.char[labels[index]])
            
    plt.show()
            
    return indexes