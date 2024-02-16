# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:17:09 2024

@author: romas
"""

import matplotlib.pyplot as plt

import numpy as np

def plot_chars(imgs, labels, classmap, plotted_elements: tuple, *args,  **kwargs):
    
    indexes = kwargs.get("indexes", [])
    
    indexes_given = len(indexes) > 0

    fig, axes_list = plt.subplots(plotted_elements[0], plotted_elements[1], figsize=(plotted_elements[0]+1, plotted_elements[0]+1))
    
    if indexes_given:
        
        prediction = kwargs.get('prediction')
        
        for ax, indexes_sublist, classes_sublist in list(zip(axes_list, indexes, prediction)):
            
            for ax_1, index, predicted_class in list(zip(ax, indexes_sublist, classes_sublist)):
                
                ax_1.axis('off')
                
                letter = imgs[index]
                
                ax_1.imshow(letter)
                
                ax_1.set_title(f"p: {predicted_class}, t: {classmap.char[labels[index]]}")
        
    else:

        for ax in axes_list:
            
            indexes_sublist = []
            
            for ax_1 in ax:
                
                ax_1.axis('off')
                
                index = np.random.randint(0, len(imgs))
                
                indexes_sublist.append(index)
                
                letter = imgs[index]
            
                ax_1.imshow(letter)
                
                ax_1.set_title(classmap.char[labels[index]])
                
            indexes.append(indexes_sublist)
            
    plt.show()
            
    return indexes