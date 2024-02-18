# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:17:55 2024

@author: romas
"""

import tensorflow.keras.layers as tkl

import tensorflow.keras.losses as tklosses

import tensorflow.keras.models as tkm

import time as tm

import numpy as np

import load_plot_data as lpd



def build_model(X_train, Y_train, X_test, Y_test, filepath, kernel_size, pool_size, classes_amount: int):
    
    model = tkm.Sequential([
        
    tkl.Rescaling(1./255),
    
     tkl.Conv2D(32, kernel_size = kernel_size, activation='relu', input_shape=(28,28,1)),
     tkl.MaxPooling2D(pool_size = pool_size),
     
     tkl.Conv2D(64, kernel_size = kernel_size, activation='relu'),
     tkl.MaxPooling2D(pool_size = pool_size),
     
     tkl.Conv2D(128, kernel_size = kernel_size, activation='relu'),
     tkl.MaxPooling2D(pool_size = pool_size),
    
     tkl.Flatten(),
     
     # try and show the inner layer workings.
     
     tkl.Dense(128, activation='relu'),
     
     tkl.Dense(64, activation='relu'),
     
     tkl.Dense(32, activation='relu'),
     
     tkl.Dense(16, activation='relu'),
     
     tkl.Dense(classes_amount)])
    
    model.compile(loss = tklosses.SparseCategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])
    
    print("Training the model")
    
    start = tm.perf_counter()
    
    model.fit(X_train, Y_train, epochs = 1) # train the CNN
    
    print(f"Training takes: {tm.perf_counter()-start} seconds.")
    
    print()
    
    print("Model accuracy on the test set")

    model.evaluate(X_test, Y_test) # test the CNN  
    
    print("Summary of the model:")
    
    model.summary()
    
    tkm.save_model(model, filepath) 
    
    with open(f'{filepath}/model_summary.txt', 'w') as f:
        
        summary = model.summary
        
        print(summary)
        
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    return model
    
def choose_and_load_model(models_built_amount: int):
    
    chosen_model = int(input(f'Choose a model number from 0 till {models_built_amount-1}: '))

    if chosen_model > models_built_amount-1:
       
       print("The number is too big, choosing the last built model")
       
       print()
       
       chosen_model = models_built_amount-1
       
    model = tkm.load_model(f'./saved_models/model_{chosen_model}', compile = True)
       
    return model, chosen_model

def make_and_plot_prediction(imgs, indexes, model, labels, classmap, elements_to_plot):
    
    imgs_to_predict = imgs[indexes].reshape(
        imgs[indexes].shape[0]*imgs[indexes].shape[1], 
        imgs[indexes].shape[2], imgs[indexes].shape[3], 1)
    
    prediction = model.predict(imgs_to_predict)
    
    classes = np.argmax(prediction, axis = 1)
    
    print("Plotting predictions")
    
    lpd.plot_chars(imgs, labels, classmap, elements_to_plot, indexes = indexes, 
               prediction = classmap.char[classes].values.reshape(elements_to_plot))