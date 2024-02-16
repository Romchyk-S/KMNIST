# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:17:55 2024

@author: romas
"""

import tensorflow.keras.layers as tkl

import tensorflow.keras.losses as tklosses

import time as tm

import tensorflow.keras.models as tkm

def build_model(X_train, Y_train, X_test, Y_test, filepath):
    
    model = tkm.Sequential([
        
    tkl.Rescaling(1./255),
    
     tkl.Conv1D(32, 3, activation='relu'),
     tkl.MaxPooling2D(),
     
     tkl.Conv1D(64, 3, activation='relu'),
     tkl.MaxPooling2D(),
     
     tkl.Conv1D(128, 3, activation='relu'),
     tkl.MaxPooling2D(),
    
     tkl.Flatten(),
     
     tkl.Dense(128, activation='relu'),
     
     tkl.Dense(64, activation='relu'),
     
     tkl.Dense(32, activation='relu'),
     
     tkl.Dense(16, activation='relu'),
     
     tkl.Dense(10)])
    
    model.compile(loss = tklosses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    
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
    
    return model
    
def choose_and_load_model(models_built_amount: int):
    
    chosen_model = int(input(f'Choose a model number from 0 till {models_built_amount-1}: '))

    if chosen_model > models_built_amount-1:
       
       print("The number is too big, choosing the last built model")
       
       print()
       
       chosen_model = models_built_amount-1
       
    model = tkm.load_model(f'./saved_models/model_{chosen_model}', compile = True)
       
    return model
