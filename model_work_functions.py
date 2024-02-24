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

import torch as torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as tudata

class NN_torch(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Softmax(dim=0),
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),            
            nn.Flatten(),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU(),
            nn.Softmax(dim=1)
            )
        
        self.double()
        
    def forward(self, x):
        
        # print("Forward")
        
        logits = self.layers(x)
        
        return logits


def build_torch_model(X_train, Y_train, X_test, Y_test, batch_size: int, epochs: int):
    
    model = NN_torch()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters())

    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], X_train.shape[1], X_train.shape[2])

    X_train = torch.tensor(X_train, dtype=float)
    
    Y_train = torch.tensor(Y_train, dtype=float)
    
    Y_train = Y_train.type(torch.LongTensor)
    
    dataset = tudata.TensorDataset(X_train, Y_train)
    
     
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[3],X_test.shape[1], X_test.shape[2])

    X_test = torch.tensor(X_test, dtype=float)
    
    Y_test = torch.tensor(Y_test, dtype=float)
    
    Y_test = Y_test.type(torch.LongTensor)
    
    test_dataset = tudata.TensorDataset(X_test, Y_test)
    
    
    training_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size)
    
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        print(epoch)
        
        running_loss = 0.0

        for epoch in range(epochs):  # loop over the dataset multiple times
        
            print(epoch)
            
            running_loss = 0.0
    
            for (i, data) in enumerate(training_loader):
                
                # print('Batch {}'.format(i + 1))
                
                # print(running_loss)
                
                # basic training loop
                
                x, y = data
                
                optimizer.zero_grad()
                
                outputs = model(x)
                
                loss = criterion(outputs, y)
                
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                
                
                # if i % 1000 == 999:    # Every 1000 mini-batches...
                
                #     print('Batch {}'.format(i + 1))
                    
                #     print(running_loss)

    for (i, data) in enumerate(test_loader):
        
        x, y = data
        
        outputs = model(x)
        
        loss = criterion(outputs, y)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()

        # print(outputs.argmax(1))
    
    

def build_model(X_train, Y_train, X_test, Y_test, filepath, kernel_size, pool_size, classes_amount: int, batch_size: int):
    
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
    
    print(X_train.shape)

    print("Training the model")
    
    start = tm.perf_counter()
    
    model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, show_metric=True) # train the CNN
    
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
    
    imgs_to_predict = imgs[indexes]
    
    prediction = model.predict(imgs_to_predict)
    
    classes = np.argmax(prediction, axis = 1)
    
    print("Plotting predictions")
    
    lpd.plot_chars(imgs, labels, classmap, elements_to_plot, indexes = indexes, 
               prediction = classmap.char[classes].values)