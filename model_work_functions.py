# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:17:55 2024

@author: romas
"""

import os as os

import tensorflow.keras.layers as tkl

import tensorflow.keras.losses as tklosses

import tensorflow.keras.models as tkm

import time as tm

import numpy as np

import load_plot_data as lpd

# import torch as torch

# import torch.nn as nn

# import torch.optim as optim

# import torch.utils.data as tudata

# import nn_torch_class as nntc


def find_model_amount(library: str) -> int:
    
    try:
        
        models_built_amount = len(os.listdir(f'./saved_models_{library}'))
        
    except FileNotFoundError:
        
        models_built_amount = 0
        
    return models_built_amount

# def build_torch_model(X_train, Y_train, X_test, Y_test, filepath: str, epochs: int,  kernel_size: tuple, pool_size: tuple, classes_amount: int,  batch_size: int):
    
#     input_channels = X_train.shape[3]
    
#     model = nntc.NN_torch(pool_size, kernel_size, classes_amount, input_channels)

#     criterion = nn.CrossEntropyLoss()

#     optimizer = optim.SGD(model.parameters())

#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], X_train.shape[1], X_train.shape[2])

#     X_train = torch.tensor(X_train, dtype=float)
    
#     Y_train = torch.tensor(Y_train, dtype=float)
    
#     Y_train = Y_train.type(torch.LongTensor)
    
#     dataset = tudata.TensorDataset(X_train, Y_train)
    
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[3],X_test.shape[1], X_test.shape[2])

#     X_test = torch.tensor(X_test, dtype=float)
    
#     Y_test = torch.tensor(Y_test, dtype=float)
    
#     Y_test = Y_test.type(torch.LongTensor)
    
#     test_dataset = tudata.TensorDataset(X_test, Y_test)
    
    
#     training_loader = torch.utils.data.DataLoader(dataset,
#                                                   batch_size=batch_size)
    
    
#     test_loader = torch.utils.data.DataLoader(test_dataset,
#                                                   batch_size=batch_size)
    
#     start = tm.perf_counter()
    
#     for epoch in range(epochs):  # loop over the dataset multiple times
    
#         total_correct, total_samples = 0, 0
    
#         start_epoch = tm.perf_counter()
    
#         print(f"Epoch {epoch}")
        
#         running_loss_train = 0.0

#         for (i, data) in enumerate(training_loader):
            
#             x, y = data
            
#             optimizer.zero_grad()
            
#             outputs = model(x)
            
#             _, predicted = torch.max(outputs, 1)

#             # Update the running total of correct predictions and samples
#             total_correct += (predicted == y).sum().item()
            
#             total_samples += y.size(0)
            
#             loss = criterion(outputs, y)
            
#             loss.backward()
            
#             optimizer.step()
            
#             running_loss_train += loss.item()
            
#         # need to make it work for unbalanced classification
        
#         print(f"Time taken to train on epoch {epoch}: {round(tm.perf_counter()-start_epoch,3)} seconds")
            
#         print(f"Loss: {round(running_loss_train/len(dataset), 3)}")
        
#         accuracy = round(100 * (total_correct / total_samples), 3)
        
#         print(f"Accuracy: {accuracy}%")
        
#         print()
            
#     print(f"Time taken to train {round(tm.perf_counter()-start, 3)} seconds")
    
#     total_correct, total_samples = 0, 0
    
#     for (i, data) in enumerate(test_loader):
        
#         running_loss_test = 0.0
        
#         x, y = data
        
#         outputs = model(x)
        
#         _, predicted = torch.max(outputs, 1)

#         # Update the running total of correct predictions and samples
#         total_correct += (predicted == y).sum().item()
        
#         total_samples += y.size(0)
        
#         running_loss_test += loss.item()
        
#     # print(x)
    
#     # print(outputs)
        
#     # print(predicted)
    
#     # print(y)
    
#     # print(model.layers[0].weight)

#     print(f"Test_set_loss: {round(running_loss_test, 3)}")
    
#     accuracy = round(100 * (total_correct / total_samples), 3)
    
#     print(f"Accuracy on test set: {accuracy}")
    
#     print()
    
#     torch.save(model, filepath)
    
#     return model
    
def build_keras_model(X_train, Y_train, X_test, Y_test, filepath: str, epochs: int, kernel_size: tuple, pool_size: tuple, classes_amount: int, batch_size: int):
    
    model = tkm.Sequential([
        
    tkl.Rescaling(1./255),
    
     tkl.Conv2D(32, kernel_size = kernel_size, activation='relu'),
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
    
    # need to make it work for unbalanced classification
    model.compile(loss = tklosses.SparseCategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])
    
    print("Training the model")
    
    start = tm.perf_counter()
    
    model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size) # train the CNN
    
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
        
        print()
        
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    return model
    
def make_and_plot_prediction(imgs, labels, indexes, model, classmap, elements_to_plot, model_name: str, dataset: str):
    
    imgs_to_predict = imgs[indexes]
    
    try:
        
    
        prediction = model.predict(imgs_to_predict)
        
    except AttributeError:
        
        imgs_to_predict = imgs_to_predict.reshape(imgs_to_predict.shape[0], imgs_to_predict.shape[3], 
                                          imgs_to_predict.shape[1], imgs_to_predict.shape[2])
        
        # imgs_to_predict = torch.tensor(imgs_to_predict, dtype=float)
        
        # prediction = model(imgs_to_predict)
        
        # prediction = prediction.detach().numpy()
    
    # classes = np.argmax(prediction, axis = 1)

    # print("Plotting predictions")
    
    # lpd.plot_chars(imgs, labels, classmap, elements_to_plot, indexes = indexes, 
    #            prediction = classmap.char[classes].values, model_name = model_name, dataset = dataset)
    
    # print()