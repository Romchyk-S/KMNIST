# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:17:55 2024

@author: romas
"""

import matplotlib.pyplot as plt
import os as os
import keras.utils as kutils
import tensorflow.keras.layers as tkl
import tensorflow.keras.losses as tklosses
import tensorflow.keras.models as tkm
import torch.optim.lr_scheduler as lr_scheduler
import sklearn.metrics as skmetrics
import time as tm
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tudata
import nn_torch_class as nntc
import load_plot_data as lpd
import sklearn.metrics as skmetrics

def find_model_amount(library: str) -> int:
    try:
        if library == 'keras':
            model_list = os.listdir(f'./saved_models_{library}')
            model_list = list(filter(lambda x: '.txt' not in x, model_list))
        else:
            model_list = os.listdir(f'./saved_models_{library}')
            
        models_built_amount = len(model_list)
    except FileNotFoundError:
        models_built_amount = 0
    return models_built_amount

def plot_learning_curve(metric: str, epoch_range: list, history: list, models_built_amount: int, library: str):    
   
    # adapt for plotting train and test curves on the same plot
    plt.plot(epoch_range, history[metric])
    plt.title(f'training_{metric} for {library} model_{models_built_amount}')
    plt.xlabel('epochs')
    plt.ylabel(f'{metric}')   
    plt.savefig(f'./plots/{metric} {library} model_{models_built_amount}')
    plt.show()

def build_keras_model(X_train, Y_train, X_test, Y_test, filepath: str, models_built_amount: int, epochs: int, 
                      kernel_size: tuple, pool_size: tuple, classes_amount: int, batch_size: int, optimizer: str = 'adam'):
    
    model = tkm.Sequential([    
    tkl.Input(X_train.shape[1:4]),
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
    
    # add more metrics
    model.compile(loss = tklosses.SparseCategoricalCrossentropy(from_logits = True), optimizer=optimizer, metrics=['accuracy'])
    
    print("Training the model")
    start = tm.perf_counter()
    history = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size) # train the CNN
    history = history.history
    epoch_range = list(range(epochs))
    epoch_range = list(map(lambda x: x+1, epoch_range))
    plot_learning_curve('accuracy', epoch_range, history, models_built_amount, 'keras')
    plot_learning_curve('loss', epoch_range, history, models_built_amount, 'keras')
    
    print(f"Training takes: {tm.perf_counter()-start} seconds.")
    print()
    print("Model accuracy on the test set")
    model.evaluate(X_test, Y_test) # test the CNN  

    kutils.plot_model(model, to_file = f'{filepath}/model_{models_built_amount}.png', show_shapes = True)
    
    print("Summary of the model:")
    model.summary()
    model.save(filepath+f'/model_{models_built_amount}.keras')
    
    with open(f'{filepath}/model_{models_built_amount}_summary.txt', 'w') as f:
        summary = model.summary
        print(summary)
        print()
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    return model

def build_torch_model(X_train, Y_train, X_test, Y_test, filepath: str, epochs: int,  
                      kernel_size: tuple, pool_size: tuple, classes_amount: int, 
                      batch_size: int, models_built_amount: int, device: str):
    
    train_loss_history, train_acc_history, test_loss_history, test_acc_history = [], [], [], []
    epoch_range = list(range(epochs))
    epoch_range = list(map(lambda x: x+1, epoch_range))
    
    input_channels = X_train.shape[3]
    model = nntc.NN_torch(pool_size, kernel_size, classes_amount, input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters())

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[3],X_train.shape[1],X_train.shape[2])
    X_train = torch.tensor(X_train, dtype=float)
    Y_train = torch.tensor(Y_train, dtype=float)
    Y_train = Y_train.type(torch.LongTensor)
    
    dataset = tudata.TensorDataset(X_train, Y_train)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[3],X_test.shape[1],X_test.shape[2])
    X_test = torch.tensor(X_test, dtype=float)
    Y_test = torch.tensor(Y_test, dtype=float)
    Y_test = Y_test.type(torch.LongTensor)
    
    test_dataset = tudata.TensorDataset(X_test, Y_test)
    training_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle = False)
    
    start = tm.perf_counter()
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor = 1.0, end_factor = 0.5, total_iters = epochs-2)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        total_correct, total_samples = 0, 0
        start_epoch = tm.perf_counter()
        print(f"Epoch {epoch}")
        running_loss_train = 0.0

        for (i, data) in enumerate(training_loader):
            
            x, y = data 
            
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            # Update the running total of correct predictions and samples
            total_correct += (predicted == y).sum().item()
            total_samples += y.size(0)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
        
        print(f"Time taken to train on epoch {epoch}: {round(tm.perf_counter()-start_epoch,3)} seconds")
        train_loss_epoch = running_loss_train/len(dataset)
        print(f"Loss: {round(train_loss_epoch, 3)}")
        train_loss_history.append(train_loss_epoch)
        train_accuracy_epoch = 100 * (total_correct / total_samples)
        print(f"Accuracy: {round(train_accuracy_epoch,3)}%")
        train_acc_history.append(train_accuracy_epoch)
        print()
            
    # plot model
    
    print(f"Time taken to train {round(tm.perf_counter()-start, 3)} seconds")
    total_correct, total_samples = 0, 0
    
    for (i, data) in enumerate(test_loader):
        running_loss_test = 0.0
        x, y = data
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        # Update the running total of correct predictions and samples
        total_correct += (predicted == y).sum().item()
        total_samples += y.size(0)
        running_loss_test += loss.item()

    print(f"Test_set_loss: {round(running_loss_test, 3)}")
    test_loss_history.append(running_loss_test)
    test_accuracy_epoch = 100 * (total_correct / total_samples)
    print(f"Accuracy on test set: {round(test_accuracy_epoch, 3)}")
    test_acc_history.append(test_accuracy_epoch)
    
    print()
    torch.save(model, filepath)
    
    history = {'train_loss': train_loss_history, 'train_accuracy': train_acc_history,
               'test_loss': test_loss_history, 'test_accuracy': test_acc_history}
    
    # find a way to plot on the same plot
    plot_learning_curve('train_loss', epoch_range, history, models_built_amount, 'torch')
    plot_learning_curve('train_accuracy', epoch_range, history, models_built_amount, 'torch')
    plot_learning_curve('test_loss', epoch_range, history, models_built_amount, 'torch')
    plot_learning_curve('test_accuracy', epoch_range, history, models_built_amount, 'torch')
    scheduler.step()
    return model
    
def make_and_plot_prediction(imgs, labels, indexes, model, classmap, elements_to_plot, model_name: str, dataset: str):
    
    imgs_to_predict = imgs[indexes]
    
    try:
        prediction = model.predict(imgs_to_predict)
    except AttributeError:
        imgs_to_predict = imgs_to_predict.reshape(imgs_to_predict.shape[0], imgs_to_predict.shape[3], 
                                          imgs_to_predict.shape[1], imgs_to_predict.shape[2])    
        imgs_to_predict = torch.tensor(imgs_to_predict, dtype=float)
        prediction = model(imgs_to_predict)
        prediction = prediction.detach().numpy()
    
    #plot loaded models
    
    classes = np.argmax(prediction, axis = 1)
    print("Plotting predictions")
    lpd.plot_chars(imgs, labels, classmap, elements_to_plot, indexes = indexes, 
               prediction = classmap.char[classes].values, model_name = model_name, dataset = dataset)
    print()