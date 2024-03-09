# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:39:46 2024

@author: romas
"""

import customtkinter as ctk

import os as os


class Program_parameters:
    
    def __init__(self):
        
        self.parameters_dict = {}
        
    def set_parameters_dict(self, var_names: list, var_values: list):
        
        self.parameters_dict = {k: v for k, v in list(zip(var_names, var_values))}


def button_command(root, prog_parameters, variables_names, variables):
    
    variables_values = [var.get() for var in variables]
    
    prog_parameters.set_parameters_dict(variables_names, variables_values)
    
    root.destroy()
    
def create_root():
    
    ctk.set_appearance_mode("system")

    new_parameters = Program_parameters()
    
    root = ctk.CTk()
    
    root.geometry("640x480")
    
    return root, new_parameters

def main_graphing(text_for_labels, datasets, rebuild_model):
    
    root, new_parameters = create_root()
    
    dataset_chosen = ctk.StringVar(value = datasets[0])
    
    label_1 = ctk.CTkLabel(root, text=text_for_labels[0])
    
    textbox_1 = ctk.CTkOptionMenu(root, values=datasets, variable=dataset_chosen)
    
    
    label_2 = ctk.CTkLabel(root, text=text_for_labels[1])
    
    if rebuild_model != 'Y':
        
        rebuild_model = ctk.BooleanVar()
    
        textbox_2 = ctk.CTkOptionMenu(root, values=['True', 'False'], variable=rebuild_model)
    
    else:
        
        rebuild_model = ctk.BooleanVar(value=1)
        
        textbox_2 = ctk.CTkOptionMenu(root, values=['True', 'False'], variable=rebuild_model, state = 'disabled')
    

    label_1.pack()
    
    textbox_1.pack()
    
    label_2.pack()
    
    textbox_2.pack()
    
    button_write_data = ctk.CTkButton(root, text=text_for_labels[-1], command = lambda: button_command(root, new_parameters, 
                                                                                                       ["dataset_chosen", "rebuild_model"], 
                                                                                                       [dataset_chosen, rebuild_model]))
    button_write_data.pack()
    
    root.mainloop()

    return new_parameters.parameters_dict

def new_learning_parameters(text_for_labels):
    
    root, new_parameters = create_root()
    
    epochs = ctk.IntVar()
    
    label_1 = ctk.CTkLabel(root, text=text_for_labels[0])
    
    textbox_1 = ctk.CTkEntry(root, textvariable=epochs)
    
    label_1.pack()
    
    textbox_1.pack()
    
    batch_size = ctk.IntVar()
    
    label_2 = ctk.CTkLabel(root, text=text_for_labels[1])
    
    textbox_2 = ctk.CTkEntry(root, textvariable=batch_size)
    
    label_2.pack()
    
    textbox_2.pack()
    
    button_write_data = ctk.CTkButton(root, text=text_for_labels[-1], command = lambda: button_command(root, new_parameters, 
                                                                                                       ["epochs", "batch_size"], 
                                                                                                       [epochs, batch_size]))
    button_write_data.pack()
    
    root.mainloop()

    return new_parameters.parameters_dict

def choose_models_to_load(text_for_labels):
    
    saved_models_keras = os.listdir("./saved_models_keras")
    
    saved_models_pytorch = os.listdir("./saved_models_pytorch")
    
    root, new_parameters = create_root()
    
    model_keras = ctk.StringVar()
    
    label_1 = ctk.CTkLabel(root, text=text_for_labels[0])
    
    textbox_1 = ctk.CTkOptionMenu(root, values=saved_models_keras, variable=model_keras)
    
    
    model_pytorch = ctk.StringVar()
    
    label_2 = ctk.CTkLabel(root, text=text_for_labels[1])
    
        
    textbox_2 = ctk.CTkOptionMenu(root, values=saved_models_pytorch, variable=model_pytorch)
    
    label_1.pack()
    
    textbox_1.pack()
    
    label_2.pack()
    
    textbox_2.pack()
    
    button_write_data = ctk.CTkButton(root, text=text_for_labels[-1], command = lambda: button_command(root, new_parameters, 
                                                                                                       ["model_keras", "model_pytorch"], 
                                                                                                       [model_keras, model_pytorch]))
    button_write_data.pack()
    
    root.mainloop()

    return new_parameters.parameters_dict


