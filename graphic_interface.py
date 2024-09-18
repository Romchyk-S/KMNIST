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

def choose_language(button_text, languages_supported):
    
    root, new_parameters = create_root()
    language = ctk.StringVar()
    
    label_1 = ctk.CTkLabel(root, text='Оберіть мову / Choose the language')
    textbox_1 = ctk.CTkOptionMenu(root, values=languages_supported, variable=language)
    label_1.pack()
    textbox_1.pack()
    
    button_write_data = ctk.CTkButton(root, text=button_text, command = lambda: button_command(root, new_parameters, 
                                                                                                       ["language"],                                                                                                 
                                                                                                       [language]))
    button_write_data.pack()
    
    root.mainloop()

    return list(new_parameters.parameters_dict.values())[0]

def choose_dataset_build_new_model(text_for_labels, datasets, build_new_model):
    
    root, new_parameters = create_root()
    
    dataset_chosen = ctk.StringVar(value = datasets[0])
    label_1 = ctk.CTkLabel(root, text=text_for_labels[0])
    textbox_1 = ctk.CTkOptionMenu(root, values=datasets, variable=dataset_chosen)
    
    label_2 = ctk.CTkLabel(root, text=text_for_labels[1])
    if build_new_model != 'Y':
        build_new_model = ctk.BooleanVar(value = True)
        textbox_2 = ctk.CTkOptionMenu(root, values=['True', 'False'], variable=build_new_model)
    else:
        build_new_model = ctk.BooleanVar(value=True)
        textbox_2 = ctk.CTkOptionMenu(root, values=['True', 'False'], variable=build_new_model, state = 'disabled')

    elements_to_plot_0 = ctk.IntVar()
    elements_to_plot_1 = ctk.IntVar()
    
    label_3 = ctk.CTkLabel(root, text=text_for_labels[2])
    elements_to_plot_list = [str(num) for num in list(range(1, 11))]
    textbox_3 = ctk.CTkOptionMenu(root, values=elements_to_plot_list, variable=elements_to_plot_0)
    textbox_4 = ctk.CTkOptionMenu(root, values=elements_to_plot_list, variable=elements_to_plot_1)

    label_1.pack()
    textbox_1.pack()
    label_2.pack()
    textbox_2.pack()
    label_3.pack()
    textbox_3.pack()
    textbox_4.pack()
    
    button_write_data = ctk.CTkButton(root, text=text_for_labels[-1], command = lambda: button_command(root, new_parameters, 
                                                                                                       ["dataset_chosen", "build_new_model", "elements_to_plot_0", "elements_to_plot_1"], 
                                                                                                       [dataset_chosen, build_new_model, elements_to_plot_0, elements_to_plot_1]))
    button_write_data.pack()
    
    root.mainloop()

    return new_parameters.parameters_dict

def new_learning_parameters(text_for_labels):
    
    # add optimizer option menu
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
    
    kernel_size = ctk.StringVar()
    label_3 = ctk.CTkLabel(root, text=text_for_labels[2])
    textbox_3 = ctk.CTkOptionMenu(root, values=['3x3', '5x5', '7x7', '9x9'], variable=kernel_size)
    label_3.pack()
    textbox_3.pack()
    
    pool_size = ctk.StringVar()
    label_4 = ctk.CTkLabel(root, text=text_for_labels[3])
    textbox_4 = ctk.CTkOptionMenu(root, values=['2x2', '3x3', '4x4', '5x5'], variable=pool_size)
    label_4.pack()
    textbox_4.pack()
    
    button_write_data = ctk.CTkButton(root, text=text_for_labels[-1], command = lambda: button_command(root, new_parameters, 
                                                                                                       ["epochs", "batch_size", "kernel_size", 'pool_size'], 
                                                                                                       [epochs, batch_size, kernel_size, pool_size]))
    button_write_data.pack()
    
    root.mainloop()
    return new_parameters.parameters_dict

def choose_making_a_prediction(text_for_labels):
    
    root, new_parameters = create_root()
  
    label_1 = ctk.CTkLabel(root, text=text_for_labels[0])
    make_a_prediction = ctk.BooleanVar(value = True)
    textbox_1 = ctk.CTkOptionMenu(root, values=['True', 'False'], variable=make_a_prediction)
    label_1.pack()
    textbox_1.pack()
    
    button_write_data = ctk.CTkButton(root, text=text_for_labels[-1], command = lambda: button_command(root, new_parameters, 
                                                                                                         ["make_a_prediction"], 
                                                                                                         [make_a_prediction]))
    button_write_data.pack()
    
    root.mainloop()

    return list(new_parameters.parameters_dict.values())[0]

def choose_models_to_load(text_for_labels):
    
    saved_models_keras = os.listdir("./saved_models_keras")
    saved_models_pytorch = os.listdir("./saved_models_pytorch")
    
    root, new_parameters = create_root()
    model_keras = ctk.StringVar()
    label_1 = ctk.CTkLabel(root, text=text_for_labels[0])
    textbox_1 = ctk.CTkOptionMenu(root, values=saved_models_keras, variable=model_keras)
    label_1.pack()
    textbox_1.pack()
    
    model_pytorch = ctk.StringVar()
    label_2 = ctk.CTkLabel(root, text=text_for_labels[1])        
    textbox_2 = ctk.CTkOptionMenu(root, values=saved_models_pytorch, variable=model_pytorch)
    label_2.pack()
    textbox_2.pack()
    
    button_write_data = ctk.CTkButton(root, text=text_for_labels[-1], command = lambda: button_command(root, new_parameters, 
                                                                                                       ["model_keras", "model_pytorch"], 
                                                                                                       [model_keras, model_pytorch]))
    button_write_data.pack()
    
    root.mainloop()

    return new_parameters.parameters_dict