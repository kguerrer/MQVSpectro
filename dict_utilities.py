# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 09:24:19 2023

@author: Hopfield
"""
import csv
from ast import literal_eval
import numpy as np
from re import sub

# DIC SAVE FUNCTIONS
def save_dic(dic, dic_name):
    with open(dic_name + '.csv', 'w') as f:
        w = csv.DictWriter(f, dic.keys())
        w.writeheader()
        w.writerow(dic)
        f.close()
        
def open_dict(filename):
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            dic = dict(row)
        dic = dict([float(a), float(x)] for a, x in zip(dic.keys(), dic.values()))
    return(dic)

def open_dict_all_str(filename):
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            dic = dict(row)
        dic = dict([a, x] for a, x in zip(dic.keys(), dic.values()))
    return(dic)

def open_dict_with_str(filename):
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            dic = dict(row)
        dic = dict([a, float(x)] for a, x in zip(dic.keys(), dic.values()))
    return(dic)

def open_dict_list_values(filename):
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            dic = dict(row)
        dic = dict([float(a), literal_eval(x)] for a, x in zip(dic.keys(), dic.values()))
    return(dic)

def open_dict_tuple_keys(filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        keys = [tuple(literal_eval(key)) for key in rows[0]]  # First row contains keys
        try: 
            values = [np.array(literal_eval(value)) for value in rows[2]]  # Third row contains values
        except:
            values = [np.array(literal_eval(sub(r'\s+', ',', value))) for value in rows[2]]  # Third row contains values
        # Combine keys and values into a dictionary
        dic = dict(zip(keys, values))
        return dic

def array_from_dict(data_dict):
    
    # Extract keys and values from the dictionary
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    m_values = np.array(sorted(list(set(x[0] for x in keys))))
    p_values = np.array(sorted(list(set(x[1] for x in keys))))
    r_values = np.array(sorted(list(set(x[2] for x in keys))))
                    
    # Determine the shape of the array
    shape = tuple(len(set(x[i] for x in keys)) for i in range(len(keys[0]))) + (len(values[0]),)

    numpy_array = np.empty(shape, dtype=float)

    # Fill the array with values from the dictionary
    for i_m in range(len(list(set(x[0] for x in keys)))):
        for i_p in range(len(list(set(x[1] for x in keys)))):
            for i_r in range(len(list(set(x[2] for x in keys)))):
                key = (m_values[i_m], p_values[i_p], r_values[i_r])
                numpy_array[i_m, i_p, i_r] = data_dict[key]
    return numpy_array