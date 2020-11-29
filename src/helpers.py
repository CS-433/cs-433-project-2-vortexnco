# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 18:40:27 2020

@author: Alexander
"""
import os

def get_label_file(filename_image):
    filename, file_extension = os.path.splitext(filename_image)
    filename_label = filename + "_label"
    return filename_label+file_extension

def has_label(filename_image, label_folder):
    filename_label = get_label_file(filename_image)
    path_label = os.path.join(label_folder, filename_label)
    return path_label.is_file()
