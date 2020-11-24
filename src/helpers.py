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