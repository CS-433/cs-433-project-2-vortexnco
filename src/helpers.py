import os
import re


def get_label_file(filename_image):
    """Get the name of the corresponding label file"""
    filename, file_extension = os.path.splitext(filename_image)
    filename_label = filename + "_label"
    return filename_label + file_extension


def get_image_file(filename_label):
    """Get the name of the corresponding image file"""
    filename, file_extension = os.path.splitext(filename_label)
    file = filename + file_extension
    filename_image = re.sub("_label" + file_extension, file_extension, file)
    return filename_image


def has_label(filename_image, label_folder):
    """Check that the image whether a corresponding label file"""
    filename_label = get_label_file(filename_image)
    path_label = os.path.join(label_folder, filename_label)
    return path_label.is_file()
