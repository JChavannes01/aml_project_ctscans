import numpy as np
import re
import os
import gzip

# Directory where the preprocessed data is stored.
INPUT_DIR = "/deepstore/datasets/course/aml/group4/All_Data"
OUTPUT_DIR = "/deepstore/datasets/course/aml/group4/models"

def add_images_from_file(images_array, file_num):
    with gzip.GzipFile(os.path.join(INPUT_DIR, 'images-{}.npy.gz'.format(file_num)), "r") as f:
        if images_array.size == 0:
            return np.load(f)
        else:
            return np.concatenate((images_array, np.load(f)), axis=0)

def load_test_val_train_files(version):
    """Load the test, validation and train labels and images from the data folder.
        Also does the basic preprocessing (converting to the right datatype, clamping and rescaling etc.)

        return images_train, images_validation, images_test, labels_train, labels_validation, labels_test"""
    # Load labels
    labels_pattern = re.compile(r'labels-(\d+).npy')
    labels_files = filter(lambda f: re.match(
        labels_pattern, f), os.listdir(INPUT_DIR))
    
    images_train, images_validation, images_test = np.array([]), np.array([]), np.array([])
    labels_train, labels_validation, labels_test = list(), list(), list()

    for f in labels_files:
        # Load images (stack all frames vertically) 
        # Loading images this way ensures that labels and images have the same order
        file_num = int(re.match(labels_pattern, f).group(1))
        
        new_labels = np.load(os.path.join(INPUT_DIR, f)).tolist()
        if file_num % 5 == 0:
            # Test file
            labels_test += new_labels
            images_test = add_images_from_file(images_test, file_num)
        elif file_num % 5 == 1:
            # Validation file
            labels_validation += new_labels
            images_validation = add_images_from_file(images_validation, file_num)
        else:
            # Train file
            labels_train += new_labels
            images_train = add_images_from_file(images_train, file_num)

    # Add new dimension (explicit mention that we have only one color channel)
    # Change range from 0-255 to 0-1 (datatype change from uint8 to float64)
    images_test = images_test[:,:,:,np.newaxis] / 255.0
    images_validation = images_validation[:,:,:,np.newaxis] / 255.0
    images_train = images_train[:,:,:,np.newaxis] / 255.0

    labels_test = np.array(labels_test, dtype=bool)
    labels_validation = np.array(labels_validation, dtype=bool)
    labels_train = np.array(labels_train, dtype=bool)
    
    return images_train, images_validation, images_test, labels_train, labels_validation, labels_test