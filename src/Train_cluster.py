import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import gzip
import pickle
from models import *

version = 1
do_training = True

# Directory where the preprocessed data is stored.
input_dir = r"data\All_Data"

def train_classifier():
    # Load labels
    labels_pattern = re.compile(r'labels-(\d+).npy')
    labels_files = filter(lambda f: re.match(
    labels_pattern, f), os.listdir(input_dir))
    labels = np.array([])
    images = np.array([])
    for f in labels_files:
        labels = np.append(labels, np.load(os.path.join(input_dir, f)))
        
        # Load images (stack all frames vertically) 
        # Loading images this way ensures that labels and images have the same order
        file_num = re.match(labels_pattern, f)[1]
        with gzip.GzipFile(os.path.join(input_dir, 'images-{}.npy.gz'.format(file_num)), "r") as file:
            if images.size == 0:
                images = np.load(file)
            else:
                images = np.concatenate((images, np.load(file)),axis=0) # (Z, X, Y)

    # Add new dimension (explicit mention that we have only one color channel)
    images = images[:,:,:,np.newaxis]
    images = images[:12]
    labels = labels[:12]
    
    # Change range from 0-255 to 0-1 (datatype change from uint8 to float64)
    images = images / 255.0

    # Change datatype labels from float64 to booleans
    labels = labels.astype(bool,copy=False)

    # Split data in train, validation and test set
    images_train, images_test, labels_train, labels_test = train_test_split(images,labels,test_size=0.2,random_state=42)
    images_train, images_validation, labels_train, labels_validation = train_test_split(images_train,labels_train,test_size=0.2,random_state=42)

    # Create model
    model = get_basic_cnn()

    # Start training
    history = model.fit(images_train, labels_train, batch_size=128, epochs=5, validation_data=(images_validation,labels_validation))

    # Determine accuracy on train, validation and test data
    accuracy_train = model.evaluate(images_train, labels_train)
    accuracy_validation = model.evaluate(images_validation, labels_validation)
    accuracy_test = model.evaluate(images_test, labels_test)
    accuracy = {"train": accuracy_train, "validation": accuracy_validation, "test": accuracy_test}

    # Determine confusion matrix for test data
    predictions = model.predict(images_test)
    predictions_list = predictions[:,1] >= 0.5

    con_matrix = confusion_matrix(labels_test, predictions_list)

    # Save model and all data
    model.save("models/CNN_v{}.h5".format(version))
    with open("models/CNN_v{}.pkl".format(version), "wb") as f:
        pickle.dump([images_train, images_test, labels_train, labels_test, history.history, accuracy, con_matrix], f)

def main():
    if do_training:
        train_classifier()

if __name__ == "__main__":
    main()