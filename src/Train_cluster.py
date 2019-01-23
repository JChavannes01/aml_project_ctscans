import os
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import gzip
import pickle
from models import *
import pandas as pd

version = 2
do_training = True

# Define experiments to run
dropout_0 = [0.05, 0.15, 0.35, 0.45]
dropout_1 = [0.05, 0.15, 0.35, 0.45]
dropout_2 = [0.05, 0.15, 0.35, 0.45]
dropout_3 = [0.3, 0.4, 0.6, 0.7]
experiments = len(dropout_0) + len(dropout_1) + len(dropout_2) + len(dropout_3)
df = pd.DataFrame(index=np.arange(experiments), columns=('dr_1', 'dr_2', 'dr_3', 'dr_4', 'accuracy_test'))

# Directory where the preprocessed data is stored.
input_dir = "/deepstore/datasets/course/aml/group4/All_Data"
output_dir = "/deepstore/datasets/course/aml/group4/models"

def train_classifier():
    global version
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
        file_num = re.match(labels_pattern, f).group(1)
        with gzip.GzipFile(os.path.join(input_dir, 'images-{}.npy.gz'.format(file_num)), "r") as file:
            if images.size == 0:
                images = np.load(file)
            else:
                images = np.concatenate((images, np.load(file)),axis=0) # (Z, X, Y)

    # Add new dimension (explicit mention that we have only one color channel)
    images = images[:,:,:,np.newaxis]
    
    # Change range from 0-255 to 0-1 (datatype change from uint8 to float64)
    images = images / 255.0

    # Change datatype labels from float64 to booleans
    labels = labels.astype(bool, copy=False)

    # Split data in train, validation and test set
    
    images_train, images_test, labels_train, labels_test, indices_train, indices_test = train_test_split(images, labels, range(images.shape[0]),test_size=0.2,random_state=42)
    images_train, images_validation, labels_train, labels_validation, indices_train, indices_validation = train_test_split(images_train, labels_train, indices_train, test_size=0.2,random_state=42)

    def run_iteration(version, i, dropout_rates):
        
        # Checks to make sure we dont accidentally override our previous models.
        if os.path.exists(os.path.join(output_dir, "v{}-exists".format(version))):
            df.loc[i] = [-1]*5
            return
        # Add a placeholder file to indicate that this version has already been trained:
        with open(os.path.join(output_dir, "v{}-exists".format(version)), 'w') as f:
            pass

        # Create model
        model = get_cnn(dropout_rates)

        # Define callbacks
        callbacks = [tf.keras.callbacks.ModelCheckpoint(monitor='val_acc',
                filepath=os.path.join(output_dir, "EditThis{}.h5".format(version)),
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                period=1),
                accuracyHistory()]

        # Start training
        history = model.fit(images_train, labels_train, batch_size=128, epochs=10, validation_data=(images_validation,labels_validation), callbacks=callbacks)

        accuracyHistory.add_validation_accuracy(self, history.history)

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
        df.loc[i] = dropout_rates + [accuracy_test]
        model_path = os.path.join(output_dir, "CNN_v{}.h5".format(version))
        model.save(model_path)
        pickle_path = os.path.join(output_dir, "CNN_v{}.pkl".format(version))
        
        with open(pickle_path, "wb") as f:
            pickle.dump([indices_train, indices_test, indices_validation, accuracyHistory.myHistory, accuracy, con_matrix], f)
        
        df.to_csv(os.path.join(output_dir, 'experiments.txt'), sep='\t')

    i = 0
    dropout_indices = [0]*len(dropout_0) + [1]*len(dropout_1) + [2]*len(dropout_2) + [3]*len(dropout_3)
    for pos, val in zip(dropout_indices, dropout_0 + dropout_1 + dropout_2 + dropout_3):
        dropout_rates = [0.25, 0.25, 0.25, 0.5]
        dropout_rates[pos] = val
        run_iteration(version, i, dropout_rates)
        version += 1
        i += 1


def main():
    if do_training:
        train_classifier()

if __name__ == "__main__":
    main()
