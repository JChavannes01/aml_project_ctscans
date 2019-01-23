import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from models import *
import pandas as pd
from util import OUTPUT_DIR, load_test_val_train_files

version = 200
do_training = True
model_id = "CNN_drr"

# Define experiments to run
dropout_0 = [0.05, 0.15, 0.35, 0.45]
dropout_1 = [0.05, 0.15, 0.35, 0.45]
dropout_2 = [0.05, 0.15, 0.35, 0.45]
dropout_3 = [0.3, 0.4, 0.6, 0.7]
experiments = len(dropout_0) + len(dropout_1) + len(dropout_2) + len(dropout_3)
df_experiments = pd.DataFrame(index=np.arange(experiments), columns=('version', 'dr_1', 'dr_2', 'dr_3', 'dr_4', 'accuracy_test'))


def train_classifier():
    global version

    images_train, images_validation, images_test, labels_train, labels_validation, labels_test = load_test_val_train_files(version)
    
    def run_iteration(version, i, dropout_rates):
        # Checks to make sure we dont accidentally override our previous models.
        if os.path.exists(os.path.join(OUTPUT_DIR, "v{}-exists".format(version))):
            df_experiments.loc[i] = [-1]*df_experiments.shape[1]
            return
        # Add a placeholder file to indicate that this version has already been trained:
        with open(os.path.join(OUTPUT_DIR, "v{}-exists".format(version)), 'w') as f:
            pass

        # Create model
        model = get_cnn(dropout_rates)

        accuracyHistory = AccuracyHistory(model, images_train, labels_train)

        # Define callbacks
        model_path = os.path.join(OUTPUT_DIR, "{}_v{}.h5".format(model_id, version)
        callbacks = [tf.keras.callbacks.ModelCheckpoint(monitor='val_acc',
                filepath=model_path),
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                period=1),
                accuracyHistory]

        # Start training
        history = model.fit(images_train, labels_train, batch_size=128, epochs=25, validation_data=(images_validation,labels_validation), callbacks=callbacks)

        # Add validation accuracy to accuracyHistory class
        accuracyHistory.add_validation_accuracy(history.history)

        # Determine accuracy on train, validation and test data
        model.load_weights(model_path)
        accuracy_train = model.evaluate(images_train, labels_train)
        accuracy_validation = model.evaluate(images_validation, labels_validation)
        accuracy_test = model.evaluate(images_test, labels_test)
        accuracy = {"train": accuracy_train, "validation": accuracy_validation, "test": accuracy_test}

        # Determine confusion matrix for test data
        predictions = model.predict(images_test)
        predictions_list = predictions[:,1] >= 0.5

        con_matrix = confusion_matrix(labels_test, predictions_list)

        # Save model and all data
        df_experiments.loc[i] = [version] + dropout_rates + [accuracy_test]
        pickle_path = os.path.join(OUTPUT_DIR, "{}_v{}.pkl".format(model_id, version))
        
        with open(pickle_path, "wb") as f:
            pickle.dump([accuracyHistory.myHistory, accuracy, con_matrix], f)
        
        df_experiments.to_csv(os.path.join(OUTPUT_DIR, 'experiments_{}.txt'.format(model_id)), sep='\t')

    i = 0
    dropout_indices = [0]*len(dropout_0) + [1]*len(dropout_1) + [2]*len(dropout_2) + [3]*len(dropout_3)
    for pos, val in zip(dropout_indices, dropout_0 + dropout_1 + dropout_2 + dropout_3):
        dropout_rates = [0.25, 0.25, 0.25, 0.5] # Defaults
        dropout_rates[pos] = val
        run_iteration(version, i, dropout_rates)
        version += 1
        i += 1

def main():
    if do_training:
        train_classifier()

if __name__ == "__main__":
    main()
