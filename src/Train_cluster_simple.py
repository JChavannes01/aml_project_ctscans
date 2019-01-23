import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from models import AccuracyHistory, get_basic_denselayers
import pandas as pd
from util import OUTPUT_DIR, load_test_val_train_files
import tensorflow as tf

version = 101
do_training = True
model_id = "basicDense"

# Define experiments to run
dropouts = [0.1*x for x in range(9)]
df_experiments = pd.DataFrame(index=np.arange(len(dropouts)), columns=('version', 'dropout_rate', 'accuracy_test'))

def train_classifier():
    global version
    
    images_train, images_validation, images_test, labels_train, labels_validation, labels_test = load_test_val_train_files(version)
    
    def run_iteration(version, i, dropout_rate):
        
        # Checks to make sure we dont accidentally override our previous models.
        if os.path.exists(os.path.join(OUTPUT_DIR, "{}_v{}-exists".format(model_id, version))):
            df_experiments.loc[i] = [-1]*df_experiments.shape[1]
            return
        # Add a placeholder file to indicate that this version has already been trained:
        with open(os.path.join(OUTPUT_DIR, "{}_v{}-exists".format(model_id, version)), 'w') as f:
            pass

        # Create model
        model = get_basic_denselayers(dropout_rate)

        accuracyHistory = AccuracyHistory(model, images_train, labels_train)

        # Define callbacks
        model_path = os.path.join(OUTPUT_DIR, "{}_v{}.h5".format(model_id, version))
        callbacks = [tf.keras.callbacks.ModelCheckpoint(monitor='val_acc',
                filepath=model_path,
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                period=1),
                accuracyHistory]

        # Start training
        print("images_validation shape: {} -- labels_validation.shape: {}".format(images_validation.shape, labels_validation.shape))
        history = model.fit(images_train, labels_train, batch_size=128, epochs=25, validation_data=(images_validation,labels_validation), callbacks=callbacks)

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
        df_experiments.loc[i] = [version, dropout_rate, accuracy_test]
        pickle_path = os.path.join(OUTPUT_DIR, "{}_v{}.pkl".format(model_id, version))
        
        with open(pickle_path, "wb") as f:
            pickle.dump([accuracyHistory.myHistory, accuracy, con_matrix], f)
        
        df_experiments.to_csv(os.path.join(OUTPUT_DIR, 'experiments_{}.txt'.format(model_id)), sep='\t')

    i = 0
    for dropout_rate in dropouts:
        run_iteration(version, i, dropout_rate)
        version += 1
        i += 1


def main():
    if do_training:
        train_classifier()

if __name__ == "__main__":
    main()
