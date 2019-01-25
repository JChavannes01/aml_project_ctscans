import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from segmentation_models import AccuracyHistory, get_unet_128
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from util import OUTPUT_DIR, load_test_val_train_files_segmentation
import random

version = 102
do_training = True
model_id = "unet_128"


def apply_brightness(x_batch, y_batch):
    return x_batch, y_batch

def apply_contrast(x_batch, y_batch):
    return x_batch, y_batch

def apply_hflip(x_batch, y_batch):
    if random.uniform(0., 1.) < 0.5:
        x_batch = np.flip(x_batch, axis=1)
        y_batch = np.flip(y_batch, axis=1)
    
    return x_batch, y_batch

def apply_vflip(x_batch, y_batch):
    if random.uniform(0., 1.) < 0.5:
        x_batch = np.flip(x_batch, axis=2)
        y_batch = np.flip(y_batch, axis=2)
    
    return x_batch, y_batch

def train_segmentation():
    # Load segmentation
    images_train, images_validation, images_test, segmentation_train, segmentation_validation, segmentation_test = load_test_val_train_files_segmentation(version)
    
    # Create model
    model = get_unet_128((128, 128, 1), 1)

    accuracyHistory = AccuracyHistory(model, images_train, segmentation_train)

    # Define callbacks
    model_path = os.path.join(OUTPUT_DIR, "{}_v{}.h5".format(model_id, version))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(monitor='val_dice_coeff',
            filepath=model_path,
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            period=1),
            accuracyHistory]

    # Start training
    BATCH_SIZE = 32
    epochs = 25
    # history = model.fit_generator(data_generator.flow(images_train, segmentation_train, batch_size=BATCH_SIZE), 
    #                                 steps_per_epoch=images_test.shape[0]/BATCH_SIZE, 
    #                                 epochs=25, verbose=2, 
    #                                 validation_data=(images_validation,segmentation_validation), 
    #                                 callbacks=callbacks, 
    #                                 workers=4, 
    #                                 use_multiprocessing=True, 
    #                                 data_format='channels_last')


    datagen = ImageDataGenerator()
    datagen.fit(images_train)

    # Manual splits because we need to apply flips to both the images (features) and segmentation (target)
    best_val_score = 0.0
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(images_train, segmentation_train, batch_size=BATCH_SIZE):

            # Apply transformations
            # Brightness
            apply_brightness(x_batch, y_batch)

            # Contrast
            apply_contrast(x_batch, y_batch)

            # Flip horizontal
            apply_hflip(x_batch, y_batch)

            # Flip Vertical
            apply_vflip(x_batch, y_batch)

            # Do training on this batch
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= images_train.shape[0] / BATCH_SIZE:
                # Get training accuracy
                accuracyHistory.on_epoch_end(e)

                # Do the evaluation and save the model if it is better than the previous best
                validation_result = model.evaluate(images_validation, segmentation_validation, verbose=0) # (loss, dice_coeff)
                accuracyHistory.add_validation_accuracy_for_epoch(validation_result[0], validation_result[1])
                
                if validation_result[1] > best_val_score:
                    model.save(model_path)
                
                break
    # history = model.fit(images_train, segmentation_train, batch_size=BATCH_SIZE, epochs=25, validation_data=(images_validation,segmentation_validation), callbacks=callbacks)

    # Add validation dice coefficient to accuracyHistory class
    # accuracyHistory.add_validation_accuracy(history.history)

    # Determine dice coefficient on train, validation and test data
    model.load_weights(model_path)
    dice_coeff_train = model.evaluate(images_train, segmentation_train)
    dice_coeff_validation = model.evaluate(images_validation, segmentation_validation)
    dice_coeff_test = model.evaluate(images_test, segmentation_test)
    dice_coeff = {"train": dice_coeff_train, "validation": dice_coeff_validation, "test": dice_coeff_test}

    # Determine confusion matrix for test data
    predictions = model.predict(images_test)
    predictions_list = predictions >= 0.5

    con_matrix = confusion_matrix(segmentation_test.flatten(), predictions_list.flatten())

    pickle_path = os.path.join(OUTPUT_DIR, "{}_v{}.pkl".format(model_id, version))
    with open(pickle_path, "wb") as f:
        pickle.dump([accuracyHistory.myHistory, dice_coeff, con_matrix], f)

def main():
    # Checks to make sure we dont accidentally override our previous models.
    if os.path.exists(os.path.join(OUTPUT_DIR, "{}_v{}-exists".format(model_id, version))):
        raise ValueError("Version {} already exists, aborting training...".format(version))
    
    # Add a placeholder file to indicate that this version has already been trained:
    with open(os.path.join(OUTPUT_DIR, "{}_v{}-exists".format(model_id, version)), 'w') as f:
        pass

    if do_training:
        train_segmentation()

if __name__ == "__main__":
    main()