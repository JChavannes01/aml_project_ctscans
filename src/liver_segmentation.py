import os
import re
import gzip
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import pickle
import matplotlib.pyplot as plt

from segmentation_models import get_unet_128

do_training = False

# Directory where the preprocessed data is stored.
output_dir = r"data\Training_Batch1\preprocessed"

def train_segmentation():
    # Load segmentation
    lab_pattern = re.compile(r'labels-(\d+).npy')
    labels_files = filter(lambda f: re.match(
    lab_pattern, f), os.listdir(output_dir))
    labels = np.array([])
    segmentation = np.array([])
    images = np.array([])
    for f in labels_files:
        # Load label so slices with no liver can be removed
        labels = np.append(labels, np.load(os.path.join(output_dir, f)))

        # Load segmentation (stack all frames vertically) 
        # Loading segmentation this way ensures that labels and images have the same order
        file_num = re.match(lab_pattern, f)[1]
        with gzip.GzipFile(os.path.join(output_dir, f'liver_segmentation-{file_num}.npy.gz'), "r") as file:
            if segmentation.size == 0:
                segmentation = np.load(file)
            else:
                segmentation = np.concatenate((segmentation, np.load(file)),axis=0) # (Z, X, Y)

        # Load images (stack all frames vertically)
        with gzip.GzipFile(os.path.join(output_dir, f'images-{file_num}.npy.gz'), "r") as file:
            if images.size == 0:
                images = np.load(file)
            else:
                images = np.concatenate((images, np.load(file)),axis=0) # (Z, X, Y)

    images = images[labels == 1,:,:]
    segmentation = segmentation[labels == 1,:,:]

    images = images[:,:,:,np.newaxis]
    segmentation = segmentation[:,:,:,np.newaxis]

    images_train, images_test, segmentation_train, segmentation_test = train_test_split(images,segmentation,test_size=0.2,random_state=42)

    print(segmentation.shape)
    print(images.shape)
    print(images_train.shape)
    print(segmentation_train.shape)
    print(images_test.shape)
    print(segmentation_test.shape)
    
    #x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = get_unet_128((128, 128, 1), 1)
    model.summary()

    '''
    i = 0
    while True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.title(str(i))
        plt.imshow(images[i,:,:,0])
        plt.subplot(1,2,2)
        plt.imshow(segmentation[i,:,:,0])
        plt.show()
        i = i + 1
    '''

    '''
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
                tf.keras.callbacks.ModelCheckpoint(monitor='val_loss',
                             filepath='models/liver_segmentation.h5',
                             save_best_only=True,
                             save_weights_only=False),
                tf.keras.callbacks.TensorBoard(log_dir='logs')]
    '''
    history = model.fit(images_train, segmentation_train, batch_size=8, epochs=8, validation_data=(images_test,segmentation_test)) #batch_size=32, callbacks=callbacks

    model.save(r"models/liver_segmentation.h5")
    with open(r"models/liver_segmentation.pkl", "wb") as f:
        pickle.dump([images_train, images_test, segmentation_train, segmentation_test, history.history], f)

def load_segmentation():
    model = get_unet_128((128, 128, 1), 1)
    model.load_weights(r"models/liver_segmentation.h5")
    
    with open(r"models/liver_segmentation.pkl", 'rb') as f:
        images_train, images_test, segmentation_train, segmentation_test, history = pickle.load(f)
    '''
    history_temp = model.fit(images_train, segmentation_train, batch_size=8, epochs=1, validation_data=(images_test,segmentation_test))
    history['loss'].append(history_temp.history['loss'][0])
    history['dice_coeff'].append(history_temp.history['dice_coeff'][0])
    history['val_loss'].append(history_temp.history['val_loss'][0])
    history['val_dice_coeff'].append(history_temp.history['val_dice_coeff'][0])

    model.save(r"models/liver_segmentation.h5")
    with open(r"models/liver_segmentation.pkl", "wb") as f:
        pickle.dump([images_train, images_test, segmentation_train, segmentation_test, history], f)
    '''
    #results = model.evaluate(images_test, segmentation_test)
    #print()
    #print(f'Result on test set: {results}')
    
    #predictions = model.predict(images_test)
    #print(f'Predictions are: {predictions}')
    #print(f'Actual values are: {segmentation_test}')
    
    # Show accuracy on train and test data
    plt.figure(figsize=[8,6])
    plt.plot(history['dice_coeff'],'r',linewidth=3.0)
    plt.plot(history['val_dice_coeff'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.grid()
    plt.show()
    
    # Show the testing image with opencv
    i = 0
    while True:
        pred = model.predict(images_test[i][np.newaxis])[0]*255
        pred = pred.astype('uint8')
        combined2 = np.zeros((128,128,3),dtype='uint8')
        combined2[:,:,0] = images_test[i][:,:,0] #Blue
        combined2[:,:,1] = segmentation_test[i][:,:,0]*255//3 #Green
        combined2[:,:,2] = pred[:,:,0]//3 # Red
        acc = np.round(model.evaluate(images_test[i][np.newaxis],segmentation_test[i][np.newaxis])[1],2)
        cv2.imshow(f'image, i: {i}, accuracy: {acc}', cv2.resize(images_test[i], dsize=(512, 512)))
        #cv2.imshow(f'prediction, i: {i}, accuracy: {acc}', cv2.resize(predictions[i]*255, dsize=(512, 512)))
        cv2.imshow(f'prediction, i: {i}, accuracy: {acc}', cv2.resize(pred, dsize=(512, 512)))
        cv2.imshow(f'segmentation, i: {i},  accuracy: {acc}', cv2.resize(segmentation_test[i]*255, dsize=(512, 512)))
        combined = cv2.addWeighted(images_test[i],0.5,pred,0.5,0)
        cv2.imshow(f'combined, i: {i}, accuracy: {acc}', cv2.resize(combined, dsize=(512, 512)))
        cv2.imshow(f'combined2, i: {i}, accuracy: {acc}', cv2.resize(combined2, dsize=(512, 512)))
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(0) & 0xFF == ord('d'):
            if i < images_test.shape[0]-1:
                i = i + 1
                cv2.destroyAllWindows()
        if cv2.waitKey(0) & 0xFF == ord('a'):
            if i > 0:
                i = i - 1
                cv2.destroyAllWindows()

def main():
    if do_training:
        train_segmentation()
    load_segmentation()

if __name__ == "__main__":
    main()