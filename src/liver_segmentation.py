import os
import re
import gzip
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from segmentation_models import get_unet_128

version = 2

# Directory where the preprocessed data is stored.
input_dir = r"data\All_Data"
output_dir = r"models"

# Old function (not usefull anymore)
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

def load_segmentation_cluster():
    # Load model
    model = get_unet_128((128, 128, 1), 1)
    model.load_weights(os.path.join(output_dir, "unet_128_v{}.h5".format(version)))
    # Load data saved by cluster
    with open(os.path.join(output_dir, "unet_128_v{}.pkl".format(version)), 'rb') as f:
        indices_train, indices_test, indices_validation, history, dice_coeff, con_matrix, predictions = pickle.load(f,encoding='bytes')
    
    # Load images and do preprocessing
    lab_pattern = re.compile(r'labels-(\d+).npy')
    labels_files = filter(lambda f: re.match(
        lab_pattern, f), os.listdir(input_dir))
    labels = np.array([])
    segmentation = np.array([])
    images = np.array([])
    for f in labels_files:
        # Load label so slices with no liver can be removed
        labels = np.append(labels, np.load(os.path.join(input_dir, f)))

        # Load segmentation (stack all frames vertically), datatype of segmentation array is uint8
        # Loading segmentation this way ensures that labels and images have the same order
        file_num = re.match(lab_pattern, f).group(1)
        with gzip.GzipFile(os.path.join(input_dir, 'liver_segmentation-{}.npy.gz'.format(file_num)), "r") as file:
            if segmentation.size == 0:
                segmentation = np.load(file)
            else:
                segmentation = np.concatenate((segmentation, np.load(file)),axis=0) # (Z, X, Y)

        # Load images (stack all frames vertically)
        with gzip.GzipFile(os.path.join(input_dir, 'images-{}.npy.gz'.format(file_num)), "r") as file:
            if images.size == 0:
                images = np.load(file)
            else:
                images = np.concatenate((images, np.load(file)),axis=0) # (Z, X, Y)

    # Filter out all slices containing no liver part
    images = images[labels == 1,:,:]
    segmentation = segmentation[labels == 1,:,:]

    # Add new dimension (explicit mention that we have only one color channel)
    images = images[:,:,:,np.newaxis]
    segmentation = segmentation[:,:,:,np.newaxis]

    # Change range from 0-255 to 0-1 (datatype change from uint8 to float64)
    images = images / 255.0

    # Change datatype labels from float64 to booleans
    labels = labels.astype(bool, copy=False)
    
    # Split data in train, validation and test set (same as done by cluster)
    images_train = images[indices_train]
    images_validation = images[indices_validation]
    images_test = images[indices_test]
    segmentation_train = segmentation[indices_train]
    segmentation_validation = segmentation[indices_validation]
    segmentation_test = segmentation[indices_test]

    # Show dice coefficients
    print("Dice coefficient on train data: {}".format(dice_coeff[b"train"][1]))
    print("Dice coefficient on validation data: {}".format(dice_coeff[b"validation"][1]))
    print("Dice coefficient on test data: {}".format(dice_coeff[b"test"][1]))

    # Show confusion matrix for test set
    print("Confusion matrix for test set: {}".format(con_matrix))

    print(history[b'val_dice_coeff'])
    print(history[b'dice_coeff'])
    
    # Show dice coefficient on train and validation data as function of epoch
    plt.figure(figsize=[8,6])
    plt.plot(history[b'dice_coeff'],'r',linewidth=3.0)
    plt.plot(history[b'val_dice_coeff'],'b',linewidth=3.0)
    plt.legend(['Training dice_coeff', 'Validation dice_coeff'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Dice coeff',fontsize=16)
    plt.title('Dice coefficient Curves',fontsize=16)
    plt.grid()
    plt.show()
    
    # Show images from the test set with openCV
    i = 0
    while True:
        predictions_t = model.predict(images_test[i][np.newaxis])
        #predictions_t = predictions[i][np.newaxis]
        combined2 = np.zeros((128,128,3))
        combined2[:,:,0] = images_test[i,:,:,0] #Blue
        combined2[:,:,1] = segmentation_test[i,:,:,0]/2 #Green
        combined2[:,:,2] = predictions_t[0,:,:,0]/2 # Red
        dice = np.round(model.evaluate(images_test[i][np.newaxis],segmentation_test[i][np.newaxis])[1],2)
        cv2.imshow(f'image, i: {i}, dice_coeff: {dice}', cv2.resize(images_test[i], dsize=(512, 512)))
        #cv2.imshow(f'prediction, i: {i}, accuracy: {acc}', cv2.resize(predictions_t[0]*255, dsize=(512, 512)))
        cv2.imshow(f'prediction, i: {i}, dice_coeff: {dice}', cv2.resize(predictions_t[0]*255., dsize=(512, 512)))
        cv2.imshow(f'segmentation, i: {i},  dice_coeff: {dice}', cv2.resize(segmentation_test[i]*255., dsize=(512, 512)))
        #combined = cv2.addWeighted(images_test[i],0.5,predictions_t[0],0.5,0)
        #cv2.imshow(f'combined, i: {i}, dice_coeff: {dice}', cv2.resize(combined, dsize=(512, 512)))
        cv2.imshow(f'combined2, i: {i}, dice_coeff: {dice}', cv2.resize(combined2, dsize=(512, 512)))
        
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
    #load_segmentation()
    load_segmentation_cluster()

if __name__ == "__main__":
    main()