import os
import re
import gzip
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import cv2
import pickle
import matplotlib.pyplot as plt

do_training = False

# Directory where the preprocessed data is stored.
#output_dir = r"data\Training_Batch1\preprocessed"
output_dir = r"data\All_Data"

def CheckPrediction(Prediction,label):
    if label >= 0.5 and Prediction[1] >= 0.5:
        return True
    elif label < 0.5 and Prediction[1] < 0.5:
        return True
    else:
        return False

def train_classifier():
    # Load labels
    labels_pattern = re.compile(r'labels-(\d+).npy')
    labels_files = filter(lambda f: re.match(
    labels_pattern, f), os.listdir(output_dir))
    labels = np.array([])
    images = np.array([])
    for f in labels_files:
        labels = np.append(labels, np.load(os.path.join(output_dir, f)))
        
        # Load images (stack all frames vertically) 
        # Loading images this way ensures that labels and images have the same order
        file_num = re.match(labels_pattern, f)[1]
        with gzip.GzipFile(os.path.join(output_dir, f'images-{file_num}.npy.gz'), "r") as file:
            if images.size == 0:
                images = np.load(file)
            else:
                images = np.concatenate((images, np.load(file)),axis=0) # (Z, X, Y)

    images_train, images_test, labels_train, labels_test = train_test_split(images,labels,test_size=0.2,random_state=42)

    print(labels.shape)
    print(images.shape)
    print(images_train.shape)
    print(labels_train.shape)
    print(images_test.shape)
    print(labels_test.shape)
    
    #x_train, x_test = x_train / 255.0, x_test / 255.0
    '''
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.Dense(2, activation=tf.nn.softmax)])
    '''
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128,128,1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    images_train = images_train[:,:,:,np.newaxis]
    images_test = images_test[:,:,:,np.newaxis]
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(images_train, labels_train, batch_size=128, epochs=25, validation_data=(images_test,labels_test))#batch_size=128

    model.save(r"models/CNN.h5")
    with open(r"models/CNN.pkl", "wb") as f:
        pickle.dump([images_train, images_test, labels_train, labels_test, history.history], f)

def load_classifier():
    #model = tf.keras.models.load_model(r"models/CNN.h5")
    
    with open(r"models/Presentation/CNN_v13.pkl", 'rb') as f:
        indices_train, indices_test, indices_validation, history, accuracy, con_matrix = pickle.load(f,encoding='bytes')
    '''
    results_test = model.evaluate(images_test, labels_test)
    print()
    print(f'Result on test set: {results_test}')
    results_train = model.evaluate(images_train, labels_train)
    print()
    print(f'Result on train set: {results_train}')
    
    predictions = model.predict(images_test)
    print(f'Predictions are: {predictions}')
    print(f'Actual values are: {labels_test}')
    
    # Find all wrong classified images
    Wrong_classified = np.array([],dtype=int)
    for i in range(images_test.shape[0]):
        if CheckPrediction(predictions[i],labels_test[i]) == False:
            Wrong_classified = np.append(Wrong_classified, i)

    # Determine confusion matrix (use one test data)
    predictions_list = np.array([])
    for i in range(images_test.shape[0]):
        if predictions[i,1] >= 0.5:
            predictions_list = np.append(predictions_list, 1.)
        else:
            predictions_list = np.append(predictions_list, 0.)

    con_matrix = confusion_matrix(labels_test, predictions_list)
    print(f'Confustion matrix using test data: {con_matrix}')

    # Determine number of slices containing liver
    N_liver_train = np.count_nonzero(labels_train)
    print(f'Number of slices containing liver (train data): {N_liver_train}')
    print(f'Number of slices containing no liver (train data): {labels_train.size - N_liver_train}')
    N_liver_test = np.count_nonzero(labels_test)
    print(f'Number of slices containing liver (test data): {N_liver_test}')
    print(f'Number of slices containing no liver (test data): {labels_test.size - N_liver_test}')
    N_liver = N_liver_train + N_liver_test
    print(f'Number of slices containing liver (all data): {N_liver}')
    print(f'Number of slices containing no liver (all data): {labels_train.size + labels_test.size - N_liver}')
    '''
    print(len(indices_test))
    print(accuracy)
    # Show accuracy on train and test data
    plt.figure(figsize=[8,6])
    plt.plot(history[b'acc'],'r',linewidth=3.0)
    plt.plot(history[b'val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.grid()
    plt.show()
    '''
    # Show the testing image with opencv
    j = 0
    i = Wrong_classified[j]
    while True:
        cv2.imshow(f'i: {i}, Pred.: {np.round(predictions[i],2)}, Lab.: {np.round(labels_test[i])}, Result: {CheckPrediction(predictions[i],labels_test[i])}', cv2.resize(images_test[i], dsize=(512, 512)))

        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(0) & 0xFF == ord('d'):
            if j < Wrong_classified.size-1:#i <labels_test.size-1:
                j = j + 1
                i = Wrong_classified[j]
                cv2.destroyAllWindows()
        if cv2.waitKey(0) & 0xFF == ord('a'):
            if j > 0:
                j = j - 1
                i = Wrong_classified[j]
                cv2.destroyAllWindows()
    '''
def main():
    if do_training:
        train_classifier()
    load_classifier()

if __name__ == "__main__":
    main()