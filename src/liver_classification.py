import os
import re
import gzip
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import pickle

do_training = True

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
    results = model.evaluate(images_test, labels_test)

    model.save(r"models/CNN.h5")
    with open(r"models/CNN.pkl", "wb") as f:
        pickle.dump([images_train, images_test, labels_train, labels_test, history.history], f)
    '''
    # Getting back the objects:
    with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
        obj0, obj1, obj2 = pickle.load(f)
    '''
    print()
    print(f'Result on test set: {results}')

    predictions = model.predict(images_test)
    print(f'Predictions are: {predictions}')
    print(f'Actual values are: {labels_test}')

    # Show the testing image with opencv
    i = 0
    while True:
        cv2.imshow(f'i: {i}, Pred.: {np.round(predictions[i],2)}, Lab.: {np.round(labels_test[i])}, Result: {CheckPrediction(predictions[i],labels_test[i])}', cv2.resize(images_test[i], dsize=(512, 512)))

        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(0) & 0xFF == ord('d'):
            if i < labels_test.size-1:
                i = i + 1
                cv2.destroyAllWindows()
        if cv2.waitKey(0) & 0xFF == ord('a'):
            if i > 0:
                i = i - 1
                cv2.destroyAllWindows()

#predictions = model.predict(x_test)
#create_figure(0, predictions, x_test, y_test)

def main():
    if do_training:
        train_classifier()

if __name__ == "__main__":
    main()