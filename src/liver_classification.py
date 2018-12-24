import os
import re
import gzip
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

do_training = True

# Directory where the preprocessed data is stored.
output_dir = r"data\Training_Batch1\preprocessed"

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

    images_train, images_test, labels_train, labels_test = train_test_split(images,labels,test_size=0.25,random_state=42)

    print(labels.shape)
    print(images.shape)
    print(images_train.shape)
    print(labels_train.shape)
    print(images_test.shape)
    print(labels_test.shape)
            
    #x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(images_train, labels_train, epochs=5)
    results = model.evaluate(images_test, labels_test)

    print()
    print(f'Result on test set: {results}')

    predictions = model.predict(images_test)
    print(f'Predictions are: {predictions}')
    print(f'Actual values are: {labels_test}')

    # Show the testing image with opencv
    i = 0
    while True:
        cv2.imshow(f'image: {i}, Prediction: {predictions[i]}, Label: {labels_test[i]}, classification: {np.rint(predictions[i,1]==np.rint(labels_test[i]))}', cv2.resize(images_test[i], dsize=(512, 512)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(25) & 0xFF == ord('d'):
            if i < labels_test.size-1:
                i = i + 1
                cv2.destroyAllWindows()
        if cv2.waitKey(25) & 0xFF == ord('a'):
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