import os
import re
import gzip
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

do_training = True

# Directory where the preprocessed data will be stored.
output_dir = r"data\Training_Batch1\preprocessed"

def train_classifier():
    # Load data
    im_pattern = re.compile(r'images-(\d+).npy.gz')
    im_files = filter(lambda f: re.match(
    im_pattern, f), os.listdir(output_dir))

    # Load images (stack all frames vertically)
    images = []
    for f in im_files:
        with gzip.GzipFile(os.path.join(output_dir, f), "r") as file:
            if images == []:
                images = np.load(file)
            else:
                images = np.concatenate((images, np.load(file)),axis=0) # (Z, X, Y)

    labels_pattern = re.compile(r'labels-(\d+).npy')
    labels_files = filter(lambda f: re.match(
    labels_pattern, f), os.listdir(output_dir))
    labels = np.array([])
    for f in labels_files:
        labels = np.append(labels, np.load(os.path.join(output_dir, f)))

    images_train, images_test, labels_train, labels_test = train_test_split(images,labels,test_size=0.2,random_state=42)

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

    print(results)


#predictions = model.predict(x_test)
#create_figure(0, predictions, x_test, y_test)

def main():
    if do_training:
        train_classifier()



if __name__ == "__main__":
    main()