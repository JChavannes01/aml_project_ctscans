import tensorflow as tf

class AccuracyHistory(tf.keras.callbacks.Callback):
    def __init__(self, model, images_train, labels_train):
        self.model = model
        self.images_train = images_train
        self.labels_train = labels_train
    def on_train_begin(self, logs=None):
        self.myHistory = {b"train_loss": [], b"train_acc": [], b"val_loss": [], b"val_acc": []}
    def on_epoch_end(self, epoch, logs=None):
        train_result = self.model.evaluate(x=self.images_train, y=self.labels_train, verbose=0)
        self.myHistory[b"train_loss"].append(train_result[0])
        self.myHistory[b"train_acc"].append(train_result[1])
    def add_validation_accuracy(self, history_from_fit_function):
        self.myHistory[b"val_loss"] = history_from_fit_function[b"val_loss"]
        self.myHistory[b"val_acc"] = history_from_fit_function[b"val_acc"]
    
def get_basic_denselayers(dropout_rate):
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(128, 128, 1)),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

def get_cnn(drrs=[0.25, 0.25, 0.25, 0.5]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128,128,1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(drrs[0]))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(drrs[1]))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(drrs[2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(drrs[3]))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model