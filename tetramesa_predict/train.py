import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import vgg16

from tensorflow.keras.utils import to_categorical



def add_top_layer(model, num_classes):
    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
    new_model.add(Flatten(input_shape=model.output_shape[1:]))
    new_model.add(Dense(512, activation='relu'))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(num_classes, activation='softmax'))
    return new_model


def pickle_to_dataframe(df):
    X = train_set.drop('target', axis=1).to_numpy().reshape( (train_set.shape[0],)+(32, 32, 3))
    image_shape = X.shape[1:]
    y = train_set['target']
    classes = np.unique(y)
    y = to_categorical(y)
    return X, y, classes, image_shape


def vgg16_model():    
    vgg16_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg16_model = add_top_layer(vgg16_model, len(classes))
    vgg16_model.compile('adam', 'categorical_crossentropy', metrics='accuracy')
    return vgg16_model


def plot_history(history):
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['accuracy'])
    plt.savefig("./train_history.png")


if __name__ == '__main__':
    train_set = pd.read_pickle("train.pkl")
    X_train, y_train, classes, image_shape = pickle_to_dataframe(train_set)
    model_1 = vgg16_model()
    hist = model_1.fit(X_train, y_train, epochs=20, batch_size=16)
    plot_history(hist.history)