import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import vgg16

from tensorflow.keras.utils import to_categorical

import keras_tuner as kt


def add_top_layer(model, num_classes, hp, config=0):
    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
    new_model.add(Flatten(input_shape=model.output_shape[1:]))
    if config == 0:
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
    if config == 1:
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
    if config == 3:
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
    if config == 4:
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
    if config == 5:
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
    if config == 6:
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dense(hp.Choice('units', [32, 64, 18, 512, 1024, 2048, 4096]), hp.Choice('activation',['relu', 'sigmoid', 'tanh'])))
        new_model.add(Dropout(hp.Choice('rate', [0.2, 0.5, 0.7, 0.9])))
    new_model.add(Dense(num_classes, activation='softmax'))
    return new_model


def preprocess_dataframe(df):
    X = train_set.drop(['target', 'target_str'], axis=1).to_numpy().reshape((train_set.shape[0],)+(32, 32, 3))
    image_shape = X.shape[1:]
    y = train_set['target']
    classes = np.unique(train_set["target_str"])
    y = to_categorical(y)
    return X, y, classes, image_shape


def vgg16_model(hp):    
    vgg16_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg16_model = add_top_layer(vgg16_model, len(classes), hp, hp.Choice('config', [0,1,2,3,4,5,6]))
    vgg16_model.compile('adam', 'categorical_crossentropy', metrics='accuracy')
    return vgg16_model


def plot_history(history):
    plt.figure()
    for key in history.keys():
        plt.plot(history[key], label=key)
    plt.legend(loc='best')
    plt.savefig("./train_history.png")

def to_numeric(results):
    return np.argmax(results,axis=1)

if __name__ == '__main__':
    # Read train set from train.pkl and test set from test.pkl
    train_set = pd.read_pickle("train.pkl")
    test_set = pd.read_pickle("test.pkl")

    # Convert from Pandas Dataframe format into one recognized by ML algorithm
    X_train, y_train, classes, image_shape = preprocess_dataframe(train_set)
    X_test, y_test, _, _ = preprocess_dataframe(test_set)

    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

    # Use BayesianOptimization fine-tuning. I use callback to track the training
    tuner = kt.BayesianOptimization(vgg16_model, objective='val_loss', max_trials=5)
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[tf.keras.callbacks.TensorBoard("training_history")])
    best_trial_id = tuner.oracle.get_best_trials()[0].trial_id
    print("Fine-Tuning Done. To see training history of best trial, run:")
    print("tensorboard --logdir training_history/%s/execution0" %best_trial_id)

    # Evaluate the best model
    best_model = tuner.get_best_models()[0]
    y_pred = to_numeric(best_model.predict(X_test))
    y_test = to_numeric(y_test)
    print(y_pred.shape, y_test.shape)
    # quit()
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    conf_mat = pd.DataFrame(conf_mat, index=classes, columns=classes)
    sns.heatmap(conf_mat, annot=True)
    plt.savefig("cofusion_matrix.png")
