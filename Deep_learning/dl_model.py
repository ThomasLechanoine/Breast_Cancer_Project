import numpy as np

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

#///////////initialize_model/////////////
def dl_initialize_model():
    """
    Initialize
    """
    #///CODE A AJOUTER///
    model = Sequential()

    model.add(layers.Input((224, 224, 3)))
    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", kernel_regularizer=l2(0.01)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_regularizer=l2(0.01)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_regularizer=l2(0.01)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(8, activation="relu"))


    model.add(layers.Dense(1, activation="sigmoid"))

    model.summary()
    print("✅ Model initialized")

    return model

#///////////compile_model/////////////
def dl_compile_model(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall'])
    """
    Compile the Neural Network
    """
    #///CODE A AJOUTER///
    model.compile(
        optimizer=optimizer,
        loss=optimizer,
        metrics=metrics
    )
    print("✅ Model compiled")

    return model

#///////////train_model/////////////
def dl_train_model(model, train_dataset, epochs, validation_dataset, callbacks, verbose)
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    train_dataset = image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="binary",
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=123
    )

    validation_dataset = image_dataset_from_directory(
        valid_path,
        labels="inferred",
        label_mode="binary",
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=123
    )

    #///CODE A AJOUTER///
    callbacks = [
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_recall', mode='max'),
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    ]

    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

#///////////evaluate_model/////////////
def dl_evaluate_model(model, test_dataset)
    """
    Evaluate trained model performance on the dataset
    """
    #///CODE A AJOUTER///
    test_loss, test_acc, test_recall = model.evaluate(test_dataset)

    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("Test recall:", test_recall)

    return test_loss, test_acc, test_recall
