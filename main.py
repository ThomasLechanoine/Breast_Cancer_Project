# ////////////////////////// IMPORT //////////////////////////

import numpy as np

from Deep_learning import ()
from Machine_learning import ()

from params import *

# ///////////////////// VISUALISATIONS //////////////////////



# //////////////////// MACHINE LEARNING /////////////////////
"""
fonctions for machine learning based on CSV
"""

def ml_preprocess_and_train()

    # Preprocess data using ml_preprocess.py


# ///////////////////// DEEP LEARNING ////////////////////
"""
fonctions for deep learning based on CSV
"""

def preprocess_and_train()

    # Preprocess images using dl_preprocess.py
    data = preprocess(data)

    # Train a model on the training set, using `ml_model.py`
    model = None
    learning_rate = # a ajouter
    batch_size = # a ajouter
    patience = # a ajouter

    model = initialize_model(input_shape=# a ajouter)
    model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(
        model,
        batch_size=batch_size,
        patience=patience,
        validation_data=# a ajouter
    )

# ///////////////////// END ////////////////////
