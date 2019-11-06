# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import os
import pathlib
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split

from typing import Iterable, List, Optional


default_n_splits = 5
default_generator = KFold(n_splits=default_n_splits)


class Data:

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            fold_generator: Iterable = default_generator
        ) -> None:
        self.x = x
        self.y = y
        self.fold_generator = fold_generator

    def __getitem__(self, index) -> "Data":
        subset = self.copy()
        subset.x = subset.x[index]
        subset.y = subset.y[index]
        return subset

    def copy(self) -> "Data":
        copy_ = Data(None, None, self.fold_generator)
        copy_.x = self.x.copy()
        copy_.y = self.y.copy()
        return copy_

    @property
    def folds(self) -> Iterable:
        def generator():
            for train_index, test_index in self.fold_generator.split(self.x):
                yield [
                        [self.x[train_index], self.y[train_index]],
                        [self.x[test_index], self.y[test_index]]
                      ]
        return generator()


def MLP(
        n_input_variables: int,
        n_output_variables: int,
        n_hidden_layers: int = 3,
        n_nodes: int = 50,
        activation: str = "tanh",
        loss: str = "mse",
        l2_strength: float = 0.0,
        l1_strength: float = 0.0,
        dropout_rate: Optional[float] = None
    ) -> tf.keras.models.Model:
    input_layer = tf.keras.layers.Input(shape=n_input_variables)
    for k in range(n_hidden_layers):
        if k == 0:
            output = input_layer
        reg = tf.keras.regularizer.l1_l2(l1=l1_strength, l2=l2_strength)
        output = tf.keras.layers.Dense(
                n_nodes,
                activation=activation,
                kernel_regularizer=reg
            )(output)
        if dropout_rate is not None:
            output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Dense(n_output_variables)(output)
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output])
    model.compile(loss=loss)
    return model


def _create_model_dir(root: str, fold: int) -> None:
    path = os.path.join(root, str(fold))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _create_callbacks(
        path: str,
        early_stopping: bool = True,
        patience: int = 10
    ) -> List:
    callbacks = []
    if early_stopping:
        callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=patience
                    )
                )
    callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True)
        )
    callbacks.append(tf.keras.callbacks.History())
    return callbacks


def train(
        model: tf.keras.models.Model,
        data: Data,
        path: str,
        batch_size: int = 100,
        train_val_split: Optional[float] = 0.25,
        early_stopping: bool = True,
        patience: int = 10
    ) -> dict:
    """ Trains a model using a k-fold cross validation scheme. Returns all
    trained models and estimates of network accuracy. """
    returns = dict("histories", [], "models", [], "metrics", [])
    for k, [train_val_set, test_set] in enumerate(data.folds):
        print("Fitting model to fold {}".format(k))
        path = os.path.join(path, str(k))
        train_x, train_y, val_x, val_y = train_test_split(
                train_val_set.x,
                train_val_set.y
            )
        history = model.fit(
                train_x,
                train_y,
                epochs=2000,
                batch_size=batch_size,
                validation_data=[val_x, val_y],
                callbacks=_create_callbacks(
                        path,
                        early_stopping=early_stopping,
                        patience=patience
                    )
            )
        model_best = tf.keras.models.load_model(path)
        metrics = model_best.evaluate(x=test_set.x, y=test_set.y)
        returns["histories"].append(history)
        returns["models"].append(model_best)
        returns["metrics"].append(metrics)
        return returns
