from datetime import datetime

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from zenml.pipelines import pipeline
from zenml.steps import Output, step


@step
def digits_data_loader() -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
):
    """Loads the digits dataset as a tuple of flattened numpy arrays."""
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test


@step
def svc_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train a sklearn SVC classifier."""
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model


@pipeline
def first_pipeline(step_1, step_2):
    (
        X_train, 
        X_test, 
        y_train, 
        y_test
    ) = step_1()

    step_2(
        X_train=X_train, 
        y_train=y_train,
    )

first_pipeline_instance = first_pipeline(
    step_1=digits_data_loader(),
    step_2=svc_trainer(),
)


first_pipeline_instance.run(run_name=f'zenml_test_pipeline-{datetime.now()}')
