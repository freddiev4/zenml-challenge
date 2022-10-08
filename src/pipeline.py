from datetime import datetime

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier

import wandb

from zenml.integrations.wandb.flavors.wandb_experiment_tracker_flavor import WandbExperimentTrackerSettings
from zenml.pipelines import pipeline
from zenml.steps import Output, step

EXPERIMENT_TRACKER = "wandb_experiment_tracker"
WANDB_SETTINGS = WandbExperimentTrackerSettings(
    settings=wandb.Settings(magic=True),
    tags=["sklearn-pipeline"]
)

TRAIN_SPLIT_RATIO = 0.60
VALIDATION_SPLIT_RATIO = 0.20
TEST_SPLIT_RATIO = 0.20

RANDOM_STATE = 42


@step
def data_loader() -> Output(
    x_train=np.ndarray,
    x_val=np.ndarray,
    x_test=np.ndarray,
    y_train=np.ndarray,
    y_val=np.ndarray,
    y_test=np.ndarray,
):
    """Loads the breast cancer dataset as numpy arrays."""
    bunch = load_breast_cancer()
    samples = bunch.data
    labels = bunch.target

    # train is now 60% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(
        samples,
        labels,
        test_size=1 - TRAIN_SPLIT_RATIO,
        random_state=RANDOM_STATE,
    )

    # validation is now 20% of the initial data set
    # test is now 20% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(
        x_test,
        y_test,
        test_size=TEST_SPLIT_RATIO / (TEST_SPLIT_RATIO + VALIDATION_SPLIT_RATIO),
        shuffle=False,
        random_state=RANDOM_STATE,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


@step
def knn_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Train a K-Nearest neighbors classifier."""
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    return model


@step(
    experiment_tracker=EXPERIMENT_TRACKER,
    settings={
        "experiment_tracker.wandb":WANDB_SETTINGS,
    },
)
def knn_evaluator(
    model: ClassifierMixin,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    predictions = model.predict(x_val)

    f1 = f1_score(y_val, predictions, average='weighted')
    avg_precision = average_precision_score(y_val, predictions, average='weighted')
    auc = roc_auc_score(y_val, predictions, average='weighted')

    wandb.log({
        "f1_score": f1,
        "avg_precision": avg_precision,
        "roc_auc": auc,
    })


@pipeline
def knn_pipeline(step_1, step_2, step_3):
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test
    ) = step_1()

    model = step_2(
        x_train=x_train,
        y_train=y_train,
    )

    # TODO: Serialize model
    step_3(
        model=model,
        x_val=x_val,
        y_val=y_val,
    )

knn_pipeline_instance = knn_pipeline(
    step_1=data_loader(),
    step_2=knn_trainer(),
    step_3=knn_evaluator(),
)


knn_pipeline_instance.run(run_name=f'knn-train-eval-pipeline{datetime.now()}')
