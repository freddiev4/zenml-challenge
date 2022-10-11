import joblib

from datetime import datetime
from typing import List

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.linear_model import LogisticRegression
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
    """
    Loads the breast cancer dataset as numpy arrays.
    Doesn't do any data cleaning / feature engineering.
    """
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
def model_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Output(
    trained_models=List[ClassifierMixin],
):
    """Train a set of classifiers."""
    models = [
        LogisticRegression(max_iter=1000),
        KNeighborsClassifier(),
        RandomForestClassifier(),
    ]
    trained_models = []
    for model in models:
        trained_models.append(
            model.fit(x_train, y_train)
        )

    return trained_models


@step(
    experiment_tracker=EXPERIMENT_TRACKER,
    settings={
        "experiment_tracker.wandb": WANDB_SETTINGS,
    },
)
def model_evaluator(
    trained_models: List[ClassifierMixin],
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Output(
    best_model=ClassifierMixin,
):
    """
    Evaluates several different models, and returns the best performing
    model based on f1 score (arbitrarily selected from classification metrics).

    Doesn't do any hyperparameter tuning.
    """
    model_to_scores = {}
    
    for model in trained_models:
        predictions = model.predict(x_val)

        score = f1_score(y_val, predictions, average='weighted')

        wandb.log({
            "f1_score": score,
        })

        model_to_scores[model] = score

    def select_best_model(model_to_scores):
        # return max(model_to_scores, key=model_to_scores.get)
        return model_to_scores.keys()[0]

    best_model = select_best_model(model_to_scores)

    wandb.log({
        "best_model_score": model_to_scores[best_model],
    })
    return best_model


@step
def serialize_model(
    model: ClassifierMixin,
) -> None:
    # TODO: Add logic for serializing model + uploading somewhere
    # only if the model performance meets certain thresholds
    joblib.dump(model, "best_model.joblib")


@pipeline(name='breast_cancer_model_selection_pipeline')
def breast_cancer_model_selection_pipeline(
    data_loader_step, 
    model_training_step, 
    model_eval_step, 
    model_serialization_step,
):
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test
    ) = data_loader_step()

    trained_models = model_training_step(
        x_train=x_train,
        y_train=y_train,
    )

    best_model = model_eval_step(
        trained_models=trained_models,
        x_val=x_val,
        y_val=y_val,
    )

    model_serialization_step(model=best_model)


# TODO: Add model deployment
pipeline_instance = breast_cancer_model_selection_pipeline(
    data_loader_step=data_loader(),
    model_training_step=model_trainer(),
    model_eval_step=model_evaluator(),
    model_serialization_step=serialize_model(),
)

from contextlib import contextmanager

@contextmanager
def postmortem_pdb():
    try:
        yield
    except Exception as exc:
        import pdb
        pdb.post_mortem()


# with postmortem_pdb():
pipeline_instance.run(run_name=f'breast-cancer-pipeline-{datetime.now()}')

