import joblib

from datetime import datetime
from typing import List

import mlflow
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

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer
from zenml.integrations.wandb.flavors.wandb_experiment_tracker_flavor import WandbExperimentTrackerSettings
from zenml.integrations.mlflow.steps import MLFlowDeployerParameters, mlflow_model_deployer_step
from zenml.pipelines import pipeline
from zenml.steps import (
    BaseParameters, 
    Output, 
    step,
)

EXPERIMENT_TRACKER = "mlflow_tracker"

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
        random_state=RANDOM_STATE,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


@step()
def regression_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Output(
    trained_regression=ClassifierMixin,
):
    """Train a LogisticRegression classifier."""
    trained_regression = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    return trained_regression



@step()
def knn_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Output(
    trained_knn=ClassifierMixin,
):
    """Train a KNN classifier."""
    trained_knn = KNeighborsClassifier().fit(x_train, y_train)
    return trained_knn


@step()
def random_forest_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Output(
    trained_random_forest=ClassifierMixin,
):
    """Train a RandomForest classifier."""
    trained_random_forest = RandomForestClassifier().fit(x_train, y_train)
    return trained_random_forest


@step(experiment_tracker=EXPERIMENT_TRACKER)
def model_evaluator(
    trained_regression: ClassifierMixin,
    trained_knn: ClassifierMixin,
    trained_random_forest: ClassifierMixin,
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
    trained_models = [trained_regression, trained_knn, trained_random_forest]

    models_to_scores = {}
    with mlflow.start_run(nested=True):
        for model in trained_models:
            # get the predictions and calculate f1 from ground truth
            predictions = model.predict_proba(x_val)

            score = 1 # f1_score(y_val, predictions, average='weighted')

            # map model to score
            models_to_scores[model] = score

            mlflow.log_metrics({"f1_score": score})

    def select_best_model(models_to_scores):
        """
        Gets the model with the highest f1-score
        """
        return max(models_to_scores, key=models_to_scores.get)

    best_model = select_best_model(models_to_scores)
    return best_model


@step
def serialize_best_model(
    best_model: ClassifierMixin,
) -> None:
    # TODO: Add logic for serializing model + uploading somewhere
    # only if the model performance meets certain thresholds
    joblib.dump(best_model, f"best_model_{type(best_model).__name__.lower()}.joblib")



@step
def model_deployment_decision_step() -> Output(
    deployment_decision=bool, 
):
    deployment_decision = True
    return deployment_decision


@pipeline(name='breast_cancer_model_selection_pipeline_v3')
def breast_cancer_model_selection_pipeline(
    data_loader_step, 
    regression_trainer_step,
    knn_trainer_step,
    random_forest_trainer_step, 
    model_eval_step, 
    model_serialization_step,
    model_deployment_decision_step,
    model_deployer_step,
):
    # load in data for train / validation / test
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test
    ) = data_loader_step()

    # train several different models in parallel
    trained_regression = regression_trainer_step(
        x_train=x_train,
        y_train=y_train,
    )
    trained_knn = knn_trainer_step(
        x_train=x_train,
        y_train=y_train,
    )
    trained_random_forest = random_forest_trainer_step(
        x_train=x_train,
        y_train=y_train,
    )

    # evaluate all the different models on the validation set
    # and return the best model
    best_model = model_eval_step(
        trained_regression=trained_regression,
        trained_knn=trained_knn,
        trained_random_forest=trained_random_forest,
        x_val=x_val,
        y_val=y_val,
    )

    # serialize the best model
    model_serialization_step(best_model=best_model)

    deploy_decision = model_deployment_decision_step()

    # spin up the inference service
    model_deployer_step(
        deploy_decision=deploy_decision,
        model=best_model,
    )


pipeline_instance = breast_cancer_model_selection_pipeline(
    data_loader_step=data_loader(),
    regression_trainer_step=regression_trainer(),
    knn_trainer_step=knn_trainer(),
    random_forest_trainer_step=random_forest_trainer(), 
    model_eval_step=model_evaluator(),
    model_serialization_step=serialize_best_model(),
    model_deployment_decision_step=model_deployment_decision_step(),
    model_deployer_step=mlflow_model_deployer_step(   
        params=MLFlowDeployerParameters(
            model_name="breast-cancer-inference-pipeline",
            workers=3,
        )
    ),
)

if __name__ == '__main__':
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

