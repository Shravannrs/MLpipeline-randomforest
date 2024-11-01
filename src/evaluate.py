import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from mlflow.models import infer_signature
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

import mlflow

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Shravannrs/pipelineportfolio.mlflow"

os.environ['MLFLOW_TRACKING_USERNAME']="Shravannrs"
os.environ['MLFLOW_TRACKING_PASSWORD']="0187c76b603d8a274635ea5783dc6da774037cde"

params=yaml.safe_load(open("params.yaml"))["train"]


def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X= data.drop(columns=["Outcome"])
    y= data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/Shravannrs/pipelineportfolio.mlflow")

    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)

    accuracy=accuracy_score(y,predictions)

    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")


if __name__== "main":
    evaluate(params["data"],params["model"])
