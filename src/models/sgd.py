from typing import List

import warnings
import numpy as np

from joblibspark import register_spark

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import parallel_backend
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from pyspark.sql.dataframe import DataFrame

warnings.filterwarnings('ignore')
register_spark()

class SGDC:
    def __init__(self):
        self.model = SGDClassifier(
            loss="hinge",
            penalty="l2",
            max_iter=2,           # chỉ 1 vòng lặp mỗi batch
            warm_start=True,      # giữ lại mô hình qua các batch
            learning_rate="optimal"
        )
        self.is_fitted = False # chỉ truyền classes qua lần huấn luyện ban đầu

    def train(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        if not self.is_fitted:
            self.model.partial_fit(X, y, classes=np.arange(10))
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)

        predictions = self.model.predict(X)

        accuracy = self.model.score(X,y)
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        return predictions, accuracy, precision, recall, f1

    def predict(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        
        predictions = self.model.predict(X)
        # predictions = np.array(predictions)
        
        accuracy = self.model.score(X,y)
        
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)
        cm = confusion_matrix(y, predictions)
        
        return predictions, accuracy, precision, recall, f1, cm