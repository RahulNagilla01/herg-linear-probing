import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LogisticProbe:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        self.scaler = StandardScaler()

    def fit_evaluate(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        self.model.fit(X_train, y_train)
        return accuracy_score(y_test, self.model.predict(X_test))


class DifferenceOfMeansProbe:
    def fit_evaluate(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Compute concept direction
        pos_mean = X_train[y_train == 1].mean(axis=0)
        neg_mean = X_train[y_train == 0].mean(axis=0)
        direction = pos_mean - neg_mean
        direction = direction / np.linalg.norm(direction)

        # Find threshold on train projections
        train_proj = X_train @ direction
        threshold = (
            train_proj[y_train == 1].mean() +
            train_proj[y_train == 0].mean()
        ) / 2

        # Evaluate on test
        test_proj = X_test @ direction
        preds = (test_proj > threshold).astype(int)
        return accuracy_score(y_test, preds)