from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import numpy as np



class RegModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        self.b = np.linalg.inv(X.T@X + self.alpha*np.eye(X.shape[1]))@X.T
        return self
    def predict(self, X):
        return X@self.b


class ARIMAX:
    def __init__(self):
        self.models = []

    def _normalize(self, X, Y):
        X_normalizers = []
        Y_normalizer = MinMaxScaler()
        Y = Y_normalizer.fit_transform(Y)
        X_normalized = []
        for x in X:
            scaler = MinMaxScaler()
            scaler.fit(x)
            X_normalized.append(scaler.transform(x))
            X_normalizers.append(scaler)

        self.X_normalizers = X_normalizers
        self.Y_normalizer = Y_normalizer
        return X_normalized, Y

    def fit(self, X, Y):
        self.models = []
        X, Y = self._normalize(X, Y)
        Y = Y.T

        for y in Y:
            self.models.append(Ridge(alpha=1.0).fit(np.hstack(X), y))

    def predict(self, X):
        preds = []
        for model in self.models:
            preds.append(model.predict(np.hstack(X)))
        return np.array(preds).T
