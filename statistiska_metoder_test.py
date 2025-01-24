import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = None
        self.n = X.shape[0]  # Number of samples
        self.d = X.shape[1]  # Number of features

    def fit(self):
        # Fit the model using Ordinary Least Squares
        X_b = np.c_[np.ones((self.n, 1)), self.X]  # Add bias term
        self.beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ self.y

    def variance(self):
        # Calculate variance of the residuals
        predictions = self.predict(self.X)
        residuals = self.y - predictions
        return np.var(residuals)

    def standard_deviation(self):
        # Calculate standard deviation of residuals
        return np.sqrt(self.variance())

    def predict(self, X):
        # Predict using the linear model
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return X_b @ self.beta

    def significance(self):
        # Report significance of the regression
        predictions = self.predict(self.X)
        residuals = self.y - predictions
        mse = self.variance()
        se = np.sqrt(mse / self.n)
        t_values = self.beta / se
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=self.n - 1)) for t in t_values]
        return p_values

    def r_squared(self):
        # Calculate R² value
        ss_total = np.sum((self.y - np.mean(self.y)) ** 2)
        ss_residual = np.sum((self.y - self.predict(self.X)) ** 2)
        return 1 - (ss_residual / ss_total)
