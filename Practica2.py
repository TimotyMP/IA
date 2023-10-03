import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


url = 'imdb_movies.csv' 
df = pd.read_csv(url)

features = df[['revenue']]
labels = df['score']

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

perceptron = Perceptron(eta=0.01, n_iter=10)
perceptron.fit(features.values, labels)

plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Puntuacion')
plt.ylabel('Ganancias')
plt.title('Entrenamiento del Perceptrón')
plt.show()
