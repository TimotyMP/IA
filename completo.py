import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """Clasificador Perceptrón.

    Parámetros
    ------------
    eta : float
        Tasa de aprendizaje (entre 0.0 y 1.0)
    n_iter : int
        Pases sobre el conjunto de entrenamiento.

    Atributos
    -----------
    w_ : 1d-array
        Pesos después del ajuste.
    errores_ : lista
        Número de clasificaciones erróneas (actualizaciones) en cada época.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Ajustar datos de entrenamiento.

        Parámetros
        ----------
        X : {similar a array}, forma = [n_muestras, n_características]
            Vectores de entrenamiento, donde n_muestras es el número de muestras y
            n_características es el número de características.
        y : similar a array, forma = [n_muestras]
            Valores objetivo.

        Devuelve
        -------
        self : objeto

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calcular entrada neta"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Devolver etiqueta de clase después del paso unitario"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


#############################################################################
## MUESTRA LOS DATOS DE ENTRENAMIENTO CON PERCEPTRON ##
print(50 * '=')
print('Sección: Entrenamiento de un modelo de Perceptrón')
print(50 * '-')
## AQUÍ VAN A TOMAR SUS DATOS DEL PROGRAMA ANTERIOR ##
df = pd.read_csv('imdb_movies.csv')
print(df.tail())

#############################################################################
print(50 * '=')
print('Gráficos de datos')
print(50 * '-')

# seleccionar calificacion y genero
y = df.iloc[0:100, 4].values
y = np.where(y == 'Horror', -1, 1)

# extraer datos de califficacion y genero
X = df.iloc[0:100, [3, 2]].values

# graficar datos   --- CLASIFICA POR COLORES
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Comun')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='No comun')

plt.xlabel('Genero')
plt.ylabel('Calificacion')
plt.legend(loc='upper left')

plt.show()

#############################################################################
print(50 * '=')
print('Entrenamiento del modelo Perceptrón')
print(50 * '-')

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Tiempo')
plt.ylabel('Número de clasificaciones erróneas')


plt.show()

#############################################################################
print(50 * '=')
print('Una función para graficar regiones de decisión')
print(50 * '-')


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # configurar generador de marcadores y mapa de colores
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # graficar la superficie de decisión
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # graficar muestras de clases
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Calificacion')
plt.ylabel('Genero')
plt.legend(loc='upper left')

plt.show()


#############################################################################
## IMPLEMENTACIÓN DE ADAPTIVE ##
print(50 * '=')
print('Implementación de un neurona lineal adaptativa en Python')
print(50 * '-')


class AdalineGD(object):
    """Clasificador ADALINE (ADAptive LInear NEuron).

    Parámetros
    ------------
    eta : float
        Tasa de aprendizaje (entre 0.0 y 1.0)
    n_iter : int
        Pases sobre el conjunto de entrenamiento.

    Atributos
    -----------
    w_ : 1d-array
        Pesos después del ajuste.
    cost_ : lista
        Valor de la función de costo de suma de cuadrados en cada época.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Ajustar datos de entrenamiento.

        Parámetros
        ----------
        X : {similar a array}, forma = [n_muestras, n_características]
            Vectores de entrenamiento, donde n_muestras es el número de muestras y
            n_características es el número de características.
        y : similar a array, forma = [n_muestras]
            Valores objetivo.

        Devuelve
        -------
        self : objeto

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calcular entrada neta"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Calcular activación lineal"""
        return self.net_input(X)

    def predict(self, X):
        """Devolver etiqueta de clase después del paso unitario"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Tiempo')
ax[0].set_ylabel('log(Costo de suma de cuadrados)')
ax[0].set_title('ADALINE - Tasa de aprendizaje 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Épocas')
ax[1].set_ylabel('Costo de suma de cuadrados')
ax[1].set_title('ADALINE - Tasa de aprendizaje 0.0001')

# plt.tight_layout()
# plt.savefig('./adaline_1.png', dpi=300)
plt.show()


print('Estandarización de características')
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('ADALINE - Descenso de gradiente')
plt.xlabel('Calificacion')
plt.ylabel('Genero')
plt.legend(loc='superior izquierda')

plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Tiempo')
plt.ylabel('Costo de suma de cuadrados')


plt.show()


#############################################################################
print(50 * '=')
print('Aprendizaje automático a gran escala y descenso de gradiente estocástico')
print(50 * '-')


class AdalineSGD(object):
    """Clasificador ADALINE (ADAptive LInear NEuron).

    Parámetros
    ------------
    eta : float
        Tasa de aprendizaje (entre 0.0 y 1.0)
    n_iter : int
        Pases sobre el conjunto de entrenamiento.
    shuffle : bool (predeterminado: True)
        Baraja los datos de entrenamiento en cada época si es True para evitar ciclos.
    random_state : int (predeterminado: None)
        Establecer un estado aleatorio para barajar e inicializar los pesos.

    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        """Ajustar datos de entrenamiento.

        Parámetros
        ----------
        X : {similar a array}, forma = [n_muestras, n_características]
            Vectores de entrenamiento, donde n_muestras es el número de muestras y
            n_características es el número de características.
        y : similar a array, forma = [n_muestras]
            Valores objetivo.

        Devuelve
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Ajustar datos de entrenamiento sin reinicializar los pesos"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Barajar datos de entrenamiento"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Inicializar pesos a cero"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Aplicar regla de aprendizaje ADALINE para actualizar los pesos"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calcular entrada neta"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Calcular activación lineal"""
        return self.net_input(X)

    def predict(self, X):
        """Devolver etiqueta de clase después del paso unitario"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('ADALINE - Descenso de gradiente estocástico')
plt.xlabel('Calificacion')
plt.ylabel('Genero')
plt.legend(loc='upper left')


plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Tiempo')
plt.ylabel('Costo promedio')


plt.show()

ada = ada.partial_fit(X_std[0, :], y[0])