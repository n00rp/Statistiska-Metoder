import numpy as np
from scipy import stats

class LinearRegression:

    def __init__(self, X, Y):
        if X.ndim != 2 or Y.ndim != 1:
            raise ValueError("X måste vara en 2D-array och Y måste vara en 1D-array.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Antalet rader i X måste matcha antalet element i Y.")
        self.X = X
        self.Y = Y

    @property
    def d(self):
        """Returnerar antalet funktioner/parametrar."""
        return self.X.shape[1]

    @property
    def n(self):
        """Returnerar storleken på urvalet."""
        return self.X.shape[0]

    def variance(self):
        """Beräknar variansen av den beroende variabeln."""
        var = np.var(self.Y)
        print("Varians:", var)
        return var

    def standard_deviation(self):
        """Beräknar standardavvikelsen av den beroende variabeln."""
        std_dev = np.std(self.Y)
        print("Standardavvikelse:", std_dev)
        return std_dev

    def significance(self):
        """Beräknar signifikansen av regressionen och returnerar p-värden."""
        n = self.n
        p = self.d
        mse = np.sum((self.Y - np.mean(self.Y))**2) / (n - p)
        se = np.sqrt(mse * np.linalg.inv(self.X.T @ self.X))
        t_values = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y / se
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - p)) for t in t_values]
        print("P-värden:", p_values)
        return p_values

    def r_squared(self):
        """Beräknar R²-värdet för regressionen."""
        Y_pred = self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        ss_res = np.sum((self.Y - Y_pred)**2)
        ss_tot = np.sum((self.Y - np.mean(self.Y))**2)
        r2 = 1 - (ss_res / ss_tot)
        print("R²-värde:", r2)
        return r2

    def fit(self):
        """Beräknar koefficienterna för regressionen."""
        coefficients = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        print("Koefficienter:", coefficients)
        return coefficients

# Läs in data från CSV-filen
data = np.genfromtxt('c:/Programering/It högskolan/Statistiska-Metoder/Data/Small-diameter-flow.csv', delimiter=',', invalid_raise=False)

# Ta bort rader som innehåller NaN-värden i någon kolumn
valid_rows = ~np.isnan(data).any(axis=1)
data = data[valid_rows]

# Definiera oberoende och beroende variabler
X1 = data[:, 0]  # Använd Kinematic som en oberoende variabel
X2 = data[:, 1]   # Använd Geometric som en annan oberoende variabel
Y = data[:, 2]         # Konvertera till numpy array

# Skapa designmatrisen X med en kolumn för interceptet
X = np.column_stack((np.ones(X1.shape[0]), X1, X2))

# Använd LinearRegression-klassen
lr = LinearRegression(X, Y)

# Utskrifter av resultaten
lr.fit()
lr.variance()
lr.standard_deviation()
lr.significance()
lr.r_squared()
