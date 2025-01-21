import numpy as np
from scipy import stats

# Läs in data från CSV-filen
data = np.genfromtxt('c:/Programering/It högskolan/Statistiska-Metoder/Data/Small-diameter-flow.csv', delimiter=',')

# Definiera oberoende och beroende variabler
X1 = data[:, 0]  # Använd Kinematic som en oberoende variabel
X2 = data[:, 1]   # Använd Geometric som en annan oberoende variabel
Y = data[:, 2]         # Konvertera till numpy array

# Kontrollera om det finns NaN-värden i datan
print("NaN-värden i X1:", np.isnan(X1).sum())
print("NaN-värden i X2:", np.isnan(X2).sum())
print("NaN-värden i Y:", np.isnan(Y).sum())

# Ta bort rader med NaN-värden
valid_indices = ~np.isnan(X1) & ~np.isnan(X2) & ~np.isnan(Y)
X1 = X1[valid_indices]
X2 = X2[valid_indices]
Y = Y[valid_indices]

# Skapa designmatrisen X med en kolumn för interceptet
X = np.column_stack((np.ones(X1.shape[0]), X1, X2))

class LinearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    @property
    def d(self):
        return self.X.shape[1]

    @property
    def n(self):
        return self.X.shape[0]

    def variance(self):
        return np.var(self.Y)

    def standard_deviation(self):
        return np.std(self.Y)

    def significance(self):
        n = self.n
        p = self.d
        mse = np.sum((self.Y - np.mean(self.Y))**2) / (n - p)
        se = np.sqrt(mse * np.linalg.inv(self.X.T @ self.X))
        t_values = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y / se
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - p)) for t in t_values]
        return p_values

    def r_squared(self):
        Y_pred = self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        ss_res = np.sum((self.Y - Y_pred)**2)
        ss_tot = np.sum((self.Y - np.mean(self.Y))**2)
        return 1 - (ss_res / ss_tot)

    def fit(self):
        return np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y

    def predict(self, X_new):
        return X_new @ self.fit()

# Beräkna koefficienterna med OLS
b = np.linalg.inv(X.T @ X) @ X.T @ Y

# Gör förutsägelser med den beräknade modellen
Y_pred = X @ b

# Beräkna residualer
residuals = Y - Y_pred

# Beräkna R^2-värdet
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Y - np.mean(Y))**2)
r_squared = 1 - (ss_res / ss_tot)

# Statistisk analys
n = len(Y)
p = X.shape[1]

# Beräkna MSE och standardfel
mse = ss_res / (n - p)
se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))

# Beräkna t-värden och p-värden
t_values = b / se
p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - p)) for t in t_values]

# Utskrift av resultaten
print("OLS-analys:")
print("Koefficienter:", b)
print("Standardfel:", se)
print("T-värden:", t_values)
print("P-värden:", p_values)
print("R^2-värde:", r_squared)

# Använd LinearRegression-klassen
lr = LinearRegression(X, Y)
print("Signifikans:", lr.significance())
print("R^2-värde:", lr.r_squared())