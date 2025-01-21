import numpy as np
from scipy import stats

# Exempeldata
# Definiera dina oberoende variabler (X1, X2, ...) och beroende variabel (Y)
X1 = np.array([1, 2, 3, 4, 5])  # Ersätt med faktiska värden för X1
X2 = np.array([2, 3, 5, 7, 11])  # Ersätt med faktiska värden för X2 (inte perfekt korrelerade)
Y = np.array([3, 5, 7, 9, 11])    # Ersätt med faktiska värden för Y

# Skapa designmatrisen X med en kolumn för interceptet
# Interceptet (β0) läggs till som en kolumn av ettor
X = np.column_stack((np.ones(X1.shape[0]), X1, X2))

# Beräkna koefficienterna med OLS
# Formeln b = (X^T X)^-1 X^T Y används för att beräkna koefficienterna
b = np.linalg.inv(X.T @ X) @ X.T @ Y

# Gör förutsägelser med den beräknade modellen
Y_pred = X @ b

# Beräkna residualer (skillnaden mellan observerade och förutsagda värden)
residuals = Y - Y_pred

# Beräkna R^2-värdet som mäter hur väl modellen förklarar variationen i Y
ss_res = np.sum(residuals**2)  # Residual sum of squares
ss_tot = np.sum((Y - np.mean(Y))**2)  # Total sum of squares
r_squared = 1 - (ss_res / ss_tot)  # R^2-värde

# Statistisk analys
n = len(Y)  # Antal observationer
p = X.shape[1]  # Antal koefficienter (inklusive intercept)

# Beräkna medelkvadratfelet (MSE) och standardfel för koefficienterna
mse = ss_res / (n - p)  # Mean squared error
se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))  # Standard errors

# Beräkna t-värden och p-värden för koefficienterna
t_values = b / se  # T-värden
p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - p)) for t in t_values]  # P-värden
