import numpy as np
from scipy import stats

class LinjärRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def n(self):
        # Antal prover
        return self.y.shape[0]

    @property
    def d(self):
        # Antal funktioner
        return len(self.b)-1

    @property
    def b(self):
        # Beräkna b med pinv för bättre stabilitet
        return np.linalg.pinv(self.x.T @ self.x) @ self.x.T @ self.y #Transponerad matris T

    def SSE(self):
        # Sum of Squared Errors (Residual sum of squares)
        return np.sum(np.square(self.y - self.prediktera(self.x)))
    
    def SSR(self):
        # Sum of Squares due to Regression
        return np.sum((self.prediktera(self.x) - np.mean(self.y))**2)
    
    def SST(self):
        # Total Sum of Squares
        return np.sum((self.y - np.mean(self.y))**2)

    def varians(self):
        # Beräkna variansen av residualerna
        return self.SSE() / (self.n - self.d - 1)

    def standardavvikelse(self):
        # Beräkna standardavvikelsen av residualerna
        return np.sqrt(self.varians())

    def signifikans(self):
        # Rapportera signifikansen av regressionen med F-test
        S = np.sqrt(self.varians())
        f_statistik = (self.SSR() / self.d) / S
        p_värde = stats.f.sf(f_statistik, self.d, self.n - self.d - 1)
        return p_värde

    def r_kvadrat(self):
        # Beräkna R²-värde
        return self.SSR() / self.SST()

    def prediktera(self, x):
        # Prediktera med hjälp av den linjära modellen
        return x @ self.b
