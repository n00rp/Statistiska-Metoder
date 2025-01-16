import pandas as pd
import numpy as np


#Läsa in CSV-filen
df = pd.read_csv("Data/Auto.csv") 

#Uppgift 1: Kontrollera om det finns tomma rader:

# Kontrollera om det finns några tomma rader
empty_rows = df[df.isnull().any(axis=1)]

# Skriv ut antalet tomma rader
print(f"Antal tomma rader: {empty_rows.shape[0]}")

#Uppgift 2: Identifiera kvantitativa och kvalitativa prediktorer:

# Identifiera kvantitativa och kvalitativa prediktorer
kvantitativa_prediktorer = df.select_dtypes(include=[np.number]).columns.tolist()
kvalitativa_prediktorer = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("Kvantitativa prediktorer:", kvantitativa_prediktorer)
print("Kvalitativa prediktorer:", kvalitativa_prediktorer)


#Uppgift 3: Beräkna intervallet för varje kvantitativ prediktor:

# Beräkna intervallet för varje kvantitativ prediktor
for prediktor in kvantitativa_prediktorer:
    min_value = np.min(df[prediktor])
    max_value = np.max(df[prediktor])
    print(f"Intervallet för {prediktor}: min = {min_value}, max = {max_value}")

# Beräkna medelvärde och standardavvikelse för varje kvantitativ prediktor
for prediktor in kvantitativa_prediktorer:
    mean_value = np.mean(df[prediktor])
    std_value = np.std(df[prediktor])
    print(f"{prediktor}: Medelvärde = {mean_value}, Standardavvikelse = {std_value}")

# Beräkna medelvärde och standardavvikelse för varje kvantitativ prediktor
for prediktor in kvantitativa_prediktorer:
    mean_value = np.mean(df[prediktor])
    std_value = np.std(df[prediktor])
    print(f"{prediktor}: Medelvärde = {mean_value}, Standardavvikelse = {std_value}")
