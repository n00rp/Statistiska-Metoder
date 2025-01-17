import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Läsa in CSV-filen
df = pd.read_csv(r"C:\Programering\It högskolan\Statistiska-Metoder\Data\Auto.csv")  # Använd råsträng för sökvägen

# Kontrollera datatyperna för kolumnerna i DataFrame
print(df.dtypes)  # Skriver ut datatyperna för varje kolumn

# Visa de första raderna av DataFrame
print(df.head())  # Skriver ut de första 5 raderna av DataFrame

# Uppgift 1: Kontrollera om det finns tomma rader:

# Kontrollera om det finns några tomma rader
empty_rows = df[df.isnull().any(axis=1)]

# Skriv ut antalet tomma rader
print(f"Antal tomma rader: {empty_rows.shape[0]}")

# Uppgift 2: Identifiera kvantitativa och kvalitativa prediktorer:

# Identifiera kvantitativa och kvalitativa prediktorer
kvantitativa_prediktorer = df.select_dtypes(include=[np.number]).columns.tolist()
kvalitativa_prediktorer = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("Kvantitativa prediktorer:", kvantitativa_prediktorer)
print("Kvalitativa prediktorer:", kvalitativa_prediktorer)

# Identifiera kolumner med datatypen "object"
object_columns = df.select_dtypes(include=["object"]).columns.tolist()

# Försök att konvertera dessa kolumner till numeriska värden
for column in object_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')  # Konvertera och sätt icke-numeriska värden till NaN

# Kontrollera datatyperna efter konvertering
print(df.dtypes)  # Skriver ut datatyperna för varje kolumn efter konvertering

# Uppgift 3: Beräkna intervallet för varje kvantitativ prediktor:

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

# Uppgift 4: Beräkna medelvärde och standardavvikelse för alla kvantitativa prediktorer
medelvärden = df[kvantitativa_prediktorer].mean()
standardavvikelser = df[kvantitativa_prediktorer].std()

print("Medelvärden för kvantitativa prediktorer:")
print(medelvärden)

print("\nStandardavvikelser för kvantitativa prediktorer:")
print(standardavvikelser)

# Uppgift 5 : Ta nu bort observationerna från den 10:e till den 85:e:

# Ta bort observationerna från den 10:e till den 85:e
subset_df = df.drop(df.index[9:85])  # Index är 0-baserat, så 9:85 tar bort 10:e till 85:e observationen

# Beräkna medelvärde, standardavvikelse och intervall för varje prediktor
medelvärden_subset = subset_df.mean()
standardavvikelser_subset = subset_df.std()
intervall_subset = subset_df.max() - subset_df.min()  # Beräkna intervallet

# Skriv ut resultaten
print("Medelvärden för kvantitativa prediktorer i delmängden:")
print(medelvärden_subset)

print("\nStandardavvikelser för kvantitativa prediktorer i delmängden:")
print(standardavvikelser_subset)

print("\nIntervall för kvantitativa prediktorer i delmängden:")
print(intervall_subset)

# Uppgift 6: Undersöka prediktorer grafiskt

# Välj specifika kvantitativa prediktorer
selected_predictors = ["cylinders", "horsepower", "acceleration", "mpg"]

# Skapa scatterplots för att analysera relationen mellan mpg och andra kvantitativa prediktorer
for predictor in selected_predictors:
    if predictor != "mpg":  # Undvik att plotta mpg mot sig själv
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=df[predictor], y=df["mpg"])
        plt.title(f"Relation mellan {predictor} och mpg")
        plt.xlabel(predictor)
        plt.ylabel("mpg")
        
        # Beräkna medelvärde och standardavvikelse
        medelvärde = df.groupby(predictor)["mpg"].mean().reset_index()
        std_avvikelse = df.groupby(predictor)["mpg"].std().reset_index()
        
        # Rita linjer för standardavvikelse
        plt.plot(medelvärde[predictor], medelvärde["mpg"], color="red", label="Medelvärde")
        plt.plot(medelvärde[predictor], medelvärde["mpg"] + std_avvikelse["mpg"], color="blue", linestyle="--", label="Medelvärde + 1 STD")
        plt.plot(medelvärde[predictor], medelvärde["mpg"] - std_avvikelse["mpg"], color="green", linestyle="--", label="Medelvärde - 1 STD")
        plt.legend()
        plt.show()

# Uppgift 7: Undersök relationen av mpg mot andra kvantitativa prediktorer

# Välj specifika kvantitativa prediktorer
selected_predictors = ["cylinders", "horsepower", "acceleration", "mpg"]

# Skapa scatterplots för att analysera relationen mellan mpg och andra kvantitativa prediktorer
for predictor in selected_predictors:
    if predictor != "mpg":  # Undvik att plotta mpg mot sig själv
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=df[predictor], y=df["mpg"])
        plt.title(f"Relation mellan {predictor} och mpg")
        plt.xlabel(predictor)
        plt.ylabel("mpg")
        
        # Beräkna medelvärde och standardavvikelse
        medelvärde = df.groupby(predictor)["mpg"].mean().reset_index()
        std_avvikelse = df.groupby(predictor)["mpg"].std().reset_index()
        
        # Rita linjer för standardavvikelse
        plt.plot(medelvärde[predictor], medelvärde["mpg"], color="red", label="Medelvärde")
        plt.plot(medelvärde[predictor], medelvärde["mpg"] + std_avvikelse["mpg"], color="blue", linestyle="--", label="Medelvärde + 1 STD")
        plt.plot(medelvärde[predictor], medelvärde["mpg"] - std_avvikelse["mpg"], color="green", linestyle="--", label="Medelvärde - 1 STD")
        plt.legend()
        plt.show()