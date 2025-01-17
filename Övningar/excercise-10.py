from ISLP import load_data
import seaborn as sns
import matplotlib.pyplot as plt



#Uppgift 10


#Ladda in dataset från ISPL
Boston = load_data("Boston")



#Uppgift B) 

#Räkna upp antal rader och kolumner
num_rows, num_columns = Boston.shape
print(f"Antal rader: {num_rows}, Antal kolumner: {num_columns}")


#Uppgift C)
# Välj ut 4 kolumner för scatterplots
selected_columns = Boston[["crim", "rm", "age", "medv"]]

# Skapa parvisa scatterplots med anpassade titlar och etiketter
sns.pairplot(selected_columns, diag_kind='kde', markers='o')
plt.suptitle("Parvisa Scatterplots av Utvalda Prediktorer i Boston Datasetet", y=1.02)
plt.xlabel("Prediktorer")
plt.ylabel("Prediktorer")
plt.show()


# Uppgift D) Analysera samband mellan prediktorer och brottslighet

correlation_with_crime = Boston.corr()['crim']
print("Korrelationskoefficienter med brottslighet:")
print(correlation_with_crime)
# Kommentar: Det finns flera prediktorer som har en signifikant korrelation med brottsligheten. 
# Till exempel har `rad` (radial highway access) en stark positiv korrelation (0.625) med brottsligheten, 
# medan `rm` (antal rum) har en negativ korrelation (-0.219), 
# vilket tyder på att fler rum är associerade med lägre brottslighet.

# Uppgift E) Identifiera förorter med höga brottsnivåer, skattesatser och elev-lärare förhållanden
high_crime = Boston[Boston['crim'] > Boston['crim'].quantile(0.75)]
high_tax = Boston[Boston['tax'] > Boston['tax'].quantile(0.75)]
high_ptratio = Boston[Boston['ptratio'] > Boston['ptratio'].quantile(0.75)]

print(f"Antal förorter med hög brottslighet: {len(high_crime)}")
print(f"Antal förorter med hög skattesats: {len(high_tax)}")
print(f"Antal förorter med hög elev-lärare förhållande: {len(high_ptratio)}")
# Kommentar: Det finns ett betydande antal förorter med hög brottslighet, medan endast ett fåtal har höga skattesatser.

# Uppgift F) Antal förorter som gränsar till Charles River
suburbs_boundary_charles_river = Boston[Boston['chas'] == 1]
num_suburbs_boundary_charles_river = len(suburbs_boundary_charles_river)
print(f"Antal förorter som gränsar till Charles River: {num_suburbs_boundary_charles_river}")

# Uppgift G) Median elev-lärare förhållande
median_ptratio = Boston['ptratio'].median()
print(f"Median elev-lärare förhållande: {median_ptratio}")

# Uppgift H) Förorten med lägsta medianvärde för ägda hem
lowest_medv_suburb = Boston.loc[Boston['medv'].idxmin()]
print(f"Förorten med lägsta medianvärde för ägda hem: {lowest_medv_suburb}")
# Kommentar: Denna förort har extremt hög brottslighet och låg medianvärde för ägda hem, vilket kan indikera socioekonomiska utmaningar.

# Uppgift I) Antal förorter med mer än sju och åtta rum
more_than_seven_rooms = len(Boston[Boston['rm'] > 7])
more_than_eight_rooms = len(Boston[Boston['rm'] > 8])
print(f"Antal förorter med mer än sju rum: {more_than_seven_rooms}")
print(f"Antal förorter med mer än åtta rum: {more_than_eight_rooms}")
# Kommentar: Det finns en betydande mängd förorter som har mer än sju rum, men endast ett fåtal som har mer än åtta rum.

