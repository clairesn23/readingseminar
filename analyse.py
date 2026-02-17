data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []


import csv
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt  
from ipywidgets import interact, FloatSlider
from datetime import datetime

import os

# Dossier de sauvegarde des figures
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Lecture des données 

# Fichier 1 : 39 DSC VH AllROI
with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_DSC_039_allROI.csv", "r") as fichier1:
    lecteur = csv.reader(fichier1)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data1.append([time_str, value])

# Fichier 2 : 39 DSC VV AllROI
with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vv_DSC_039_allROI.csv", "r") as fichier2:
    lecteur = csv.reader(fichier2)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data2.append([time_str, value])

# Fichier 3 : 39 DSC VH TropiScat
with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_DSC_039_TropiScat.csv", "r") as fichier3:
    lecteur = csv.reader(fichier3)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data3.append([time_str, value])


# Fichier 4 : 39 DSC VV TropiScat
with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vv_DSC_039_TropiScat.csv", "r") as fichier4:
    lecteur = csv.reader(fichier4)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data4.append([time_str, value])

# Fichier 5 : 47 ASC VH AllROI
with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_ASC_047_allROI.csv", "r") as fichier5:
    lecteur = csv.reader(fichier5)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data5.append([time_str, value])

# Fichier 6 : 47 ASC VV AllROI
with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vv_ASC_047_allROI.csv", "r") as fichier6:
    lecteur = csv.reader(fichier6)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data6.append([time_str, value])

# Fichier 7 : 47 ASC VH TropiScat
with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_ASC_047_TropiScat.csv", "r") as fichier7:
    lecteur = csv.reader(fichier7)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data7.append([time_str, value])

# Fichier 8 : 47 ASC VV TropiScat
with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vv_ASC_047_TropiScat.csv", "r") as fichier8:
    lecteur = csv.reader(fichier8)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data8.append([time_str, value])


# Première mise en forme 

# Fichier 1 
data_a1 = np.array(data1, dtype=object)
T1 = data_a1[:, 0]            # colonne temps
values1 = data_a1[:, 1].astype(float)  # colonne valeurs

dates1 = [datetime.strptime(t, "%Y-%m-%d") for t in T1]
days1 = np.array([(d - dates1[0]).days for d in dates1])

# Calcul de la fréquence d'échantillonnage
intervals1 = np.diff(days1)
fs1 = 1 / np.mean(intervals1)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs1:.4f} échantillons/jour")

# Fichier 2
data_a2 = np.array(data2, dtype=object)
T2 = data_a2[:, 0]            # colonne temps
values2 = data_a2[:, 1].astype(float)  # colonne valeurs

dates2 = [datetime.strptime(t, "%Y-%m-%d") for t in T2]
days2 = np.array([(d - dates2[0]).days for d in dates2])

# Calcul de la fréquence d'échantillonnage
intervals2 = np.diff(days2)
fs2 = 1 / np.mean(intervals2)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs2:.4f} échantillons/jour")

# Fichier 3
data_a3 = np.array(data3, dtype=object)
T3 = data_a3[:, 0]            # colonne temps
values3 = data_a3[:, 1].astype(float)  # colonne valeurs

dates3 = [datetime.strptime(t, "%Y-%m-%d") for t in T3]
days3 = np.array([(d - dates3[0]).days for d in dates3])

# Calcul de la fréquence d'échantillonnage
intervals3 = np.diff(days3)
fs3 = 1 / np.mean(intervals3)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs3:.4f} échantillons/jour")

# Fichier 4
data_a4 = np.array(data4, dtype=object)
T4 = data_a4[:, 0]            # colonne temps
values4 = data_a4[:, 1].astype(float)  # colonne valeurs

dates4 = [datetime.strptime(t, "%Y-%m-%d") for t in T4]
days4 = np.array([(d - dates4[0]).days for d in dates4])

# Calcul de la fréquence d'échantillonnage
intervals4 = np.diff(days4)
fs4 = 1 / np.mean(intervals4)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs4:.4f} échantillons/jour")

# Fichier 5
data_a5 = np.array(data5, dtype=object)
T5 = data_a5[:, 0]            # colonne temps
values5 = data_a5[:, 1].astype(float)  # colonne valeurs

dates5 = [datetime.strptime(t, "%Y-%m-%d") for t in T5]
days5 = np.array([(d - dates5[0]).days for d in dates5])

# Calcul de la fréquence d'échantillonnage
intervals5 = np.diff(days5)
fs5 = 1 / np.mean(intervals5)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs5:.4f} échantillons/jour")

# Fichier 6
data_a6 = np.array(data6, dtype=object)
T6 = data_a6[:, 0]            # colonne temps
values6 = data_a6[:, 1].astype(float)  # colonne valeurs

dates6 = [datetime.strptime(t, "%Y-%m-%d") for t in T6]
days6 = np.array([(d - dates6[0]).days for d in dates6])

# Calcul de la fréquence d'échantillonnage
intervals6 = np.diff(days6)
fs6 = 1 / np.mean(intervals6)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs6:.4f} échantillons/jour")

# Fichier 7
data_a7 = np.array(data7, dtype=object)
T7 = data_a7[:, 0]            # colonne temps
values7 = data_a7[:, 1].astype(float)  # colonne valeurs

dates7 = [datetime.strptime(t, "%Y-%m-%d") for t in T7]
days7 = np.array([(d - dates7[0]).days for d in dates7])

# Calcul de la fréquence d'échantillonnage
intervals7 = np.diff(days7)
fs7 = 1 / np.mean(intervals7)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs7:.4f} échantillons/jour")

# Fichier 8
data_a8 = np.array(data8, dtype=object)
T8 = data_a8[:, 0]            # colonne temps
values8 = data_a8[:, 1].astype(float)  # colonne valeurs

dates8 = [datetime.strptime(t, "%Y-%m-%d") for t in T8]
days8 = np.array([(d - dates8[0]).days for d in dates8])

# Calcul de la fréquence d'échantillonnage
intervals8 = np.diff(days8)
fs8 = 1 / np.mean(intervals8)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs8:.4f} échantillons/jour")



# La série temporelle
dates=dates8
values=values8
fs=fs8
plt.figure(figsize=(10,4))
plt.plot(dates, values, marker='o')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Série temporelle")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "serie_temporelle8.png"), dpi=300)
plt.close()  
#window_lengths=[730, 900, 1000]
##wl_samples = int(np.floor(wl / 12))  # window en nombre de samples
            # nperseg = min(wl_samples, len(y))   # nb échantillons dans la fenêtre (taille de la fenêtre ou nombre d'échantillons restants si on est proche de la fin)
            # noverlap = int(nperseg * ol)    # overlap en nombre de samples

            # freqs, psd = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap)    # fonction welch de scipy.signal


wl_samples = int(np.floor(730 / 12))  # window de 730 jours convertie en nombre de samples
nperseg = min(wl_samples, len(values))   # nb échantillons dans la fenêtre (taille de la fenêtre ou nombre d'échantillons restants si on est proche de la fin)
noverlap = int(nperseg * 0.5)    # overlap de 50 

frequencies, psd = signal.welch(values, fs=fs, nperseg=nperseg)

plt.figure(figsize=(10,4))
plt.semilogy(frequencies, psd)
plt.xlabel("Fréquence (cycles/jour)")
plt.ylabel("Densité spectrale de puissance")
plt.title("PSD Welch")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "psd_welch8.png"), dpi=300)
plt.close()
plt.show()
