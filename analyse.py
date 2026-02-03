data1 = []


import csv
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt  
from ipywidgets import interact, FloatSlider
from datetime import datetime

with open("data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_DSC_039_allROI.csv", "r") as fichier1:
    lecteur = csv.reader(fichier1)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data1.append([time_str, value])

data_a1 = np.array(data1, dtype=object)
times = data_a1[:, 0]            # colonne temps
values = data_a1[:, 1].astype(float)  # colonne valeurs

dates = [datetime.strptime(t, "%Y-%m-%d") for t in times]
days = np.array([(d - dates[0]).days for d in dates])

# Calcul de la fréquence d'échantillonnage
intervals = np.diff(days)
fs = 1 / np.mean(intervals)  # échantillons par jour
print(f"Fréquence d'échantillonnage fs = {fs:.4f} échantillons/jour")

# La série temporelle
plt.figure(figsize=(10,4))
plt.plot(dates, values, marker='o')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Série temporelle")
plt.grid(True)

frequencies, psd = signal.welch(values, fs=fs, nperseg=len(values)//2)

plt.figure(figsize=(10,4))
plt.semilogy(frequencies, psd)
plt.xlabel("Fréquence (cycles/jour)")
plt.ylabel("Densité spectrale de puissance")
plt.title("PSD Welch")
plt.grid(True)
plt.show()