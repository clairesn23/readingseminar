data1 = []


import csv
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt  
from ipywidgets import interact, FloatSlider

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

def plot_psd(fs=3000):
    frequencies, psd = signal.welch(values, fs=fs, nperseg=len(values)//2)
    plt.figure(figsize=(10,5))
    plt.semilogy(frequencies, psd)
    plt.xlabel("Fréquence (cycles/unit)")
    plt.ylabel("Densité spectrale de puissance")
    plt.title(f"PSD Welch avec fs = {fs}")
    plt.grid(True)
    plt.show()

interact(plot_psd, fs=FloatSlider(value=1.0, min=0.01, max=10.0, step=0.01))

plt.plot(times, values)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Série temporelle")

