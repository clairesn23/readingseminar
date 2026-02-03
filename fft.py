data1 = []

import csv
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt  
from ipywidgets import interact, FloatSlider
from datetime import datetime

with open("readingseminar/data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_DSC_039_allROI.csv", "r") as fichier1:
    lecteur = csv.reader(fichier1)
    next(lecteur)  # sauter l'entête
    for row in lecteur:
        time_str = row[0]
        value = float(row[1])
        data1.append([time_str, value])

data_a1 = np.array(data1, dtype=object)
times_str = data_a1[:, 0]
values = data_a1[:, 1].astype(float)

# Convertir les dates
dates = [datetime.strptime(t, '%Y-%m-%d') for t in times_str]
timestamps = np.array([(d - dates[0]).days for d in dates])

# Pas de temps moyen et fréquence d'échantillonnage
dt_days = np.mean(np.diff(timestamps))
fs_year = 365.25 / dt_days  # cycles par an

def plot_psd(fs=fs_year):
    frequencies, psd = signal.welch(values, fs=fs, nperseg=len(values)//2)
    plt.figure(figsize=(10,5))
    plt.semilogy(frequencies, psd)
    plt.xlabel("Fréquence (cycles/an)")
    plt.ylabel("Densité spectrale de puissance")
    plt.title(f"PSD Welch - Rétrodiffusion radar")
    plt.grid(True)
    plt.show()

interact(plot_psd, fs=FloatSlider(value=fs_year, min=0.01, max=50.0, step=0.1))

# FFT simplifiée
N = len(values)
fft_values = np.fft.fft(values)
fft_freq = np.fft.fftfreq(N, d=1/fs_year)

# Fréquences et amplitudes positives uniquement
mask = fft_freq > 0
freq_positive = fft_freq[mask]
amplitude_positive = np.abs(fft_values[mask])

plt.figure(figsize=(12,5))
plt.plot(freq_positive, amplitude_positive)
plt.xlabel("Fréquence (cycles/an)")
plt.ylabel("Amplitude")
plt.title("FFT - Rétrodiffusion radar")
plt.xlim(0, 10)
plt.grid(True)
plt.show()

# Série temporelle
plt.figure(figsize=(12,5))
plt.plot(dates, values)
plt.xlabel("Date")
plt.ylabel("Rétrodiffusion (dB)")
plt.title("Série temporelle - Rétrodiffusion radar VH")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()