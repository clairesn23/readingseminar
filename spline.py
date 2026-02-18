# IMPORTS

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import BSpline
from numpy.linalg import solve
from datetime import datetime
from ipywidgets import interact, FloatLogSlider
import os
from scipy.integrate import trapezoid

# Dossier de sauvegarde des figures
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# LECTURE ET PRÉTRAITEMENT DES DONNÉES

def read_csv_series(filepath):
    data = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([row[0], float(row[1])])
    
    data = np.array(data, dtype=object)
    dates = [datetime.strptime(t, "%Y-%m-%d") for t in data[:,0]]
    values = data[:,1].astype(float)
    days = np.array([(d - dates[0]).days for d in dates])
    
    intervals = np.diff(days)
    fs = 1 / np.mean(intervals)
    
    return dates, days, values, fs

# Chargement des 8 séries
files = [
"data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_DSC_039_allROI.csv",
"data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vv_DSC_039_allROI.csv",
"data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_DSC_039_TropiScat.csv",
"data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vv_DSC_039_TropiScat.csv",
"data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_ASC_047_allROI.csv",
"data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vv_ASC_047_allROI.csv",
"data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vh_ASC_047_TropiScat.csv",
"data/Data/S1A_ASC-DSC_VV-VH_Paracou/s1a_vv_ASC_047_TropiScat.csv"
]

series = [read_csv_series(f) for f in files]

# Paramètre global : nombre de nœuds
n_knots_global = 60  # <-- tu peux changer ce nombre pour complexifier ou simplifier la spline

# CONSTRUCTION BASE B-SPLINE
def build_bspline_basis(t, degree=3, n_knots=n_knots_global):
    knots_internal = np.linspace(t.min(), t.max(), n_knots)
    knots = np.concatenate(([t.min()] * degree,
                            knots_internal,
                            [t.max()] * degree))
    
    K = len(knots) - degree - 1
    B = np.zeros((len(t), K))
    
    for j in range(K):
        coeff = np.zeros(K)
        coeff[j] = 1
        spline_basis = BSpline(knots, coeff, degree)
        B[:, j] = spline_basis(t)
    
    return B

# SPLINE PÉNALISÉE QUADRATIQUE
def penalized_spline_fit(B, Y, lam):
    beta = solve(B.T @ B + lam * np.eye(B.shape[1]), B.T @ Y)
    Y_hat = B @ beta
    return beta, Y_hat

def smoothing_matrix(B, lam):
    return B @ solve(B.T @ B + lam * np.eye(B.shape[1]), B.T)

def compute_edf(B, lam):
    return np.trace(smoothing_matrix(B, lam))

# GCV POUR OPTIMISATION AUTOMATIQUE
def gcv_score(B, Y, lam):
    n = len(Y)
    _, Y_hat = penalized_spline_fit(B, Y, lam)
    residual = np.sum((Y - Y_hat)**2)
    edf = compute_edf(B, lam)
    return (n * residual) / ((n - edf)**2)

def optimize_lambda_gcv(B, Y):
    lambdas = np.logspace(-4, 4, 100)
    scores = np.array([gcv_score(B, Y, lam) for lam in lambdas])
    lambda_opt = lambdas[np.argmin(scores)]
    return lambda_opt

# FONCTION PRINCIPALE MODÉLISATION RÉGIONALE
def regional_penalized_model(t, Y_reg, lam=None, auto_lambda=False):
    B = build_bspline_basis(t)
    
    if auto_lambda:
        lam = optimize_lambda_gcv(B, Y_reg)
        print("Lambda optimal (GCV) =", lam)
    elif lam is None:
        lam = 1.0
    
    beta, Y_hat = penalized_spline_fit(B, Y_reg, lam)
    EDF = compute_edf(B, lam)
    omega_c = lam**(-1/4)
    
    print("EDF =", EDF)
    print("Fréquence de coupure ω_c =", omega_c)
    
    return Y_hat, lam, EDF, omega_c

# Créer le dossier pour les splines
SPLINE_DIR = "splines"
os.makedirs(SPLINE_DIR, exist_ok=True)

# Fonction pour générer et sauvegarder les splines avec nombre de noeuds dans le nom
def plot_and_save_spline(dates, values, index, lam=1.0, n_knots=n_knots_global):
    t_numeric = np.array([(d - dates[0]).days for d in dates])
    B = build_bspline_basis(t_numeric, n_knots=n_knots)
    _, Y_hat = penalized_spline_fit(B, values, lam)
    
    plt.figure(figsize=(10,4))
    plt.plot(dates, values, 'o', alpha=0.5, label="Signal original")
    plt.plot(dates, Y_hat, linewidth=2, label=f"Spline pénalisée λ={lam}")
    plt.title(f"Spline pénalisée série {index} | n_knots={n_knots}")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    
    filename = os.path.join(SPLINE_DIR, f"spline{index}_knots{n_knots}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Spline série {index} sauvegardée → {filename}")

# Boucle sur toutes les séries
for i, (dates, t, values, fs) in enumerate(series, start=1):
    B = build_bspline_basis(t, n_knots=n_knots_global)
    lam_opt = optimize_lambda_gcv(B, values)
    plot_and_save_spline(dates, values, index=i, lam=lam_opt, n_knots=n_knots_global)
