# ---- LIB.PY ----

# --- Imports ---
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import find_peaks
from scipy import stats

plt.rcParams.update({
    'font.size': 14,          # Taille de base
    'axes.titlesize': 20,     # Taille du titre
    'axes.labelsize': 16,     # Taille des labels (x et y)
    'xtick.labelsize': 14,    # Taille des graduations X
    'ytick.labelsize': 14,    # Taille des graduations Y
    'legend.fontsize': 14,    # Taille de la légende
    'figure.titlesize': 22    # Titre de la figure
})

# --- Files preparation ---
# on supprime les données infinies et les NaN, on convertit le temps en datetime, on trie par date

def prepa_files(path,tolerance=100):
    df = pd.read_csv(path)
    df = df.replace(-np.inf, np.nan).dropna()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    mean = df['value'].mean()
    std = df['value'].std()
    low, high = mean - tolerance*std, mean + tolerance*std
    df_clean = df[(df['value'] >= low) & (df['value'] <= high)]
    return df_clean

# --- STL Decomposition ---
# on resample les données toutes les 12j pour avoir une périodicité annuelle, puis on applique la décomposition STL

def ST(df,period=30.4):             # prise de valeur tout les 12j, 12*30.4 ~= 365j
    serie_resampled = df['value'].resample('12D').interpolate()
    stl = STL(serie_resampled, period, robust=True)
    res = stl.fit()
    return stl,res

# --- Affichage ---
# on affiche les séries temporelles, les tendances, les saisonnalités et les résidus avec STL (en utilisant la fonction précédente)

def affichage(df_asc,stl_asc,res_asc,df_dsc,stl_dsc,res_dsc):
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(df_asc.index, df_asc['value'], label='ASC', color='steelblue')
    axs[0].plot(df_dsc.index, df_dsc['value'], label='DSC', color='orange')
    axs[0].set_title('Time series All')

    axs[1].plot(res_asc.trend, label='ASC', color='steelblue')
    axs[1].plot(res_dsc.trend, label='DSC', color='orange')
    axs[1].set_ylabel('Trend (dB)')
    axs[1].legend()
    axs[1].set_title('Décomposition STL')

    axs[2].plot(res_asc.seasonal, label='ASC', color='steelblue')
    axs[2].plot(res_dsc.seasonal, label='DSC', color='orange')
    axs[2].set_ylabel('Seasonal')

    axs[3].plot(res_asc.resid, label='ASC', color='steelblue')
    axs[3].plot(res_dsc.resid, label='DSC', color='orange')
    axs[3].set_ylabel('Resid')
    axs[3].set_xlabel('Date')

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    return 0

# --- PSD Analysis ---
# on calcule la PSD avec la méthode de Welch

# fonction pour obtenir les n périodes dominantes (n=2 par défaut puisqu'on observe 2 pics marqués)
def get_top_periods(freqs, psd, n_peaks=2):
    peaks, _ = find_peaks(psd)
    top_peaks = peaks[np.argsort(psd[peaks])[-n_peaks:]]  # Top 2
    top_freqs = freqs[top_peaks]
    top_periods = 1 / top_freqs
    return top_periods

# calcul de la DSP avec Welch

import os
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

def psd_welch_annee(df, nom, window_lengths=[730, 900, 1000], overlap_ratios=[0.30, 0.40, 0.50], n_peaks=2, outdir="figures_welch", prefix="PSD", show=True):
    y = df['value'].values
    fs = 1/12  # Sentinel-1, un point/12 jours
    os.makedirs(outdir, exist_ok=True)

    # Boucle sur l'overlap et sur la wl
    for ol in overlap_ratios:
        plt.figure(figsize=(10, 6))
        print(f"\n===== Overlap Analysis = {ol*100:.0f}% =====")

        for wl in window_lengths:
            wl_samples = int(np.floor(wl / 12))  # window en nombre de samples
            nperseg = min(wl_samples, len(y))   # nb échantillons dans la fenêtre (taille de la fenêtre ou nombre d'échantillons restants si on est proche de la fin)
            noverlap = int(nperseg * ol)    # overlap en nombre de samples

            freqs, psd = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap)    # fonction welch de scipy.signal
            freqs_annee = freqs * 365
            mask = (freqs_annee > 0.1) & (freqs_annee < 10) # filtrage des fréquences utiles

            plt.semilogx(freqs_annee[mask], psd[mask], label=f'Window: {wl} days')

            top_periods = get_top_periods(freqs[mask], psd[mask], n_peaks=n_peaks)
            top_periods_years = [p/365 for p in top_periods]
            print(f" WL = {wl} days — Dominant periods ≈ " f"{[round(period,3) for period in top_periods_years]} years" f" or, {top_periods}  days")

        # Affichage à chaque overlap
        plt.title(f"DSP Welch, Overlap {ol*100:.0f}%")
        plt.xlabel("Frequency (cycles/year) [log]")
        plt.ylabel("Power Spectral Density")
        plt.tick_params(axis='both', which='major')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = f"{prefix}_overlap_{int(ol*100)}pct_{nom}.png"
        fpath = os.path.join(outdir, fname)

        # Sauvegarde
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"✔ Figure sauvegardée : {fpath}")

        if show:
            plt.show()
        else:
            plt.close()
        plt.show()

# --- Seasonal Functions ---
# fonctions à fit avec curve_fit. On peut choisir entre 1, 2 ou 4 fréquences

#def seasonal_1freq(t, a,b,A,B,T):                                     PAS A JOUR
#    return a + b*t + A*np.sin(2*np.pi*t/T) + B*np.cos(2*np.pi*t/T)

def seasonal_2freq(t, a, b, A1, B1, w1, A2, B2, w2):
    return (a + b*t
            + A1*np.sin(t*w1) + B1*np.cos(t*w1)
            + A2*np.sin(t*w2) + B2*np.cos(t*w2))

def seasonal_4freq(t, a, b, A1, B1, w1, A2, B2, w2, A3, B3, w3, A4, B4, w4):
    return (a + b*t
            + A1*np.sin(t*w1) + B1*np.cos(t*w1)
            + A2*np.sin(t*w2) + B2*np.cos(t*w2)
            + A3*np.sin(t*w3) + B3*np.cos(t*w3)
            + A4*np.sin(t*w4) + B4*np.cos(t*w4))

def seasonal_2freq_quad(t, a, b, c, A1, B1, w1, A2, B2, w2):
    return (a + b*t + c*t**2
            + A1*np.sin(t*w1) + B1*np.cos(t*w1)
            + A2*np.sin(t*w2) + B2*np.cos(t*w2))

def seasonal_4freq_quad(t, a, b, c, A1, B1, w1, A2, B2, w2, A3, B3, w3, A4, B4, w4):
    return (a + b*t + c*t**2
            + A1*np.sin(t*w1) + B1*np.cos(t*w1)
            + A2*np.sin(t*w2) + B2*np.cos(t*w2)
            + A3*np.sin(t*w3) + B3*np.cos(t*w3)
            + A4*np.sin(t*w4) + B4*np.cos(t*w4))

# modèles avec les fréquences fixées (pour les ROI)

def make_model_fixed_T_2freq(w1, w2):
    def model(t, a, b, A1, B1, A2, B2):
        return (a + b*t
                + A1*np.sin(t*w1) + B1*np.cos(t*w1)
                + A2*np.sin(t*w2) + B2*np.cos(t*w2))
    return model

def make_model_fixed_T_4freq(w1, w2, w3, w4):
    def model(t, a, b, A1, B1, A2, B2, A3, B3, A4, B4):
        return (a + b*t
                + A1*np.sin(t*w1) + B1*np.cos(t*w1)
                + A2*np.sin(t*w2) + B2*np.cos(t*w2)
                + A3*np.sin(t*w3) + B3*np.cos(t*w3)
                + A4*np.sin(t*w4) + B4*np.cos(t*w4))
    return model

def make_model_fixed_T_2freq_quad(w1, w2):
    def model(t, a, b, c, A1, B1, A2, B2):
        return (a + b*t + c*t**2 +
                + A1*np.sin(t*w1) + B1*np.cos(t*w1)
                + A2*np.sin(t*w2) + B2*np.cos(t*w2))
    return model

def make_model_fixed_T_4freq_quad(w1, w2, w3, w4):
    def model(t, a, b, c, A1, B1, A2, B2, A3, B3, A4, B4):
        return (a + b*t + c*t**2 +
                + A1*np.sin(t*w1) + B1*np.cos(t*w1)
                + A2*np.sin(t*w2) + B2*np.cos(t*w2)
                + A3*np.sin(t*w3) + B3*np.cos(t*w3)
                + A4*np.sin(t*w4) + B4*np.cos(t*w4))
    return model

# --- Curve Fitting ---
# fit de la série temporelle avec curve_fit, on peut choisir entre 2 ou 4 fréquences (modèles sans fréquences fixées)

def fit_curve_n_freq(df1,df_gx_daily, w_init,nom,n_freq=2,quad=False,save_fig=True, outdir="figures_model", prefix="SAR_fit", dpi=300):
    os.makedirs(outdir, exist_ok=True)
    t = (df1.index - df1.index[0]).days.values
    print("t days:", t[-10:-1])
    y = df1['value'].values

    ymin = np.min(y)-0.1
    ymax = np.max(y)+0.1

    if n_freq==2:
        w1_init, w2_init = w_init[0], w_init[1]
        if quad == False :
            p0 = [np.mean(y), 0, 0.5, 0.5, w1_init, 0.3, 0.3, w2_init]
            model = seasonal_2freq
        else :
            p0 = [np.mean(y), 0, 0, 0.5, 0.5, w1_init, 0.3, 0.3, w2_init]
            model = seasonal_2freq_quad
    elif n_freq==4:
        if quad == False :
            w1_init, w2_init, w3_init, w4_init = w_init[0], w_init[1], w_init[2], w_init[3]
            p0 = [np.mean(y), 0, 0.5, 0.5, w1_init, 0.3, 0.3, w2_init, 0.2, 0.2, w3_init, 0.1, 0.1, w4_init]
            model = seasonal_4freq
        else :
            w1_init, w2_init, w3_init, w4_init = w_init[0], w_init[1], w_init[2], w_init[3]
            p0 = [np.mean(y), 0, 0, 0.5, 0.5, w1_init, 0.3, 0.3, w2_init, 0.2, 0.2, w3_init, 0.1, 0.1, w4_init]
            model = seasonal_4freq_quad
    params, covariance = curve_fit(model, t, y, p0=p0)
    sigma = np.sqrt(np.diag(covariance))        # incertitude des paramètres

    if n_freq==2:
        if quad == False :
            y_fit = seasonal_2freq(t, *params)
        else :
            y_fit = seasonal_2freq_quad(t, *params)
    elif n_freq==4:
        if quad == False :
            y_fit = seasonal_4freq(t, *params)
        else :
            y_fit = seasonal_4freq_quad(t, *params)

    r, p_value = stats.pearsonr(y, y_fit)
    #r = np.corrcoef(y, y_fit)[0,1]
    print(f"coefficient de corrélation linéaire de Pearson entre y et y_fit = {r:.3f}")
    print(f"p-value associée = {p_value}")

    # Calcul des résidus
    residus = y - y_fit

    # Visualisation
    plt.figure(figsize=(12, 6))
    plt.plot(df1.index, residus)
    plt.title("Analyse des résidus")
    plt.ylabel("Erreur (Original - Reconstruit)")
    plt.axhline(0, color='black', linestyle='--')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylim(ymin, ymax)
    #--- FOND VPD PAR COLORMAP ---

    dates = df_gx_daily.index
    vpd = df_gx_daily["amplitude_j_vpd"].values

    vpd_matrix = vpd.reshape(1, -1)

    extent = [dates.min(), dates.max(),ymin, ymax]

    vmin = 0
    vmax = 25

    ax.imshow(vpd_matrix,aspect='auto',cmap='Greys',extent=extent,alpha=0.35,origin="lower",vmin=vmin,vmax=vmax)
    cbar = plt.colorbar(ax.imshow(vpd_matrix, aspect='auto', cmap='Greys',extent=extent, alpha=0.35, origin="lower",vmin=vmin, vmax=vmax),
                        ax=ax,orientation="horizontal",pad=0.12)
    cbar.set_label("daily Vapor Pressure Deficit amplitude (55 m)")

    ax.plot(df1.index, df1["value"], color="black",alpha=0.4, linewidth=1.5, label="Time Series")
    ax.set_ylabel("(dB)")
    ax.grid(True)

    ax.plot(df1.index, y, "o", alpha=0.4, color="black", label="Obs S1")
    ax.plot(df1.index, y_fit, "-", color="red", linewidth=2, label=f"Model {n_freq} frequencies")

    ax_rain = ax.twinx()

    ax_rain.bar(df_gx_daily.index, df_gx_daily["pluie"],
            width=1, alpha=0.9, color="royalblue", label="Daily rainfall (mm)")
    ax_rain.set_ylabel("Rainfall (mm/day)", color="royalblue")

    ax_rain.tick_params(axis='y', labelcolor="royalblue")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_rain.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Time Series + Fit + Rainfall + VPD")
    plt.tight_layout()
    if save_fig:
        start = df1.index.min().strftime("%Y%m%d")
        end = df1.index.max().strftime("%Y%m%d")

        quad_str = "quad" if quad else "lin"
        fname = (
            f"{prefix}_"
            f"{n_freq}freq_"
            f"{quad_str}_"
            f"{nom}_"
            f"{start}_{end}.png"
        )

        fpath = os.path.join(outdir, fname)

        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"✔ Figure sauvegardée : {fpath}")
    plt.show()

    k = 0               # décalage des paramètres si modèle quadratique
    if quad == True:
        k = 1
    if n_freq==2:
        w = [params[4+k], params[7+k]]
        all_parameters = params
        params = list(params[0:4+k])+list(params[5+k:7+k])
    elif n_freq==4:
        w = [params[4+k], params[7+k], params[10+k], params[13+k]]
        all_parameters = params
        params = list(params[0:4+k])+list(params[5+k:7+k])+list(params[8+k:10+k])+list(params[11+k:13+k])
    return w,sigma,params,all_parameters

# pareil avec modèle à fréquences fixées (pour les ROI)
def fit_curve_n_freq_fixed(df,df_gx_daily, w ,init_params,nom,quad=False,save_fig=True, outdir="figures_model", prefix="SAR_fit_ROI", dpi=300):
    os.makedirs(outdir, exist_ok=True)
    n_freq = len(w)
    t = (df.index - df.index[0]).days.values
    y = df['value'].values

    ymin = np.min(y)-0.1
    ymax = np.max(y)+0.1
    
    p0 = init_params

    if n_freq==2:
        if quad == False :
            model = make_model_fixed_T_2freq(w[0], w[1])
        else :
            model = make_model_fixed_T_2freq_quad(w[0], w[1])
        params, covariance= curve_fit(model, t, y, p0=p0)
    if n_freq==4:
        if quad == False :
            model = make_model_fixed_T_4freq(w[0], w[1], w[2], w[3])
        else :
            model = make_model_fixed_T_4freq_quad(w[0], w[1], w[2], w[3])
        params, covariance = curve_fit(model, t, y, p0=p0)
    sigma = np.sqrt(np.diag(covariance))        # incertitude des paramètres
    
    y_fit = model(t, *params)

    r, p_value = stats.pearsonr(y, y_fit)
    #r = np.corrcoef(y, y_fit)[0,1]
    print(f"coefficient de corrélation linéaire de Pearson entre y et y_fit = {r:.3f}")
    print(f"p-value associée = {p_value}")

    # Calcul des résidus
    residus = y - y_fit

    # Visualisation
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, residus)
    plt.title("Analyse des résidus")
    plt.ylabel("Erreur (Original - Reconstruit)")
    plt.axhline(0, color='black', linestyle='--')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylim(ymin, ymax)
    #--- FOND VPD PAR COLORMAP ---

    dates = df_gx_daily.index
    vpd = df_gx_daily["amplitude_j_vpd"].values

    vmin = 0
    vmax = 25

    vpd_matrix = vpd.reshape(1, -1)

    extent = [
        dates.min(), dates.max(),
        ymin, ymax
    ]

    ax.imshow(
        vpd_matrix,
        aspect='auto',
        cmap='Greys',      # ou 'viridis', 'plasma', etc.
        extent=extent,
        alpha=0.35,
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )

    cbar = plt.colorbar(
        ax.imshow(vpd_matrix, aspect='auto', cmap='Greys',
                extent=extent, alpha=0.35, origin="lower",
                vmin=vmin, vmax=vmax),
        ax=ax,
        orientation="horizontal",
        pad=0.12
    )
    cbar.set_label("Daily Vapor Pressure Deficit amplitude (55 m)")

    ax.plot(df.index, df["value"], color="black", alpha=0.4, linewidth=1.5, label="Time Series")
    ax.set_ylabel("(dB)")
    ax.grid(True)

    ax.plot(df.index, y, "o", alpha=0.4, color="black", label="Obs S1")
    ax.plot(df.index, y_fit, "-", color="red", linewidth=2, label=f"Model {n_freq} frequencies")

    ax_rain = ax.twinx()
    ax_rain.bar(df_gx_daily.index, df_gx_daily["pluie"],
            width=1, alpha=0.9, color="royalblue", label="Daily rainfall (mm)")
    ax_rain.set_ylabel("Rainfall (mm/day)", color="royalblue")
    ax_rain.tick_params(axis='y', labelcolor="royalblue")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_rain.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Time Series + Fit + Rainfall + VPD")
    plt.tight_layout()
    if save_fig:
        start = df.index.min().strftime("%Y%m%d")
        end = df.index.max().strftime("%Y%m%d")

        quad_str = "quad" if quad else "lin"
        fname = (
            f"{prefix}_"
            f"{n_freq}freq_"
            f"{quad_str}_"
            f"{nom}_"
            f"{start}_{end}.png"
        )

        fpath = os.path.join(outdir, fname)

        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"✔ Figure sauvegardée : {fpath}")
    plt.show()

    return sigma,params

# Parel pour les ROI avec fréquences fixées
# def afficher_diff_signaux_fixe(df_vh, df_vv, w_vh, params_vh, w_vv, params_vv, df_gx_daily, n_freq=2):
#     if n_freq==2:
#         params_vh = list(params_vh[0:4])+[w_vh[0]]+list(params_vh[4:6])+[w_vh[1]]
#         params_vv = list(params_vv[0:4])+[w_vv[0]]+list(params_vv[4:6])+[w_vv[1]]
#     elif n_freq==4:
#         params_vh = list(params_vh[0:4])+[w_vh[0]]+list(params_vh[4:6])+[w_vh[1]]+list(params_vh[6:8])+[w_vh[2]]+list(params_vh[8:10])+[w_vh[3]]
#         params_vv = list(params_vv[0:4])+[w_vv[0]]+list(params_vv[4:6])+[w_vv[1]]+list(params_vv[6:8])+[w_vv[2]]+list(params_vv[8:10])+[w_vv[3]]
#     # on fait une jointure interne pour ne garder que les dates communes aux deux DataFrames
#     df_merged = pd.merge(df_vh[['value']], df_vv[['value']],
#                          left_index=True, right_index=True,
#                          how='inner', suffixes=('_vh', '_vv'))

#     y_vh = df_merged["value_vh"].values
#     y_vv = df_merged["value_vv"].values

#     t0 = df_merged.index.min()
#     t = (df_merged.index - t0).days.values

#     if n_freq == 2:
#         y_vh_fit = seasonal_2freq(t, *params_vh)
#         y_vv_fit = seasonal_2freq(t, *params_vv)
#     elif n_freq == 4:
#         y_vh_fit = seasonal_4freq(t, *params_vh)
#         y_vv_fit = seasonal_4freq(t, *params_vv)

#     diff_obs = y_vh - y_vv
#     diff_fit = y_vh_fit - y_vv_fit

#     r = np.corrcoef(diff_obs, diff_fit)[0, 1]
#     print(f"corr de Pearson entre signal reconstruit et signal original = {r:.3f}")

#     fig, ax = plt.subplots(figsize=(12, 6))

#     ax.plot(df_merged.index, diff_obs,
#             color="lightgray", linewidth=1.5,
#             label="Différence Observée (VH - VV)")
#     ax.plot(df_merged.index, diff_obs, "o",
#             alpha=0.4, color="black")

#     ax.plot(df_merged.index, diff_fit,
#             "-", linewidth=2, color="red",
#             label="Différence Reconst. (Fit VH - Fit VV)")

#     ax.set_ylabel("Différence (dB)")
#     ax.grid(True)

#     ax_rain = ax.twinx()
#     ax_rain.bar(df_gx_daily.index, df_gx_daily["pluie"],
#                 width=1, alpha=0.9,
#                 color="royalblue",
#                 label="Pluie journalière (mm)")
#     ax_rain.set_ylabel("Pluie (mm/jour)", color="royalblue")
#     ax_rain.tick_params(axis='y', labelcolor="royalblue")

#     h1, l1 = ax.get_legend_handles_labels()
#     h2, l2 = ax_rain.get_legend_handles_labels()
#     ax.legend(h1 + h2, l1 + l2, loc="upper left")

#     plt.title("Différence VH - VV (Observée vs Reconstruite)")
#     plt.tight_layout()
#     plt.show()

def plot_signal_with_fit(ax, dates, y_obs, y_fit, 
                         label_obs="Observé", label_fit="Reconstruit",
                         color_obs="black", color_fit="red"):
    
    ax.plot(dates, y_obs, "o", alpha=0.4, color=color_obs, label=label_obs)
    ax.plot(dates, y_fit, "-", linewidth=2, color=color_fit, label=label_fit)
    ax.grid(True)
    ax.legend(loc="upper left")

def plot_rain(ax, df_gx_daily):
    ax_rain = ax.twinx()
    ax_rain.bar(df_gx_daily.index, df_gx_daily["pluie"],
                width=1, alpha=0.8, color="royalblue",
                label="Pluie (mm)")
    ax_rain.set_ylabel("Pluie (mm/jour)", color="royalblue")
    ax_rain.tick_params(axis='y', labelcolor="royalblue")
    return ax_rain

def afficher_diff_signaux_fixe(df1, df2, w1, params1, w2, params2, df_gx_daily,noms, n_freq=2):
    if n_freq==2:
        params1 = list(params1[0:4])+[w1[0]]+list(params1[4:6])+[w1[1]]
        params2 = list(params2[0:4])+[w2[0]]+list(params2[4:6])+[w2[1]]
    elif n_freq==4:
        params1 = list(params1[0:4])+[w1[0]]+list(params1[4:6])+[w1[1]]+list(params1[6:8])+[w1[2]]+list(params1[8:10])+[w1[3]]
        params2 = list(params2[0:4])+[w2[0]]+list(params2[4:6])+[w2[1]]+list(params2[6:8])+[w2[2]]+list(params2[8:10])+[w2[3]]
    # on fait une jointure interne pour ne garder que les dates communes aux deux DataFrames
    df_merged = pd.merge(df1[['value']], df2[['value']],
                         left_index=True, right_index=True,
                         how='inner', suffixes=('_1', '_2'))

    y1 = df_merged["value_1"].values
    y2 = df_merged["value_2"].values
    dates = df_merged.index

    t0 = df_merged.index.min()
    t = (df_merged.index - t0).days.values

    if n_freq == 2:
        y1_fit = seasonal_2freq(t, *params1)
        y2_fit = seasonal_2freq(t, *params2)
    elif n_freq == 4:
        y1_fit = seasonal_4freq(t, *params1)
        y2_fit = seasonal_4freq(t, *params2)

    diff_obs = y1 - y2
    diff_fit = y1_fit - y2_fit

    r = np.corrcoef(diff_obs, diff_fit)[0, 1]
    print(f"corr de Pearson entre signal reconstruit et signal original = {r:.3f}")

    # affichage
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # plot 1 : différence VH - VV
    plot_signal_with_fit(ax1, dates, diff_obs, diff_fit,
                         label_obs="Diff Observée", label_fit="Diff Reconst.",
                         color_obs="black", color_fit="red")
    ax1.set_title(f"Différence {noms[0]} - {noms[1]}")
    ax1.set_ylabel("(dB)")

    plot_rain(ax1, df_gx_daily)

    # plot 2 : Signaux VH et VV
    plot_signal_with_fit(ax2, dates, y1, y1_fit,
                         label_obs=f"{noms[0]} Observé", label_fit=f"{noms[0]} Reconstruit",
                         color_obs="darkgreen", color_fit="green")
    plot_signal_with_fit(ax2, dates, y2, y2_fit,
                         label_obs=f"{noms[1]} Observé", label_fit=f"{noms[1]} Reconstruit",
                         color_obs="darkorange", color_fit="orange")

    ax2.set_title(f"Signaux {noms[0]} et {noms[1]}")
    ax2.set_ylabel("(dB)")

    plt.tight_layout()
    plt.show()

def afficher_diff_signaux(df1, df2, params1, params2, df_gx_daily, noms, n_freq=2):
    df_merged = pd.merge(df1[['value']], df2[['value']],
                         left_index=True, right_index=True,
                         how='inner', suffixes=('_1', '_2'))

    y1 = df_merged["value_1"].values
    y2 = df_merged["value_2"].values
    dates = df_merged.index

    t0 = dates.min()
    t = (dates - t0).days.values

    # modèles
    if n_freq == 2:
        y1_fit = seasonal_2freq(t, *params1)
        y2_fit = seasonal_2freq(t, *params2)
    else:
        y1_fit = seasonal_4freq(t, *params1)
        y2_fit = seasonal_4freq(t, *params2)

    diff_obs = y1 - y2
    diff_fit = y1_fit - y2_fit

    r = np.corrcoef(diff_obs, diff_fit)[0, 1]
    print(f"Corrélation {noms[0]}–{noms[1]} reconstruite vs observée = {r:.3f}")

    # affichage
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # plot 1 : différence VH - VV
    plot_signal_with_fit(ax1, dates, diff_obs, diff_fit,
                         label_obs="Diff Observée", label_fit="Diff Reconst.",
                         color_obs="black", color_fit="red")
    ax1.set_title(f"Différence {noms[0]}–{noms[1]}")
    ax1.set_ylabel("(dB)")

    plot_rain(ax1, df_gx_daily)

    # plot 2 : Signaux VH et VV
    plot_signal_with_fit(ax2, dates, y1, y1_fit,
                         label_obs=f"{noms[0]} Observé", label_fit=f"{noms[0]} Reconstruit",
                         color_obs="darkgreen", color_fit="green")
    plot_signal_with_fit(ax2, dates, y2, y2_fit,
                         label_obs=f"{noms[1]} Observé", label_fit=f"{noms[1]} Reconstruit",
                         color_obs="darkorange", color_fit="orange")

    ax2.set_title(f"Signaux {noms[0]} et {noms[1]}")
    ax2.set_ylabel("(dB)")

    plt.tight_layout()
    plt.show()



# --- Affichage des différences VH - VV ---
# on affiche la différence observée et la différence reconstruite entre VH et VV

# def afficher_diff_signaux(df_vh, df_vv, params_vh, params_vv, df_gx_daily, n_freq=2):

#     # Jointure interne sur les dates communes
#     df_merged = pd.merge(df_vh[['value']], df_vv[['value']],
#                          left_index=True, right_index=True,
#                          how='inner', suffixes=('_vh', '_vv'))

#     y_vh = df_merged["value_vh"].values
#     y_vv = df_merged["value_vv"].values

#     t0 = df_merged.index.min()
#     t = (df_merged.index - t0).days.values

#     # Fit individuellement
#     if n_freq == 2:
#         y_vh_fit = seasonal_2freq(t, *params_vh)
#         y_vv_fit = seasonal_2freq(t, *params_vv)
#     elif n_freq == 4:
#         y_vh_fit = seasonal_4freq(t, *params_vh)
#         y_vv_fit = seasonal_4freq(t, *params_vv)

#     # Différences
#     diff_obs = y_vh - y_vv
#     diff_fit = y_vh_fit - y_vv_fit

#     # Corrélation
#     r = np.corrcoef(diff_obs, diff_fit)[0, 1]
#     print(f"corr de Pearson entre signal reconstruit et signal original = {r:.3f}")

#     # ---------------------------------------------------------
#     # FIGURE AVEC 2 SUBPLOTS VERTICAUX
#     # ---------------------------------------------------------
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     # =========================================================
#     # 🔹 SUBPLOT 1 : DIFFÉRENCE VH–VV (observée & reconstruite)
#     # =========================================================
#     ax1.plot(df_merged.index, diff_obs,
#              color="lightgray", linewidth=1.5,
#              label="Différence Observée (VH - VV)")
#     ax1.plot(df_merged.index, diff_obs, "o",
#              alpha=0.4, color="black")

#     ax1.plot(df_merged.index, diff_fit,
#              "-", linewidth=2, color="red",
#              label="Différence Reconst. (Fit VH - Fit VV)")

#     ax1.set_ylabel("Différence (dB)")
#     ax1.grid(True)

#     ax1_rain = ax1.twinx()
#     ax1_rain.bar(df_gx_daily.index, df_gx_daily["pluie"],
#                  width=1, alpha=0.8, color="royalblue",
#                  label="Pluie journalière (mm)")
#     ax1_rain.set_ylabel("Pluie (mm/jour)", color="royalblue")
#     ax1_rain.tick_params(axis='y', labelcolor="royalblue")

#     h1, l1 = ax1.get_legend_handles_labels()
#     h2, l2 = ax1_rain.get_legend_handles_labels()
#     ax1.legend(h1 + h2, l1 + l2, loc="upper left")

#     ax1.set_title("Différence VH - VV (Observée vs Reconstruite)")

#     # =========================================================
#     # 🔹 SUBPLOT 2 : LES SIGNUX VH & VV (obs + fit)
#     # =========================================================
#     ax2.plot(df_merged.index, y_vh, "o", alpha=0.4,
#              color="darkgreen", label="VH observé")
#     ax2.plot(df_merged.index, y_vh_fit, "-",
#              color="green", linewidth=2, label="VH reconstruit")

#     ax2.plot(df_merged.index, y_vv, "o", alpha=0.4,
#              color="darkorange", label="VV observé")
#     ax2.plot(df_merged.index, y_vv_fit, "-",
#              color="orange", linewidth=2, label="VV reconstruit")

#     ax2.set_ylabel("Signal (dB)")
#     ax2.grid(True)

#     h3, l3 = ax2.get_legend_handles_labels()
#     ax2.legend(h3, l3, loc="upper left")

#     ax2.set_title("Signaux VH et VV (Observés vs Reconstruits)")

#     # ---------------------------------------------------------
#     plt.tight_layout()
#     plt.show()