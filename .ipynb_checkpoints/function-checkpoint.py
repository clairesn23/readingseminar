import csv
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt  
from ipywidgets import interact, FloatSlider
from datetime import datetime
import pandas as pd
import os

# Imports supplémentaires pour Fourier et résidus
from scipy.optimize import curve_fit
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.interpolate import BSpline
import seaborn as sns
from scipy.linalg import lstsq
from scipy.stats import pearsonr


def prepadataframe(Paracou=1, allROI=0): 
    #### PARACOU=1 : on fait sur paracou, si PARACOU=0 on fait sur trinité
    #### ALLROI : si on fait sur paracou, on voit si on utilise ALLROI ou Tropiscat grace a ALLROI
    """Charge les données SAR Paracou ou Trinité."""
    if Paracou==1:
        Data_file = "data/Data/S1A_ASC-DSC_VV-VH_Paracou"        
        df_VH_ASC_TropiScat = pd.read_csv(f"{Data_file}/s1a_vh_ASC_047_TropiScat.csv")
        df_VH_DSC_TropiScat = pd.read_csv(f"{Data_file}/s1a_vh_DSC_039_TropiScat.csv")
        df_VV_ASC_TropiScat = pd.read_csv(f"{Data_file}/s1a_vv_ASC_047_TropiScat.csv")
        df_VV_DSC_TropiScat = pd.read_csv(f"{Data_file}/s1a_vv_DSC_039_TropiScat.csv")
        
        dfs_TropiScat = {
            "VV_ASC": df_VV_ASC_TropiScat,
            "VH_ASC": df_VH_ASC_TropiScat,
            "VV_DSC": df_VV_DSC_TropiScat,
            "VH_DSC": df_VH_DSC_TropiScat,
        }

        df_VH_ASC_allROI = pd.read_csv(f"{Data_file}/s1a_vh_ASC_047_allROI.csv")
        df_VH_DSC_allROI = pd.read_csv(f"{Data_file}/s1a_vh_DSC_039_allROI.csv")
        df_VV_ASC_allROI = pd.read_csv(f"{Data_file}/s1a_vv_ASC_047_allROI.csv")
        df_VV_DSC_allROI = pd.read_csv(f"{Data_file}/s1a_vv_DSC_039_allROI.csv")
        
        dfs_allROI = {
            "VV_ASC": df_VV_ASC_allROI,
            "VH_ASC": df_VH_ASC_allROI,
            "VV_DSC": df_VV_DSC_allROI,
            "VH_DSC": df_VH_DSC_allROI,
        }
        
        value_TropiScat = {
            "VV_ASC": df_VV_ASC_TropiScat.select_dtypes(include='number').to_numpy(),
            "VH_ASC": df_VH_ASC_TropiScat.select_dtypes(include='number').to_numpy(),
            "VV_DSC": df_VV_DSC_TropiScat.select_dtypes(include='number').to_numpy(),
            "VH_DSC": df_VH_DSC_TropiScat.select_dtypes(include='number').to_numpy(),
        }
        
        value_allROI = {
            "VV_ASC": df_VV_ASC_allROI.select_dtypes(include='number').to_numpy(),
            "VH_ASC": df_VH_ASC_allROI.select_dtypes(include='number').to_numpy(),
            "VV_DSC": df_VV_DSC_allROI.select_dtypes(include='number').to_numpy(),
            "VH_DSC": df_VH_DSC_allROI.select_dtypes(include='number').to_numpy(),
        }
        
        date_TropiScat = {
            "VV_ASC": np.array(df_VV_ASC_TropiScat)[:,0],
            "VH_ASC": np.array(df_VH_ASC_TropiScat)[:,0],
            "VV_DSC": np.array(df_VV_DSC_TropiScat)[:,0],
            "VH_DSC": np.array(df_VH_DSC_TropiScat)[:,0],
        }
        date_allROI = {
            "VV_ASC": np.array(df_VV_ASC_allROI)[:,0],
            "VH_ASC": np.array(df_VH_ASC_allROI)[:,0],
            "VV_DSC": np.array(df_VV_DSC_allROI)[:,0],
            "VH_DSC": np.array(df_VH_DSC_allROI)[:,0],
        }
        if allROI==1:
            return value_allROI, date_allROI, dfs_allROI
        else:
            return value_TropiScat, date_TropiScat, dfs_TropiScat
            
    else:
        Data_file="data/Data/StudyCase_S1_Trinité"
        df_VH_ASC_Trinit = pd.read_csv(f"{Data_file}/ts_s1a_21NZF_vh_ASC_120_867km2.csv")
        df_VV_ASC_defoliated_Trinit = pd.read_csv(f"{Data_file}/ts_s1a_21NZF_vv_ASC_120_1ha-defoliated.csv")
        df_VV_ASC_Trinit = pd.read_csv(f"{Data_file}/ts_s1a_21NZF_vv_ASC_120_867km2.csv")

        dfs_Trinit = {
            "VV_ASC": df_VV_ASC_Trinit,
            "VH_ASC": df_VH_ASC_Trinit,
            "VV_DEFOL": df_VV_ASC_defoliated_Trinit,
        }

        moy_masked_Trinit = {
            "VV_ASC": df_VV_ASC_Trinit.select_dtypes(include='number').to_numpy()[:,0],
            "VH_ASC": df_VH_ASC_Trinit.select_dtypes(include='number').to_numpy()[:,0],
            "VV_DEFOL": df_VV_ASC_defoliated_Trinit.select_dtypes(include='number').to_numpy()[:,0],
        }
        moy_unmasked_Trinit = {
            "VV_ASC": df_VV_ASC_Trinit.select_dtypes(include='number').to_numpy()[:,1],
            "VH_ASC": df_VH_ASC_Trinit.select_dtypes(include='number').to_numpy()[:,1],
            "VV_DEFOL": df_VV_ASC_defoliated_Trinit.select_dtypes(include='number').to_numpy()[:,1],
        }
        
        date_Trinit = {
            "VV_ASC": np.array(df_VV_ASC_Trinit)[:,0],
            "VH_ASC": np.array(df_VH_ASC_Trinit)[:,0],
            "VV_DEFOL": np.array(df_VV_ASC_defoliated_Trinit)[:,0],
        }
        return moy_masked_Trinit, moy_unmasked_Trinit, date_Trinit, dfs_Trinit



def SerieTemp(value,date,nom='VV_ASC'): 
        
    T=date[nom]
    dates= [datetime.strptime(t, "%Y-%m-%d") for t in T]
    days = np.array([(d - dates[0]).days for d in dates])
    intervals=np.diff(days)
    fs=1/np.mean(intervals)
    
    values=value[nom]
    
    plt.figure(figsize=(10,4))
    plt.plot(dates, values, marker='o')
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Série temporelle")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()  



def plot_signal_par_mois_single(df, dates_dict, nom, title=None):
    """
    Trace le signal en concaténant les mois sur l'axe X :
    d'abord tous les janvier (triés par année), puis tous les février, etc.
    Avec une régression linéaire par mois.
    """
    values = df[nom][:, 0]
    dates_raw = [datetime.strptime(t, "%Y-%m-%d") for t in dates_dict[nom]]
    
    df_plot = pd.DataFrame({
        "date": dates_raw,
        "value": values,
        "month": [d.month for d in dates_raw],
        "year": [d.year for d in dates_raw],
    }).sort_values(["month", "date"]).reset_index(drop=True)
    
    mois_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    
    fig, ax = plt.subplots(figsize=(18, 5))
    
    x_cursor = 0
    xticks_pos, xticks_labels = [], []
    
    for m in range(1, 13):
        df_month = df_plot[df_plot["month"] == m].copy()
        if df_month.empty:
            continue
        
        n_pts = len(df_month)
        x_local = np.arange(x_cursor, x_cursor + n_pts)
        y = df_month["value"].values
        
        # Plot points
        ax.scatter(x_local, y, color=colors[m-1], s=20, zorder=3, label=mois_labels[m-1])
        ax.plot(x_local, y, color=colors[m-1], alpha=0.4, linewidth=0.8)
        
        # Régression linéaire
        coeffs = np.polyfit(x_local, y, 1)
        reg_line = np.polyval(coeffs, x_local)
        ax.plot(x_local, reg_line, color=colors[m-1], linewidth=2.5, linestyle="--", zorder=4)
        
        # Séparateur et label
        ax.axvline(x_cursor - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        xticks_pos.append(x_cursor + n_pts / 2)
        xticks_labels.append(mois_labels[m-1])
        
        x_cursor += n_pts
    
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels, fontsize=11)
    ax.set_ylabel("Backscatter")
    ax.set_title(title or f"Signal groupé par mois avec régression linéaire — {nom}")
    ax.legend(loc="upper right", ncol=6, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




def get_top_periods(freqs, psd, n_peaks=3):
    """Identifie les n périodes dominantes dans le spectre."""
    indices = np.argsort(psd)[::-1][:n_peaks]
    top_freqs = freqs[indices]
    top_psds = psd[indices]
    top_periods = 1 / top_freqs
    return top_periods, top_freqs, top_psds

def psd_welch_annee(df, nom, window_lengths=[730, 910, 1095, 1280, 1460], 
                     overlap_ratios=[0.70, 0.80, 0.90], n_peaks=3, 
                     outdir="figures_welch", prefix="PSD", show=False):
    
    #Calcule et affiche la DSP par méthode de Welch avec variation des paramètres.
    
    #Paramètres :
    #-----------
    #df : dict
    #    Dictionnaire contenant les valeurs SAR pour chaque série
    #nom : str
    #    Nom de la série à analyser (ex: 'VH_DSC')
    #window_lengths : list
    #    Durées de fenêtre en jours
    #overlap_ratios : list
    #    Ratios de recouvrement à tester
    #n_peaks : int
    #    Nombre de pics dominants à identifier
    
    os.makedirs(outdir, exist_ok=True)
    y = df[nom][:, 0]
    fs = 1 / 12  # Sentinel-1 : ~12 jours de revisite

    for ol in overlap_ratios:
        print(f"\n===== Overlap Analysis = {int(ol*100)}% =====")
        fig, ax = plt.subplots(figsize=(18, 5))

        for wl in window_lengths:
            wl_samples = int(np.floor(wl / 12))
            nperseg = min(wl_samples, len(y))
            noverlap = int(nperseg * ol)
            freqs, psd = signal.welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap)
            freqs_annee = freqs * 365
            mask = (freqs_annee > 0.1) & (freqs_annee < 10)

            line, = ax.semilogx(freqs_annee[mask], psd[mask], label=f'Window: {wl} days')
            color = line.get_color()

            top_periods, top_freqs, top_psds = get_top_periods(freqs[mask], psd[mask], n_peaks=n_peaks)
            top_freqs_annee = top_freqs * 365

            ax.scatter(top_freqs_annee, top_psds, color=color, s=80, zorder=5)
            for f_ann, p_ann, period in zip(top_freqs_annee, top_psds, top_periods):
                ax.annotate(f"{period:.0f}d", xy=(f_ann, p_ann),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=7, color=color)

            top_periods_years = [p/365 for p in top_periods]
            print(f" WL = {wl} days — Dominant periods ≈ "
                  f"{[round(period,3) for period in top_periods_years]} years "
                  f"or, {[int(p) for p in top_periods]} days")

        ax.set_title(f"DSP Welch, Overlap {ol*100:.0f}%")
        ax.set_xlabel("Frequency (cycles/year) [log]")
        ax.set_ylabel("Power Spectral Density")
        ax.tick_params(axis='both', which='major')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        fname = f"{prefix}_overlap_{int(ol*100)}pct_{nom}.png"
        fpath = os.path.join(outdir, fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"✔ Figure sauvegardée : {fpath}")
        if show:
            plt.show()
        else:
            plt.close()





# Variable globale pour stocker les fréquences ω
omegas = None

def fourier_model(t, a, b, *coeffs):
    """
    Modèle de Fourier : y(t) = a + b*t + Σ[Aᵢ*sin(ωᵢ*t) + Bᵢ*cos(ωᵢ*t)]
    
    Les fréquences ω sont fixées d'avance (variable globale).
    """
    global omegas
    y = a + b * t
    n_freq = len(coeffs) // 2
    for i in range(n_freq):
        omega = omegas[i]
        A = coeffs[2*i]
        B = coeffs[2*i + 1]
        y += A * np.sin(omega * t) + B * np.cos(omega * t)
    return y

def fit_fourier(y_signal, dates_str, n_freq, nom):
    """
    Ajuste un modèle de Fourier à n_freq harmoniques.
    
    Retourne :
    ---------
    dict avec : params, t, y, y_fit, residuals, r_squared, omegas, periods
    """
    global omegas
    
    # Conversion dates → jours depuis origine
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates_str]
    t0 = dates_dt[0]
    t = np.array([(d - t0).days for d in dates_dt], dtype=float)
    y = y_signal
    
    # Extraction des fréquences dominantes via Welch
    fs = 1 / 12
    nperseg = min(int(730 / 12), len(y))
    freqs, psd = signal.welch(y, fs=fs, window="hann",
                              nperseg=nperseg, noverlap=nperseg // 2,
                              scaling="density")
    top_freqs_idx = np.argsort(psd[1:])[::-1][:n_freq] + 1
    top_freqs = freqs[top_freqs_idx]
    omegas = 2 * np.pi * top_freqs
    
    # Estimation initiale
    p0 = [y.mean(), 0.0] + [0.1, 0.1] * n_freq
    
    # Ajustement par moindres carrés
    try:
        params, pcov = curve_fit(fourier_model, t, y, p0=p0, maxfev=10000)
    except RuntimeError:
        print(f"    ⚠ Convergence échouée pour {nom} avec n_freq={n_freq}")
        return None
    
    # Reconstruction et résidus
    y_fit = fourier_model(t, *params)
    residuals = y - y_fit
    
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        "params": params,
        "pcov": pcov,
        "t": t,
        "y": y,
        "y_fit": y_fit,
        "residuals": residuals,
        "r_squared": r_squared,
        "omegas": omegas.copy(),
        "periods": 2 * np.pi / omegas,
        "dates": dates_dt
    }


def build_bspline_basis(t, n_knots, degree=3):
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

def penalized_spline_fit(B, Y, lam):
    beta = solve(B.T @ B + lam * np.eye(B.shape[1]), B.T @ Y)
    Y_hat = B @ beta
    return beta, Y_hat  

def smoothing_matrix(B, lam):
    return B @ solve(B.T @ B + lam * np.eye(B.shape[1]), B.T)

def compute_edf(B, lam):
    return np.trace(smoothing_matrix(B, lam))

def plot_fourier_fit(fit_result, nom, n_freq, outdir="figures_fourier"):
    """Visualise le signal, le fit de Fourier et les résidus."""
    os.makedirs(outdir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    
    # Signal + reconstruction
    ax = axes[0]
    ax.plot(fit_result["dates"], fit_result["y"], "-", lw=1.2, alpha=0.7,
            label="Données SAR", color="steelblue")
    ax.plot(fit_result["dates"], fit_result["y_fit"], "-", lw=1.8,
            label=f"Fourier (n={n_freq})", color="orangered")
    ax.set_ylabel("σ⁰ [dB]", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_title(f"{nom}  |  R² = {fit_result['r_squared']:.4f}", fontsize=12, fontweight="bold")
    
    # Résidus
    ax = axes[1]
    ax.plot(fit_result["dates"], fit_result["residuals"], "-", lw=0.8,
            alpha=0.7, color="gray")
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_ylabel("Résidus [dB]", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    fname = f"fourier_fit_{nom}_n{n_freq}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"    → {fname}")

def analyze_residuals(residuals, nom, outdir="figures_residuals", n_freq=None, method="fourier"):
    """
    Analyse complète des résidus (Fourier ou Spline pénalisée).
    """
    os.makedirs(outdir, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # ── Histogramme ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(residuals, bins=30, density=True, alpha=0.7,
             color="steelblue", edgecolor="black", label="Résidus")
    mu, sigma = 0, residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 200)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2,
             label=f"N(0, {sigma:.3f})")
    ax1.set_xlabel("Résidus [dB]", fontsize=10)
    ax1.set_ylabel("Densité", fontsize=10)
    title_suffix = f"n={n_freq}" if n_freq is not None else method
    ax1.set_title(f"Distribution des résidus – {nom} ({title_suffix})", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ── Q-Q plot ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Q-Q plot (normalité)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ── Autocorrélation (ACF) ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    plot_acf(residuals, lags=min(40, len(residuals) // 2), ax=ax3, alpha=0.05)
    ax3.set_title("Autocorrélation des résidus", fontsize=10)
    ax3.set_xlabel("Lag", fontsize=9)
    
    # ── Statistiques textuelles ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
    lb_p_min = lb_test["lb_pvalue"].min()
    
    interp_norm = ("Résidus compatibles avec une loi normale (p > 0.05)" 
                   if shapiro_p > 0.05 
                   else "Résidus NON gaussiens (p < 0.05)")
    interp_acorr = ("Pas d'autocorrélation significative (p > 0.05)" 
                    if lb_p_min > 0.05 
                    else "Autocorrélation détectée (p < 0.05) → signal non capturé")
    
    method_line = f"n_freq = {n_freq}" if n_freq is not None else f"méthode = {method}"
    
    text = f"""
    STATISTIQUES DES RÉSIDUS  –  {nom}  ({method_line})
    {'─'*70}
    
    Normalité (Shapiro-Wilk) :
        • W = {shapiro_stat:.4f}
        • p-value = {shapiro_p:.4e}
        • Interprétation : {interp_norm}
    
    Autocorrélation (Ljung-Box, 10 lags) :
        • p-value min = {lb_p_min:.4e}
        • Interprétation : {interp_acorr}
    
    Statistiques descriptives :
        • Moyenne    = {residuals.mean():.4f} dB  (doit être ≈ 0)
        • Écart-type = {residuals.std():.4f} dB
        • Min / Max  = {residuals.min():.4f} / {residuals.max():.4f} dB
    """
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    
    fname = f"residuals_analysis_{nom}_{method_line.replace(' ', '').replace('=','')}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"    → {fname}")

def gcv_score(B, Y, lam):
    n = len(Y)
    _, Y_hat = penalized_spline_fit(B, Y, lam)
    residual = np.sum((Y - Y_hat)**2)
    edf = compute_edf(B, lam)
    return (n * residual) / ((n - edf)**2)

def optimize_lambda_gcv(B, Y):
    lambdas = np.logspace(-4, 4, 100)
    scores = np.array([gcv_score(B, Y, lam) for lam in lambdas])
    return lambdas[np.argmin(scores)]

    # FONCTION PRINCIPALE MODÉLISATION RÉGIONALE
def regional_penalized_model(t, Y_reg,n, lam=None, auto_lambda=False):
    B = build_bspline_basis(t,n)
    
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

def stepwise_knot_selection_bspline(x, y, alpha=3.0, verbose=True):
    n = len(x)
    min_knots = 5
    n_init = max(min_knots, n // 3)
    quantiles = np.linspace(0, 100, n_init + 2)[1:-1]
    J = len(quantiles)

    B_full = build_bspline_basis(x, J)
    lam_full = optimize_lambda_gcv(B_full, y)
    _, y_hat_full = penalized_spline_fit(B_full, y, lam_full)
    rss_full = np.sum((y - y_hat_full)**2)
    p_full = J + 4
    sigma2_hat = rss_full / max(n - p_full, 1)

    Cp_init = rss_full + alpha * p_full * sigma2_hat
    history = {"Cp": [Cp_init], "RSS": [rss_full], "n_knots": [J]}

    if verbose:
        print(f"  {'# nœuds':>8}  {'p=nœuds+4':>10}  {'RSS':>14}  {'Cp':>14}")
        print("  " + "-" * 54)
        print(f"  {J:>8}  {p_full:>10}  {rss_full:>14.4f}  {Cp_init:>14.4f}  ← modèle initial")

    for step in range(J - 1, min_knots - 1, -1):
        p_j = step + 4
        B   = build_bspline_basis(x, step)
        lam = optimize_lambda_gcv(B, y)
        _, y_hat = penalized_spline_fit(B, y, lam)
        rss = np.sum((y - y_hat)**2)
        Cp  = rss + alpha * p_j * sigma2_hat

        history["Cp"].append(Cp)
        history["RSS"].append(rss)
        history["n_knots"].append(step)

        if verbose:
            marker = "  ← optimal" if Cp == min(history["Cp"]) else ""
            print(f"  {step:>8}  {p_j:>10}  {rss:>14.4f}  {Cp:>14.4f}{marker}")

    best_idx     = np.argmin(history["Cp"])
    best_n_knots = history["n_knots"][best_idx]

    if verbose:
        print(f"\n  ✓ Nœuds optimal : {best_n_knots}  |  Cp = {history['Cp'][best_idx]:.4f}")

    return best_n_knots, history
