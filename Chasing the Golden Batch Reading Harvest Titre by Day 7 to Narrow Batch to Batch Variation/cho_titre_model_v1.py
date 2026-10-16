"""
BMX5102 Team 3 (Schrodinger's CHO) - harvest titre prediction with a CHO mass balance
====================================================================================

Data:   AstraZeneca upstream dataset, Gangadharan et al. (2021), Computers & Chemical
        Engineering 146:107226, supplementary workbook "1-s2.0-S0098135421000041-mmc3.xlsx"
        (106 CHO cultures, 4 L to 500 L, every value rescaled 0 to 1 by the authors).

What each section answers (V10 proposal):
    Sections 2-4  build one row per culture per decision day, with mass-balance features
    Section 5     Q2: how well do day-5, day-7 and day-9 data predict harvest titre at bench scale?
    Section 6     Lesson 7 checks on the chosen model (coefficients, VIF, influence, residual plots)
    Section 7     Q3: the bench model taken unchanged to the 11 pilot and plant runs (transfer penalty)
    Section 8     Q4: calibrate only a level shift and a scale factor on k large runs, test on the rest

How to run in VS Code:
    1. pip install pandas numpy scipy matplotlib openpyxl statsmodels lmdiag
    2. Put the workbook next to this file (or change DATA_FILE in Section 1).
    3. Run cell by cell (each "# %%" is a cell) or run the whole file. Outputs go to ./outputs.

Judgement calls made in this script (check each one, they are yours to defend):
    J1  The outcome is the LAST titre measured in the RAW sheet (harvest, day 13 to 18).
        The gap-filled sheet is not used for the outcome because its missing titres were
        filled in by the authors' own model.
    J2  Inputs on day d use only data up to day d. Gaps are filled by carrying the last
        measured value forward (never backward). The authors' gap-filled sheet is NOT the
        default because its filling may use later days, which would leak the future into a
        day-d prediction. Set INPUT_SOURCE = "gapfilled" to compare.
    J3  Mass balance (product): dP/dt = qP x Xv, so P(t) = P(0) + qP x IVCD(t). Titre on days 1 to 3
        averages 0.004 on the 0-1 scale, so P(0) is taken as 0 and the cumulative specific
        productivity is qP_cum = titre / IVCD at the last titre reading (one reading is enough).
        Features built from it: titre to date, qP_cum, current viable cells, net growth rate,
        and a projection to the planned harvest day, P(d) + qP_cum x VCD(d) x (harvest - d).
    J4  Nutrient balances do not close: feed additions are not recorded, so glucose and
        lactate rates are "apparent" (consumption mixed with feeding). Used as features only.
    J5  Values are rescaled 0 to 1, so units are hidden. Differences between days are exact;
        ratios and integrals (IVCD, qP, VCD/TCD) are correct only up to a constant factor,
        which a regression absorbs. Viability is not in the file; VCD/TCD is used as an index.
    J6  Scale is back-calculated from the rescaled volume column: 4, 4.9, 5, 50, 100, 250, 500 L.
        Bench = 4 to 5 L (95 runs, training). Large = 50 to 500 L (11 runs, never trained on).
    J7  The 23 four-litre runs behave like a different product, so a 0/1 product marker is
        included (set PRODUCT_MARKER = False to drop it).
    J8  Feature sets are kept small on purpose: about 95 bench runs cannot support 20 inputs.
    J9  Osmolality is left out of the default sets: it is missing on 28% of culture-days, which
        drops two large runs, and one 4 L run has an extreme early value that wrecks day-5 fits.
        It stays in the feature table (osm_d) if you want to test it.
    J10 Every feature set and every decision day is scored on the SAME runs, so differences in
        accuracy come from the model and the day, not from which runs happened to be included.
"""

# %% 1. Settings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILE = Path("1-s2.0-S0098135421000041-mmc3.xlsx")
OUT = Path("outputs")
DECISION_DAYS = [5, 7, 9]
PLANNED_HARVEST_DAY = 15          # most runs are harvested on day 15 (63 of 103 with a titre)
INPUT_SOURCE = "raw_locf"         # "raw_locf" (default, see J2) or "gapfilled"
PRODUCT_MARKER = True             # see J7
CHOSEN_SET = "mass_balance"       # model used in Sections 6 to 8 (best transfer to large scale in Q3)
CHOSEN_DAY = 7
SEED = 20261016
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# Colours: reference categorical palette, fixed order (blue, orange, aqua, yellow, magenta)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f2a37", "#5b6b7b", "#e3e8ee"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9d1da", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})

# %% 2. Load and tidy
def load_sheet(name):
    df = pd.read_excel(DATA_FILE, sheet_name=name)
    df.columns = [c.strip() for c in df.columns]
    df["run"] = df["Culture ID"].str.extract(r"^(C\d+)_")[0]
    df["day"] = df["Culture ID"].str.extract(r"_d(\d+)")[0].astype(int)
    return df.sort_values(["run", "day"]).reset_index(drop=True)

raw = load_sheet("Raw Data")
gap = load_sheet("Gap-Filled Data")

# J6: back-calculate tank volume (litres) from the rescaled column
VOLUME_MAP = {0.0: 4.0, 0.001815: 4.9, 0.002016: 5.0, 0.092742: 50.0, 0.193548: 100.0, 0.495968: 250.0, 1.0: 500.0}
raw["volume_L"] = raw["Culture Volume"].round(6).map(VOLUME_MAP)
assert raw["volume_L"].notna().all(), "Unexpected culture volume value: check VOLUME_MAP"
runs = raw.groupby("run").agg(volume_L=("volume_L", "first"), last_day=("day", "max"))
runs["scale"] = np.where(runs["volume_L"] <= 5, "bench", "large")
print(f"{len(runs)} cultures, {len(raw)} culture-days")
print(runs["volume_L"].value_counts().sort_index().rename("cultures").to_string())

# %% 3. Outcome: harvest titre (J1)
titre_obs = raw.dropna(subset=["[mAb]"])
harvest = titre_obs.groupby("run").agg(harvest_titre=("[mAb]", "last"), harvest_day=("day", "last"))
runs = runs.join(harvest)
print(f"\nRuns with no titre at all (excluded): {runs['harvest_titre'].isna().sum()} -> "
      f"{list(runs.index[runs['harvest_titre'].isna()])}")
print("Harvest day:", runs["harvest_day"].value_counts().sort_index().to_dict())

# %% 4. Mass-balance features on day d
VARS = ["VCD", "TCD", "[mAb]", "pH", "Temperature", "pO2", "pCO2", "Osmolality",
        "[Glucose]", "[Glutamine]", "[Glutamate]", "[Lactate]", "[NH3]", "EGN", "CPDL"]
EPS = 1e-4

def run_series(run_id):
    """Daily series for one run, inputs per INPUT_SOURCE (J2); titre always from the raw sheet."""
    src = gap if INPUT_SOURCE == "gapfilled" else raw
    s = src.loc[src["run"] == run_id, ["day"] + VARS].set_index("day").sort_index()
    if INPUT_SOURCE != "gapfilled":
        s = s.ffill()                                    # carry forward only, never backward
    s["titre_raw"] = raw.loc[raw["run"] == run_id].set_index("day")["[mAb]"]
    return s

def trapezoid(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x))) if len(y) > 1 else 0.0

def features_at_day(run_id, d):
    s = run_series(run_id)
    s = s.loc[s.index <= d]
    if s.empty or d not in s.index:
        return None
    vcd = s["VCD"].clip(lower=EPS)
    days = s.index.values
    ivcd = np.array([trapezoid(vcd.loc[:t], days[days <= t]) for t in days])
    ivcd_s = pd.Series(ivcd, index=days)
    t_meas = s["titre_raw"].dropna()                      # titre measured up to day d
    f = {"run": run_id, "day": d}
    f["titre_d"] = float(t_meas.iloc[-1]) if len(t_meas) else np.nan
    f["IVCD_d"] = float(ivcd_s.loc[d])
    f["VCD_d"] = float(vcd.loc[d])
    f["viab_index"] = float(vcd.loc[d] / max(s["TCD"].loc[d], EPS))
    earlier = days[days <= d - 2]                         # 2-day window for rates (or the nearest earlier day)
    d0 = earlier.max() if len(earlier) else days.min()
    f["mu_net"] = float((np.log(vcd.loc[d]) - np.log(vcd.loc[d0])) / max(d - d0, 1))
    if len(t_meas):                                       # J3: cumulative qP, P(0) taken as 0
        tb = t_meas.index[-1]
        iv_tb = ivcd_s.loc[ivcd_s.index <= tb].iloc[-1]    # the two sheets do not always share day labels
        f["qP_cum"] = float(t_meas.iloc[-1] / iv_tb) if iv_tb > 1e-3 else np.nan
    else:
        f["qP_cum"] = np.nan
    f["prod_rate"] = f["qP_cum"] * f["VCD_d"]            # titre being made per day right now
    f["titre_proj"] = f["titre_d"] + f["prod_rate"] * (PLANNED_HARVEST_DAY - d)   # projection (J3)
    dI2 = ivcd_s.loc[d] - ivcd_s.loc[d0]
    for v, name in [("[Glucose]", "qGlc_app"), ("[Lactate]", "qLac_app"), ("[NH3]", "qNH3_app")]:
        dv = s[v].loc[d] - s[v].loc[d0]
        f[name] = float((-dv if name == "qGlc_app" else dv) / dI2) if dI2 > EPS else np.nan   # J4
    for v, name in [("pH", "pH_mean"), ("Temperature", "T_mean"), ("pO2", "pO2_mean"), ("pCO2", "pCO2_mean")]:
        f[name] = float(s[v].mean())                      # average condition so far
    for v, name in [("Osmolality", "osm_d"), ("[Glucose]", "glc_d"), ("[Lactate]", "lac_d"), ("[NH3]", "nh3_d"),
                    ("[Glutamine]", "gln_d"), ("CPDL", "cpdl_d"), ("EGN", "egn_d")]:
        f[name] = float(s[v].loc[d])
    return f

rows = [r for run_id in runs.index for d in DECISION_DAYS if (r := features_at_day(run_id, d)) is not None]
table = pd.DataFrame(rows).merge(runs, left_on="run", right_index=True)
table["product_4L"] = (table["volume_L"] == 4.0).astype(int)
table = table.dropna(subset=["harvest_titre"])
table.to_csv(OUT / "feature_table.csv", index=False)
print(f"\nFeature table: {len(table)} rows ({table['run'].nunique()} runs x {len(DECISION_DAYS)} days) -> outputs/feature_table.csv")

# %% 5. Q2: fit and cross-validate at bench scale
FEATURE_SETS = {
    "baseline_titre_only": ["titre_d"],
    "raw_readings": ["titre_d", "VCD_d", "viab_index", "glc_d", "lac_d", "nh3_d"],
    "mass_balance": ["titre_d", "qP_cum", "VCD_d", "mu_net", "viab_index"],
    "mass_balance_plus_conditions": ["titre_d", "qP_cum", "VCD_d", "mu_net", "viab_index",
                                     "pH_mean", "T_mean", "pO2_mean", "pCO2_mean"],
    "portable": ["titre_d", "qP_cum", "VCD_d", "mu_net", "glc_d", "lac_d", "pH_mean", "pO2_mean"],
    "projection_corrected": ["titre_proj", "mu_net", "viab_index"],   # mass balance first, regression corrects it
}
if PRODUCT_MARKER:
    for k in FEATURE_SETS:
        if k != "baseline_titre_only":
            FEATURE_SETS[k] = FEATURE_SETS[k] + ["product_4L"]
ALL_FEATS = sorted({f for v in FEATURE_SETS.values() for f in v})

def design(df, feats):
    return np.column_stack([np.ones(len(df))] + [df[f].values for f in feats])

def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def loo_predictions(df, feats):
    """Leave-one-run-out: each run is predicted by a model that never saw it."""
    X, y = design(df, feats), df["harvest_titre"].values
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        pred[i] = X[i] @ ols_fit(X[m], y[m])
    return pred

def scores(y, p):
    resid = y - p
    return {"RMSE": float(np.sqrt(np.mean(resid ** 2))), "MAE": float(np.mean(np.abs(resid))),
            "Q2_out_of_sample_R2": float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)),
            "Spearman_rank": float(stats.spearmanr(y, p)[0]), "bias_mean_error": float(np.mean(p - y))}

bench_all = table[table["scale"] == "bench"]
# J10: keep only bench runs that have every feature on every decision day
complete = bench_all.dropna(subset=ALL_FEATS).groupby("run")["day"].nunique()
common_runs = complete.index[complete == len(DECISION_DAYS)]
dropped = sorted(set(bench_all["run"]) - set(common_runs))
print(f"\nQ2 sample: {len(common_runs)} bench runs with every feature on days {DECISION_DAYS}; "
      f"{len(dropped)} dropped, mostly for having no titre reading by day {min(DECISION_DAYS)}")
cv_rows = []
for d in DECISION_DAYS:
    bd = bench_all[(bench_all["day"] == d) & bench_all["run"].isin(common_runs)]
    y = bd["harvest_titre"].values
    cv_rows.append({"day": d, "feature_set": "mean_only (no model)", "n_runs": len(bd), "n_inputs": 0,
                    **scores(y, np.array([np.mean(np.delete(y, i)) for i in range(len(y))]))})
    for name, feats in FEATURE_SETS.items():
        cv_rows.append({"day": d, "feature_set": name, "n_runs": len(bd), "n_inputs": len(feats),
                        **scores(y, loo_predictions(bd, feats))})
cv = pd.DataFrame(cv_rows)
cv.to_csv(OUT / "q2_cross_validation.csv", index=False)
print("\nQ2 leave-one-run-out accuracy at bench scale (titre on the 0-1 scale):")
print(cv.pivot(index="feature_set", columns="day", values="RMSE").round(3).to_string())
print("\nOut-of-sample R2:")
print(cv.pivot(index="feature_set", columns="day", values="Q2_out_of_sample_R2").round(2).to_string())

fig, ax = plt.subplots(figsize=(8.6, 4))
for i, name in enumerate(FEATURE_SETS):
    sub = cv[cv["feature_set"] == name]
    ax.plot(sub["day"], sub["RMSE"], marker="o", lw=2, ms=6, color=PALETTE[i], label=name.replace("_", " "))
ref = cv[cv["feature_set"] == "mean_only (no model)"]
ax.plot(ref["day"], ref["RMSE"], ls="--", color=MUTED, lw=1, label="no model (mean)")
ax.set_xticks(DECISION_DAYS); ax.set_xlabel("decision day"); ax.set_ylabel("prediction error, RMSE (0-1 titre scale)")
ax.set_title("Q2 - How early can harvest titre be predicted? (bench runs, leave-one-run-out)", loc="left", fontsize=9.5, color=INK)
ax.grid(color=GRID); ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(4.6, 9.4)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.text(0.01, 0.01, "Source: Gangadharan et al. (2021) supplementary data, AstraZeneca CHO cultures", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "q2_accuracy_by_day.png", dpi=200)

# %% 6. Lesson 7 checks on the chosen model
feats = FEATURE_SETS[CHOSEN_SET]
bd = bench_all[(bench_all["day"] == CHOSEN_DAY) & bench_all["run"].isin(common_runs)].reset_index(drop=True)
X, y = design(bd, feats), bd["harvest_titre"].values
beta = ols_fit(X, y)
fitted = X @ beta; resid = y - fitted
n, p = X.shape
sigma2 = resid @ resid / (n - p)
XtX_inv = np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(XtX_inv) * sigma2)
tval = beta / se
pval = 2 * stats.t.sf(np.abs(tval), n - p)
tcrit = stats.t.ppf(0.975, n - p)
coef = pd.DataFrame({"coef": beta, "std_err": se, "t": tval, "p_value": pval,
                     "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se}, index=["Intercept"] + feats)
r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
print(f"\nChosen model: {CHOSEN_SET}, day {CHOSEN_DAY}, n = {n} bench runs, in-sample R2 = {r2:.3f}, "
      f"residual SD = {np.sqrt(sigma2):.3f}")
print(coef.round(4).to_string())
coef.to_csv(OUT / "q2_chosen_model_coefficients.csv")

# Variance inflation: how much each coefficient's variance is inflated by its correlated colleagues
vif = {}
for j, fname in enumerate(feats):
    others = [g for g in feats if g != fname]
    Xo = design(bd, others); yj = bd[fname].values
    rj = yj - Xo @ ols_fit(Xo, yj)
    r2j = 1 - rj @ rj / np.sum((yj - yj.mean()) ** 2)
    vif[fname] = 1 / max(1 - r2j, 1e-12)
vif = pd.Series(vif, name="VIF").sort_values(ascending=False)
print("\nVariance inflation (worry above 5 to 10):"); print(vif.round(1).to_string())
vif.to_csv(OUT / "q2_vif.csv")

# Influence: leverage, standardised residuals, Cook's distance
H = X @ XtX_inv @ X.T
lev = np.diag(H)
std_resid = resid / np.sqrt(sigma2 * (1 - lev))
cooks = std_resid ** 2 / p * lev / (1 - lev)
infl = pd.DataFrame({"run": bd["run"], "volume_L": bd["volume_L"], "harvest_titre": y, "fitted": fitted,
                     "std_resid": std_resid, "leverage": lev, "cooks_D": cooks}).sort_values("cooks_D", ascending=False)
print(f"\nMost influential runs (average leverage = {p / n:.3f}):"); print(infl.head(5).round(3).to_string(index=False))
infl.to_csv(OUT / "q2_influence.csv", index=False)

# The four residual charts (same as R's plot(model)), built by hand
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
axs[0, 0].scatter(fitted, resid, s=18, color=PALETTE[0]); axs[0, 0].axhline(0, color=MUTED, lw=0.8)
axs[0, 0].set(title="Residuals vs fitted (linearity)", xlabel="fitted", ylabel="residual")
osm, osr = stats.probplot(std_resid, dist="norm")[0]
axs[0, 1].scatter(osm, osr, s=18, color=PALETTE[0]); axs[0, 1].plot(osm, osm, color=MUTED, lw=0.8)
axs[0, 1].set(title="Normal Q-Q (normal errors)", xlabel="theoretical quantile", ylabel="standardised residual")
axs[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), s=18, color=PALETTE[0])
axs[1, 0].set(title="Scale-location (equal variance)", xlabel="fitted", ylabel="sqrt |standardised residual|")
axs[1, 1].scatter(lev, std_resid, s=18, color=PALETTE[0]); axs[1, 1].axhline(0, color=MUTED, lw=0.8)
axs[1, 1].axvline(3 * p / n, color=PALETTE[1], ls=":", lw=1)
for _, r in infl.head(3).iterrows():
    i = bd.index[bd["run"] == r["run"]][0]
    axs[1, 1].annotate(r["run"], (lev[i], std_resid[i]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK)
axs[1, 1].set(title="Residuals vs leverage (influence)", xlabel="leverage", ylabel="standardised residual")
for a in axs.flat:
    a.grid(color=GRID); a.spines[["top", "right"]].set_visible(False); a.title.set_fontsize(9)
fig.suptitle(f"Lesson 7 health check: {CHOSEN_SET.replace('_', ' ')}, day {CHOSEN_DAY}", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(OUT / "q2_residual_checks.png", dpi=200)

# Cross-check with statsmodels and lmdiag, the course tools (skipped if not installed)
try:
    import statsmodels.formula.api as smf
    safe = {f: f.replace("[", "").replace("]", "") for f in feats}
    sm_df = bd.rename(columns=safe)
    sm_model = smf.ols("harvest_titre ~ " + " + ".join(safe.values()), data=sm_df).fit()
    print("\nstatsmodels cross-check (coefficients should match the table above):")
    print(sm_model.summary())
    try:
        import lmdiag
        lm_fig = lmdiag.plot(sm_model); lm_fig.savefig(OUT / "q2_lmdiag.png", dpi=200)
    except ImportError:
        print("lmdiag not installed: the hand-built residual charts are in outputs/q2_residual_checks.png")
except ImportError:
    print("\nstatsmodels not installed: coefficient table above is computed directly with numpy/scipy.")

# %% 7. Q3: the bench model, unchanged, on the 11 pilot and plant runs
large_all = table[(table["scale"] == "large") & (table["day"] == CHOSEN_DAY)]
large = large_all.dropna(subset=feats).reset_index(drop=True)
miss = sorted(set(large_all["run"]) - set(large["run"]))
if miss:
    print(f"\nLarge runs without every input on day {CHOSEN_DAY} (left out of Q3/Q4): {miss}")
large["pred_bench_model"] = design(large, feats) @ beta
t3 = scores(large["harvest_titre"].values, large["pred_bench_model"].values)
print(f"\nQ3 transfer penalty, {len(large)} large runs (50 to 500 L), day {CHOSEN_DAY}:")
print({k: round(v, 3) for k, v in t3.items()})
print(f"Bench leave-one-run-out RMSE for the same model: "
      f"{cv[(cv['feature_set'] == CHOSEN_SET) & (cv['day'] == CHOSEN_DAY)]['RMSE'].iloc[0]:.3f}")
print(large[["run", "volume_L", "harvest_titre", "pred_bench_model"]].round(3).to_string(index=False))
large.to_csv(OUT / "q3_transfer_large_runs.csv", index=False)

transfer_rows = []
for d in DECISION_DAYS:
    bdd = bench_all[(bench_all["day"] == d) & bench_all["run"].isin(common_runs)]
    lg = table[(table["scale"] == "large") & (table["day"] == d)]
    for name, fs in FEATURE_SETS.items():
        b_ = bdd.dropna(subset=fs); l_ = lg.dropna(subset=fs)
        if len(l_) < 3:
            continue
        bb = ols_fit(design(b_, fs), b_["harvest_titre"].values)
        transfer_rows.append({"day": d, "feature_set": name, "n_large": len(l_),
                              **scores(l_["harvest_titre"].values, design(l_, fs) @ bb)})
pd.DataFrame(transfer_rows).to_csv(OUT / "q3_transfer_all_sets.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 4.6))
ax.scatter(large["pred_bench_model"], large["harvest_titre"], s=42, color=PALETTE[1], edgecolor="white", zorder=3)
large["abs_error"] = (large["pred_bench_model"] - large["harvest_titre"]).abs()
to_label = set(large.nlargest(3, "abs_error")["run"]) | set(large.loc[large["volume_L"] == 500, "run"])
for _, r in large[large["run"].isin(to_label)].iterrows():       # label only the 500 L run and the 3 biggest misses
    ax.annotate(f"{r['run']} ({r['volume_L']:.0f} L)", (r["pred_bench_model"], r["harvest_titre"]),
                xytext=(5, -10), textcoords="offset points", fontsize=7, color=INK)
lim = [0, max(1.0, large[["pred_bench_model", "harvest_titre"]].values.max() * 1.05)]
ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set(xlabel="predicted by the bench model (0-1)", ylabel="actual harvest titre (0-1)")
ax.set_title(f"Q3 - Bench model on pilot and plant runs, day {CHOSEN_DAY}", loc="left", fontsize=9.5, color=INK)
ax.grid(color=GRID); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "q3_transfer_scatter.png", dpi=200)

# %% 8. Q4: calibrate a level shift and a scale factor on k large runs, slopes fixed
# Two numbers cannot be estimated reliably from one or two runs, so k = 1 and 2 fit the level only;
# from k = 3 the level and the scale are both fitted. A level-only curve is reported for every k.
yL, pL, tL = large["harvest_titre"].values, large["pred_bench_model"].values, large["titre_d"].values
nL = len(yL)
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
cal_rows = []
for k in range(0, nL):
    e_cal, e_lvl, e_plant = [], [], []
    combos = [()] if k == 0 else list(combinations(range(nL), k))
    for idx in combos:
        tr = np.array(idx, dtype=int)
        te = np.array([i for i in range(nL) if i not in idx])
        if k == 0:
            e_cal.append(rmse(yL[te], pL[te])); e_lvl.append(e_cal[-1]); continue
        shift = (yL[tr] - pL[tr]).mean()
        e_lvl.append(rmse(yL[te], pL[te] + shift))                       # level shift only
        if k < 3:
            e_cal.append(e_lvl[-1])
        else:
            a, b = np.linalg.lstsq(np.column_stack([np.ones(k), pL[tr]]), yL[tr], rcond=None)[0]
            e_cal.append(rmse(yL[te], a + b * pL[te]))                   # level and scale
            c0, c1 = np.linalg.lstsq(np.column_stack([np.ones(k), tL[tr]]), yL[tr], rcond=None)[0]
            e_plant.append(rmse(yL[te], c0 + c1 * tL[te]))               # comparator: plant runs alone
    cal_rows.append({"k_plant_runs_used": k, "n_splits": len(combos),
                     "RMSE_calibrated_mean": float(np.mean(e_cal)), "RMSE_calibrated_median": float(np.median(e_cal)),
                     "RMSE_calibrated_p10": float(np.percentile(e_cal, 10)),
                     "RMSE_calibrated_p90": float(np.percentile(e_cal, 90)),
                     "RMSE_level_only_mean": float(np.mean(e_lvl)),
                     "RMSE_level_only_p10": float(np.percentile(e_lvl, 10)),
                     "RMSE_level_only_p90": float(np.percentile(e_lvl, 90)),
                     "RMSE_plant_only_mean": float(np.mean(e_plant)) if e_plant else np.nan})
cal = pd.DataFrame(cal_rows)
cal.to_csv(OUT / "q4_calibration_curve.csv", index=False)
print("\nQ4 calibration (every possible choice of k calibration runs, tested on the rest):")
print(cal.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.6, 4))
ax.fill_between(cal["k_plant_runs_used"], cal["RMSE_level_only_p10"], cal["RMSE_level_only_p90"],
                color=PALETTE[0], alpha=0.15, lw=0)
ax.plot(cal["k_plant_runs_used"], cal["RMSE_level_only_mean"], marker="o", lw=2, color=PALETTE[0], label="bench model + level shift (one number)")
ls_ = cal[cal["k_plant_runs_used"] >= 3]
ax.plot(ls_["k_plant_runs_used"], ls_["RMSE_calibrated_mean"], marker="o", lw=1.5, ls="--", color=PALETTE[2], label="bench model + level and scale (two numbers)")
pc_ = cal.dropna(subset=["RMSE_plant_only_mean"])
ax.plot(pc_["k_plant_runs_used"], pc_["RMSE_plant_only_mean"], marker="o", lw=2, color=PALETTE[1], label="plant runs alone (titre to date)")
bench_rmse = cv[(cv["feature_set"] == CHOSEN_SET) & (cv["day"] == CHOSEN_DAY)]["RMSE"].iloc[0]
ax.axhline(bench_rmse, color=MUTED, ls="--", lw=1)
ax.text(0.1, bench_rmse - 0.004, "accuracy at bench scale", va="top", ha="left", fontsize=7.5, color=MUTED)
ax.set(xlabel="plant runs used for calibration (k)", ylabel="error on the other large runs, RMSE (0-1)")
ax.set_title(f"Q4 - How many plant batches does it take? (day {CHOSEN_DAY}; band: 10th to 90th percentile, level shift)",
             loc="left", fontsize=9.5, color=INK)
ax.grid(color=GRID); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "q4_calibration_curve.png", dpi=200)

# %% 9. What to check before you trust any of this
print("""
Before using these numbers:
  - Read J1 to J10 at the top and agree or change each one.
  - Q2: does the best feature set beat the titre-only baseline by a margin that matters for the decision?
  - Section 6: any VIF above 10? Any run with Cook's D far above the rest? Refit without it and show both.
  - Q3/Q4: only 11 large runs, one at 500 L. Read the per-run table, not just the average.
  - Re-run with INPUT_SOURCE = "gapfilled" and PRODUCT_MARKER = False to see how much each choice moves the answer.
""")
