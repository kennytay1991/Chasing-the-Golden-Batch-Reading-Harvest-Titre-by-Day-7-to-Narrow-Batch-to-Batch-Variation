"""
BMX5102 Team 3 (Schrodinger's CHO) - harvest titre prediction with a CHO mass balance
====================================================================================

Data:   AstraZeneca upstream dataset, Gangadharan et al. (2021), Computers & Chemical
        Engineering 146:107226, supplementary workbook "1-s2.0-S0098135421000041-mmc3.xlsx"
        (106 CHO cultures, 4 L to 500 L, every value rescaled 0 to 1 by the authors).

What each section answers (V10 proposal):
    Sections 2-4  build one row per culture per decision day, with mass-balance features
    Section 5     Q2: how well do day-5, day-7 and day-9 data predict harvest titre at bench scale?
                  (the day is when the forecast is MADE; the target is always harvest titre)
    Section 6     Lesson 7 checks on the chosen model (coefficients, VIF, influence, residual plots)
    Section 7     Q3: the bench model taken unchanged to the 11 pilot and plant runs (transfer penalty)
    Section 8     Q4: calibrate only a level shift and a scale factor on k large runs, test on the rest
    Section 9     Kinetic layer: growth, death and product kinetics per run (Luedeking-Piret),
                  pooled Monod tests with and without ammonia/lactate inhibition, apparent yields,
                  bench vs large comparison of the kinetic parameters, and a kinetic forecast of
                  harvest titre used alone and inside a hybrid regression
    Section 10    Rolling forecast: (a) harvest titre forecast made on every day from 3 to 14, and
                  (b) the daily titre curve, from the forecast day to harvest, checked against every
                  titre reading in the file. forecast_daily_titre(run, day) gives the curve for one run
    Section 11    What to check before you trust any of this

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
    J11 Viability is rebuilt from the rescaled VCD/TCD ratio. VCD and TCD were rescaled
        separately, so VCD/TCD = viability x c. The highest ratios (99.5th percentile, about 1.24)
        are taken as PEAK_VIABILITY = 99%, which fixes c. Dead cells = Xv x (1/viability - 1).
    J12 The kinetic forecast learns the BIOLOGY (how cultures typically grow, plateau and die,
        day by day) from bench runs only, and takes each run's own STATE on day d (viable cells,
        titre, specific productivity). Inside cross-validation the typical profile is rebuilt
        without the held-out run, so no run helps predict itself.
    J13 Monod and yield results are exploratory. Feeds top up glucose by unrecorded amounts
        and rescaling hides where true zero is, so half-saturation constants are biased.
        Apparent yields use days 1 to 2 only, before glucose shows the first feed on day 3.
    J14 Titre is measured on 68% of culture-days (about 10 readings per run), and 21 runs have no
        reading before day 6. The mass balance needs one titre reading, so a forecast on day d uses
        only runs with a reading by day d. Section 10 reports two samples: every run available that
        day (n changes), and the same bench runs on every day from 5 to 13 (like for like, as J10).
    J15 The daily titre curve takes its END POINT from the mass-balance regression for that day and
        its SHAPE from the mass balance: titre rises in proportion to the integral of viable cells
        (Luedeking-Piret in Section 9 puts 93% of production there), with future viable cells
        projected from the typical bench growth and death profile. The harvest day is treated as
        known in advance, because it is a planned decision. Two comparators are reported: the mass
        balance alone, and a straight line to the same end point.
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
DECISION_DAYS = [5, 7, 9]          # days the forecast is MADE in Q2 to Q4 (target is always harvest titre)
PLANNED_HARVEST_DAY = 15          # most runs are harvested on day 15 (63 of 103 with a titre)
INPUT_SOURCE = "raw_locf"         # "raw_locf" (default, see J2) or "gapfilled"
PRODUCT_MARKER = True             # see J7
CHOSEN_SET = "mass_balance"       # model used in Sections 6 to 8 (best transfer to large scale in Q3)
CHOSEN_DAY = 7
PEAK_VIABILITY = 0.99             # see J11
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

# %% 9. Kinetic layer (J11 to J13)
# Unstructured CHO kinetics, written as mass balances on viable cells Xv, dead cells Xd and product P:
#     dXv/dt = (mu - kd) * Xv          growth minus first-order death
#     dXd/dt = kd * Xv                 dead cells accumulate (lysis ignored)
#     dP/dt  = alpha * dXv/dt + beta * Xv   Luedeking-Piret: growth-associated + non-growth-associated
#     mu     = mu_max * Glc/(Ks + Glc) * KI_amm/(KI_amm + NH3) * KI_lac/(KI_lac + Lac)   Monod with inhibition
# All quantities are on the authors' 0-1 scales, so parameters compare runs and scales; they are not
# literature values in g/L or cells/mL.
from scipy.optimize import curve_fit

# 9a. Viability and dead cells from the rescaled VCD/TCD ratio (J11)
ratio_all = (raw["VCD"] / raw["TCD"]).where(raw["TCD"] > 0.05)
VIAB_C = float(np.nanpercentile(ratio_all, 99.5) / PEAK_VIABILITY)
print(f"\nKinetic layer: viability scale c = {VIAB_C:.3f} "
      f"(a VCD/TCD ratio of {VIAB_C * PEAK_VIABILITY:.3f} is taken as {PEAK_VIABILITY:.0%} viable)")

def kinetic_frame(run_id):
    """Daily states for one run: viable cells, viability, dead cells, titre, substrates and wastes."""
    s = run_series(run_id)
    k = pd.DataFrame(index=s.index)
    k["Xv"] = s["VCD"].clip(lower=EPS)
    k["viab"] = (s["VCD"] / s["TCD"].clip(lower=EPS) / VIAB_C).clip(0.01, 1.0)
    k["Xd"] = k["Xv"] * (1 / k["viab"] - 1)
    k["P"] = s["titre_raw"]
    for v, name in [("[Glucose]", "glc"), ("[Glutamine]", "gln"), ("[Lactate]", "lac"), ("[NH3]", "nh3")]:
        k[name] = s[v]
    days = k.index.values.astype(float)
    k["IVCD"] = [trapezoid(k["Xv"].loc[:t], days[days <= t]) for t in days]
    return k

def intervals(k):
    """Day-to-day specific rates: net growth, death (from the dead-cell balance) and gross growth."""
    d = k.index.values.astype(float)
    xv0, xv1 = k["Xv"].values[:-1], k["Xv"].values[1:]
    out = pd.DataFrame({"t": d[:-1].astype(int), "dt": np.diff(d), "xavg": (xv0 + xv1) / 2,
                        "dXd": np.diff(k["Xd"].values)})
    out["mu_net"] = np.log(xv1 / xv0) / out["dt"]
    out["kd"] = out["dXd"] / (out["xavg"] * out["dt"])
    out["mu_gross"] = out["mu_net"] + out["kd"].clip(lower=0)
    for c in ["glc", "gln", "lac", "nh3", "viab"]:
        out[c] = k[c].values[:-1]
    return out

KF = {r: kinetic_frame(r) for r in runs.index}
IV = pd.concat([intervals(KF[r]).assign(run=r, scale=runs.loc[r, "scale"]) for r in runs.index], ignore_index=True)
IV = IV[np.isfinite(IV["mu_net"])]

# 9b. Kinetic fingerprint of every run, and bench vs large comparison
def fingerprint(run_id):
    k = KF[run_id]; iv = IV[IV["run"] == run_id]
    peak = int(k["Xv"].idxmax())
    f = {"run": run_id, "peak_day": peak, "peak_Xv": float(k["Xv"].max()), "viab_harvest": float(k["viab"].iloc[-1])}
    early = k.loc[k.index <= min(5, peak)]                                   # exponential phase
    f["mu_exp"] = float(np.polyfit(early.index.values, np.log(early["Xv"].values), 1)[0]) if len(early) >= 3 else np.nan
    post = iv[iv["t"] >= peak]                                               # decline phase
    f["kd_decline"] = float(post["dXd"].sum() / (post["xavg"] * post["dt"]).sum()) if len(post) >= 2 else np.nan
    pm = k.dropna(subset=["P"]); pm = pm[pm.index >= 5]                       # production phase
    f["qP_prod"] = float(np.polyfit(pm["IVCD"], pm["P"], 1)[0]) if len(pm) >= 3 else np.nan
    f["Y_X_glc_app"] = f["Y_lac_glc_app"] = np.nan                           # J13: days 1 to 2 only
    if {1, 2} <= set(k.index) and k.loc[[1, 2], ["glc", "lac"]].notna().all().all():
        dg = k.loc[1, "glc"] - k.loc[2, "glc"]
        if dg > 0.01:
            f["Y_X_glc_app"] = float((k.loc[2, "Xv"] - k.loc[1, "Xv"]) / dg)
            f["Y_lac_glc_app"] = float((k.loc[2, "lac"] - k.loc[1, "lac"]) / dg)
    return f

fp = pd.DataFrame([fingerprint(r) for r in runs.index]).merge(runs[["volume_L", "scale"]], left_on="run", right_index=True)
fp.to_csv(OUT / "k1_kinetic_fingerprints.csv", index=False)
FP_PARAMS = {"mu_exp": "growth rate, exponential phase (per day)", "peak_day": "day of peak viable cells",
             "peak_Xv": "peak viable cell density (0-1)", "kd_decline": "death rate after the peak (per day)",
             "qP_prod": "specific productivity, production phase", "viab_harvest": "viability at harvest",
             "Y_X_glc_app": "apparent cell yield on glucose, days 1-2", "Y_lac_glc_app": "apparent lactate yield on glucose, days 1-2"}
cmp_rows = []
for p_, label in FP_PARAMS.items():
    b = fp.loc[fp["scale"] == "bench", p_].dropna(); b_ex4 = fp.loc[(fp["scale"] == "bench") & (fp["volume_L"] > 4), p_].dropna()
    l = fp.loc[fp["scale"] == "large", p_].dropna()
    pv = stats.mannwhitneyu(b, l, alternative="two-sided").pvalue if len(b) > 2 and len(l) > 2 else np.nan
    cmp_rows.append({"parameter": p_, "meaning": label, "n_bench": len(b), "bench_median": b.median(),
                     "bench_median_excl_4L": b_ex4.median(), "n_large": len(l), "large_median": l.median(),
                     "large_vs_bench_ratio": l.median() / b.median() if b.median() else np.nan, "MannWhitney_p": pv})
fpc = pd.DataFrame(cmp_rows)
fpc.to_csv(OUT / "k1_fingerprint_bench_vs_large.csv", index=False)
print("\nKinetic fingerprint, bench vs large (does the biology change with scale?):")
print(fpc.drop(columns="meaning").round(3).to_string(index=False))

fig, axs = plt.subplots(2, 3, figsize=(10, 6))
for a, p_ in zip(axs.flat, ["mu_exp", "kd_decline", "qP_prod", "peak_day", "peak_Xv", "viab_harvest"]):
    for j, (sc_, col) in enumerate([("bench", PALETTE[0]), ("large", PALETTE[1])]):
        v = fp.loc[fp["scale"] == sc_, p_].dropna().values
        a.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=14 if sc_ == "bench" else 28, color=col, alpha=0.7, edgecolor="white", lw=0.5)
        a.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color=INK, lw=2)
    pv = fpc.loc[fpc["parameter"] == p_, "MannWhitney_p"].iloc[0]
    a.set_xticks([0, 1]); a.set_xticklabels(["bench\n4-5 L", "large\n50-500 L"])
    a.set_title(f"{FP_PARAMS[p_]}\nMann-Whitney p = {pv:.2f}", fontsize=8, color=INK)
    a.grid(axis="y", color=GRID); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Kinetic fingerprint by scale (bar = median). Does the biology transfer?", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(OUT / "k1_fingerprint_by_scale.png", dpi=200)

# Typical day-by-day kinetics, bench vs large
prof_b = IV[IV["scale"] == "bench"].groupby("t")[["mu_net", "kd", "viab"]].median()
prof_l = IV[IV["scale"] == "large"].groupby("t")[["mu_net", "kd", "viab"]].median()
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for a, c, lab in zip(axs, ["mu_net", "kd", "viab"], ["net growth rate (per day)", "death rate (per day)", "viability"]):
    a.plot(prof_b.index, prof_b[c], marker="o", ms=4, lw=2, color=PALETTE[0], label="bench, median of 95 runs")
    a.plot(prof_l.index, prof_l[c], marker="o", ms=4, lw=2, color=PALETTE[1], label="large, median of 11 runs")
    a.axhline(0, color=MUTED, lw=0.6) if c != "viab" else None
    a.set(xlabel="culture day", title=lab); a.title.set_fontsize(9)
    a.grid(color=GRID); a.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False, fontsize=7.5)
fig.suptitle("Typical kinetics by culture day", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(OUT / "k1_kinetic_profiles.png", dpi=200)

# 9c. Pooled kinetic laws on bench runs: Monod tests and Luedeking-Piret
g = IV[(IV["scale"] == "bench") & (IV["t"] <= 12)].dropna(subset=["mu_gross", "glc", "nh3", "lac"])
g = g[np.isfinite(g["mu_gross"])]
y_mu = g["mu_gross"].values
def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
def monod(X, mumax, ks): return mumax * X[0] / (ks + X[0])
def monod_inhib(X, mumax, ks, ka, kl): return mumax * X[0] / (ks + X[0]) * ka / (ka + X[1]) * kl / (kl + X[2])
mon_rows = [{"model": "constant growth rate (reference)", "R2": 0.0, "params": f"mu = {y_mu.mean():.3f}"}]
Xg = (np.clip(g["glc"].values, 1e-4, None), np.clip(g["nh3"].values, 0, None), np.clip(g["lac"].values, 0, None))
try:
    pm1, _ = curve_fit(monod, Xg, y_mu, p0=[0.6, 0.05], bounds=([0, 1e-4], [3, 2]), maxfev=20000)
    mon_rows.append({"model": "Monod on glucose", "R2": r2(y_mu, monod(Xg, *pm1)),
                     "params": f"mu_max = {pm1[0]:.3f}, Ks_glc = {pm1[1]:.4f}" + ("  (Ks at bound)" if pm1[1] <= 1.1e-4 or pm1[1] >= 1.99 else "")})
except RuntimeError:
    mon_rows.append({"model": "Monod on glucose", "R2": np.nan, "params": "did not converge"})
try:
    lo2, hi2 = [0, 1e-4, 1e-3, 1e-3], [3, 2, 5, 5]
    pm2, _ = curve_fit(monod_inhib, Xg, y_mu, p0=[0.8, 0.05, 0.2, 0.5], bounds=(lo2, hi2), maxfev=50000)
    at_bound = [n_ for n_, v_, a_, b_ in zip(["mu_max", "Ks_glc", "KI_amm", "KI_lac"], pm2, lo2, hi2) if v_ <= a_ * 1.1 + 1e-12 or v_ >= b_ * 0.99]
    mon_rows.append({"model": "Monod on glucose with ammonia and lactate inhibition", "R2": r2(y_mu, monod_inhib(Xg, *pm2)),
                     "params": f"mu_max = {pm2[0]:.3f}, Ks_glc = {pm2[1]:.4f}, KI_amm = {pm2[2]:.3f}, KI_lac = {pm2[3]:.3f}"
                               + (f"  (AT BOUND, not identifiable: {', '.join(at_bound)})" if at_bound else "")})
except RuntimeError:
    mon_rows.append({"model": "Monod with inhibition", "R2": np.nan, "params": "did not converge"})
mon = pd.DataFrame(mon_rows); mon.to_csv(OUT / "k2_monod_tests.csv", index=False)
print(f"\nMonod tests on {len(g)} bench day-intervals (gross growth rate, days 1 to 12), in-sample R2:")
print(mon.round(3).to_string(index=False))

lp = []                                                                      # intervals between titre readings
for r in runs.index[runs["scale"] == "bench"]:
    k = KF[r].dropna(subset=["P"])
    for (t1, a1), (t2, a2) in zip(k.iterrows(), list(k.iterrows())[1:]):
        lp.append({"run": r, "dP": a2["P"] - a1["P"], "dX": a2["Xv"] - a1["Xv"], "dI": a2["IVCD"] - a1["IVCD"]})
lp = pd.DataFrame(lp)

def fit_lp(df):
    A = df[["dX", "dI"]].values; yv = df["dP"].values
    coef_ = np.linalg.lstsq(A, yv, rcond=None)[0]
    res = yv - A @ coef_; s2 = res @ res / (len(yv) - 2)
    se_ = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * s2)
    return coef_, se_

(alpha_all, beta_all), (se_a, se_b) = fit_lp(lp)
share_alpha = abs(alpha_all * lp["dX"]).sum() / (abs(alpha_all * lp["dX"]).sum() + abs(beta_all * lp["dI"]).sum())
lp_tab = pd.DataFrame({"term": ["alpha (growth-associated)", "beta (non-growth-associated, = qP)"], "estimate": [alpha_all, beta_all],
                       "std_err": [se_a, se_b], "p_value": [2 * stats.t.sf(abs(alpha_all / se_a), len(lp) - 2), 2 * stats.t.sf(abs(beta_all / se_b), len(lp) - 2)]})
lp_tab.to_csv(OUT / "k2_luedeking_piret.csv", index=False)
print(f"\nLuedeking-Piret on {len(lp)} bench titre intervals: dP = alpha x dXv + beta x dIVCD")
print(lp_tab.round(4).to_string(index=False))
print(f"Share of product formation attributed to the growth-associated term: {share_alpha:.0%}")

# 9d. Kinetic forecast of harvest titre, alone and inside a hybrid regression (J12)
def typical_net_growth(exclude=()):
    sub = IV[(IV["scale"] == "bench") & (IV["dt"] == 1) & ~IV["run"].isin(exclude)]
    return sub.groupby("t")["mu_net"].median()

def lp_alpha(exclude=()):
    return fit_lp(lp[~lp["run"].isin(exclude)])[0][0]

def kinetic_forecast(row, prof, alpha):
    """Integrate dXv/dt = mu_net(t) x Xv from day d to harvest with the typical bench growth/death profile,
    then dP = alpha x dXv + qP x Xv dt with this run's own specific productivity."""
    d = int(row["day"]); xv = [row["VCD_d"]]
    for t in range(d, PLANNED_HARVEST_DAY):
        xv.append(xv[-1] * np.exp(prof.get(t, prof.iloc[-1])))
    ivcd_future = trapezoid(xv, np.arange(d, PLANNED_HARVEST_DAY + 1))
    return row["titre_d"] + row["qP_cum"] * ivcd_future + alpha * (xv[-1] - xv[0])

HYBRID_FEATS = ["P_kin"] + FEATURE_SETS["mass_balance"]     # the kinetic forecast added on top of the mass-balance inputs
kin_rows = []
for d in DECISION_DAYS:
    bd_k = bench_all[(bench_all["day"] == d) & bench_all["run"].isin(common_runs)].reset_index(drop=True)
    yk = bd_k["harvest_titre"].values; nk = len(yk)
    p_kin, p_hyb = np.empty(nk), np.empty(nk)
    for i in range(nk):                                  # leave-one-run-out, profile rebuilt without run i
        held = bd_k.loc[i, "run"]
        prof_i, alpha_i = typical_net_growth(exclude=[held]), lp_alpha(exclude=[held])
        pk_all = bd_k.apply(lambda r_: kinetic_forecast(r_, prof_i, alpha_i), axis=1).values
        p_kin[i] = pk_all[i]
        tr = np.arange(nk) != i
        Xh = np.column_stack([np.ones(nk), pk_all] + [bd_k[f].values for f in HYBRID_FEATS[1:]])
        p_hyb[i] = Xh[i] @ ols_fit(Xh[tr], yk[tr])
    mb = cv[(cv["feature_set"] == "mass_balance") & (cv["day"] == d)].iloc[0]
    kin_rows.append({"day": d, "model": "mass-balance regression (Section 5)", "where": "bench", "RMSE": mb["RMSE"], "Spearman_rank": mb["Spearman_rank"], "bias_mean_error": mb["bias_mean_error"]})
    kin_rows.append({"day": d, "model": "kinetic forecast alone", "where": "bench", **{k_: v for k_, v in scores(yk, p_kin).items() if k_ in ("RMSE", "Spearman_rank", "bias_mean_error")}})
    kin_rows.append({"day": d, "model": "hybrid: kinetics + regression", "where": "bench", **{k_: v for k_, v in scores(yk, p_hyb).items() if k_ in ("RMSE", "Spearman_rank", "bias_mean_error")}})
    # Transfer to the large runs: biology from all bench runs, state from each large run
    prof_all, alpha_full = typical_net_growth(), lp_alpha()
    lg = table[(table["scale"] == "large") & (table["day"] == d)].dropna(subset=["VCD_d", "titre_d", "qP_cum", "mu_net"]).reset_index(drop=True)
    bd_k["P_kin"] = bd_k.apply(lambda r_: kinetic_forecast(r_, prof_all, alpha_full), axis=1)
    lg["P_kin"] = lg.apply(lambda r_: kinetic_forecast(r_, prof_all, alpha_full), axis=1)
    bh = ols_fit(design(bd_k, HYBRID_FEATS), yk)
    lg["P_hybrid"] = design(lg, HYBRID_FEATS) @ bh
    mbs = pd.read_csv(OUT / "q3_transfer_all_sets.csv").query("feature_set == 'mass_balance' and day == @d").iloc[0]
    kin_rows.append({"day": d, "model": "mass-balance regression (Section 5)", "where": "large", "RMSE": mbs["RMSE"], "Spearman_rank": mbs["Spearman_rank"], "bias_mean_error": mbs["bias_mean_error"]})
    for m_, col in [("kinetic forecast alone", "P_kin"), ("hybrid: kinetics + regression", "P_hybrid")]:
        sc_ = scores(lg["harvest_titre"].values, lg[col].values)
        kin_rows.append({"day": d, "model": m_, "where": "large", "RMSE": sc_["RMSE"], "Spearman_rank": sc_["Spearman_rank"], "bias_mean_error": sc_["bias_mean_error"]})
    if d == CHOSEN_DAY:
        lg_chosen = lg.copy()
kin = pd.DataFrame(kin_rows); kin.to_csv(OUT / "k3_kinetic_vs_regression.csv", index=False)
print("\nKinetic forecast and hybrid vs the mass-balance regression (RMSE on the 0-1 titre scale):")
print(kin.pivot_table(index=["where", "model"], columns="day", values="RMSE").round(3).to_string())
print("\nRank correlation (does it order runs correctly?):")
print(kin.pivot_table(index=["where", "model"], columns="day", values="Spearman_rank").round(2).to_string())
lg_chosen[["run", "volume_L", "harvest_titre", "P_kin", "P_hybrid"]].to_csv(OUT / "k3_kinetic_transfer_large_runs.csv", index=False)

# Level-shift calibration of the hybrid on k large runs (same test as Section 8)
yH, pH_ = lg_chosen["harvest_titre"].values, lg_chosen["P_hybrid"].values
cal_k = []
for kk in range(0, len(yH)):
    errs = []
    for idx in ([()] if kk == 0 else combinations(range(len(yH)), kk)):
        te = np.array([i for i in range(len(yH)) if i not in idx]); tr = np.array(idx, dtype=int)
        shift = (yH[tr] - pH_[tr]).mean() if kk else 0.0
        errs.append(rmse(yH[te], pH_[te] + shift))
    cal_k.append({"k_plant_runs_used": kk, "RMSE_hybrid_level_shift_mean": float(np.mean(errs)),
                  "RMSE_p10": float(np.percentile(errs, 10)), "RMSE_p90": float(np.percentile(errs, 90))})
cal_k = pd.DataFrame(cal_k).merge(cal[["k_plant_runs_used", "RMSE_level_only_mean", "RMSE_plant_only_mean"]], on="k_plant_runs_used")
cal_k.to_csv(OUT / "k3_hybrid_calibration.csv", index=False)
print(f"\nLevel-shift calibration on k large runs, day {CHOSEN_DAY} (hybrid vs Section 8 mass-balance model):")
print(cal_k.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for a, wh in zip(axs, ["bench", "large"]):
    for j, m_ in enumerate(["mass-balance regression (Section 5)", "kinetic forecast alone", "hybrid: kinetics + regression"]):
        sub = kin[(kin["where"] == wh) & (kin["model"] == m_)]
        a.plot(sub["day"], sub["RMSE"], marker="o", lw=2, color=PALETTE[[0, 3, 2][j]], label=m_)
    a.set_title("bench runs, leave-one-run-out" if wh == "bench" else "11 large runs, model learned on bench only", fontsize=9, color=INK)
    a.set_xticks(DECISION_DAYS); a.set_xlabel("decision day"); a.grid(color=GRID); a.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("prediction error, RMSE (0-1 titre scale)"); axs[1].legend(frameon=False, fontsize=7.5)
fig.suptitle("Does adding CHO kinetics improve the forecast?", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(OUT / "k3_kinetic_vs_regression.png", dpi=200)

# %% 10. Rolling forecast: harvest titre from every day, and the daily titre curve (J14, J15)
# 10a asks: standing on day d (3 to 14), how well can we forecast HARVEST titre?
# 10b asks: standing on day d, what will titre be on EVERY remaining day up to harvest?
ROLL_DAYS = list(range(3, 15))
MB_FEATS = FEATURE_SETS[CHOSEN_SET]

roll = pd.DataFrame([r for run_id in runs.index for d in ROLL_DAYS if (r := features_at_day(run_id, d)) is not None])
roll = roll.merge(runs, left_on="run", right_index=True)
roll["product_4L"] = (roll["volume_L"] == 4.0).astype(int)
roll = roll.dropna(subset=["harvest_titre"] + MB_FEATS)                  # J14: needs a titre reading by day d
roll = roll[roll["day"] < roll["harvest_day"]].reset_index(drop=True)    # a forecast is made before harvest

# 10a. Harvest forecast made on every day
# Two samples: (1) every bench run with a titre reading by day d, so early days have fewer runs;
# (2) the same bench runs on every day from FIXED_DAYS, so the curve compares days like for like (J10, J14).
FIXED_DAYS = list(range(5, 14))
bench_roll = roll[roll["scale"] == "bench"]
cnt = bench_roll[bench_roll["day"].isin(FIXED_DAYS)].groupby("run")["day"].nunique()
fixed_runs = cnt.index[cnt == len(FIXED_DAYS)]
roll["H_hat"] = np.nan              # harvest forecast: bench = leave-one-run-out, large = bench model unchanged
BETA_BY_DAY, r1_rows = {}, []
for d in ROLL_DAYS:
    b = roll[(roll["day"] == d) & (roll["scale"] == "bench")]
    l = roll[(roll["day"] == d) & (roll["scale"] == "large")]
    if len(b) < 20:
        continue
    y = b["harvest_titre"].values
    p_all = loo_predictions(b, MB_FEATS)
    roll.loc[b.index, "H_hat"] = p_all
    BETA_BY_DAY[d] = ols_fit(design(b, MB_FEATS), y)
    if len(l):
        roll.loc[l.index, "H_hat"] = design(l, MB_FEATS) @ BETA_BY_DAY[d]
    yl, pl = l["harvest_titre"].values, roll.loc[l.index, "H_hat"].values
    rec = {"day": d, "days_left_to_harvest": PLANNED_HARVEST_DAY - d,
           "n_bench_all": len(b), "RMSE_mass_balance_all": rmse(y, p_all),
           "n_bench_fixed": np.nan, "RMSE_mass_balance_fixed": np.nan, "RMSE_titre_only_fixed": np.nan,
           "RMSE_mean_only_fixed": np.nan, "R2_mass_balance_fixed": np.nan,
           "n_large": len(l), "RMSE_large_transfer": rmse(yl, pl) if len(l) else np.nan,
           "Spearman_large": float(stats.spearmanr(yl, pl)[0]) if len(l) > 2 else np.nan}
    if d in FIXED_DAYS:
        bf = b[b["run"].isin(fixed_runs)]
        yf = bf["harvest_titre"].values
        p_mb = loo_predictions(bf, MB_FEATS)
        rec.update({"n_bench_fixed": len(bf), "RMSE_mass_balance_fixed": rmse(yf, p_mb),
                    "RMSE_titre_only_fixed": rmse(yf, loo_predictions(bf, ["titre_d"])),
                    "RMSE_mean_only_fixed": rmse(yf, np.array([np.mean(np.delete(yf, i)) for i in range(len(yf))])),
                    "R2_mass_balance_fixed": scores(yf, p_mb)["Q2_out_of_sample_R2"]})
    r1_rows.append(rec)
r1 = pd.DataFrame(r1_rows)
r1.to_csv(OUT / "r1_harvest_forecast_every_day.csv", index=False)
print(f"\nRolling forecast of HARVEST titre, made on each day (RMSE on the 0-1 titre scale).")
print(f"'all' = every bench run with a titre reading by that day (n changes); "
      f"'fixed' = the same {len(fixed_runs)} bench runs on days {FIXED_DAYS[0]} to {FIXED_DAYS[-1]} (J14):")
print(r1.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
a = axs[0]
fx = r1.dropna(subset=["RMSE_mass_balance_fixed"])
for j, (col, lab) in enumerate([("RMSE_mass_balance_fixed", "mass-balance regression"), ("RMSE_titre_only_fixed", "titre-to-date only"),
                                ("RMSE_mean_only_fixed", "no model (average run)")]):
    a.plot(fx["day"], fx[col], marker="o", lw=2, color=[PALETTE[0], PALETTE[3], MUTED][j], label=lab)
a.set_title(f"same {len(fixed_runs)} bench runs every day, leave-one-run-out", fontsize=9, color=INK)
a.legend(frameon=False, fontsize=7.5)
a = axs[1]
a.plot(r1["day"], r1["RMSE_large_transfer"], marker="o", lw=2, color=PALETTE[1], label="bench model on the 10-11 large runs")
a.plot(fx["day"], fx["RMSE_mass_balance_fixed"], lw=1, ls="--", color=PALETTE[0], label="same model at bench scale")
a.set_title("pilot and plant runs, model learned on bench only", fontsize=9, color=INK)
a.legend(frameon=False, fontsize=7.5)
for a in axs:
    a.set_xlabel("day the forecast is made"); a.grid(color=GRID)
    a.spines[["top", "right"]].set_visible(False)
axs[0].set_xticks(fx["day"]); axs[1].set_xticks(r1["day"])
axs[0].set_ylabel("harvest titre error, RMSE (0-1 scale)")
fig.suptitle("Harvest forecast: as good on day 5 as on day 10, with about half the titre-only error",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.text(0.01, 0.005, "Source: AstraZeneca CHO dataset (Gangadharan et al. 2021). n per day in r1_harvest_forecast_every_day.csv.",
         fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "r1_harvest_forecast_every_day.png", dpi=200)

# 10b. Daily titre curve from day d to harvest (J15)
bench_ids = runs.index[runs["scale"] == "bench"]
PROF_LOO = {r: typical_net_growth(exclude=[r]) for r in bench_ids}       # bench runs never see their own biology
ALPHA_LOO = {r: lp_alpha(exclude=[r]) for r in bench_ids}
PROF_ALL, ALPHA_ALL = typical_net_growth(), lp_alpha()

def daily_curve(row, H_hat, prof, alpha):
    """Titre forecast for every day from d to harvest. Returns days and three curves:
    mass balance alone, mass balance anchored to the regression's harvest forecast, straight line to that forecast."""
    r_, d, T = row["run"], int(row["day"]), int(row["harvest_day"])
    k = KF[r_]
    tb = k["P"].loc[:d].dropna().index[-1]                     # day of the last titre reading
    iv_gap = float(k["IVCD"].loc[d] - k["IVCD"].loc[tb])      # cells already grown since that reading
    xv = [row["VCD_d"]]
    for t in range(d, T):                                     # dXv/dt = mu_net(t) x Xv, typical bench profile
        xv.append(xv[-1] * np.exp(prof.get(t, prof.iloc[-1])))
    xv, days_ = np.array(xv), np.arange(d, T + 1)
    dI = np.concatenate([[0.0], np.cumsum((xv[1:] + xv[:-1]) / 2)])     # future IVCD from day d
    p_mb = row["titre_d"] + row["qP_cum"] * (iv_gap + dI) + alpha * (xv - xv[0])
    shape = (iv_gap + dI) / (iv_gap + dI[-1]) if (iv_gap + dI[-1]) > EPS else np.ones_like(dI)
    p_anch = row["titre_d"] + (H_hat - row["titre_d"]) * shape
    p_line = row["titre_d"] + (H_hat - row["titre_d"]) * (days_ - tb) / max(T - tb, 1)
    return days_, p_mb, p_anch, p_line

CURVES = {"mass balance alone": 0, "regression end point + mass-balance shape": 1, "straight line to regression end point": 2}
traj_rows = []
for _, row in roll.dropna(subset=["H_hat"]).iterrows():
    bench_run = row["scale"] == "bench"
    prof = PROF_LOO[row["run"]] if bench_run else PROF_ALL
    alpha = ALPHA_LOO[row["run"]] if bench_run else ALPHA_ALL
    days_, *curves = daily_curve(row, row["H_hat"], prof, alpha)
    actual = KF[row["run"]]["P"]
    for i, t in enumerate(days_):
        if t <= row["day"] or t not in actual.index or pd.isna(actual.loc[t]):
            continue
        traj_rows.append({"run": row["run"], "scale": row["scale"], "volume_L": row["volume_L"],
                          "forecast_day": int(row["day"]), "target_day": int(t), "days_ahead": int(t - row["day"]),
                          "actual_titre": float(actual.loc[t]),
                          **{name: float(curves[j][i]) for name, j in CURVES.items()}})
traj = pd.DataFrame(traj_rows)
traj.to_csv(OUT / "r2_daily_titre_forecasts.csv", index=False)

def err_table(df, by):
    out = []
    for key, g in df.groupby(by):
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec["n_points"] = len(g)
        for name in CURVES:
            rec[name] = rmse(g["actual_titre"].values, g[name].values)
        out.append(rec)
    return pd.DataFrame(out)

r3 = err_table(traj, ["scale", "days_ahead"])
r3.to_csv(OUT / "r3_daily_error_by_days_ahead.csv", index=False)
r3_day = err_table(traj, ["scale", "forecast_day"])
r3_day.to_csv(OUT / "r3_daily_error_by_forecast_day.csv", index=False)
print(f"\nDaily titre curve: {len(traj)} forecasts checked against actual titre readings "
      f"({traj['run'].nunique()} runs, forecasts made on days {ROLL_DAYS[0]} to {ROLL_DAYS[-1]}).")
print("Overall RMSE by method (0-1 titre scale):")
print(err_table(traj, ["scale"]).round(3).to_string(index=False))
print("\nRMSE by how many days ahead the forecast looks:")
print(r3.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for a, sc_ in zip(axs, ["bench", "large"]):
    sub = r3[(r3["scale"] == sc_) & (r3["n_points"] >= 10)]
    for j, name in enumerate(CURVES):
        a.plot(sub["days_ahead"], sub[name], marker="o", lw=2, color=[PALETTE[3], PALETTE[0], MUTED][j], label=name)
    a.set_title("bench runs, leave-one-run-out" if sc_ == "bench" else "pilot and plant runs, model learned on bench only",
                fontsize=9, color=INK)
    a.set_xlabel("days ahead of the forecast day"); a.grid(color=GRID); a.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("daily titre error, RMSE (0-1 scale)"); axs[0].legend(frameon=False, fontsize=7.5)
fig.suptitle("Daily titre forecast: mass balance alone for the next few days at scale, regression end point beyond", x=0.01, ha="left", fontsize=10, color=INK)
fig.text(0.01, 0.005, "Source: AstraZeneca CHO dataset (Gangadharan et al. 2021). Points with fewer than 10 checks are not plotted.",
         fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "r3_daily_error_by_days_ahead.png", dpi=200)

# Example curves: one typical bench run and the two largest runs, forecasts made on days 5, 7 and 9
big = runs[runs["scale"] == "large"].sort_values("volume_L", ascending=False).index
has_579 = traj[traj["forecast_day"].isin([5, 7, 9])].groupby("run")["forecast_day"].nunique()
cand = [r for r in has_579.index[has_579 == 3] if runs.loc[r, "scale"] == "bench"]
bench_ex = (runs.loc[cand, "harvest_titre"] - runs.loc[bench_ids, "harvest_titre"].median()).abs().idxmin()
examples = [bench_ex] + [r for r in big if r in set(traj["run"])][:2]
fig, axs = plt.subplots(1, len(examples), figsize=(10, 3.8), sharey=True, sharex=True)
for a, r_ in zip(np.atleast_1d(axs), examples):
    act = KF[r_]["P"].dropna()
    a.scatter(act.index, act.values, s=16, color=INK, zorder=3, label="actual titre")
    for j, d in enumerate([5, 7, 9]):
        rw = roll[(roll["run"] == r_) & (roll["day"] == d)].dropna(subset=["H_hat"])
        if rw.empty:
            continue
        rw = rw.iloc[0]
        prof = PROF_LOO[r_] if rw["scale"] == "bench" else PROF_ALL
        alpha = ALPHA_LOO[r_] if rw["scale"] == "bench" else ALPHA_ALL
        days_, _, p_anch, _ = daily_curve(rw, rw["H_hat"], prof, alpha)
        a.plot(days_, p_anch, lw=2, color=PALETTE[j], label=f"forecast made on day {d}")
        a.axvline(d, color=PALETTE[j], lw=0.6, ls=":")
    a.set_title(f"{r_}, {runs.loc[r_, 'volume_L']:g} L", fontsize=9, color=INK)
    a.set_xlabel("culture day"); a.grid(color=GRID); a.spines[["top", "right"]].set_visible(False)
np.atleast_1d(axs)[0].set_ylabel("titre (0-1 scale)"); np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7.5, loc="upper left")
np.atleast_1d(axs)[0].set_xticks(range(0, 19, 3))
fig.suptitle("Daily titre forecast: on track at bench scale, overshoots late in the run at plant scale", x=0.01, ha="left", fontsize=10, color=INK)
fig.text(0.01, 0.005, "Source: AstraZeneca CHO dataset (Gangadharan et al. 2021). Large runs: model learned on bench runs only.",
         fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "r2_example_daily_curves.png", dpi=200)

def forecast_daily_titre(run_id, day):
    """Live use: on `day`, forecast titre for every remaining day to harvest, using the model learned
    on all bench runs for that day. For a bench run this is in-sample, so use it for demos, not for scoring."""
    rw = roll[(roll["run"] == run_id) & (roll["day"] == day)]
    if rw.empty or day not in BETA_BY_DAY:
        return f"{run_id}: no titre reading by day {day}, or day {day} has too few bench runs"
    rw = rw.iloc[0]
    H = float(np.r_[1.0, rw[MB_FEATS].astype(float).values] @ BETA_BY_DAY[day])
    days_, p_mb, p_anch, _ = daily_curve(rw, H, PROF_ALL, ALPHA_ALL)
    out = pd.DataFrame({"day": days_, "forecast_titre": p_anch, "mass_balance_alone": p_mb})
    out["actual_titre"] = [KF[run_id]["P"].get(t, np.nan) for t in days_]
    return out.round(3)

print(f"\nExample of live use, forecast_daily_titre('{big[0]}', {CHOSEN_DAY}):")
print(forecast_daily_titre(big[0], CHOSEN_DAY).to_string(index=False))

# %% 11. What to check before you trust any of this
print("""
Before using these numbers:
  - Read J1 to J15 at the top and agree or change each one.
  - Q2: does the best feature set beat the titre-only baseline by a margin that matters for the decision?
  - Section 6: any VIF above 10? Any run with Cook's D far above the rest? Refit without it and show both.
  - Q3/Q4: only 11 large runs, one at 500 L. Read the per-run table, not just the average.
  - Section 10: choose the decision day from r1 (error vs days left to act), not from three days alone.
    At plant scale the daily curve overshoots late in the run; that is the Q3 bias that Q4 calibrates.
  - Re-run with INPUT_SOURCE = "gapfilled" and PRODUCT_MARKER = False to see how much each choice moves the answer.
""")
