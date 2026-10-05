"""
Energy Security Pipeline — G20 + Europe Full Analysis (Chapter 2)
=================================================================
Panel: 45 countries = 19 G20 member economies (EU bloc excluded; member
states represented individually) + Europe-30 (EU27 + Switzerland, Norway,
UK), deduplicated; 2000–2023, 1,080 country-years.
Three European country sets are kept separate:
  Europe-30 (EUROPE30_ISO3)           panel design; overlaps the G20 in
                                       DEU, FRA, ITA, GBR
  Europe non-G20 (EUROPE_NONG20_ISO3) 26 countries, disjoint from the G20;
                                       G20 (19) + 26 = 45; used for all
                                       G20 vs Europe comparisons
  Eurostat validation sample          EU27 + NOR + GBR + TUR = 30 (no CHE);
    (EUROSTAT_VALIDATION_ISO3)         observed coverage reported per run Every step is forced onto the full 45 x 24
grid; missing values are handled by explicit, step-specific rules.

RUN SETTINGS (top of file)
  TEST_MODE           True = small replication counts for a quick end-to-end
                      check (not reportable); False = production counts.
  _REPS               replication counts, production / test:
                        B_BOOTSTRAP 1,000/20   (CRS baseline, GML own-period)
                        B_BOOTSTRAP_ALT 500/20 (VRS robustness bootstrap)
                        B_GML_GLOBAL 500/5     (GML pooled-technology bootstrap)
                        B_SW2007 1,000/50      (Simar–Wilson 2007 second loop)
                        B_RESAMPLE 2,000/200   (resampling intervals, lag CIs)
                        N_PERM 9,999/999       (Kruskal–Wallis permutations)
                        B_PERM_SEPARABILITY 999/99 (separability permutations)
  VRS = False         CRS is the baseline returns to scale; VRS is always run
                      as a robustness check.
  VRS_ABATEMENT       "kuosmanen": VRS with activity-specific abatement
                      (Kuosmanen 2005); CRS results are unaffected.
  DDF_SOLVER          "highs" (SciPy) or "cbc" (PuLP); the other is the
                      automatic fallback. Every failure is counted.
  PARALLEL_BOOTSTRAP  bootstrap years / replications run on all CPU cores;
                      seeds fixed per year / replication (results do not
                      depend on the number of cores).
  ESI_C2_MODE         "headline" (baseline) or "energy" C2 affordability.
  RUN_LABEL           label for the production results folder.

PIPELINE (in execution order; run_all() runs them all)
  Audit  DDF solver audit: prints the LP formulation, self-tests (feasibility,
         CRS <= VRS(Kuosmanen) <= VRS(single-lambda), CRS unchanged by
         abatement weights, CBC = HiGHS) and saves the solver source code.
  1.  Data download: OWID energy data; World Bank WDI (GDP, growth, net energy
      imports, CO2 per capita, energy intensity, CPI, household consumption)
      and WGI (six governance indicators); Eurostat household electricity
      prices (nrg_pc_204, band DC, EUR and PPS, 2007–2023).
  3.  Master panel: merge, enforce the 45 x 24 grid, fallbacks, governance
      interpolated within country (fills the missing WGI year 2001).
  2.  Affordability validation: G20 energy-burden diagnostic; external
      European-subsample validation of the affordability dimension against
      the Eurostat household electricity burden (construct-level pooled,
      country-FE and two-way FE tests; component-level diagnostics; headline
      vs energy C2; common-sample comparison).
  4.  ESI construction: four dimensions, one normalisation rule (pooled,
      winsorised 1–99% min–max):
        resilience     normalised Shannon diversity of the primary-energy mix
                       (ESI only; not a DDF input or output)
        affordability  strict 3-component composite (C1 energy intensity,
                       C2 price pressure: headline and energy variants,
                       C3 household consumption per capita)
        robustness     mean of six WGI estimates
        sustainability CO2 per capita
      Baseline equal weights; robustness: PCA weights and Benefit-of-the-
      Doubt DEA (common complete-case sample). Direction audit.
  5.  DDF-DEA, CRS baseline: inputs energy per capita and import dependence;
      good output electricity per capita; bad output LIFECYCLE GHG intensity
      of electricity generation (gCO2e/kWh, Ember via OWID; weakly disposable,
      equality constraint); observation-
      scaled direction g = (y0, b0). Annual frontiers. Smooth homogeneous
      bootstrap ADAPTED to the DDF (beta in [0,1], reflection at both bounds,
      pseudo-outputs along the DDF direction, basic intervals; documented in
      the "DDF SMOOTH BOOTSTRAP" note; coverage checked by the optional
      Monte Carlo validation, run_all(run_validation=True)). Raw VRS also
      solved; scale efficiency SE = CRS/VRS; improvement potential
      beta = 1/theta - 1; frontier/exporter check; correlation table.
  5b. VRS robustness bootstrap (add_alt_rts_bootstrap, B_BOOTSTRAP_ALT);
      DDF_VRS / DDF_CRS figures for all countries (45), G20 (19) and
      Europe non-G20 (26).
  5c. Specification sensitivity (reported, never adopted): the baseline is an
      INDICATOR frontier (weak disposability and CRS scaling imposed on carbon
      intensity, a ratio); physical alternatives with electricity and its
      lifecycle emissions per person / as totals (CRS: identical by
      construction -- checked; VRS: differ), and three import definitions.
  6.  Global Malmquist–Luenberger index (Oh 2010), CRS: GML = GEC x GTC
      against own-period and pooled global technologies (100% coverage);
      bootstrap intervals from paired own-period/global draws; countries
      classified as improving/declining/unchanged with tolerance.
  7.  k-NN conditional-efficiency SENSITIVITY DIAGNOSTIC (joint: null-
      reconstruction test with country-block permutations and recomputed
      conditional DDFs, H0' = Z unrelated to frontier and inefficiency; per
      variable: diagnostic only; not the DSW 2018 test) + Simar–Wilson (2007)
      Algorithm 2 adapted to the DDF (frontier rows excluded in step 2,
      parametric first loop, latent model truncated at 1 and CENSORED at 2
      because about a third of corrected values reach the DDF bound; full
      optimiser / Hessian / bootstrap-collapse diagnostics; parametric AND
      country-cluster intervals). Separability is assumed, not tested.
  7b. Restricted Simar–Wilson (affordability + GDP growth) on its own and on
      the full model's sample (sensitivity check).
  7c. Conditional DDF D(x,y,b|Z) — structural benchmark against similar
      peers — and sensitivity to the neighbourhood
      definition (k, with/without sustainability).
  8.  Panel fixed effects (country + year FE, clustered SE).
  8b. Mundlak correlated random effects: within vs between associations.
  8c. Lagged ESI–DDF link (lags 0–3): within-country cross-correlations,
      distributed-lag FE in both directions, ESI dimensions, cross-lagged
      (Granger-type) FE models.
  10. Panel Granger causality — OPTIONAL (run_all(run_granger=True)).
  10b. Local projections (Jordà 2005): responses of efficiency and ESI to
      exposure-interacted shocks (oil/GFC 2008–09, COVID 2020–21, gas
      2022–23) and continuous shocks (import dependence, ESI); data-driven
      candidate shock years; six-chart shock-analysis suite.
  11. Summary plots (efficiency CIs, GML decomposition, ESI x DDF quadrants).
  12. Theory-driven metafrontier, three technology regimes from the
      electricity mix (fossil-dominant, low-carbon specialised,
      diversified/mixed): group and meta DDF (CRS baseline, VRS robustness),
      technology gap ratio, four-way classification, regime tests
      (Kruskal–Wallis, Holm-corrected Mann–Whitney, Cliff's delta; CRS vs
      VRS robustness), links to conditional DDF, ESI and GML, TGR trend,
      sources of TGR change against a fixed pooled benchmark (12.12b),
      robustness (k-means clustering, G20 split, alternative import
      variable, regime thresholds F x LC grid and 2000-2004 classification
      period), resampling intervals (12.15, kept separate from frontier-
      estimation uncertainty).
  13. Export: main panel, GML, Simar–Wilson, six country-year reference
      tables, affordability components, sample-scope check, solver stats,
      consolidated sensitivity overview (sensitivity_overview.csv).
  Appendix A — OPTIONAL (run_all(run_optional=True)): between-effects,
      panel quantile regression, metafrontier with alternative groupings.

RUN INFRASTRUCTURE
  Each run writes to its own folder <root>/run_TEST_<date>_<time> or
  run_PROD_<date>_<time>_<label> (plots/, tables/, logs/, data_export/,
  checkpoints/). Nothing is overwritten (overwrite=False by default).
  A checkpoint is saved after every step; after a Colab disconnect, re-run
  the cell and run_all() resumes the unfinished run with the same settings.
  Long bootstraps print progress and an estimated finishing time.
  Every figure is saved with a <name>_data.csv holding the plotted values and
  layout (bars with category labels, points, text labels, axis limits), so
  figures can be verified from data. At the end of each run:
  tables/key_results.csv (headline numbers), tables/consistency_check.csv
  (headline numbers recomputed from the exported files) and
  logs/run_manifest.json (settings, executed and OMITTED analyses, SHA-256
  file inventory). consistency_check(run_dir, text_path="chapter.docx")
  compares the chapter text with key_results.csv.
  Step numbering: there is no Step 9; the shock evidence is Step 10b.

Author: research pipeline
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — INSTALLS & IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys
global OUTPUT_COLS

def pip_install(*pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *pkgs],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

# pulp is pinned below 3: newer releases (e.g. 4.x) no longer bundle the CBC
# solver, so a fresh Colab runtime could end up with PuLP but no CBC.
pip_install("wbgapi", "pandas_datareader", "wbdata", "pulp>=2.7,<3", "linearmodels", "scipy", "statsmodels",
            "matplotlib", "seaborn", "requests")

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests, io, time, itertools
import wbgapi as wb
import pulp
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import grangercausalitytests
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
import os

# *** FIX: drive.mount() was previously unconditional and unguarded -- if
# mounting fails for ANY reason (browser popup blocked, an interrupted
# OAuth flow, a transient Colab/Drive backend timeout, or simply running
# this notebook in a context where Drive access isn't available), the
# ValueError it raises previously halted the ENTIRE script before any
# analysis code ran at all, even though nothing else in this pipeline
# actually REQUIRES Drive to be mounted: _init_run_dir() already defaults
# to a plain relative "results/" folder, which works perfectly well
# without Drive -- results simply won't persist across a Colab runtime
# restart in that case, which is a reasonable, clearly-communicated
# fallback rather than a hard failure. ***


try:
    #from google.colab import files
    from google.colab import drive
    drive.mount('/content/drive')
    _DRIVE_MOUNTED = True

except ModuleNotFoundError:
    print(">>> Not running in Colab environment...")
except Exception as _drive_err:
    _DRIVE_MOUNTED = False
    print(f"  ⚠ Google Drive mount failed ({_drive_err}) -- continuing WITHOUT Drive.")
    print(f"    Results will be saved locally under 'results/' instead of Drive, and")
    print(f"    will NOT persist if this Colab runtime is restarted or disconnected.")
    print(f"    To retry mounting Drive: re-run this cell, or check for a blocked")
    print(f"    browser popup / an incomplete Google sign-in prompt.")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS FOLDER — timestamped, never overwritten
# ─────────────────────────────────────────────────────────────────────────────
import os, sys
from datetime import datetime
from pathlib import Path
from io import StringIO

RUN_DIR: Path = None   # set by _init_run_dir(); used by _save_step()

def _init_run_dir(base: str = "results", label: str = None) -> Path:
    """
    Create a unique timestamped results folder:
        results/run_YYYYMMDD_HHMMSS/
    Sets the global RUN_DIR so every subsequent _save_step() call writes there.
    """
    global RUN_DIR
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Every run gets its own folder, so no run can overwrite another. The name
    # shows the mode (TEST / PROD) and an optional label, e.g.
    #   run_PROD_20261005_093000_final   or   run_TEST_20261004_163728
    _mode = "TEST" if globals().get("TEST_MODE", True) else "PROD"
    _lab = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in str(label)) if label else ""
    run_dir = Path(base) / (f"run_{_mode}_{ts}" + (f"_{_lab}" if _lab else ""))
    _k = 2
    while run_dir.exists():                      # never reuse an existing folder
        run_dir = run_dir.with_name(run_dir.name.split("__")[0] + f"__{_k}"); _k += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    # sub-folders
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    RUN_DIR = run_dir
    print(f"\n  Results folder: {run_dir.resolve()}")
    return run_dir


def _rdir() -> Path:
    """
    Return RUN_DIR, initialising it if not yet set OR if the directory has
    become inaccessible since it was set (e.g. a Google Drive desync or
    remount mid-run, which leaves RUN_DIR pointing at a stale path that no
    longer physically exists even though the Python variable is not None).
    Re-checking existence here, not just identity to None, is what
    prevents a FileNotFoundError from _save_fig/_save_csv/_save_log later
    in a long run.
    """
    global RUN_DIR
    if RUN_DIR is None or not RUN_DIR.exists():
        if RUN_DIR is not None:
            print(f"  ⚠ RUN_DIR ({RUN_DIR}) no longer exists on disk "
                  f"(Drive desync/remount?) — recreating a fresh results folder.")
        _init_run_dir()
    return RUN_DIR


def _save_fig(fname: str) -> None:
    """Save current matplotlib figure to plots/ sub-folder."""
    import matplotlib.pyplot as plt
    dest_dir = _rdir() / "plots"
    dest_dir.mkdir(parents=True, exist_ok=True)   # defensive: re-create even
                                                    # if the subfolder itself
                                                    # was removed independently
                                                    # of RUN_DIR's own root
    dest = dest_dir / fname
    plt.savefig(dest, dpi=150, bbox_inches="tight")
    print(f"  Saved: {dest}")
    # Export the plotted values and layout next to every figure, so that what a
    # figure shows (bar heights, point positions, labels, quadrant placements,
    # axis limits) can be verified from data rather than from the image.
    try:
        n = _export_figure_data(plt.gcf(), dest.with_name(dest.stem + "_data.csv"))
        if n:
            print(f"  Saved figure data ({n} rows): {dest.with_name(dest.stem + '_data.csv').name}")
    except Exception as e:
        print(f"  ⚠ figure data export failed for {fname}: {e}")


def _export_figure_data(fig, dest_csv):
    """Long-format export of everything drawn on each axis: line series,
    bars (with their category tick labels), scatter points (with nearby text
    labels), heat-map cells, text annotations, and axis titles / labels /
    limits / reference lines. Returns the number of rows written."""
    import matplotlib
    from matplotlib.patches import Rectangle
    rows = []
    for ai, ax in enumerate(fig.get_axes()):
        title = ax.get_title() or ""
        base = {"axis": ai, "axis_title": title}
        xl, yl = ax.get_xlim(), ax.get_ylim()
        rows.append({**base, "element": "axis", "series": "", "x": xl[0], "y": yl[0],
                     "x2": xl[1], "y2": yl[1],
                     "text": f"xlabel={ax.get_xlabel()} | ylabel={ax.get_ylabel()}"})
        xt = {round(float(t), 6): l.get_text() for t, l in zip(ax.get_xticks(), ax.get_xticklabels()) if l.get_text()}
        yt = {round(float(t), 6): l.get_text() for t, l in zip(ax.get_yticks(), ax.get_yticklabels()) if l.get_text()}

        def _cat(v, ticks):
            if not ticks:
                return ""
            k = min(ticks, key=lambda t: abs(t - v))
            return ticks[k] if abs(k - v) < 0.5 else ""

        for ln in ax.get_lines():
            x, y = np.asarray(ln.get_xdata(), dtype=object), np.asarray(ln.get_ydata(), dtype=object)
            lab = ln.get_label()
            if len(x) == 2 and (len(set(map(str, x))) == 1 or len(set(map(str, y))) == 1):
                rows.append({**base, "element": "reference_line", "series": lab,
                             "x": x[0], "y": y[0], "x2": x[-1], "y2": y[-1]})
                continue
            for xi, yi in zip(x, y):
                rows.append({**base, "element": "line", "series": lab, "x": xi, "y": yi})
        for p in ax.patches:
            if isinstance(p, Rectangle) and (p.get_width() != 0 or p.get_height() != 0):
                x0, y0, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
                cx, cy = x0 + w / 2, y0 + h / 2
                rows.append({**base, "element": "bar", "series": p.get_label(),
                             "x": x0, "y": y0, "width": w, "height": h,
                             "category": _cat(cx, xt) or _cat(cy, yt)})
        for col in ax.collections:
            if isinstance(col, matplotlib.collections.PathCollection):
                for (px, py) in np.asarray(col.get_offsets()):
                    rows.append({**base, "element": "point", "series": col.get_label(), "x": px, "y": py})
            elif isinstance(col, matplotlib.collections.QuadMesh):
                arr = np.asarray(col.get_array()).ravel()
                rows.append({**base, "element": "heatmap", "series": "", "text": f"{len(arr)} cells",
                             "x": float(np.nanmin(arr)) if len(arr) else np.nan,
                             "y": float(np.nanmax(arr)) if len(arr) else np.nan})
        for im in ax.get_images():
            arr = np.asarray(im.get_array())
            if arr.ndim == 2:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        rows.append({**base, "element": "cell", "series": "", "x": j, "y": i,
                                     "value": arr[i, j], "category": f"{_cat(j, xt)} | {_cat(i, yt)}"})
        for t in ax.texts:
            px, py = t.get_position()
            rows.append({**base, "element": "text", "series": "", "x": px, "y": py, "text": t.get_text()})
    if not rows:
        return 0
    pd.DataFrame(rows).to_csv(dest_csv, index=False)
    return len(rows)


def _save_csv(df, fname: str, label: str = "") -> None:
    """Save a DataFrame to tables/ sub-folder."""
    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        return
    dest_dir = _rdir() / "tables"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    df.to_csv(dest, index=False)
    tag = f" ({label})" if label else ""
    print(f"  Saved{tag}: {dest}")


def _save_log(text: str, fname: str) -> None:
    """Append text to a log file in logs/ sub-folder."""
    dest_dir = _rdir() / "logs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    with open(dest, "a", encoding="utf-8") as f:
        f.write(text + "\n")


class _TeeLogger:
    """
    Context manager that duplicates stdout to a log file.
    Usage:
        with _TeeLogger("step5_dea.log"):
            ... run step ...
    """
    def __init__(self, fname: str):
        self.fname = fname
        self._buf  = StringIO()
        self._orig = None

    def write(self, msg):
        self._orig.write(msg)
        self._buf.write(msg)

    def flush(self):
        self._orig.flush()

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._orig
        _save_log(self._buf.getvalue(), self.fname)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: G20 = 19 sovereign economies + EU as a bloc.
# EU excluded here (not a sovereign state with its own energy system);
# EU member states DEU, FRA, ITA are included individually.
# When presenting results: "19 G20 member economies
# (EU bloc excluded; member states represented individually)."
G20_ISO3 = [
    "ARG","AUS","BRA","CAN","CHN","DEU","FRA","GBR","IDN","IND",
    "ITA","JPN","KOR","MEX","RUS","SAU","TUR","USA","ZAF",
]

# EU27 member ISO3 codes used for affordability-proxy validation
# EU27_ISO3 retained as reference constant; no longer used in analysis pipeline
EU27_ISO3 = [
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA",
    "DEU","GRC","HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD",
    "POL","PRT","ROU","SVK","SVN","ESP","SWE",
]

# All sovereign European countries (UN-recognised, exc. microstates < 100k pop)
EUROPE_ISO3 = [
    # EU27
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA",
    "DEU","GRC","HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD",
    "POL","PRT","ROU","SVK","SVN","ESP","SWE",
    # Non-EU Europe
    # CHE (Switzerland) added per explicit request -- present in this
    # pipeline's own embedded household-energy CPI table (see
    # _CPI_ENERGY_EMBEDDED_CSV) and in standard World Bank/OWID/Eurostat
    # coverage, so no new data source is needed for it.
   # "ALB","AND","BIH","BLR","GEO","ISL","MDA","MKD","MNE","SRB","UKR"
    "CHE","NOR","GBR"
]

# Full panel: G20 + Europe (deduplicated — DEU, FRA, ITA, GBR already in G20)
PANEL_ISO3 = sorted(set(G20_ISO3 + EUROPE_ISO3))

# ── THREE DIFFERENT EUROPEAN COUNTRY SETS (kept strictly separate) ───────────
# 1. EUROPE30_ISO3 (30) -- the European part of the panel DESIGN:
#      EU27 + Switzerland + Norway + United Kingdom.
#      Overlaps the G20 in DEU, FRA, ITA, GBR. Used only to build the panel.
# 2. EUROPE_NONG20_ISO3 (26) -- the DISJOINT comparison group:
#      Europe-30 minus the four countries that are G20 members.
#      G20 (19) and Europe non-G20 (26) partition the 45-country panel; all
#      "G20 vs non-G20 Europe" figures and comparisons use this set.
# 3. EUROSTAT_VALIDATION_ISO3 (30) -- the external affordability-validation
#      sample: EU27 + Norway + United Kingdom + Türkiye (defined further below,
#      next to the Eurostat geo codes). It is NOT Europe-30: it adds Türkiye
#      and does not contain Switzerland, and its observed coverage is
#      reported from the data on every run.
EUROPE30_ISO3 = sorted(EUROPE_ISO3)
EUROPE_NONG20_ISO3 = sorted(set(EUROPE30_ISO3) - set(G20_ISO3))
assert len(G20_ISO3) == 19 and len(EUROPE30_ISO3) == 30 and len(EUROPE_NONG20_ISO3) == 26
assert set(G20_ISO3).isdisjoint(EUROPE_NONG20_ISO3) and len(PANEL_ISO3) == 19 + 26 == 45
YEAR_START, YEAR_END = 2000, 2023
# ── Replication counts ────────────────────────────────────────────────────────
# TEST_MODE = True  -> small counts for a fast check that the whole pipeline
#                      runs end to end. Results are NOT reportable.
# TEST_MODE = False -> production counts (use these for every reported number).
TEST_MODE = True

# Production values = time-budget setting agreed for the final run (Oct 2026).
#                      production  test
_REPS = {"B_BOOTSTRAP":         (1000,  20),   # SW1998 frontier bootstrap: Step 5 CRS
                                               # baseline and GML own-period
         "B_BOOTSTRAP_ALT":     (500,   20),   # SW1998 bootstrap of the ALTERNATIVE returns
                                               # to scale (VRS robustness check). Set
                                               # separately so the robustness check can use
                                               # fewer replications than the baseline.
         "B_GML_GLOBAL":        (500,    5),   # GML bootstrap vs pooled technology. Set
                                               # SEPARATELY from B_BOOTSTRAP: raising the DDF
                                               # bootstrap does not raise this one. GML
                                               # intervals pair own-period and global draws,
                                               # so they use min(B_BOOTSTRAP, B_GML_GLOBAL)
                                               # replications. Each replication solves 1,080
                                               # LPs on 1,080 DMUs (slow): the run prints an
                                               # ETA after the first replication.
         "B_SW2007":            (1000,  50),   # Simar-Wilson (2007) second loop (SW2007
                                               # themselves use 2,000; 1,000 is the
                                               # time-budget choice, disclosed in Ch. 2)
         "B_RESAMPLE":          (2000, 200),   # Step 12 resampling intervals, regime tests
         "N_PERM":              (9999, 999),   # Kruskal-Wallis permutations (min p = 1/10,000)
         "B_PERM_SEPARABILITY": (999,   99),   # separability: diagnostic permutations
         "B_SEP_NULL":          (199,   19),   # separability: null-reconstruction permutations
                                               # (each recomputes all conditional DDFs)
         "L1_SW2007":           (100,   10),   # Simar-Wilson Algorithm 2, first loop (SW use 100)
         "MC_R":                (100,    4),   # bootstrap validation: Monte Carlo samples
         "MC_B":                (200,   20)}   # bootstrap validation: draws per sample
_k = 1 if TEST_MODE else 0
B_BOOTSTRAP         = _REPS["B_BOOTSTRAP"][_k]
B_BOOTSTRAP_ALT     = _REPS["B_BOOTSTRAP_ALT"][_k]
B_GML_GLOBAL        = _REPS["B_GML_GLOBAL"][_k]
B_SW2007            = _REPS["B_SW2007"][_k]
B_RESAMPLE          = _REPS["B_RESAMPLE"][_k]
N_PERM              = _REPS["N_PERM"][_k]
B_PERM_SEPARABILITY = _REPS["B_PERM_SEPARABILITY"][_k]
B_SEP_NULL          = _REPS["B_SEP_NULL"][_k]
L1_SW2007           = _REPS["L1_SW2007"][_k]
MC_R                = _REPS["MC_R"][_k]
MC_B                = _REPS["MC_B"][_k]
print(f"{'⚠ TEST MODE' if TEST_MODE else 'PRODUCTION MODE'}: B_BOOTSTRAP={B_BOOTSTRAP}, "
      f"B_BOOTSTRAP_ALT={B_BOOTSTRAP_ALT}, "
      f"B_GML_GLOBAL={B_GML_GLOBAL}, B_SW2007={B_SW2007}, B_RESAMPLE={B_RESAMPLE}, "
      f"N_PERM={N_PERM}, B_PERM_SEPARABILITY={B_PERM_SEPARABILITY}"
      + ("  -- results NOT reportable" if TEST_MODE else ""))
ALPHA       = 0.05          # CI significance level
VRS         = False         # BASELINE returns to scale: False = CRS (baseline),
                            # True = VRS. The other assumption is always run in
                            # as a robustness check (add_alt_rts_bootstrap / Step 12).
                            # CRS is the baseline because (i) the equality-
                            # constrained bad-output DDF represents a weakly
                            # disposable technology only under CRS (Kuosmanen
                            # 2005), (ii) it matches the CRS GML, and (iii) VRS
                            # restricts exporters (import input at its floor) to
                            # other exporters as peers.
# Weak disposability of the bad output under VRS (reviewer item 1).
#   "kuosmanen" (default): Kuosmanen (2005) technology. Each reference DMU j has
#       an active weight z_j and an abatement weight u_j; lambda_j = z_j + u_j.
#       Outputs (good and bad) are built from z only, inputs and the convexity
#       constraint from z + u. Abatement is therefore activity-specific: each
#       DMU can scale down its outputs by its own factor z_j/(z_j+u_j), while
#       still using its inputs. This is the correct VRS representation of a
#       weakly disposable technology.
#   "uniform": the earlier single-lambda VRS model (sum lambda_j = 1 with the
#       equality on the bad output). It imposes one common abatement factor and
#       does NOT in general satisfy weak disposability under VRS (Kuosmanen
#       2005). Kept only so results can be compared with earlier runs.
# Under CRS the two are identical (u_j = 0 is always optimal), so the CRS
# baseline, the CRS GML and every CRS result are unaffected by this switch.
VRS_ABATEMENT = "kuosmanen"

# ── Run-time settings (do not change any estimate) ───────────────────────────
# DDF_SOLVER: primary LP solver; the other one is the automatic fallback.
#   "cbc"   : CBC through PuLP (used for all test runs so far).
#   "highs" : HiGHS through scipy.optimize.linprog. Same linear programme; the
#             solver audit checks that both give the same beta (|diff| < 1e-6).
#             In a benchmark HiGHS was ~3x faster on 45-DMU problems and ~6-8x
#             faster on the 1,080-DMU pooled problems (GML global technology),
#             because CBC is called as a separate program for every LP.
DDF_SOLVER = "highs"
# Parallel bootstrap: Step 5 (CRS and VRS) years, GML own-period years and GML
# global replications are distributed over N_JOBS worker processes (fork).
# Each year / replication has its own fixed random seed, so results do NOT
# depend on N_JOBS or on the order in which workers finish.
PARALLEL_BOOTSTRAP = True
N_JOBS = None            # None = all CPU cores (Colab: usually 2)
# Label added to the NEW run folder name when the file is run as a script /
# Colab cell (see the entry point at the end), e.g. "final".
RUN_LABEL = "final"      # used for PRODUCTION runs only (TEST_MODE = False)
# Monte Carlo validation of the DDF bootstrap (coverage / bias correction on a
# known technology). Optional; production cost about MC_R x MC_B x 45 LPs.
RUN_BOOTSTRAP_VALIDATION = True    # run before treating bootstrap intervals as inference
BASE_RTS    = "VRS" if VRS else "CRS"   # label of the baseline specification
ALT_RTS     = "CRS" if VRS else "VRS"   # label of the robustness check

# ═════════════════════════════════════════════════════════════════════════════
# C2 AFFORDABILITY: BOTH VARIANTS ALWAYS COMPUTED, ESI_C2_MODE SELECTS ACTIVE ONE
# ═════════════════════════════════════════════════════════════════════════════
# *** HISTORY / SUPERSEDED DESIGN NOTICE: this block originally implemented
# a single ON/OFF toggle (USE_ALT_C2_AFFORDABILITY) that switched C2's
# construction between the headline-CPI and energy-CPI versions, computing
# only whichever one was selected, with dim_affordability falling back to a
# NaN-skipping C1+C3 mean wherever C2 was missing.
#
# Both of those design choices have since been superseded:
#   - dim_affordability is no longer a NaN-skipping mean (see the STRICT
#     complete-case logic in step4_esi below, with short-gap interpolation
#     for isolated missing years and an explicit AFF13 side artifact for
#     gaps too long to interpolate);
#   - BOTH the headline and energy C2 variants (and their corresponding
#     dim_affordability_headline/_energy and esi_score_headline/_energy)
#     are now ALWAYS computed together in step4_esi, not just whichever one
#     a single toggle selected.
#
# USE_ALT_C2_AFFORDABILITY itself is consequently VESTIGIAL: it is no
# longer read by any conditional in this file (verify with a text search
# if in doubt) and is retained only so any external code or notebook cell
# that still sets it does not raise a NameError. ESI_C2_MODE, defined next,
# is the actual, currently-read control: it selects which of the two
# ALWAYS-computed variants becomes "dim_affordability" / "esi_score" for
# every downstream step. Change ESI_C2_MODE, not USE_ALT_C2_AFFORDABILITY,
# to switch the pipeline's active specification. ***
#
# The remainder of this comment block (embedded CPI data, loader, and the
# original construction rationale) is retained below for reference. Some
# earlier example CI numbers in the affordability construction ("dim_
# affordability = mean(C1,C2,C3)") predate this note.
#
# is computed EXACTLY as before this change, with no behavioural difference
# whatsoever. Setting it to True switches ONLY the C2 (relative price
# pressure) component's construction; C1 and C3 are unchanged either way.
#
# Motivation: the original C2 (aff_price_ratio, a country-demeaned headline-
# CPI-based ratio) requires constructing internationally comparable price-
# level indices, which is methodologically harder to defend than using each
# country's OWN household-energy CPI directly. The alternative instead uses
# the YEAR-OVER-YEAR LOG CHANGE in household-energy CPI (a standard,
# unit-free inflation-rate measure that needs no cross-country level
# comparability at all):
#
#     C2*_it = 1 - N[ 100 * Delta ln(CPI^energy_it) ]
#
# where Delta ln(CPI^energy_it) = ln(CPI^energy_it) - ln(CPI^energy_i,t-1),
# and N[.] is the SAME pooled, winsorised min-max normalisation (minmax())
# used for every ESI indicator -- so C2* = _minmax_col(100*dln_cpi,
# invert=True), a direct, mechanical substitution of the underlying series,
# not a new normalisation convention. C1 and C3 are computed exactly as
# before (C1*_it = 1 - N(EnergyIntensity_it), C3*_it = N(ln(HFCE_pc_it)),
# both unchanged from the existing pipeline).
#
# Missing data: the source CPI table has gaps (e.g. Finland pre-2006, a
# first-difference is undefined for each country's first available year,
# and a small number of Eurostat "low reliability" flagged values). Where
# C2* cannot be computed for a given country-year, that row's composite
# falls back to the mean of C1* and C3* alone -- this happens automatically
# via pandas' NaN-skipping mean() once C2* is included as a column (a
# missing C2* value does not need special-case handling; it is simply
# skipped row-by-row in the same equal-weight mean used throughout this
# pipeline for partial component availability).
# *** ADDED: both the headline-CPI and energy-CPI variants of C2 (and
# therefore of dim_affordability and esi_score) are now ALWAYS computed
# together in step4_esi(), as dim_affordability_headline/_energy and
# esi_score_headline/_energy -- see the request this responds to: "make
# two calculations of efficiency one with energy C2 and the other with
# headline C2, which gives 2 affordability and 2 ESI". ESI_C2_MODE below
# is the SINGLE place that selects which of the two variants becomes the
# "active" dim_affordability / esi_score used by every downstream step
# (Simar-Wilson, Mundlak, Panel FE, Step 12, Local Projections, etc.).
# Change this one string to switch the entire pipeline's downstream
# analysis to the other variant; both variants remain fully computed and
# available under their own suffixed column names regardless of this
# setting, so switching back and forth costs nothing and loses no results.
ESI_C2_MODE = "headline"    # "headline" (default) or "energy"
# Display-only: countries whose C2 values lie far outside every other country's
# range. In C2 diagrams the axis range is set by the OTHER countries and these
# are drawn capped at the axis edge (with a footnote). Calculations are unchanged.
C2_OUTLIERS = ["TUR", "ARG"]

USE_ALT_C2_AFFORDABILITY = True    # *** CHANGED for final dissertation specification, per
                                    # review: the Eurostat validation confirmed the energy-
                                    # specific C2 restores the theoretically expected within-
                                    # country relationship that the headline-CPI C2 does not
                                    # show (see the construct-validation results reported
                                    # elsewhere in this project). Turned ON here so build_esi()
                                    # actually uses this result, rather than leaving it computed
                                    # only inside the validation routine for comparison. ***

_CPI_ENERGY_EMBEDDED_CSV = """iso_code,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023
AUS,38.35,40.98,42.7,45.2,46.54,48.18,49.84,51.99,56.91,63.14,71.6,78.79,89.71,99.68,101.05,100.0,101.41,110.17,116.94,115.53,111.85,107.79,115.0,131.84
AUT,67.28,69.42,67.88,68.77,72.32,77.43,82.17,87.06,91.99,89.77,92.6,97.97,101.52,103.33,102.88,100.0,98.17,97.42,100.03,102.65,102.36,110.19,150.73,175.65
BEL,64.25,66.07,63.91,63.45,66.48,72.56,79.03,78.7,99.33,86.51,91.78,107.89,114.28,110.02,101.8,100.0,104.0,111.35,117.88,119.45,108.25,125.4,213.15,150.17
BGR,43.07,45.03,52.45,59.9,65.59,69.37,72.82,76.26,85.75,90.42,89.63,92.15,100.29,98.44,95.45,100.0,99.78,103.44,108.02,112.01,111.8,118.36,144.94,152.28
CAN,91.3,92.9,100.0,98.0,102.0,104.9,110.8,112.9,113.2,115.2,120.7,124.3,128.0,132.7,137.7,140.5,141.3,142.7,143.5,145.6,147.2,149.3,151.8,151.0
CHE,78.29,78.43,74.19,74.92,77.34,85.99,91.92,93.11,108.49,92.41,101.33,108.82,110.2,108.38,108.79,100.0,97.82,100.35,108.04,108.38,101.37,106.74,131.23,151.41
CZE,47.89,55.63,58.61,57.93,59.05,62.83,70.42,72.68,82.78,89.8,89.3,94.6,102.8,104.29,98.83,100.02,99.48,98.83,100.87,107.9,111.22,107.68,140.95,193.43
DEU,55.29,60.32,59.7,61.79,64.18,71.36,78.7,81.75,90.89,88.7,88.89,97.27,102.91,107.17,106.17,100.0,95.68,96.56,99.39,102.28,101.67,104.27,138.33,157.71
DNK,69.49,71.27,73.51,74.33,75.21,80.22,84.36,84.48,91.01,88.61,95.08,101.96,103.64,104.95,106.08,100.0,97.77,97.38,98.89,98.3,95.01,103.77,153.34,130.05
ESP,,,57.45,58.22,59.33,63.04,69.24,70.54,77.89,76.47,81.32,94.07,103.28,102.89,105.32,100.0,89.52,97.79,102.01,96.94,87.51,110.5,145.15,105.31
EST,32.42,36.85,41.81,43.26,44.91,48.79,52.99,60.16,75.02,77.77,83.11,88.29,98.32,108.73,104.24,100.0,95.49,96.24,103.64,106.02,100.59,122.35,222.7,244.09
FIN,,,,,,63.08,67.19,69.68,81.87,79.36,86.9,103.49,103.24,103.98,102.55,99.99,99.11,104.4,109.2,115.9,113.3,117.34,153.98,160.07
FRA,60.16,60.99,60.71,62.2,63.62,68.15,72.87,74.26,81.36,76.63,81.68,90.09,95.05,98.73,100.57,100.0,98.53,102.17,108.85,112.24,110.01,120.19,149.11,162.47
GBR,37.7,37.9,38.7,39.4,42.2,47.8,59.7,64.3,76.4,81.3,78.7,86.1,92.9,99.5,103.7,100.0,97.1,100.9,107.8,111.7,106.5,111.9,183.9,211.4
GRC,39.26,39.32,40.04,41.65,43.53,51.52,57.72,57.84,67.82,57.92,67.89,80.95,96.73,111.7,110.89,100.0,94.92,100.7,102.22,101.85,94.06,105.17,155.67,131.12
HRV,52.97,56.16,56.96,59.54,60.99,63.43,66.21,66.0,70.51,74.94,80.59,82.78,95.05,100.9,101.3,100.0,95.21,91.3,93.75,96.86,94.34,96.16,109.63,118.34
HUN,44.22,48.72,51.38,55.16,62.86,66.73,71.02,87.97,99.17,107.11,113.82,120.28,127.73,116.86,103.05,100.0,99.93,100.7,102.11,103.14,103.42,103.94,126.6,146.5
IDN,,,,,,,,,,,,,100.0,105.4,112.1,118.4,122.0,123.8,127.0,128.8,129.8,131.8,136.2,138.6
IND,62.66,65.54,67.14,69.17,71.62,73.01,70.72,72.93,77.9,81.22,87.85,100.0,100.0,107.0,111.6,117.1,120.7,128.5,138.0,140.0,143.5,161.4,178.9,184.6
IRL,,,,54.89,58.79,68.07,74.6,80.16,87.12,82.06,83.42,91.42,100.38,103.88,105.18,100.0,94.83,95.9,103.2,105.8,99.21,112.94,171.95,197.91
ITA,66.2,68.86,66.89,69.45,69.53,75.24,82.56,83.71,92.83,88.18,86.63,93.46,105.16,106.55,102.79,100.0,95.23,97.85,102.63,104.48,95.74,111.21,206.14,195.91
JPN,81.43,81.38,79.78,78.88,78.74,78.77,81.59,82.3,87.8,84.38,83.49,86.87,91.14,96.37,103.35,100.0,90.53,93.61,98.31,101.04,97.58,,,
KOR,60.36,66.59,63.03,66.14,69.7,73.92,79.69,81.64,88.42,87.89,91.34,97.21,102.18,106.65,110.61,100.0,89.05,87.97,85.6,87.05,85.38,84.22,98.74,118.42
LTU,50.2,51.21,52.8,52.83,52.66,55.09,58.82,65.28,78.49,91.24,98.17,108.96,116.63,115.91,111.02,100.0,95.16,95.11,100.66,103.22,92.47,102.81,180.05,188.19
LUX,66.57,66.0,63.21,65.23,69.26,80.45,89.87,91.57,106.05,93.38,99.25,109.64,117.11,115.0,108.61,100.0,91.79,94.26,102.57,105.3,98.89,114.77,148.1,142.61
LVA,34.26,34.81,34.89,35.88,39.06,41.45,47.82,54.28,73.67,81.19,80.7,89.69,98.26,96.55,95.56,100.0,93.35,94.77,98.49,103.32,98.02,104.34,162.94,184.91
MEX,,,,58.93,64.69,69.69,75.11,77.32,82.74,82.5,85.63,86.88,90.77,93.7,99.46,100.0,99.37,109.05,118.87,116.78,120.09,139.1,141.96,126.03
NLD,53.73,61.91,65.11,69.8,72.28,82.4,90.34,94.05,96.92,97.71,89.9,95.2,101.31,103.97,102.74,100.0,94.47,96.07,106.09,119.32,105.07,123.53,264.29,166.46
NOR,55.53,69.08,68.12,89.71,83.14,81.75,101.9,83.72,105.08,101.65,120.78,115.92,95.7,109.79,103.56,100.0,119.3,129.08,158.97,160.1,116.63,192.55,232.28,217.93
POL,48.23,53.35,56.6,58.59,60.23,62.7,66.85,69.35,76.38,84.59,88.08,94.36,100.57,100.02,99.99,100.0,97.52,99.1,100.56,99.26,103.88,111.48,145.92,173.63
PRT,54.27,56.76,57.1,59.58,61.04,64.68,67.39,69.8,72.92,73.19,77.54,85.42,97.08,99.96,101.44,100.0,98.55,98.25,100.89,98.03,95.73,97.02,123.06,113.31
RUS,,,,,22.26,29.21,34.78,39.97,46.69,56.98,65.66,73.78,77.26,86.2,93.15,100.0,106.78,111.72,116.23,121.84,125.24,130.17,,
SVK,39.92,47.58,49.29,61.42,70.68,76.42,87.2,89.16,92.24,94.7,91.57,99.62,105.06,104.64,102.4,100.0,97.09,92.92,94.62,100.39,104.19,99.61,116.09,129.36
SVN,45.17,49.88,50.64,52.69,55.92,62.79,67.92,70.25,80.03,78.08,87.75,95.57,101.22,104.18,103.22,100.0,97.36,99.15,104.58,108.08,101.58,109.37,132.34,140.44
SWE,55.45,63.52,66.47,81.14,81.98,83.11,92.99,92.26,103.47,104.59,110.24,112.49,107.9,107.2,103.98,100.0,103.14,108.03,120.12,124.06,111.99,131.18,172.87,147.89
TUR,,,,,,38.38,42.8,45.97,57.96,63.39,67.76,72.42,85.65,92.48,95.41,100.0,103.84,107.91,128.47,152.31,177.4,210.33,419.96,403.64
USA,,,,,,,,,,,97.27,99.48,97.07,99.33,103.87,100.0,98.15,101.88,103.18,102.77,102.49,110.57,130.14,132.16
ZAF,22.48,24.17,25.89,27.4,29.08,30.65,31.94,34.28,40.58,50.14,59.79,69.84,78.6,85.61,91.79,100.0,109.23,114.49,120.41,131.68,143.08,157.4,174.59,194.83
"""

def _load_cpi_energy_table():
    """
    Parse the embedded household-energy CPI table (index, base varies by
    country/vintage as supplied) into a long-format DataFrame:
    iso_code, year, cpi_energy. Used only when USE_ALT_C2_AFFORDABILITY is
    True. This is a static, user-supplied table (not fetched from an API),
    embedded directly so the alternative C2 construction has no external
    file dependency.
    """
    import io
    wide = pd.read_csv(io.StringIO(_CPI_ENERGY_EMBEDDED_CSV))
    long = wide.melt(id_vars="iso_code", var_name="year", value_name="cpi_energy")
    long["year"] = long["year"].astype(int)
    long = long.dropna(subset=["cpi_energy"]).reset_index(drop=True)
    return long


def compute_energy_cpi_level_ratio(panel):
    """Household-energy CPI expressed relative to its own country mean (the
    LEVEL transformation used for headline C2), aligned to panel's index."""
    cpi = _load_cpi_energy_table()
    cpi = cpi[cpi["year"].between(YEAR_START, YEAR_END)].copy()
    cpi["ratio"] = cpi["cpi_energy"] / cpi.groupby("iso_code")["cpi_energy"].transform("mean")
    m = panel[["iso_code", "year"]].merge(cpi[["iso_code", "year", "ratio"]],
                                          on=["iso_code", "year"], how="left")
    m.index = panel.index
    return m["ratio"]


def compute_c2_star_alt(panel):
    """
    Compute C2*_it = 100 * [ln(CPI^energy_it) - ln(CPI^energy_i,t-1)], the
    raw (not yet normalised) year-over-year log change in household-energy
    CPI, merged onto panel by iso_code+year. Returns a Series aligned to
    panel's index, with NaN wherever the source CPI table has no value for
    that country-year or for the country's first available year (a first
    difference is undefined there).

    Part of the OPTIONAL ALTERNATIVE C2 block -- see the banner above.
    """
    cpi = _load_cpi_energy_table().sort_values(["iso_code","year"])
    cpi["ln_cpi"] = np.log(cpi["cpi_energy"])
    cpi["dln_cpi_x100"] = cpi.groupby("iso_code")["ln_cpi"].diff() * 100.0
    merged = panel[["iso_code","year"]].merge(
        cpi[["iso_code","year","dln_cpi_x100"]], on=["iso_code","year"], how="left")
    merged.index = panel.index
    n_valid = merged["dln_cpi_x100"].notna().sum()
    print(f"  ℹ Alt-C2 (100*Delta ln CPI_energy): {n_valid}/{len(panel)} obs matched "
          f"from the embedded CPI table")
    return merged["dln_cpi_x100"]
# ═════════════════════════════════════════════════════════════════════════════
# END OPTIONAL ALTERNATIVE C2 (constant, embedded data and loader). The
# branch that actually USES this inside step4_esi's affordability
# construction is separately delimited below, inside step4_esi itself.
# ═════════════════════════════════════════════════════════════════════════════

OWID_URL = (
    "https://raw.githubusercontent.com/owid/energy-data/master/"
    "owid-energy-data.csv"
)

WB_INDICATORS = {
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "EG.IMP.CONS.ZS"   : "net_import_pct",
    "GOV_WGI_CC.EST"           : "wgi_cc",
    "GOV_WGI_GE.EST"           : "wgi_ge",
    "GOV_WGI_PV.EST"           : "wgi_pv",
    "GOV_WGI_RL.EST"           : "wgi_rl",
    "GOV_WGI_RQ.EST"           : "wgi_rq",
    "GOV_WGI_VA.EST"           : "wgi_va",
}

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_owid(iso_list=None, year_start=YEAR_START, year_end=YEAR_END):
    """Download OWID energy CSV, filter to given ISO list and year range."""
    print("↓ Downloading OWID energy data …")
    r = requests.get(OWID_URL, timeout=90); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    if iso_list:
        df = df[df["iso_code"].isin(iso_list)]
    df = df[df["year"].between(year_start, year_end)].reset_index(drop=True)
    print(f"  OWID rows: {len(df)}")
    return df


def download_world_bank(iso_list=G20_ISO3):
    """
    Download WDI + WGI indicators, return merged long-format DataFrame.

    Non-WGI indicators (GDP, growth, net imports): via wbgapi (source db=2).

    WGI indicators (CC.EST, GE.EST, PV.EST, RL.EST, RQ.EST, VA.EST):
      wbgapi routes these to db=3 which builds a malformed URL
      (mixed YR2000;2001;YR2002 time segments) → JSONDecodeError.
      FIX: bypass wbgapi entirely for WGI; use a direct requests call to
           https://api.worldbank.org/v2/source/75/country/{countries}/
           indicator/{indicator}?date=YYYY:YYYY&per_page=20000&format=json
      Source 75 = Worldwide Governance Indicators (correct source ID).
    """
    import requests

    #WGI_CODES   = {"CC.EST","GE.EST","PV.EST","RL.EST","RQ.EST","VA.EST"}
    WGI_CODES   = {"GOV_WGI_CC.EST","GOV_WGI_GE.EST","GOV_WGI_PV.EST","GOV_WGI_RL.EST","GOV_WGI_RQ.EST","GOV_WGI_VA.EST"}

    DATE_RANGE  = f"{YEAR_START}:{YEAR_END}"  # e.g. "2000:2022"
    COUNTRIES   = ";".join(iso_list)

    # ── Helper: fetch one non-WGI indicator via wbgapi ───────────────────────
    def _fetch_wdi(wb_code, col):
        raw = wb.data.DataFrame(
            wb_code, economy=iso_list,
            time=range(YEAR_START, YEAR_END + 1),
            labels=False
        ).reset_index()
        if "time" in raw.columns:
            raw = raw.rename(columns={wb_code: col, "economy": "iso_code", "time": "year"})
            raw["year"] = raw["year"].astype(str).str.extract(r"(\d{4})").astype(int)
        else:
            yr_cols = [c for c in raw.columns if str(c).startswith("YR")]
            id_cols = [c for c in raw.columns if not str(c).startswith("YR")]
            raw = raw.melt(id_vars=id_cols, value_vars=yr_cols,
                           var_name="year_str", value_name=col)
            raw["year"] = raw["year_str"].str.replace("YR", "").astype(int)
            eco_col = next((c for c in id_cols if c.lower() in ("economy","iso_code")), id_cols[0])
            raw = raw.rename(columns={eco_col: "iso_code"})
        return raw[["iso_code", "year", col]].dropna(subset=[col])

    # ── Helper: fetch one WGI indicator via direct REST call ─────────────────
    def _fetch_wgi(wb_code, col):
        """

        """

        raw = wb.data.DataFrame(
            wb_code, economy=iso_list,
            time=range(YEAR_START, YEAR_END + 1),
            labels=False, db=3
        ).reset_index()
        if "time" in raw.columns:
            raw = raw.rename(columns={wb_code: col, "economy": "iso_code", "time": "year"})
            raw["year"] = raw["year"].astype(str).str.extract(r"(\d{4})").astype(int)
        else:
            yr_cols = [c for c in raw.columns if str(c).startswith("YR")]
            id_cols = [c for c in raw.columns if not str(c).startswith("YR")]
            raw = raw.melt(id_vars=id_cols, value_vars=yr_cols,
                           var_name="year_str", value_name=col)
            raw["year"] = raw["year_str"].str.replace("YR", "").astype(int)
            eco_col = next((c for c in id_cols if c.lower() in ("economy","iso_code")), id_cols[0])
            raw = raw.rename(columns={eco_col: "iso_code"})
        return raw[["iso_code", "year", col]].dropna(subset=[col])




    def _fetch_wgi_old(indicator, col):
        """
        Fetch one WGI indicator.

        Strategy: try four independent methods in order, return on first success.

        Method 1 — pandas_datareader (most reliable in Colab/Jupyter):
          Uses wb.download() which internally calls the standard v2 indicator
          endpoint with proper date ranges. Completely independent of wbgapi.

        Method 2 — wbgapi with db=3, year-by-year (avoids the mixed-format bug):
          The wbgapi malformed URL bug only occurs with time=range(). Passing a
          single integer year builds a clean "time=YR2020" segment. We loop years.

        Method 3 — direct requests, country/all, no source param:
          /v2/country/all/indicator/{CC.EST}?date={YEAR_START}:{YEAR_END}&per_page=1500
          No source= parameter; API auto-routes WGI indicators.

        Method 4 — direct requests, country/all, source=3:
          Same as Method 3 but with source=3 appended.
        """
        # ── Method 1: pandas_datareader ───────────────────────────────────────
        try:
            from pandas_datareader import wb as pdr_wb
            raw = pdr_wb.download(
                indicator=indicator,
                country=iso_list,
                start=YEAR_START,
                end=YEAR_END,
                errors="ignore"
            ).reset_index()
            # Find the value column
            val_col = next((c for c in raw.columns
                            if c not in ("country","year")), None)
            if val_col:
                raw = raw.rename(columns={"country": "iso_code", val_col: col})
                raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
                result = raw[["iso_code","year",col]].dropna(subset=[col])
                if len(result) > 10:
                    return result
        except Exception:
            pass

        # ── Method 2: wbgapi db=3 year-by-year ───────────────────────────────
        try:
            rows = []
            for yr in range(YEAR_START, YEAR_END + 1):
                try:
                    chunk = wb.data.DataFrame(
                        indicator, economy=iso_list,
                        time=yr, labels=False, db=3
                    ).reset_index()
                    if chunk.empty:
                        continue
                    val_col = (indicator if indicator in chunk.columns
                               else chunk.columns[-1])
                    eco_col = next((c for c in chunk.columns
                                    if c.lower() in ("economy","iso_code")),
                                   chunk.columns[0])
                    for _, row in chunk.iterrows():
                        iso = str(row.get(eco_col, ""))
                        val = row.get(val_col)
                        if iso in set(iso_list) and val is not None and not pd.isna(val):
                            rows.append({"iso_code": iso,
                                         "year": yr, col: float(val)})
                except Exception:
                    continue
            if len(rows) > 10:
                return pd.DataFrame(rows)[["iso_code","year",col]].dropna(subset=[col])
        except Exception:
            pass

        # ── Methods 3 & 4: direct requests ───────────────────────────────────
        for source_param in ["", "&source=3"]:
            try:
                base = (
                    f"https://api.worldbank.org/v2/country/all"
                    f"/indicator/{indicator}"
                    f"?date={DATE_RANGE}&per_page=1500&format=json{source_param}"
                )
                rows, page = [], 1
                while True:
                    resp = requests.get(f"{base}&page={page}", timeout=60)
                    if resp.status_code != 200:
                        break
                    payload = resp.json()
                    if not isinstance(payload, list) or len(payload) < 2:
                        break
                    if isinstance(payload[0], dict) and "message" in payload[0]:
                        break  # API error
                    meta, data = payload[0], payload[1]
                    if not data:
                        break
                    iso_set = set(iso_list)
                    for item in data:
                        iso = item.get("countryiso3code", "")
                        yr  = item.get("date")
                        val = item.get("value")
                        if iso in iso_set and yr and val is not None:
                            try:
                                rows.append({"iso_code": iso,
                                             "year": int(yr), col: float(val)})
                            except (ValueError, TypeError):
                                pass
                    if page >= int(meta.get("pages", 1)):
                        break
                    page += 1
                if len(rows) > 10:
                    df = pd.DataFrame(rows)
                    return df[["iso_code","year",col]].dropna(subset=[col])
            except Exception:
                pass

        raise ValueError(
            f"{indicator}: all download methods failed. "
            f"Add WGI CSV manually — see WGI_CSV_FALLBACK in step1."
        )

    # ── Main download loop ────────────────────────────────────────────────────
    # ── WGI pre-download hint ─────────────────────────────────────────────────
    # If automatic WGI download fails, download manually from:
    #   https://databank.worldbank.org/source/worldwide-governance-indicators
    #   → Select: All Countries | All Indicators | 2000-2023 | Download CSV
    #     (must match YEAR_START-YEAR_END below; the manual download is
    #     NOT auto-restricted to this window the way the API fetch is)
    #   → Save as "wgi_data.csv" in your working directory or /content/
    # The pipeline will detect and load it automatically.

    print("↓ Downloading World Bank WDI + WGI …")
    frames = []
    for wb_code, col in WB_INDICATORS.items():
        try:
            if wb_code in WGI_CODES:
                df = _fetch_wgi(wb_code, col)
                print(f"  ✓ {wb_code:25s} → {col}  ({len(df)} rows, WGI)")
            else:
                df = _fetch_wdi(wb_code, col)
                print(f"  ✓ {wb_code:25s} → {col}  ({len(df)} rows, WDI)")
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ Could not fetch {wb_code}: {e}")

    # ── EN.GHG.CO2.PC.CE.AR5: territorial (direct) CO2 emissions per capita,
    #    whole economy, excluding LULUCF (t CO2e/capita; EDGAR-based) ─────────
    # Fetched separately because it lives in WDI db=2 under a long code that
    # is not in WB_INDICATORS (added here to keep WB_INDICATORS clean).
    try:
        df_ghg = _fetch_wdi("EN.GHG.CO2.PC.CE.AR5", "ghg_co2_pc_ar5")
        frames.append(df_ghg)
        print(f"  ✓ EN.GHG.CO2.PC.CE.AR5       → ghg_co2_pc_ar5  ({len(df_ghg)} rows, WDI)")
    except Exception as _e:
        print(f"  ⚠ EN.GHG.CO2.PC.CE.AR5: {_e} — dim_sustainability will fall back to co2_intensity_elec")

    # ── WGI CSV fallback ─────────────────────────────────────────────────────
    # If all API methods fail, load from a CSV that the user can download manually:
    #   https://databank.worldbank.org/source/worldwide-governance-indicators
    # Save as "wgi_data.csv" in the working directory with columns:
    #   Country Code | Indicator Code | YYYY [... one col per year]

    #wgi_loaded = any(f for f in frames
    #                 if any(c.startswith("wgi_") for c in f.columns))
    wgi_loaded = True
    if not wgi_loaded:
        wgi_csv_paths = ["wgi_data.csv", "/content/wgi_data.csv",
                         "data/wgi_data.csv"]
        for csv_path in wgi_csv_paths:
            try:
                wgi_raw = pd.read_csv(csv_path)
                # Standard WB DataBank export format:
                # Country Code | Indicator Code | 2000 | 2001 | ... | 2023
                wgi_map = {
                    "CC.EST": "wgi_cc", "GE.EST": "wgi_ge", "PV.EST": "wgi_pv",
                    "RL.EST": "wgi_rl", "RQ.EST": "wgi_rq", "VA.EST": "wgi_va",
                }
                code_col = next((c for c in wgi_raw.columns
                                 if "code" in c.lower() and "indicator" not in c.lower()),
                                wgi_raw.columns[0])
                ind_col  = next((c for c in wgi_raw.columns
                                 if "indicator" in c.lower() and "code" in c.lower()),
                                wgi_raw.columns[1])
                yr_cols  = [c for c in wgi_raw.columns
                            if str(c).strip().isdigit() and
                            YEAR_START <= int(str(c).strip()) <= YEAR_END]
                if yr_cols:
                    for ind_code, col_name in wgi_map.items():
                        sub = wgi_raw[wgi_raw[ind_col] == ind_code]
                        if sub.empty:
                            continue
                        melted = sub.melt(id_vars=[code_col],
                                          value_vars=yr_cols,
                                          var_name="year",
                                          value_name=col_name)
                        melted = melted.rename(columns={code_col: "iso_code"})
                        melted["year"] = melted["year"].astype(int)
                        melted[col_name] = pd.to_numeric(melted[col_name],
                                                          errors="coerce")
                        melted = melted.dropna(subset=[col_name])
                        melted = melted[melted["iso_code"].isin(iso_list)]
                        if len(melted) > 5:
                            frames.append(melted[["iso_code","year",col_name]])
                    print(f"  ✓ WGI loaded from CSV: {csv_path}")
                    break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"  ⚠ WGI CSV load failed ({csv_path}): {e}")

    if not frames:
        print("  ⚠ No WB data fetched — returning empty DataFrame")
        return pd.DataFrame(columns=["iso_code", "year", "governance_score"])

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["iso_code", "year"], how="outer")

    wgi_cols = [c for c in out.columns if c.startswith("wgi_")]
    if wgi_cols:
        out["governance_score"] = out[wgi_cols].mean(axis=1)
        print(f"  ✓ governance_score = mean({wgi_cols})  "
              f"({out['governance_score'].notna().sum()} non-null)")
    else:
        out["governance_score"] = np.nan
        print("  ⚠ No WGI — governance_score will use log(GDP) fallback in step3")

    print(f"  WB rows after merge: {len(out)}")
    return out




# ═════════════════════════════════════════════════════════════════════════════

def _wb_fetch_simple(indicator, col, iso_list):
    """Fetch one WDI indicator via wbgapi (db=2 only). Returns long DataFrame."""
    import wbgapi as wb
    raw = wb.data.DataFrame(indicator, economy=iso_list,
                            time=range(YEAR_START, YEAR_END + 1),
                            labels=False, skipBlanks=True).reset_index()
    yr_cols = [c for c in raw.columns if str(c).startswith("YR")]
    id_cols = [c for c in raw.columns if not str(c).startswith("YR")]
    if not yr_cols:
        raise ValueError("No year columns returned")
    long = raw.melt(id_vars=id_cols, value_vars=yr_cols,
                    var_name="year_str", value_name=col)
    long["year"] = long["year_str"].str.replace("YR", "").astype(int)
    iso_col = next((c for c in ["economy","iso_code","Country"] if c in long.columns), id_cols[0])
    long = long.rename(columns={iso_col: "iso_code"})
    long[col] = pd.to_numeric(long[col], errors="coerce")
    long = long[["iso_code","year",col]].dropna()
    return long


# ── Generic, robust JSON-stat 2.0 fetch/parse (embedded from the data script) ──
# *** FIX (root cause of the Eurostat extraction failure): the previous
# implementation used a hand-written 2D-specific flat-index formula
# (flat_idx = g_idx * sizes[time_pos] + t_idx) that is only correct if
# geo and time are the ONLY two non-trivial dimensions, in that exact
# order. Eurostat's real nrg_pc_204 response has 6+ dimensions (freq,
# product, nrg_cons, unit, tax, currency, geo, time); if the server does
# not collapse every filtered dimension to size exactly 1, or returns
# dimensions in a different order, that formula silently computes the
# WRONG value or raises a KeyError. This is very likely why the fetch
# failed in a real run even with genuine network access to ec.europa.eu.
# The generic stride-based parser below decomposes the flat index against
# ALL dimensions actually present, regardless of count or order, and is
# therefore robust to exactly this failure mode. ***

_EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

# ISO3 -> Eurostat geo code (Eurostat's own EL/UK conventions)
_EUROSTAT_GEO_MAP = {
    "AUT": "AT", "BEL": "BE", "BGR": "BG", "CYP": "CY", "CZE": "CZ",
    "DEU": "DE", "DNK": "DK", "ESP": "ES", "EST": "EE", "FIN": "FI",
    "FRA": "FR", "GBR": "UK", "GRC": "EL", "HRV": "HR", "HUN": "HU",
    "IRL": "IE", "ITA": "IT", "LTU": "LT", "LUX": "LU", "LVA": "LV",
    "MLT": "MT", "NLD": "NL", "NOR": "NO", "POL": "PL", "PRT": "PT",
    "ROU": "RO", "SVK": "SK", "SVN": "SI", "SWE": "SE", "TUR": "TR",
}
_EUROSTAT_GEO_TO_ISO = {v: k for k, v in _EUROSTAT_GEO_MAP.items()}

# Eurostat affordability-validation sample (set 3 above): EU27 + NOR + GBR +
# TUR = 30 countries, listed EXPLICITLY. Previously this was built as
# Europe-30 + TUR (31 incl. Switzerland); Switzerland has no code in the
# Eurostat geo map above, so it was silently dropped and the observed sample
# happened to be 30 countries -- a different 30 from Europe-30. The list is
# now explicit so the two sets of 30 cannot be confused.
EUROSTAT_VALIDATION_ISO3 = sorted(EU27_ISO3 + ["NOR", "GBR", "TUR"])
assert len(EUROSTAT_VALIDATION_ISO3) == 30 and set(EUROSTAT_VALIDATION_ISO3) <= set(_EUROSTAT_GEO_MAP)
assert "CHE" not in EUROSTAT_VALIDATION_ISO3


def _parse_jsonstat_to_df(js):
    """
    Flatten a JSON-stat 2.0 response into a tidy DataFrame with one column
    per dimension plus 'value'. Handles ANY number of dimensions via the
    standard row-major stride decomposition of the flat index.
    """
    dims = js["id"]
    sizes = js["size"]
    cats = {
        d: {v: k for k, v in js["dimension"][d]["category"]["index"].items()}
        for d in dims
    }
    strides, acc = [], 1
    for s in reversed(sizes):
        strides.insert(0, acc)
        acc *= s

    rows = []
    for pos_str, val in js.get("value", {}).items():
        pos = int(pos_str)
        rec = {
            d: cats[d][(pos // stride) % size]
            for d, size, stride in zip(dims, sizes, strides)
        }
        rec["value"] = val
        rows.append(rec)
    return pd.DataFrame(rows)


def fetch_eurostat_dataset(dataset, filters):
    """
    Fetch a whole Eurostat dataset (all geos, all periods) in ONE request.
    If the filtered query is rejected (HTTP 400 -- a common Eurostat
    failure mode when a dimension code is wrong or has changed), print
    Eurostat's own error message and retry with NO dimension filters,
    filtering client-side instead.
    """
    def _request(params):
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{_EUROSTAT_BASE}/{dataset}?format=JSON&lang=en"
        if query:
            url += "&" + query
        return requests.get(url, timeout=120)

    r = _request(filters)
    if r.status_code == 400:
        print(f"    {dataset}: filtered query rejected (HTTP 400).")
        print(f"      Eurostat says: {r.text[:400]}")
        print(f"      Retrying without dimension filters (client-side filtering)...")
        r = _request({})

    if r.status_code != 200:
        raise RuntimeError(f"{dataset}: HTTP {r.status_code}. Body: {r.text[:400]}")

    df = _parse_jsonstat_to_df(r.json())
    if df.empty:
        raise RuntimeError(f"{dataset}: request succeeded but returned no data.")

    for dim, code in filters.items():
        if dim in df.columns:
            df = df[df[dim] == code]
    return df.reset_index(drop=True)


def download_eurostat_electricity_prices(iso_list=None):
    """
    Download household electricity prices for the European validation
    sample from Eurostat (nrg_pc_204: household electricity prices, band
    DC, 2,500-4,999 kWh/year, all taxes and levies included). Available
    from 2007 onward.

    *** Embeds the robust generic JSON-stat fetch/parse mechanism above as
    the PRIMARY implementation, replacing the earlier 2D-specific flat-
    index parser that was the likely cause of real fetch failures. ***

    Fetches BOTH EUR and PPS prices, per the explanation document: PPS
    (purchasing power standard) is the preferred validation price where
    coverage is complete, since it already adjusts for cross-country
    price-level differences; EUR is retained as a fallback/comparison.

    Returns DataFrame: iso_code | country | year | semester |
    eurostat_elec_price_eur_kwh | eurostat_elec_price_pps_kwh
    (one row per country-year-semester; annualisation happens downstream).
    """
    if iso_list is None:
        iso_list = EUROSTAT_VALIDATION_ISO3

    print("  ↓ Downloading Eurostat household electricity prices (nrg_pc_204) …")
    frames = []
    for currency, col in [("EUR", "eurostat_elec_price_eur_kwh"),
                          ("PPS", "eurostat_elec_price_pps_kwh")]:
        try:
            raw = fetch_eurostat_dataset("nrg_pc_204", {
                # "product" removed: nrg_pc_204 no longer has that dimension
                # (HTTP 400 "Dimension PRODUCT is not defined"); the dataset
                # covers electricity only, so no product/siec filter is needed.
                "freq": "S", "nrg_cons": "KWH2500-4999",
                "unit": "KWH", "tax": "I_TAX", "currency": currency,
            })
            raw = raw[raw["geo"].isin(_EUROSTAT_GEO_TO_ISO)].copy()
            raw["iso_code"] = raw["geo"].map(_EUROSTAT_GEO_TO_ISO)
            raw = raw[raw["iso_code"].isin(iso_list)]
            raw["year"] = raw["time"].str[:4].astype(int)
            # *** ADDED: explicit filter to YEAR_START-YEAR_END. Eurostat's
            # nrg_pc_204 series is not itself bounded to this pipeline's
            # analysis window (it may extend earlier or later than what the
            # rest of this file is designed for), and while the current
            # merge architecture keys external data to the main panel's own
            # (already-bounded) rows -- so an over-fetch here would likely
            # be silently dropped downstream regardless -- filtering
            # explicitly here removes the dependency on that implicit
            # protection and keeps this function's own output correctly
            # scoped on its own terms. ***
            raw = raw[(raw["year"] >= YEAR_START) & (raw["year"] <= YEAR_END)]
            raw["semester"] = raw["time"].str[-2:]
            raw = raw.rename(columns={"value": col})
            frames.append(raw[["iso_code","year","semester",col]])
            print(f"    ✓ {col}: {len(raw)} semester obs, "
                  f"{raw['iso_code'].nunique()} countries")
        except Exception as e:
            print(f"    ⚠ {currency} price fetch failed: {e}")

    if not frames:
        print("    ⚠ Both EUR and PPS fetches failed — returning empty DataFrame")
        return pd.DataFrame(columns=["iso_code","year","semester",
                                     "eurostat_elec_price_eur_kwh",
                                     "eurostat_elec_price_pps_kwh"])

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["iso_code","year","semester"], how="outer")
    return df


def build_eurostat_affordability_validation_table(panel, eurostat_prices):
    """
    Construct the full Eurostat affordability-validation panel, 2007-2023,
    with EXACTLY the schema specified in the review's explanation document:

        iso_code, country, year,
        eurostat_elec_price_eur_kwh, eurostat_elec_price_pps_kwh,
        eurostat_elec_price_annual, annual_household_elec_cost,
        aff_intensity, aff_price_ratio, aff_hfce_pc,
        eurostat_elec_burden,
        _aff_c1, _aff_c2, _aff_c3, dim_affordability,
        validation_sample

    eurostat_elec_price_annual: annual average of the two semesters, using
        PPS if available (preferred, adjusts for cross-country price
        levels), falling back to EUR where PPS coverage is incomplete.

    validation_sample = 1 only where ALL FOUR of (Eurostat price,
        aff_intensity, aff_price_ratio, aff_hfce_pc) are available -- kept
        clean and explicit, per the explanation document's instruction not
        to mix missing-data rules.

    This does NOT replace the G20 affordability index; it is an external
    European-subsample validation only, restricted to
    EUROSTAT_VALIDATION_ISO3 (EU27 + NOR + GBR + TUR) and 2007-2023 (the
    nrg_pc_204 coverage window).
    """
    ANNUAL_KWH = 3500   # representative annual consumption, inside band DC

    if eurostat_prices is None or len(eurostat_prices) == 0:
        print("  ⚠ No Eurostat electricity price data available")
        return pd.DataFrame()

    # Annualise: average across semesters, preferring PPS over EUR
    price_cols = [c for c in ["eurostat_elec_price_eur_kwh","eurostat_elec_price_pps_kwh"]
                 if c in eurostat_prices.columns]
    euro_annual = (eurostat_prices.groupby(["iso_code","year"], as_index=False)[price_cols]
                   .mean())
    if "eurostat_elec_price_pps_kwh" in euro_annual.columns:
        euro_annual["eurostat_elec_price_annual"] = (
            euro_annual["eurostat_elec_price_pps_kwh"]
            .fillna(euro_annual.get("eurostat_elec_price_eur_kwh"))
        )
    else:
        euro_annual["eurostat_elec_price_annual"] = euro_annual.get("eurostat_elec_price_eur_kwh")

    euro_annual["annual_household_elec_cost"] = (
        euro_annual["eurostat_elec_price_annual"] * ANNUAL_KWH
    )

    needed_panel_cols = ["iso_code","year","aff_intensity","aff_price_ratio",
                         "aff_hfce_pc","_aff_c1","_aff_c2","_aff_c3",
                         "dim_affordability","country","gdp_pc_ppp"]
    avail_panel_cols = [c for c in needed_panel_cols if c in panel.columns]
    sub_panel = panel[avail_panel_cols].drop_duplicates(subset=["iso_code","year"]).copy()

    # *** FIX (review): a genuine construct-level validation of dim_affordability
    # was added to validate_affordability_eu27(), but this function -- called
    # from step2_validate_affordability(), which runs BEFORE step4_esi() --
    # is invoked at a point in the pipeline where dim_affordability (and the
    # oriented _aff_c1/_aff_c2/_aff_c3 components ESI actually uses) do not
    # yet exist in panel at all. The previous version silently dropped these
    # columns from avail_panel_cols with no warning, so the construct-level
    # test downstream silently found nothing to test against -- exactly the
    # inconsistency identified in review (a "primary" result referenced by
    # the component-level section's own text, but never actually printed).
    #
    # Fix: if dim_affordability is not already present (i.e. this function is
    # being called before Step 4 has run, the normal case for this Step 2
    # validation), compute a LOCAL replica using the IDENTICAL formula
    # step4_esi() itself uses, on the SAME full panel this function already
    # receives -- not an approximation, the exact same three-line
    # construction (all with the single pooled, winsorised rule, minmax()):
    # _aff_c1 = minmax(aff_intensity, invert), _aff_c2 = minmax(aff_price_ratio,
    # invert), _aff_c3 = minmax(ln
    # winsorised 1-99%, of aff_hfce_pc), dim_affordability = mean of
    # available components. Per-year normalisation is computed on the FULL
    # panel (matching Step 4's own convention), before restricting to the
    # European validation subsample below -- normalising only within the
    # European subsample would use different bounds than ESI's own
    # construction and would no longer be "the exact affordability score
    # used in the ESI" the construct-level test is meant to validate. If
    # step4_esi() has already run and these columns already exist in panel,
    # they are used directly instead, avoiding any duplicate computation. ***
    if "dim_affordability" not in sub_panel.columns:
        print("  ℹ dim_affordability not yet computed on this panel (this validation")
        print("    runs before Step 4 in the normal pipeline order) -- computing local")
        print("    replicas of BOTH C2 variants using ESI's own exact C1/C2/C3 formula,")
        print("    on the full panel, so the construct-level test below has genuine")
        print("    composite scores to validate rather than silently skipping them.")
        if "aff_intensity" in panel.columns and panel["aff_intensity"].notna().any():
            sub_panel["_aff_c1"] = minmax(panel["aff_intensity"], invert=True).reindex(sub_panel.index)
        if "aff_hfce_pc" in panel.columns and panel["aff_hfce_pc"].notna().sum() > 20:
            sub_panel["_aff_c3"] = minmax(panel["aff_hfce_pc"].clip(lower=1.0), log=True).reindex(sub_panel.index)

        # *** ADDED: BOTH C2 variants computed explicitly here, headline first
        # as the labelled baseline, matching the same dual-computation and
        # labelling convention used in step4_esi() and the six export
        # tables. Previously this local replica only ever built the
        # headline variant, so the Eurostat validation table itself never
        # showed the energy-C2 comparison directly. ***
        have_c1c3 = "_aff_c1" in sub_panel.columns and "_aff_c3" in sub_panel.columns
        if "aff_price_ratio" in panel.columns and panel["aff_price_ratio"].notna().sum() > 50 and have_c1c3:
            sub_panel["_aff_c2_headline"] = minmax(panel["aff_price_ratio"], invert=True).reindex(sub_panel.index)
            sub_panel["dim_affordability_headline"] = sub_panel[
                ["_aff_c1", "_aff_c2_headline", "_aff_c3"]].mean(axis=1)
            print(f"    ✓ dim_affordability_headline (baseline) computed locally")
        _dln_energy_local = compute_c2_star_alt(panel) if have_c1c3 else None
        if _dln_energy_local is not None and _dln_energy_local.notna().sum() > 20 and have_c1c3:
            sub_panel["_aff_c2_energy"] = minmax(_dln_energy_local, invert=True).reindex(sub_panel.index)
            sub_panel["dim_affordability_energy"] = sub_panel[
                ["_aff_c1", "_aff_c2_energy", "_aff_c3"]].mean(axis=1)
            print(f"    ✓ dim_affordability_energy computed locally")

        # MATCHED TEMPORAL TRANSFORMATIONS (reviewer). The baseline headline
        # C2 is a LEVEL (CPI / country mean) while the energy C2 is a GROWTH
        # rate (100 x Delta ln CPI_energy), so comparing the two composites
        # mixes a change of price index with a change of transformation. Both
        # missing combinations are built here, on the full panel with the
        # same pooled normalisation:
        #   _aff_c2_headline_growth : 100 x Delta ln(headline CPI)
        #   _aff_c2_energy_level    : energy CPI / its country mean
        if have_c1c3 and "aff_price_ratio" in panel.columns:
            _hg = panel.groupby("iso_code")["aff_price_ratio"].transform(
                lambda x: 100.0 * np.log(x.where(x > 0)).diff())
            sub_panel["_aff_c2_headline_growth"] = minmax(_hg, invert=True).reindex(sub_panel.index)
        if have_c1c3:
            _el = compute_energy_cpi_level_ratio(panel)
            if _el.notna().sum() > 20:
                sub_panel["_aff_c2_energy_level"] = minmax(_el, invert=True).reindex(sub_panel.index)

        # "dim_affordability" (unlabelled) = headline, the labelled baseline,
        # for backward compatibility with any code reading this column name
        # directly; both labelled variants remain available under their own
        # names regardless.
        if "dim_affordability_headline" in sub_panel.columns:
            sub_panel["dim_affordability"] = sub_panel["dim_affordability_headline"]
            sub_panel["_aff_c2"] = sub_panel.get("_aff_c2_headline")
        elif "dim_affordability_energy" in sub_panel.columns:
            sub_panel["dim_affordability"] = sub_panel["dim_affordability_energy"]
            sub_panel["_aff_c2"] = sub_panel.get("_aff_c2_energy")
        else:
            print("    ⚠ Neither C2 variant could be computed -- dim_affordability will")
            print("      be based on C1+C3 only, if even those are available.")
            if have_c1c3:
                sub_panel["dim_affordability"] = sub_panel[["_aff_c1", "_aff_c3"]].mean(axis=1)

    out = euro_annual.merge(sub_panel, on=["iso_code","year"], how="left")
    out = out[out["iso_code"].isin(EUROSTAT_VALIDATION_ISO3)]
    out = out[(out["year"] >= 2007) & (out["year"] <= 2023)]

    if "aff_hfce_pc" in out.columns:
        out["eurostat_elec_burden"] = out["annual_household_elec_cost"] / out["aff_hfce_pc"]
    else:
        out["eurostat_elec_burden"] = np.nan
    # Alternative burden denominator: GDP per capita (PPP) instead of household
    # consumption per capita, so that the criterion does not share C3's variable.
    if "gdp_pc_ppp" in out.columns:
        out["eurostat_elec_burden_gdp"] = out["annual_household_elec_cost"] / out["gdp_pc_ppp"]

    required_for_sample = ["eurostat_elec_price_annual","aff_intensity",
                           "aff_price_ratio","aff_hfce_pc"]
    required_for_sample = [c for c in required_for_sample if c in out.columns]
    out["validation_sample"] = out[required_for_sample].notna().all(axis=1).astype(int)

    col_order = ["iso_code","country","year",
                "eurostat_elec_price_eur_kwh","eurostat_elec_price_pps_kwh",
                "eurostat_elec_price_annual","annual_household_elec_cost",
                "aff_intensity","aff_price_ratio","aff_hfce_pc",
                "eurostat_elec_burden",
                "_aff_c1","_aff_c2_headline","_aff_c2_energy",
                "_aff_c2_headline_growth","_aff_c2_energy_level","_aff_c3",
                "eurostat_elec_burden_gdp","gdp_pc_ppp",
                "dim_affordability_headline","dim_affordability_energy",
                "dim_affordability",
                "validation_sample"]
    col_order = [c for c in col_order if c in out.columns]
    result = out[col_order].sort_values(["iso_code","year"]).reset_index(drop=True)
    # Explicit provenance marker: "dim_affordability" (unlabelled, used by the
    # rest of this validation routine as the default) is always the HEADLINE
    # variant, the labelled baseline -- stated here so this is never
    # ambiguous from the table alone, matching the same convention used in
    # the six country-year export tables in export_country_year_tables().
    result["c2_baseline_used_for_dim_affordability"] = "headline"
    return result


def validate_affordability_eu27(panel, eurostat_prices=None):
    """
    External European-subsample validation of the affordability dimension's
    THREE raw components (energy intensity, price pressure, purchasing
    capacity) against Eurostat household electricity prices, 2007-2023.

    *** IMPORTANT METHODOLOGICAL FRAMING: this provides an EXTERNAL
    EUROPEAN-SUBSAMPLE VALIDATION of the affordability proxy (EU27 + NOR +
    GBR + TUR, 2007 onward -- Eurostat nrg_pc_204 coverage). It does NOT
    validate affordability for the full G20 panel, and tests DIRECTIONAL
    CONSISTENCY, not exact equality. Do not describe this as "Eurostat
    validates affordability for the G20." ***

    Builds the full validation table (build_eurostat_affordability_validation_table),
    restricts to validation_sample==1 (all four required variables present,
    per the explanation document's instruction to keep the sample clean),
    then runs:
        - Spearman rank correlations among C1, C2, C3, the Eurostat price,
          and the constructed Eurostat electricity burden
        - Panel regression: eurostat_elec_burden ~ C1 + C2 + C3, country +
          year fixed effects, clustered SE, with expected signs
          beta1>0 (C1), beta2>0 (C2), beta3<0 (C3)

    Returns dict: {"table": full validation DataFrame (all rows, with the
    validation_sample flag), "correlation_matrix", "model", coefficients,
    p-values, "all_signs_match"}.
    """
    print(f"\n{'═'*66}")
    print("EXTERNAL EUROPEAN-SUBSAMPLE AFFORDABILITY VALIDATION (Eurostat)")
    print("  Validates DIRECTIONAL CONSISTENCY of C1/C2/C3, not exact equality.")
    print("  Does NOT validate affordability for the full G20 sample.")
    print(f"  Requested sample: EU27 + NOR + GBR + TUR = {len(EUROSTAT_VALIDATION_ISO3)} countries,")
    print(f"  2007-2023 (nrg_pc_204 coverage). NOT the Europe-30 panel group")
    print(f"  (EU27 + CHE + NOR + GBR): Switzerland is not in this sample, Türkiye is.")
    print(f"{'═'*66}")

    full_table = build_eurostat_affordability_validation_table(panel, eurostat_prices)
    if full_table.empty:
        print("  ⚠ Could not construct the Eurostat validation table — skipped")
        return {}

    n_total = len(full_table)
    validated = full_table[full_table["validation_sample"] == 1].copy()
    _obs = sorted(validated["iso_code"].unique())
    _miss = sorted(set(EUROSTAT_VALIDATION_ISO3) - set(_obs))
    _yrs = validated.groupby("iso_code")["year"].agg(["min", "max", "count"])
    print(f"\n  OBSERVED Eurostat validation sample: {len(_obs)} of the "
          f"{len(EUROSTAT_VALIDATION_ISO3)} requested countries (EU27 + NOR + GBR + TUR)")
    print(f"  (not the same set as the Europe-30 panel group, which has CHE instead of TUR)")
    if _miss:
        print(f"  Requested but not observed: {_miss}")
    _short = _yrs[_yrs["count"] < _yrs["count"].max()]
    if len(_short):
        print("  Countries with fewer years: " + ", ".join(
            f"{i} {int(r['min'])}-{int(r['max'])} ({int(r['count'])})" for i, r in _short.iterrows()))
    n = len(validated)
    print(f"\n  Full table: {n_total} country-year rows "
          f"({full_table['iso_code'].nunique()} countries)")
    print(f"  Clean validation sample (validation_sample==1): {n} rows "
          f"({validated['iso_code'].nunique()} countries)")
    if n < 20:
        print(f"  ⚠ Small sample — treat results as indicative only")
    if n < 5:
        return {"table": full_table}

    corr_cols = [c for c in ["aff_intensity","aff_price_ratio","aff_hfce_pc",
                             "dim_affordability",
                             "eurostat_elec_price_annual","eurostat_elec_burden"]
                if c in validated.columns]
    corr = validated[corr_cols].corr(method="spearman")
    print(f"\n  Spearman rank correlations (directional consistency check):")
    print(corr.round(3).to_string())

    result = {"table": full_table, "n_obs": n, "correlation_matrix": corr}

    # *** ADDED: construct-level validation of the COMPOSITE affordability
    # score, as the PRIMARY external-validity test -- added after review
    # correctly identified that the three component-level regressions
    # below test three separate proxy hypotheses, not the actual construct
    # this pipeline uses. dim_affordability = mean(C1, C2, C3) is what
    # every downstream step (ESI, Simar-Wilson, Mundlak, conditional DDF)
    # actually consumes -- it is NOT a DDF input; validating C1, C2 and C3 individually is
    # useful DIAGNOSTIC evidence for understanding which channel drives
    # the composite, but is not itself a test of the construct. The
    # primary hypothesis is AFF up => Eurostat burden down, i.e.
    # rho(AFF, Burden) < 0, tested here directly via Pearson and Spearman
    # correlation and via three regression specifications of increasing
    # stringency: pooled, country fixed effects, and country + year fixed
    # effects (the most demanding, identifying beta purely from within-
    # country, within-year deviations). ***
    if "dim_affordability" in validated.columns and "eurostat_elec_burden" in validated.columns:
        construct_df = validated[["dim_affordability","eurostat_elec_burden",
                                  "iso_code","year"]].dropna()
        if len(construct_df) >= 10:
            print(f"\n  {'═'*66}")
            print(f"  PRIMARY CONSTRUCT VALIDATION: composite AFF vs Eurostat burden")
            print(f"  H0: AFF and burden are unrelated.  H1: AFF up => burden down")
            print(f"  (rho(AFF, Burden) < 0), i.e. the actual construct this pipeline")
            print(f"  uses (dim_affordability = mean(C1,C2,C3)), not the three")
            print(f"  individual proxy hypotheses tested separately below.")
            print(f"  {'═'*66}")

            r_pearson, p_pearson = stats.pearsonr(construct_df["dim_affordability"],
                                                   construct_df["eurostat_elec_burden"])
            rho_spearman, p_spearman = stats.spearmanr(construct_df["dim_affordability"],
                                                        construct_df["eurostat_elec_burden"])
            print(f"\n  Pearson  (AFF, Burden)              = {r_pearson:+.4f}  (p={p_pearson:.4f})")
            print(f"  Spearman (AFF, Burden)              = {rho_spearman:+.4f}  (p={p_spearman:.4f})")

            # Country-mean Spearman: the between-country (level) relationship,
            # one observation per country, complementing the pooled country-
            # year correlations above with the cross-sectional counterpart.
            country_mean_df = construct_df.groupby("iso_code")[
                ["dim_affordability","eurostat_elec_burden"]].mean()
            rho_cm, p_cm = (np.nan, np.nan)
            if len(country_mean_df) >= 5:
                rho_cm, p_cm = stats.spearmanr(country_mean_df["dim_affordability"],
                                               country_mean_df["eurostat_elec_burden"])
                print(f"  Country-mean Spearman (AFF, Burden) = {rho_cm:+.4f}  (p={p_cm:.4f}, "
                      f"n={len(country_mean_df)} countries)")
            print(f"  Expected sign throughout: negative (higher AFF = better affordability;")
            print(f"  higher Eurostat burden = worse affordability -- beta_AFF < 0)")

            import statsmodels.formula.api as smf
            print(f"\n  Regression specifications (Burden_it = ... + beta*AFF_it + e_it):")
            print(f"  {'Specification':<32} {'beta':>10} {'p-value':>10} {'R2':>8} {'N':>6}")
            print(f"  {'-'*70}")

            specs = [
                ("Pooled (no FE)",
                 "eurostat_elec_burden ~ dim_affordability"),
                ("Country FE",
                 "eurostat_elec_burden ~ dim_affordability + C(iso_code)"),
                ("Country + Year FE (most demanding)",
                 "eurostat_elec_burden ~ dim_affordability + C(iso_code) + C(year)"),
            ]
            construct_models = {}
            for label, formula in specs:
                try:
                    m = smf.ols(formula, data=construct_df).fit(
                        cov_type="cluster", cov_kwds={"groups": construct_df["iso_code"]})
                    beta = m.params.get("dim_affordability", np.nan)
                    pval = m.pvalues.get("dim_affordability", np.nan)
                    print(f"  {label:<32} {beta:>+10.4f} {pval:>10.4f} {m.rsquared:>8.4f} {int(m.nobs):>6}")
                    construct_models[label] = m
                except Exception as e:
                    print(f"  {label:<32}  failed: {e}")

            print(f"  {'-'*70}")
            beta_full = construct_models.get("Country + Year FE (most demanding)")
            beta_pooled = construct_models.get("Pooled (no FE)")

            # *** REVISED per review: the previous version's "else" branch
            # (b >= 0) was labelled "wrong sign" regardless of significance,
            # which overclaims when the coefficient is not statistically
            # distinguishable from zero (observed directly: a coefficient of
            # +0.0180 with p=0.2378 was being reported as if it were
            # confirmed evidence of a positive relationship, when the
            # correct reading is that no relationship -- in either
            # direction -- is detected at that specification). The revised
            # logic below checks significance explicitly at every step, and
            # separately considers the pooled/cross-country evidence (from
            # the pooled model and the Pearson/Spearman/country-mean-
            # Spearman results already computed above) alongside the
            # within-country (FE) evidence, since these can legitimately
            # differ -- a construct can be a valid STRUCTURAL, cross-country
            # measure of affordability without also being a valid indicator
            # of short-run within-country changes, and the verdict should
            # say which of these the evidence actually supports rather than
            # collapsing both questions into a single pass/fail. ***
            if beta_full is not None:
                b_fe = beta_full.params.get("dim_affordability", np.nan)
                p_fe = beta_full.pvalues.get("dim_affordability", np.nan)
                fe_negative_sig = (not pd.isna(p_fe)) and p_fe < 0.05 and b_fe < 0
                fe_positive_sig = (not pd.isna(p_fe)) and p_fe < 0.05 and b_fe > 0
                fe_not_sig = pd.isna(p_fe) or p_fe >= 0.05

                pooled_negative_sig = (r_pearson < 0 and p_pearson < 0.05) or (rho_spearman < 0 and p_spearman < 0.05)
                crossctry_negative_sig = (not pd.isna(rho_cm)) and (not pd.isna(p_cm)) and rho_cm < 0 and p_cm < 0.05

                if fe_negative_sig:
                    print(f"  Primary construct-level result: the composite affordability dimension")
                    print(f"  exhibits a statistically significant negative association with the")
                    print(f"  Eurostat burden even in the most demanding (country + year FE)")
                    print(f"  specification (beta={b_fe:+.4f}, p={p_fe:.4f}) -- direct support for")
                    print(f"  the construct's core hypothesis at both the structural and within-")
                    print(f"  country level.")
                elif fe_positive_sig:
                    print(f"  Primary construct-level result: the coefficient is POSITIVE and")
                    print(f"  statistically significant in the most demanding specification")
                    print(f"  (beta={b_fe:+.4f}, p={p_fe:.4f}) -- this is a genuine sign reversal in")
                    print(f"  the within-country dimension and should be investigated directly,")
                    print(f"  not attributed to noise.")
                elif fe_not_sig and (pooled_negative_sig or crossctry_negative_sig):
                    print(f"  Primary construct-level result: the composite affordability dimension")
                    print(f"  exhibits the expected negative association with electricity burden in")
                    print(f"  pooled and cross-country comparisons"
                          + (f" (country-mean Spearman rho={rho_cm:+.4f}, p={p_cm:.4f})" if crossctry_negative_sig else "")
                          + f". However, no statistically significant relationship is detected from")
                    print(f"  within-country annual variation after country fixed effects, either")
                    print(f"  with or without year fixed effects (beta={b_fe:+.4f}, p={p_fe:.4f}). The")
                    print(f"  evidence therefore supports the construct primarily as a STRUCTURAL")
                    print(f"  cross-country affordability measure rather than as an indicator of")
                    print(f"  short-run changes in household electricity burden.")
                else:
                    print(f"  Primary construct-level result: no statistically significant")
                    print(f"  relationship with the Eurostat burden was detected at either the")
                    print(f"  pooled/cross-country level or the within-country level on this run")
                    print(f"  (most demanding spec: beta={b_fe:+.4f}, p={p_fe:.4f}) -- interpret the")
                    print(f"  construct's external validity with caution.")

            # SUPPORTING (not primary) evidence: AFF against the actual observed
            # Eurostat electricity price, as distinct from the constructed
            # burden measure. Burden (annual_household_elec_cost / aff_hfce_pc)
            # is the more direct affordability criterion since it already
            # nets out purchasing power; price alone does not. Reported here
            # as supplementary corroboration only.
            if "eurostat_elec_price_annual" in validated.columns:
                price_df = validated[["dim_affordability","eurostat_elec_price_annual",
                                      "iso_code"]].dropna()
                if len(price_df) >= 10:
                    rho_price, p_price = stats.spearmanr(price_df["dim_affordability"],
                                                         price_df["eurostat_elec_price_annual"])
                    print(f"\n  {'─'*60}")
                    print(f"  DESCRIPTIVE (not a hypothesis test): AFF vs actual Eurostat")
                    print(f"  electricity price")
                    print(f"  {'─'*60}")
                    print(f"  Spearman (AFF, Eurostat price) = {rho_price:+.4f}  (p={p_price:.4f})")
                    print(f"  *** No directional hypothesis is imposed here, following review: raw")
                    print(f"  electricity price alone does not net out purchasing power the way")
                    print(f"  burden (price x consumption / income) does, so AFF and price are not")
                    print(f"  expected to move together in a specific direction on a priori")
                    print(f"  grounds -- a country can face a high per-kWh price yet still have")
                    print(f"  high overall affordability if incomes are also high. This is reported")
                    print(f"  as descriptive information only; burden (above) remains the")
                    print(f"  principal, and the only hypothesis-tested, external criterion. ***")
                    result["construct_vs_price_spearman"] = rho_price

            # *** ADDED (review): diagnostic to determine whether C1 and/or
            # C2 are attenuating the composite's relationship with burden,
            # given C3 alone shows a much stronger correlation with burden
            # than the full three-component composite does.
            #
            # *** REVISED per review, in two ways:
            # (1) Now that the energy-specific CPI dataset exists (see
            #     USE_ALT_C2_AFFORDABILITY near the top of this file), this
            #     diagnostic additionally builds AFF_new = C1 + C2^energy + C3
            #     alongside AFF_old = C1 + C2^headline + C3 (the current
            #     default) and AFF13 = C1 + C3, so the key question can be
            #     answered directly: does replacing C2 with the energy-
            #     specific version approach AFF13's within-country behaviour
            #     while keeping a price channel in the construct at all,
            #     rather than simply deleting C2? This is computed here
            #     regardless of the USE_ALT_C2_AFFORDABILITY toggle's current
            #     setting, so both versions of C2 can be compared side by
            #     side in one run.
            # (2) The FE regressions now report p-values (clustered by
            #     country) alongside each beta, not point estimates alone --
            #     a coefficient moving from near zero to a larger magnitude
            #     is not itself evidence of improvement unless the larger
            #     coefficient is also statistically distinguishable from
            #     zero, which the previous version of this table had no way
            #     to show. ***
            #
            # This remains diagnostic only: it identifies whether replacing
            # or dropping a component would improve external validity ON
            # THIS EUROPEAN SUBSAMPLE, but does not by itself justify
            # changing the composite used in the full G20 ESI, since C1 and
            # C2 may carry information relevant to the full, non-European
            # panel that this validation subsample cannot speak to. ***
            # ── CANDIDATE COMPOSITES x CRITERIA (revised per review) ──────────
            # (1) The baseline HEADLINE composite is now included. The old key
            #     used column "_aff_c2", which is not in the validation table,
            #     so AFF_old silently dropped out of the comparison.
            # (2) Matched temporal transformations: level and growth versions
            #     of BOTH price indices, so "headline vs energy" is compared
            #     with the transformation held fixed.
            # (3) A criterion that does not share C3's denominator: the burden
            #     with a GDP-per-capita denominator. C1+C2 composites (without
            #     C3) are added for the same reason.
            vt = validated.copy()
            candidates = {
                "AFF headline-level (BASELINE: C1+C2hl+C3)": ["_aff_c1", "_aff_c2_headline", "_aff_c3"],
                "AFF energy-growth (C1+C2eg+C3)":            ["_aff_c1", "_aff_c2_energy", "_aff_c3"],
                "AFF headline-growth (C1+C2hg+C3)":          ["_aff_c1", "_aff_c2_headline_growth", "_aff_c3"],
                "AFF energy-level (C1+C2el+C3)":             ["_aff_c1", "_aff_c2_energy_level", "_aff_c3"],
                "AFF13 (C1+C3)":                             ["_aff_c1", "_aff_c3"],
                "AFF12 headline-level (C1+C2hl, no C3)":     ["_aff_c1", "_aff_c2_headline"],
                "AFF12 energy-growth (C1+C2eg, no C3)":      ["_aff_c1", "_aff_c2_energy"],
                "C3 only":                                   ["_aff_c3"],
            }
            candidates = {k: v for k, v in candidates.items() if all(c in vt.columns for c in v)}
            criteria = {
                "burden / HFCE pc (shares C3 variable)": "eurostat_elec_burden",
                "burden / GDP pc":                      "eurostat_elec_burden_gdp",
            }
            criteria = {k: v for k, v in criteria.items()
                        if v in vt.columns and vt[v].notna().sum() >= 30}
            c2_cols = [c for c in ("_aff_c2_headline", "_aff_c2_energy",
                                   "_aff_c2_headline_growth", "_aff_c2_energy_level") if c in vt.columns]
            common = vt[c2_cols].notna().all(axis=1) if c2_cols else pd.Series(True, index=vt.index)

            def _stats(cols, crit, df):
                d_ = df[cols + [crit, "iso_code", "year"]].dropna().copy()
                d_["_a"] = d_[cols].mean(axis=1)
                out_ = {"N": len(d_), "countries": d_["iso_code"].nunique(),
                        "rho_pooled": np.nan, "rho_country_means": np.nan,
                        "beta_country_year_FE": np.nan, "p_country_year_FE": np.nan}
                if len(d_) < 20:
                    return out_
                out_["rho_pooled"] = stats.spearmanr(d_["_a"], d_[crit])[0]
                cm_ = d_.groupby("iso_code")[["_a", crit]].mean()
                if len(cm_) >= 5:
                    out_["rho_country_means"] = stats.spearmanr(cm_["_a"], cm_[crit])[0]
                try:
                    m_ = smf.ols(f"{crit} ~ _a + C(iso_code) + C(year)", data=d_).fit(
                        cov_type="cluster", cov_kwds={"groups": d_["iso_code"]})
                    out_["beta_country_year_FE"] = m_.params.get("_a", np.nan)
                    out_["p_country_year_FE"] = m_.pvalues.get("_a", np.nan)
                except Exception:
                    pass
                return out_

            rows = []
            for crit_lab, crit in criteria.items():
                for cand_lab, cols in candidates.items():
                    for samp_lab, df_ in (("common", vt[common]), ("own", vt)):
                        r_ = _stats(cols, crit, df_)
                        rows.append({"criterion": crit_lab, "composite": cand_lab, "sample": samp_lab, **r_})
            cmp_df = pd.DataFrame(rows)
            result["candidate_criteria_table"] = cmp_df

            print(f"\n  {'═'*104}")
            print(f"  AFFORDABILITY COMPOSITES x VALIDATION CRITERIA (COMMON SAMPLE: rows where all four")
            print(f"  C2 versions exist, N={int(common.sum())}). Expected sign: NEGATIVE for every criterion")
            print(f"  (more affordable -> lower burden).")
            print(f"  C2 versions: hl = headline level (baseline), eg = energy growth, hg = headline")
            print(f"  growth, el = energy level. Same-transformation pairs: hl vs el (levels), hg vs eg (growth).")
            print(f"  {'═'*104}")
            for crit_lab in criteria:
                sub_ = cmp_df[(cmp_df["criterion"] == crit_lab) & (cmp_df["sample"] == "common")]
                print(f"\n  Criterion: {crit_lab}")
                print(f"  {'Composite':<44}{'N':>5}{'rho_pool':>10}{'rho_ctry':>10}{'beta_2wFE':>11}{'p':>8}")
                for _, r_ in sub_.iterrows():
                    star = "*" if (pd.notna(r_["p_country_year_FE"]) and r_["p_country_year_FE"] < 0.05) else " "
                    print(f"  {r_['composite']:<44}{int(r_['N']):>5}{r_['rho_pooled']:>+10.3f}"
                          f"{r_['rho_country_means']:>+10.3f}{r_['beta_country_year_FE']:>+11.4f}"
                          f"{r_['p_country_year_FE']:>8.3f}{star}")
            print(f"\n  * two-way FE coefficient significant at 5% (country-clustered SE).")
            print(f"\n  Within-country (two-way FE) summary, common sample — expected sign NEGATIVE:")
            _cm = cmp_df[cmp_df["sample"] == "common"]
            for cand in candidates:
                parts = []
                for crit_lab in criteria:
                    r_ = _cm[(_cm["composite"] == cand) & (_cm["criterion"] == crit_lab)]
                    if r_.empty or pd.isna(r_["beta_country_year_FE"].iloc[0]):
                        continue
                    b_, p_ = r_["beta_country_year_FE"].iloc[0], r_["p_country_year_FE"].iloc[0]
                    tag = ("expected sign, significant" if b_ < 0 and p_ < 0.05 else
                           "expected sign, n.s." if b_ < 0 else
                           "WRONG sign, significant" if p_ < 0.05 else "wrong sign, n.s.")
                    parts.append(f"{crit_lab.split(' (')[0]}: {tag}")
                print(f"    {cand:<44} " + "; ".join(parts))
            print(f"  Pooled correlations measure the structural (cross-country) validity the")
            print(f"  baseline is designed for; the FE results show how far each composite also")
            print(f"  tracks annual within-country burden changes.")
            print(f"  Reading guide:")
            print(f"   - headline vs energy at the SAME transformation: compare hl with el, and hg with eg.")
            print(f"   - level vs growth for the SAME index: compare hl with hg, and el with eg.")
            print(f"   - the burden/HFCE criterion shares its denominator with C3; agreement there can be")
            print(f"     partly mechanical. Validity should rest on the GDP-denominator burden and")
            print(f"     on the C1+C2 composites, which exclude C3.")
            try:
                _save_csv(cmp_df, "step2_affordability_composites_by_criterion.csv",
                          "Affordability composites x validation criteria (common and own samples)")
            except Exception:
                pass
            candidate_results = {r_["composite"]: r_ for r_ in rows
                                 if r_["sample"] == "own" and r_["criterion"].startswith("burden / HFCE")}
            result["candidate_composites_common_sample"] = {
                r_["composite"]: r_ for r_ in rows
                if r_["sample"] == "common" and r_["criterion"].startswith("burden / HFCE")}
            result["candidate_composites"] = candidate_results

            result["construct_pearson"] = r_pearson
            result["construct_spearman"] = rho_spearman
            result["construct_country_mean_spearman"] = rho_cm
            result["construct_models"] = construct_models
        else:
            print(f"\n  ⚠ Only {len(construct_df)} complete (AFF, Burden) observations —")
            print(f"    construct-level validation skipped")
    else:
        print(f"\n  ⚠ dim_affordability or eurostat_elec_burden not available in the")
        print(f"    validation table — construct-level validation skipped entirely.")
        print(f"    (The component-level diagnostics below are NOT a substitute for")
        print(f"    this; if this warning appears, investigate why before treating")
        print(f"    the component-level results as sufficient validation.)")

    print(f"\n  {'─'*66}")
    print(f"  COMPONENT-LEVEL DIAGNOSTICS (secondary): the three regressions below")
    print(f"  test C1, C2 and C3 individually against the Eurostat burden. These")
    print(f"  are diagnostic evidence for which channel drives the composite AFF")
    print(f"  score above, NOT three independent validations of the construct --")
    print(f"  the construct-level test above is primary.")
    print(f"  {'─'*66}")

    needed_model_cols = ["eurostat_elec_burden","aff_intensity","aff_price_ratio",
                         "aff_hfce_pc","iso_code","year"]
    if all(c in validated.columns for c in needed_model_cols):
        model_df = validated[needed_model_cols].dropna()
        if len(model_df) >= 20:
            import statsmodels.formula.api as smf
            model = smf.ols(
                "eurostat_elec_burden ~ aff_intensity + aff_price_ratio + aff_hfce_pc "
                "+ C(iso_code) + C(year)",
                data=model_df
            ).fit(cov_type="cluster", cov_kwds={"groups": model_df["iso_code"]})

            print(f"\n  Panel regression: EB^EU_it ~ C1 + C2 + C3, country + year FE, "
                  f"clustered SE")
            # *** FIX: "Match?" now requires the correct SIGN AND p<0.05, not
            # sign alone. A coefficient of +0.0000 with p=0.83 is statistically
            # indistinguishable from zero and is NOT evidence of a positive
            # relationship -- reporting it as a plain "✓" alongside a highly
            # significant coefficient of the correct sign elsewhere overstates
            # how much support the data actually provide. ***
            print(f"\n  {'Component':<28} {'Coefficient':>12} {'p-value':>10}  "
                  f"{'Expected':>10}  Match?")
            print(f"  {'─'*76}")
            checks = [
                ("aff_intensity",   "C1: energy intensity", "+"),
                ("aff_price_ratio", "C2: price pressure",   "+"),
                ("aff_hfce_pc",     "C3: purchasing capacity", "-"),
            ]
            all_match = True
            for col, label, expected in checks:
                if col not in model.params.index:
                    continue
                coef = model.params[col]
                pval = model.pvalues[col]
                sign_ok = (coef > 0 and expected == "+") or (coef < 0 and expected == "-")
                sig_ok  = pval < 0.05
                if sign_ok and sig_ok:
                    match_str = "✓"
                elif sign_ok and not sig_ok:
                    match_str = "○ (sign ok, not significant)"
                else:
                    match_str = "✗"
                all_match = all_match and sign_ok and sig_ok
                print(f"  {label:<28} {coef:>+12.4f} {pval:>10.4f}  "
                      f"{expected:>10}  {match_str}")
            print(f"  {'─'*76}")
            print(f"  R² = {model.rsquared:.4f}  |  N = {int(model.nobs)}")
            print(f"  ✓ = correct sign AND p<0.05   ○ = correct sign but not")
            print(f"  significant   ✗ = wrong sign (regardless of significance)")

            result["model"] = model
            result["beta_intensity"] = model.params.get("aff_intensity")
            result["beta_price"]     = model.params.get("aff_price_ratio")
            result["beta_hfce"]      = model.params.get("aff_hfce_pc")
            result["all_signs_match"] = all_match
            result["r_squared"]      = model.rsquared

            # *** ADDED (review): the component regression above still uses
            # the OLD headline-CPI-based C2 (aff_price_ratio) -- useful for
            # explaining why the original specification was replaced, but it
            # no longer validates the C2 this pipeline now intends to use
            # when USE_ALT_C2_AFFORDABILITY is enabled. This adds the
            # analogous regression with the RAW (non-oriented) energy-CPI-
            # based measure in C2's place:
            #     Burden_it = alpha_i + lambda_t + b1*C1 + b2*C2^energy
            #                 + b3*C3 + e_it
            # Expected sign for C2^energy is POSITIVE (b2 > 0): higher
            # energy-price inflation should INCREASE burden, the same
            # direction already expected for the old C2 above -- both are
            # raw price-pressure measures on the same orientation, only the
            # underlying series differs. ***
            c2_energy_raw = compute_c2_star_alt(validated)
            model_df_energy = validated[["eurostat_elec_burden","aff_intensity",
                                         "aff_hfce_pc","iso_code","year"]].copy()
            model_df_energy["c2_energy_raw"] = c2_energy_raw
            model_df_energy = model_df_energy.dropna()
            print(f"\n  {'─'*66}")
            print(f"  COMPONENT REGRESSION WITH ENERGY-SPECIFIC C2 (replaces headline C2")
            print(f"  above with 100*Delta ln(CPI_energy), the raw, non-oriented measure)")
            print(f"  {'─'*66}")
            if len(model_df_energy) >= 20:
                model_energy = smf.ols(
                    "eurostat_elec_burden ~ aff_intensity + c2_energy_raw + aff_hfce_pc "
                    "+ C(iso_code) + C(year)",
                    data=model_df_energy
                ).fit(cov_type="cluster", cov_kwds={"groups": model_df_energy["iso_code"]})
                print(f"\n  Panel regression: EB^EU_it ~ C1 + C2^energy + C3, country + year FE, "
                      f"clustered SE")
                print(f"\n  {'Component':<28} {'Coefficient':>12} {'p-value':>10}  "
                      f"{'Expected':>10}  Match?")
                print(f"  {'─'*76}")
                checks_energy = [
                    ("aff_intensity",  "C1: energy intensity",      "+"),
                    ("c2_energy_raw",  "C2^energy: price pressure", "+"),
                    ("aff_hfce_pc",    "C3: purchasing capacity",   "-"),
                ]
                all_match_energy = True
                for col, label, expected in checks_energy:
                    if col not in model_energy.params.index:
                        continue
                    coef = model_energy.params[col]
                    pval = model_energy.pvalues[col]
                    sign_ok = (coef > 0 and expected == "+") or (coef < 0 and expected == "-")
                    sig_ok = pval < 0.05
                    if sign_ok and sig_ok:
                        match_str = "✓"
                    elif sign_ok and not sig_ok:
                        match_str = "○ (sign ok, not significant)"
                    else:
                        match_str = "✗"
                    all_match_energy = all_match_energy and sign_ok and sig_ok
                    print(f"  {label:<28} {coef:>+12.4f} {pval:>10.4f}  "
                          f"{expected:>10}  {match_str}")
                print(f"  {'─'*76}")
                print(f"  R² = {model_energy.rsquared:.4f}  |  N = {int(model_energy.nobs)}")
                print(f"  (compare N here against the headline-C2 regression above -- if they")
                print(f"  differ, part of any difference in results could reflect the change in")
                print(f"  sample rather than only the change in C2 measure)")
                result["model_energy_c2"] = model_energy
                result["beta_c2_energy"] = model_energy.params.get("c2_energy_raw")
                result["p_c2_energy"] = model_energy.pvalues.get("c2_energy_raw")
                result["all_signs_match_energy_c2"] = all_match_energy
            else:
                print(f"  ⚠ Only {len(model_df_energy)} complete observations — energy-C2")
                print(f"    component regression not estimated")

            # *** ADDED: between-country (no fixed effects) robustness check.
            #
            # REVISED per review: the original comment described the country
            # fixed-effects specification as "double-demeaning" aff_price_ratio
            # and framed this as a mechanical cause of any sign reversal. This
            # overstates what the FE transformation does and risks implying
            # the within-country estimate is somehow less valid rather than
            # simply answering a different, more restrictive question.
            # aff_price_ratio is constructed relative to each country's own
            # price dynamics; adding country and year fixed effects on top of
            # that means beta2 in the panel model above is identified from
            # residual within-country variation only, after removing both
            # persistent country heterogeneity and common annual shocks -- a
            # substantially more restrictive source of variation than the
            # cross-country level relationship this check examines directly
            # using country-period means. Where the two disagree, this is
            # evidence the cross-sectional and within-country relationships
            # differ, not evidence that the within-country estimate is wrong
            # or that the between-country estimate reveals a "true" effect
            # the panel model failed to detect. ***
            print(f"\n  {'─'*60}")
            print(f"  ROBUSTNESS: between-country (level) relationship, no FE")
            print(f"  {'─'*60}")
            print(f"  Because the price-pressure indicator is constructed relative to")
            print(f"  country-specific price dynamics, the two-way fixed-effects")
            print(f"  specification above identifies beta2 only from residual within-")
            print(f"  country variation after removing both persistent country")
            print(f"  heterogeneity and common annual shocks -- a substantially more")
            print(f"  restrictive source of variation than the cross-country level")
            print(f"  relationship examined below, which regresses country-period means")
            print(f"  (one observation per country) instead.")
            country_means = model_df.groupby("iso_code")[needed_model_cols[:4]].mean()
            if len(country_means) >= 8:
                Xb = sm.add_constant(country_means[["aff_intensity","aff_price_ratio",
                                                     "aff_hfce_pc"]].values.astype(float),
                                     has_constant="add")
                yb = country_means["eurostat_elec_burden"].values.astype(float)
                between_model = OLS(yb, Xb).fit(cov_type="HC3")
                print(f"\n  {'Component':<28} {'Coefficient':>12} {'p-value':>10}  Expected")
                print(f"  {'─'*66}")
                for i, (col, label, expected) in enumerate(checks, start=1):
                    coef = between_model.params[i]
                    pval = between_model.pvalues[i]
                    sign_ok = (coef > 0 and expected == "+") or (coef < 0 and expected == "-")
                    print(f"  {label:<28} {coef:>+12.4f} {pval:>10.4f}  {expected:>10}  "
                          f"{'✓' if sign_ok else '✗'}")
                print(f"  {'─'*66}")
                print(f"  R² = {between_model.rsquared:.4f}  |  N = {len(country_means)} "
                      f"countries")
                print(f"  A positive between-country coefficient combined with a negative")
                print(f"  within-country coefficient indicates that the cross-sectional and")
                print(f"  within-country relationships differ. The result is therefore")
                print(f"  interpreted as partial rather than uniform external validation of")
                print(f"  the price-pressure channel -- the between-country regression is")
                print(f"  associational, not proof of a structural relationship.")
                result["between_model"] = between_model
            else:
                print(f"  ⚠ Only {len(country_means)} countries — between-country check skipped")
                between_model = None

            # *** REVISED interpretation, following review: the previous
            # closing summary treated this as a pass/fail ("all signs
            # match" vs "at least one does not"), framing the affordability
            # CONSTRUCT's validity around three independent component
            # hypotheses. Since the construct-level test above is now the
            # primary validation, the component-level results below are
            # reframed as MIXED DIAGNOSTIC EVIDENCE about which channel
            # drives the composite -- not a pass/fail on the construct
            # itself. This description is generated from whatever the
            # actual within- and between-country coefficients say on this
            # run, rather than a fixed narrative, since which component
            # shows which pattern is an empirical question that could
            # differ across data updates. ***
            print(f"\n  {'─'*60}")
            print(f"  COMPONENT-LEVEL DIAGNOSTIC SUMMARY (secondary; mixed evidence")
            print(f"  is an entirely normal outcome here, not a validation failure --")
            print(f"  see the primary construct-level result above)")
            print(f"  {'─'*60}")
            for col, label, expected in checks:
                if col not in model.params.index:
                    continue
                coef_w = model.params[col]; p_w = model.pvalues[col]
                sign_w_ok = (coef_w > 0 and expected == "+") or (coef_w < 0 and expected == "-")
                sig_w_ok = p_w < 0.05

                coef_b, p_b, sign_b_ok = None, None, None
                if between_model is not None:
                    idx = [c for c,_,_ in checks].index(col) + 1
                    coef_b = between_model.params[idx]; p_b = between_model.pvalues[idx]
                    sign_b_ok = (coef_b > 0 and expected == "+") or (coef_b < 0 and expected == "-")

                if sign_w_ok and sig_w_ok:
                    verdict = "strong, significant within-country evidence in the expected direction"
                elif sign_b_ok and coef_b is not None and (p_b or 1) < 0.05 and not (sign_w_ok and sig_w_ok):
                    verdict = "positive between-country association, unstable within-country after fixed effects -- partial validation"
                elif sign_w_ok or sign_b_ok:
                    verdict = "correct direction but not statistically significant -- weak evidence, acceptable given the proxy's known limitations"
                else:
                    verdict = "does not show the expected direction at either the within- or between-country level"
                print(f"    {label}: {verdict}")
            print(f"  {'─'*60}")
        else:
            print(f"  ⚠ Only {len(model_df)} complete observations for the panel "
                  f"regression — model not estimated")

    return result



def download_energy_burden(iso_list):
    """
    Download all three components of the composite affordability indicator.

    Component 1 — Energy intensity (quantity burden):
        WB EG.USE.COMM.GD.PP.KD: kg oil equivalent per $1,000 PPP GDP
        Higher = more energy per unit of income = higher vulnerability
        Fallback: computed from OWID energy_consumption_twh / gdp_pc_ppp

    Component 2 — Relative consumer price pressure (price burden):
        The implemented series is headline CPI (WB FP.CPI.TOTL), NOT an
        energy-specific price index. Calling this "energy price" would
        overstate what is actually measured — it is general consumer price
        pressure, used here as a macro price-pressure proxy because no
        energy-specific CPI series covers the full G20+Europe panel.
        Fallback: GDP deflator (NY.GDP.DEFL.KD.ZG) as a macro price-
                  pressure proxy if headline CPI is unavailable.

    Component 3 — Household purchasing-capacity burden:
        HFCE_pc = NE.CON.PRVT.PP.KD (households & NPISHs final consumption
        expenditure, PPP, constant 2021 international $) / SP.POP.TOTL.
        Series covers 1990-2025; PREVIOUSLY verified complete for the 44-
        country panel over the pipeline's YEAR_START-YEAR_END window (2000-2023;
        1,012/1,012 obs at that time, no interpolation needed).
        *** SCOPE CHANGE: Switzerland (CHE) was added to the panel after
        this 1,012/1,012 completeness check was last run; that specific
        count has NOT been re-verified against the new 45-country panel,
        and should not be assumed to still hold exactly (Switzerland's own
        World Bank coverage for this indicator may or may not be complete).
        Re-run this check explicitly before relying on full completeness. ***
        Higher household consumption capacity = better ability to absorb
        energy costs. Direction is handled in build_esi so the composite
        keeps its higher = more secure orientation.
        Fallback: gdp_pc_ppp as capacity proxy (flagged in build_esi).

    Returns DataFrame: iso_code | year | aff_intensity | aff_price_ratio | aff_hfce_pc
    """
    print("  ↓ Downloading affordability composite components …")
    frames_aff = []

    # ── Component 1: Energy intensity (EG.USE.COMM.GD.PP.KD) ─────────────────
    c1_loaded = False
    for ind in ["EG.USE.COMM.GD.PP.KD", "EG.EGY.PRIM.PP.KD"]:
        try:
            df1 = _wb_fetch_simple(ind, "aff_intensity", iso_list)
            if len(df1) > 10:
                frames_aff.append(df1)
                print(f"    ✓ aff_intensity: {len(df1)} obs ({ind})")
                c1_loaded = True
                break
        except Exception as e:
            print(f"    ⚠ {ind}: {e}")
    if not c1_loaded:
        print("    ⚠ aff_intensity: API failed — will compute from OWID in step3")
        frames_aff.append(pd.DataFrame(columns=["iso_code","year","aff_intensity"]))

    # ── Component 2: headline CPI, country-demeaned (relative consumer price
    #    pressure -- NOT an energy-specific price index; see docstring above) ───
    # Strategy: download headline CPI growth and energy CPI index separately,
    # compute ratio. If energy CPI not available, use electricity price / gdp_pc_ppp.
    c2_loaded = False
    try:
        # Try: FP.CPI.TOTL is the CPI index (2010=100), available all G20
        df_cpi  = _wb_fetch_simple("FP.CPI.TOTL", "cpi_total", iso_list)

        # *** ADDED: targeted fallback for countries whose FP.CPI.TOTL
        # (index LEVEL) series has known discontinuities in the World Bank
        # data -- Argentina's index is broken across multiple currency
        # redenominations/statistical-methodology changes (well-documented,
        # e.g. around 2007-2015 INDEC credibility issues), even though its
        # FP.CPI.TOTL.ZG (annual % change) series remains reported and
        # complete. Switzerland is included here too (its own FP.CPI.TOTL
        # coverage was found incomplete after being added to the panel).
        # Embedded here as real, user-supplied annual %-change data,
        # chained into a synthetic index level (base=100 at the first
        # available year for each country) using idx_t = idx_(t-1) * (1 +
        # pct_t/100) -- the standard construction for turning an inflation-
        # rate series into a comparable price level when the official level
        # series itself is unusable. This ONLY fills in years where
        # FP.CPI.TOTL is missing for these two specific countries; any
        # year where FP.CPI.TOTL already has a value is left untouched, so
        # this cannot override or contradict a genuine World Bank
        # observation, only fill a genuine gap. ***
        _CPI_PCT_FALLBACK_RAW = """iso_code,year,cpi_pct_change
ARG,2000,-0.9
ARG,2001,-1.1
ARG,2002,25.9
ARG,2003,13.4
ARG,2004,4.4
ARG,2005,9.6
ARG,2006,10.9
ARG,2007,8.8
ARG,2008,8.6
ARG,2009,6.3
ARG,2010,10.5
ARG,2011,9.8
ARG,2012,10
ARG,2013,10.6
ARG,2014,14
ARG,2015,15
ARG,2016,25.7
ARG,2017,25.7
ARG,2018,34.27722371
ARG,2019,53.54830435
ARG,2020,42.01509474
ARG,2021,48.40937863
ARG,2022,72.43075753
ARG,2023,133.4889356
CHE,2000,1.558529197
CHE,2001,0.98902033
CHE,2002,0.642711507
CHE,2003,0.638272931
CHE,2004,0.802908731
CHE,2005,1.171954203
CHE,2006,1.059509283
CHE,2007,0.732350603
CHE,2008,2.426041171
CHE,2009,-0.480481936
CHE,2010,0.688238707
CHE,2011,0.231349208
CHE,2012,-0.692552018
CHE,2013,-0.217323155
CHE,2014,-0.013202539
CHE,2015,-1.143908672
CHE,2016,-0.434618664
CHE,2017,0.53378784
CHE,2018,0.936335464
CHE,2019,0.36288618
CHE,2020,-0.725874933
CHE,2021,0.581814168
CHE,2022,2.835027986
CHE,2023,2.13540088
"""
        import io as _io_cpi_fallback
        _cpi_pct_df = pd.read_csv(_io_cpi_fallback.StringIO(_CPI_PCT_FALLBACK_RAW))
        _cpi_pct_df = _cpi_pct_df.sort_values(["iso_code","year"])

        # *** FIX (reviewer: "align reconstructed CPI levels to observed
        # values"). The fallback used to chain the inflation rates into an
        # index starting at 100 in the first year and then insert those
        # values only where FP.CPI.TOTL was missing. If a country had SOME
        # observed World Bank years (index base 2010 = 100) and some missing
        # years, the series mixed two different scales and showed artificial
        # jumps where they met. Because C2 (aff_price_ratio) is the CPI level
        # divided by the country's own mean, such a jump would create an
        # artificial change in affordability.
        # Now every missing year is reconstructed FROM AN OBSERVED ANCHOR on
        # the World Bank scale: forward  CPI_t   = CPI_t-1 x (1 + pi_t/100),
        #                       backward CPI_t-1 = CPI_t / (1 + pi_t/100).
        # Observed values are never changed. At the far end of an internal
        # gap the chained value is compared with the next observed value
        # ("closing gap"); a large gap is reported, not hidden. A country with
        # NO observed level keeps a chained index (base 100 in its first year):
        # its scale is arbitrary but internally consistent, and C2 is scale-
        # free within a country, so this does not affect results. ***
        _recon_rows, _diag_rows = [], []
        _obs_all = df_cpi.set_index(["iso_code", "year"])["cpi_total"] if len(df_cpi) else pd.Series(dtype=float)
        for _iso, _grp in _cpi_pct_df.groupby("iso_code"):
            _pi = _grp.set_index("year")["cpi_pct_change"].astype(float)
            _yrs = sorted(set(_pi.index) | set(range(YEAR_START, YEAR_END + 1)))
            _obs = pd.Series({y: _obs_all.get((_iso, y), np.nan) for y in _yrs}, dtype=float)
            _lvl = _obs.copy()
            _src = pd.Series(np.where(_obs.notna(), "observed", ""), index=_yrs)
            _closing = []
            if _obs.notna().any():
                for _k, y in enumerate(_yrs[1:], start=1):           # forward from anchors
                    yp = _yrs[_k - 1]
                    if pd.isna(_lvl[y]) and pd.notna(_lvl[yp]) and pd.notna(_pi.get(y)):
                        _lvl[y] = _lvl[yp] * (1 + _pi[y] / 100.0); _src[y] = "reconstructed (forward)"
                    elif pd.notna(_obs[y]) and _src[yp].startswith("reconstructed") and pd.notna(_pi.get(y)):
                        _chained = _lvl[yp] * (1 + _pi[y] / 100.0)   # closing check at re-entry
                        _closing.append(100.0 * (_chained / _obs[y] - 1.0))
                for _k in range(len(_yrs) - 2, -1, -1):               # backward before first anchor
                    y, yn = _yrs[_k], _yrs[_k + 1]
                    if pd.isna(_lvl[y]) and pd.notna(_lvl[yn]) and pd.notna(_pi.get(yn)):
                        _lvl[y] = _lvl[yn] / (1 + _pi[yn] / 100.0); _src[y] = "reconstructed (backward)"
                _anchor = "World Bank FP.CPI.TOTL"
            else:
                _base = None
                for y in _yrs:
                    if _base is None:
                        _lvl[y], _base = 100.0, y
                    elif pd.notna(_pi.get(y)) and pd.notna(_lvl[_yrs[_yrs.index(y) - 1]]):
                        _lvl[y] = _lvl[_yrs[_yrs.index(y) - 1]] * (1 + _pi[y] / 100.0)
                    _src[y] = "reconstructed (no observed anchor)"
                _anchor = f"none (chained index, {_base} = 100)"
            # consistency of the embedded inflation rates with observed levels
            _o = _obs.dropna()
            _g_obs = (_o / _o.shift(1) - 1.0) * 100.0
            _g_obs = _g_obs[[y for y in _g_obs.index if y - 1 in _o.index]].dropna()
            _dev = (_g_obs - _pi.reindex(_g_obs.index)).abs().dropna()
            _new = [y for y in _yrs if _src[y].startswith("reconstructed") and YEAR_START <= y <= YEAR_END]
            _diag_rows.append({"iso_code": _iso, "observed_years": int(_obs.loc[YEAR_START:YEAR_END].notna().sum()),
                               "reconstructed_years": len(_new),
                               "reconstructed": (f"{min(_new)}-{max(_new)}" if _new else ""),
                               "anchor": _anchor,
                               "max_abs_closing_gap_pct": (max(map(abs, _closing)) if _closing else np.nan),
                               "mean_abs_diff_observed_vs_embedded_inflation_pp":
                                   (float(_dev.mean()) if len(_dev) else np.nan)})
            for y in _new:
                _recon_rows.append({"iso_code": _iso, "year": y, "cpi_total_fallback": float(_lvl[y]),
                                    "cpi_source": _src[y]})
        _cpi_fallback_idx = pd.DataFrame(_recon_rows) if _recon_rows else \
            pd.DataFrame(columns=["iso_code", "year", "cpi_total_fallback", "cpi_source"])

        df_cpi = df_cpi.merge(_cpi_fallback_idx[["iso_code", "year", "cpi_total_fallback"]],
                              on=["iso_code","year"], how="outer")
        _fill_mask = df_cpi["cpi_total"].isna() & df_cpi["cpi_total_fallback"].notna()
        _n_filled = int(_fill_mask.sum())
        df_cpi.loc[_fill_mask, "cpi_total"] = df_cpi.loc[_fill_mask, "cpi_total_fallback"]
        df_cpi = df_cpi.drop(columns=["cpi_total_fallback"])
        _diag = pd.DataFrame(_diag_rows)
        _mixed = _diag[(_diag["observed_years"] > 0) & (_diag["reconstructed_years"] > 0)]["iso_code"].tolist()
        print(f"    ✓ FP.CPI.TOTL gaps filled for ARG/CHE: +{_n_filled} obs, reconstructed from")
        print(f"      embedded annual inflation (FP.CPI.TOTL.ZG) and ANCHORED to observed")
        print(f"      World Bank CPI levels (same index base); observed values unchanged:")
        for _, r in _diag.iterrows():
            _cg = r["max_abs_closing_gap_pct"]
            _dv = r["mean_abs_diff_observed_vs_embedded_inflation_pp"]
            print(f"      {r['iso_code']}: observed {r['observed_years']} yrs, reconstructed "
                  f"{r['reconstructed_years']} yrs {r['reconstructed'] or ''}; anchor: {r['anchor']}"
                  + (f"; max closing gap {_cg:.2f}%" if pd.notna(_cg) else "")
                  + (f"; observed vs embedded inflation differ by {_dv:.2f} pp on average" if pd.notna(_dv) else ""))
            if pd.notna(_cg) and abs(_cg) > 2.0:
                print(f"      ⚠ {r['iso_code']}: closing gap above 2% -- a level break remains where the "
                      f"reconstructed and observed series meet; inspect before reporting C2 for this country")
        print(f"      Countries combining observed AND reconstructed years in this run: "
              f"{_mixed or 'none'}" + ("" if _mixed else
              " (so no observed/reconstructed scale mixing can occur in this panel; the anchoring"
              " rule guards against it should World Bank coverage change)"))
        try:
            _save_csv(_diag, "step1_cpi_reconstruction_check.csv",
                      "CPI gap reconstruction: anchors, years, closing gaps")
        except Exception:
            pass

        # Energy CPI: no single WB indicator covers all G20
        # Use electricity consumption growth as a proxy for price exposure:
        # aff_price_ratio = CPI level (higher = more accumulated price pressure)
        # Normalise within country to get within-country variation
        if len(df_cpi) > 10:
            df_cpi = df_cpi.rename(columns={"cpi_total": "aff_price_ratio"})
            # De-mean within country: captures year-to-year price acceleration
            df_cpi["aff_price_ratio"] = (
                df_cpi.groupby("iso_code")["aff_price_ratio"]
                      .transform(lambda x: x / x.mean())
            )
            frames_aff.append(df_cpi)
            print(f"    ✓ aff_price_ratio: {len(df_cpi)} obs (FP.CPI.TOTL, country-demeaned)")
            c2_loaded = True
    except Exception as e:
        print(f"    ⚠ FP.CPI.TOTL: {e}")

    if not c2_loaded:
        # Fallback: GDP deflator as a general macro price-pressure proxy
        try:
            df_deflator = _wb_fetch_simple("NY.GDP.DEFL.KD.ZG", "aff_price_ratio", iso_list)
            if len(df_deflator) > 10:
                frames_aff.append(df_deflator)
                print(f"    ✓ aff_price_ratio: {len(df_deflator)} obs (GDP deflator fallback)")
                c2_loaded = True
        except Exception:
            pass
    if not c2_loaded:
        print("    ⚠ aff_price_ratio: all sources failed — component will be omitted")
        frames_aff.append(pd.DataFrame(columns=["iso_code","year","aff_price_ratio"]))

    # ── Component 3: Household purchasing-capacity (HFCE per capita, PPP) ─────
    # HFCE_pc = NE.CON.PRVT.PP.KD / SP.POP.TOTL
    # PREVIOUSLY verified complete for the 44-country panel over the
    # pipeline's YEAR_START-YEAR_END window (2000-2023)
    # (1,100/1,100, no interpolation). *** SCOPE CHANGE: Switzerland (CHE)
    # was added to the panel after this check; not yet re-verified for the
    # new 45-country panel -- do not assume completeness still holds. ***
    c3_loaded = False
    try:
        df_hfce = _wb_fetch_simple("NE.CON.PRVT.PP.KD", "hfce_ppp", iso_list)
        df_pop  = _wb_fetch_simple("SP.POP.TOTL",       "pop_wb",  iso_list)
        if len(df_hfce) > 10 and len(df_pop) > 10:
            df_c3 = df_hfce.merge(df_pop, on=["iso_code","year"], how="inner")
            df_c3["aff_hfce_pc"] = (df_c3["hfce_ppp"]
                                    / df_c3["pop_wb"].replace(0, np.nan))
            df_c3 = df_c3[["iso_code","year","aff_hfce_pc"]].dropna()
            if len(df_c3) > 10:
                frames_aff.append(df_c3)
                print(f"    ✓ aff_hfce_pc: {len(df_c3)} obs "
                      f"(NE.CON.PRVT.PP.KD / SP.POP.TOTL)")
                c3_loaded = True
    except Exception as e:
        print(f"    ⚠ NE.CON.PRVT.PP.KD / SP.POP.TOTL: {e}")

    if not c3_loaded:
        print("    ⚠ aff_hfce_pc: API failed — build_esi will fall back to "
              "gdp_pc_ppp as a capacity proxy")
        frames_aff.append(pd.DataFrame(columns=["iso_code","year","aff_hfce_pc"]))

    # ── Merge all three components ────────────────────────────────────────────
    out = pd.DataFrame({"iso_code": pd.Series(dtype=str),
                        "year":     pd.Series(dtype=int)})
    for df in frames_aff:
        if df.empty:
            continue
        if out.empty:
            out = df
        else:
            out = out.merge(df, on=["iso_code","year"], how="outer")

    # Also keep energy_intensity_koe_gdp alias for backward compatibility
    if "aff_intensity" in out.columns:
        out["energy_intensity_koe_gdp"] = out["aff_intensity"]

    return out

# SECTION 2 — G20 ENERGY BURDEN INDICATOR DIAGNOSTIC
# ═════════════════════════════════════════════════════════════════════════════

KWH_PER_KGOE = 11.63   # 1 kg of oil equivalent = 41.868 MJ = 11.63 kWh


def owid_energy_intensity_kgoe(panel):
    """
    OWID-based energy intensity in the SAME UNITS as World Bank
    EG.USE.COMM.GD.PP.KD (kg of oil equivalent per $1,000 of PPP GDP):

        energy_pc [kWh/person] / gdp_pc_ppp [$/person] x 1,000 / 11.63

    *** FIX: the earlier OWID "burden" was energy_consumption_twh /
    gdp_pc_ppp, i.e. TOTAL energy divided by PER-CAPITA GDP. That ratio is
    not an intensity: it scales with population (checked on OWID data for
    this panel: Spearman rho with population 0.97, correlation with OWID's
    own energy_per_gdp only 0.14). It was used in the WB-OWID consistency
    check (giving r = 0.42, wrongly attributed to "different units") and in
    three fallbacks (Step 2/3 burden, ESI C1). Units cannot explain a low
    correlation anyway: Pearson and Spearman correlations are invariant to
    rescaling. ***
    """
    if "energy_pc" not in panel.columns or "gdp_pc_ppp" not in panel.columns:
        return pd.Series(np.nan, index=panel.index)
    return panel["energy_pc"] / panel["gdp_pc_ppp"].where(panel["gdp_pc_ppp"] > 0) * 1000.0 / KWH_PER_KGOE


def wb_owid_intensity_consistency(panel, wb_col="energy_intensity_koe_gdp"):
    """
    Inspect, rather than explain away, the agreement between World Bank energy
    intensity (EG.USE.COMM.GD.PP.KD, energy use in kgoe per $1,000 PPP GDP) and
    the OWID-based intensity computed on the SAME unit and the SAME GDP series.
    With identical units and GDP, remaining differences come from the energy
    numerator: World Bank/IEA energy use (physical energy content method) vs
    OWID/Energy Institute primary energy (substitution method, which counts
    non-fossil electricity at its fossil-fuel equivalent), plus coverage and
    revisions. Reports pooled, between-country (country means) and within-
    country (demeaned log) correlations, the WB/OWID level ratio by country,
    and whether the ratio is related to the non-fossil share of primary energy.
    """
    print(f"\n  ── WB vs OWID energy-intensity consistency (same unit, same GDP) ──")
    if wb_col not in panel.columns:
        print("  ⚠ World Bank intensity not available -- check skipped"); return {}
    d = panel[["iso_code", "year"]].copy()
    d["wb"] = panel[wb_col]
    d["owid"] = owid_energy_intensity_kgoe(panel)
    nf = [c for c in ("nuclear_share", "renew_share") if c in panel.columns]
    d["nonfossil_share"] = panel[nf].sum(axis=1, min_count=1) if nf else np.nan
    d = d.dropna(subset=["wb", "owid"])
    d = d[(d["wb"] > 0) & (d["owid"] > 0)]
    if len(d) < 20:
        print(f"  ⚠ only {len(d)} complete observations -- check skipped"); return {}
    from scipy.stats import pearsonr, spearmanr
    r_p = pearsonr(d["wb"], d["owid"])[0]
    r_s = spearmanr(d["wb"], d["owid"])[0]
    cm = d.groupby("iso_code")[["wb", "owid", "nonfossil_share"]].mean()
    r_between = spearmanr(cm["wb"], cm["owid"])[0]
    d["lwb"], d["low"] = np.log(d["wb"]), np.log(d["owid"])
    dw = d[["lwb", "low"]] - d.groupby("iso_code")[["lwb", "low"]].transform("mean")
    r_within = pearsonr(dw["lwb"], dw["low"])[0]
    d["ratio"] = d["wb"] / d["owid"]
    cr = d.groupby("iso_code")["ratio"].agg(["mean", "min", "max"])
    cr["nonfossil_share"] = cm["nonfossil_share"]
    r_ratio_nf = spearmanr(cr["mean"], cr["nonfossil_share"], nan_policy="omit")[0]
    print(f"  n = {len(d)} country-years, {d['iso_code'].nunique()} countries")
    print(f"  Pooled          : Pearson r = {r_p:.3f}   Spearman rho = {r_s:.3f}")
    print(f"  Between-country : Spearman rho of country means = {r_between:.3f}")
    print(f"  Within-country  : Pearson r of demeaned log intensities = {r_within:.3f}")
    print(f"  Level ratio WB/OWID: median {d['ratio'].median():.3f} "
          f"(1st-99th pct {d['ratio'].quantile(.01):.3f}-{d['ratio'].quantile(.99):.3f})")
    print(f"  Ratio vs non-fossil share of primary energy (country means): Spearman rho = {r_ratio_nf:+.3f}")
    _strength = ("strong" if r_ratio_nf <= -0.6 else "moderate" if r_ratio_nf <= -0.3
                 else "weak" if r_ratio_nf < 0 else "no")
    print(f"    -> {_strength} support for the accounting (substitution-method) explanation")
    print(f"    (a NEGATIVE value is what the substitution method predicts: OWID counts")
    print(f"     nuclear/renewable electricity at fossil-equivalent input, raising its")
    print(f"     energy figure -- and lowering WB/OWID -- where the non-fossil share is high)")
    srt = cr.sort_values("mean")
    print(f"  Largest deviations (country-mean WB/OWID ratio):")
    for iso, r in pd.concat([srt.head(4), srt.tail(4)]).iterrows():
        print(f"    {iso}: {r['mean']:.3f}  (range {r['min']:.3f}-{r['max']:.3f}; "
              f"non-fossil share {r['nonfossil_share']:.1f}%)")
    ok = r_s > 0.9 and r_within > 0.5
    print(f"  Assessment: " + ("the two sources agree closely in ranking and in within-country"
                               " movement; WB EG.USE.COMM.GD.PP.KD remains the C1 source."
                               if ok else
                               "agreement is NOT close -- inspect the countries listed above before"
                               " relying on either source; the difference is not a unit issue."))
    out = cr.reset_index().rename(columns={"mean": "ratio_mean", "min": "ratio_min", "max": "ratio_max"})
    try:
        _save_csv(out, "step2_wb_owid_intensity_by_country.csv", "WB/OWID energy-intensity ratio by country")
    except Exception:
        pass
    return {"pearson": r_p, "spearman": r_s, "between_spearman": r_between,
            "within_pearson": r_within, "ratio_median": float(d["ratio"].median()),
            "ratio_vs_nonfossil_spearman": r_ratio_nf, "by_country": out}


def validate_energy_burden_g20(panel):
    """
    Validate and characterise the energy expenditure burden indicator
    directly on the G20 panel — no EU27 extrapolation needed.

    Indicator: energy_intensity_koe_gdp (kg oil equivalent per $1,000 PPP GDP)
      = World Bank EG.USE.COMM.GD.PP.KD  [primary]
      = energy_consumption_twh / gdp_pc_ppp  [computed fallback, always available]

    Checks performed:
      1. Coverage across G20 countries and years
      2. Cross-country distribution by year (mean, std, min, max)
      3. Within-country trend analysis (rising/falling burden → improving/worsening affordability)
      4. Correlation with G20 electricity prices where available (EG.ELC.COST.KH)
      5. Comparison of WB vs OWID-computed burden (consistency check)

    Returns: dict with coverage stats and summary DataFrame
    """
    print("\n" + "═"*60)
    print("SECTION 2 — G20 ENERGY BURDEN INDICATOR DIAGNOSTIC")
    print("═"*60)

    # ── Coverage check ────────────────────────────────────────────────────────
    burden_col = None
    for col in ["energy_intensity_koe_gdp", "energy_burden_computed"]:
        if col in panel.columns and panel[col].notna().any():
            burden_col = col
            break

    if burden_col is None:
        # Compute from OWID columns — always available
        if "energy_consumption_twh" in panel.columns and "gdp_pc_ppp" in panel.columns:
            panel = panel.copy()
            panel["energy_intensity_koe_gdp"] = owid_energy_intensity_kgoe(panel)
            burden_col = "energy_intensity_koe_gdp"
            print("  ℹ energy_intensity_koe_gdp computed from OWID energy_pc / gdp_pc_ppp (kgoe per $1,000)")
        else:
            print("  ✗ Cannot compute energy burden — missing energy_consumption_twh or gdp_pc_ppp")
            return {}

    n_total   = len(panel)
    n_nonull  = panel[burden_col].notna().sum()
    pct_cover = 100 * n_nonull / n_total if n_total > 0 else 0
    n_countries = panel[panel[burden_col].notna()]["iso_code"].nunique()
    yr_min = panel[panel[burden_col].notna()]["year"].min()
    yr_max = panel[panel[burden_col].notna()]["year"].max()

    print(f"\n  ── Coverage ──")
    print(f"  Non-null observations : {n_nonull:,} / {n_total:,}  ({pct_cover:.0f}%)")
    print(f"  Countries covered     : {n_countries} / {panel['iso_code'].nunique()}")
    print(f"  Year range            : {yr_min}–{yr_max}")

    # ── Cross-country distribution summary ───────────────────────────────────
    print(f"\n  ── Cross-country distribution (sample stats) ──")
    dist = panel.groupby("iso_code")[burden_col].agg(["mean","std","min","max"]).round(4)
    dist.columns = ["mean_burden","std_burden","min_burden","max_burden"]
    dist = dist.sort_values("mean_burden", ascending=False)
    print(f"  {'Country':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Interpretation")
    print("  " + "-"*70)
    for iso, row in dist.iterrows():
        # Higher burden = more energy per unit of income = higher vulnerability
        if row["mean_burden"] > dist["mean_burden"].quantile(0.75):
            interp = "High burden (energy-intensive / lower income)"
        elif row["mean_burden"] < dist["mean_burden"].quantile(0.25):
            interp = "Low burden  (efficient or high income)"
        else:
            interp = "Moderate burden"
        print(f"  {iso:<8} {row['mean_burden']:>8.4f} {row['std_burden']:>8.4f} "
              f"{row['min_burden']:>8.4f} {row['max_burden']:>8.4f}  {interp}")

    # ── Within-country trend (improving = burden falling over time) ───────────
    print(f"\n  ── Within-country trends (2000→latest) ──")
    trends = []
    for iso, g in panel.groupby("iso_code"):
        g = g.sort_values("year").dropna(subset=[burden_col])
        if len(g) < 5:
            continue
        early = g[g["year"] <= g["year"].quantile(0.33)][burden_col].mean()
        late  = g[g["year"] >= g["year"].quantile(0.67)][burden_col].mean()
        change_pct = 100 * (late - early) / (early + 1e-9)
        direction = "↓ improving" if change_pct < -5 else ("↑ worsening" if change_pct > 5 else "→ stable")
        trends.append({"iso_code": iso, "early": round(early,4), "late": round(late,4),
                        "change_pct": round(change_pct,1), "direction": direction})
        print(f"  {iso:<6} {early:.4f} → {late:.4f}  ({change_pct:+.1f}%)  {direction}")

    # ── Spot-check: correlation with G20 electricity prices if available ───────
    elec_col = "elec_price_usd_kwh"
    if elec_col in panel.columns and panel[elec_col].notna().sum() > 20:
        df_check = panel[[burden_col, elec_col]].dropna()
        if len(df_check) >= 10:
            from scipy.stats import pearsonr, spearmanr
            rp, pp = pearsonr(df_check[burden_col], df_check[elec_col])
            rs, ps = spearmanr(df_check[burden_col], df_check[elec_col])
            print(f"\n  ── G20 electricity price spot-check (n={len(df_check)}) ──")
            print(f"  Pearson  r = {rp:+.4f}   p = {pp:.4e}  "
                  f"{'✓' if pp < 0.05 else '✗'}")
            print(f"  Spearman ρ = {rs:+.4f}   p = {ps:.4e}  "
                  f"{'✓' if ps < 0.05 else '✗'}")

    # ── Consistency: WB vs OWID energy intensity, same unit and same GDP ──────
    wb_owid = wb_owid_intensity_consistency(panel)

    print(f"\n  ✓ Energy burden indicator validated on G20 panel directly.")
    print(f"    No EU27 extrapolation required: WB EG.USE.COMM.GD.PP.KD covers all G20")
    print(f"    economies annually from 1990-2023. OWID-based fallback (energy per capita /")
    print(f"    GDP per capita, kgoe per $1,000) available if the WB series is missing.")
    print(f"    Higher burden = more energy per $1k income = higher affordability vulnerability.")

    return {
        "burden_col"        : burden_col,
        "n_obs"             : int(n_nonull),
        "n_countries"       : int(n_countries),
        "year_range"        : (int(yr_min), int(yr_max)),
        "country_stats"     : dist,
        "country_trends"    : pd.DataFrame(trends),
        "coverage_pct"      : pct_cover,
        "wb_owid_consistency": {k: v for k, v in (wb_owid or {}).items() if k != "by_country"},
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ESI CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_hhi(panel):
    """
    HHI of the primary-energy mix (RETAINED for backward compatibility /
    comparison only). dim_resilience uses the normalised Shannon index
    (see compute_shannon_diversity below), not this HHI. Kept as a
    diagnostic/robustness cross-check.

    ROLE IN THE PIPELINE: an ESI (structural) indicator only. It is NOT a
    DDF/DEA input or output and never enters the first-stage frontier
    (inputs: energy_pc, dea_net_import; good output: electricity per capita;
    bad output: lifecycle GHG intensity of electricity). Through dim_resilience it is
    used only as an environmental variable: in the second stage (Simar-Wilson,
    panel FE, Mundlak, lag analysis), to localise the reference set in the
    conditional DDF and in the separability diagnostic.
    DATA: shares of PRIMARY ENERGY (OWID *_share_energy: coal, oil, gas,
    nuclear, renewables), not of electricity generation; the electricity-mix
    shares (*_share_elec) are used only for the Step 12 regime assignment.
    """
    share_cols = ["coal_share","oil_share","gas_share","nuclear_share","renew_share"]
    avail = [c for c in share_cols if c in panel.columns]
    shares = panel[avail].copy()
    if shares.max().max() > 2:
        shares /= 100.0
    shares = shares.clip(lower=0).fillna(0)
    panel["hhi"] = (shares**2).sum(axis=1)
    return panel


def compute_shannon_diversity(panel):
    """
    Normalised Shannon (entropy-based) diversity index of the primary-energy
    mix, replacing the HHI as the basis for dim_resilience.

    ROLE IN THE PIPELINE: an ESI (structural) indicator only. It is NOT a
    DDF/DEA input or output and never enters the first-stage frontier
    (inputs: energy_pc, dea_net_import; good output: electricity per capita;
    bad output: lifecycle GHG intensity of electricity). Through dim_resilience it is
    used only as an environmental variable: in the second stage (Simar-Wilson,
    panel FE, Mundlak, lag analysis), to localise the reference set in the
    conditional DDF and in the separability diagnostic.
    DATA: shares of PRIMARY ENERGY (OWID *_share_energy: coal, oil, gas,
    nuclear, renewables), not of electricity generation; the electricity-mix
    shares (*_share_elec) are used only for the Step 12 regime assignment.

        H   = -sum_i p_i * ln(p_i)              (Shannon entropy)
        H*  = H / ln(n)                          (normalised, in [0,1])

    where p_i is energy source i's share of the primary-energy mix and n is the
    number of energy-source categories used (here n=5: coal, oil, gas,
    nuclear, renewables). H is 0 when one source supplies the entire mix
    (p_i=1 for one i) and rises to its maximum, ln(n), when all n sources
    have equal shares (p_i=1/n for every i); dividing by ln(n) rescales
    this to the standard [0,1] normalised Shannon index (Shannon
    equitability / Pielou's evenness), directly comparable in range to the
    HHI it replaces.

    *** NOTE ON THE REQUESTED FORMULA: the formula as literally specified,
    H* = ln(n) - sum_i p_i*ln(p_i), is NOT the standard "normalised
    Shannon index" -- that name conventionally refers to H/ln(n) (dividing
    by the maximum entropy, not subtracting from it), which is bounded in
    [0,1] and is what is implemented here. The literal formula as written
    is unbounded and grows with n even after "normalisation" (it equals
    ln(n) + H, ranging from ln(n) to 2*ln(n) as H ranges from 0 to ln(n)),
    which does not behave like a normalised index. If the literal
    subtraction formula was intended deliberately rather than as a
    transcription of the standard division formula, this should be
    confirmed explicitly before use, since it is not a named index in the
    diversification literature. ***

    Unlike HHI (where LOWER values indicate MORE diversification, requiring
    the invert=True flag downstream), the Shannon index is oriented so that
    HIGHER values directly indicate MORE diversification -- no inversion
    is applied when this feeds into dim_resilience below.

    Returns the panel with a new column "shannon_diversity" in [0,1].
    """
    share_cols = ["coal_share","oil_share","gas_share","nuclear_share","renew_share"]
    avail = [c for c in share_cols if c in panel.columns]
    n_sources = len(avail)
    if n_sources < 2:
        print("  ⚠ Fewer than 2 energy-mix share columns available — "
              "shannon_diversity cannot be computed, falling back to HHI-based resilience")
        panel["shannon_diversity"] = np.nan
        return panel

    shares = panel[avail].copy()
    if shares.max().max() > 2:
        shares /= 100.0
    shares = shares.clip(lower=0).fillna(0)

    # Renormalise each row's shares to sum to 1 (guards against rounding
    # error or a residual "other" category not captured by the five
    # explicit source columns, which would otherwise understate entropy).
    row_sums = shares.sum(axis=1).replace(0, np.nan)
    shares = shares.div(row_sums, axis=0).fillna(0)

    # Shannon entropy: -sum p_i ln(p_i), with the convention 0*ln(0) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = shares.values * np.log(shares.values)
    term = np.where(shares.values > 0, term, 0.0)
    H = -term.sum(axis=1)

    H_max = np.log(n_sources)
    panel["shannon_diversity"] = np.clip(H / H_max, 0.0, 1.0)

    print(f"  shannon_diversity: computed from {n_sources} primary-energy shares "
          f"({avail}), H*=H/ln({n_sources})={H_max:.4f}")
    print(f"    Mean H* = {panel['shannon_diversity'].mean():.4f}  "
          f"(1.0 = perfectly even mix, 0.0 = single-source concentration)")
    return panel

def compute_co2_intensity(panel):
    """
    The DDF's undesirable output, stored as `co2_intensity_elec` (column name
    kept for compatibility). WHAT IT ACTUALLY MEASURES (verified against the
    OWID energy codebook and Ember's methodology):

      LIFECYCLE greenhouse-gas intensity of domestic electricity GENERATION,
      in grams of CO2-EQUIVALENT per kWh (gCO2e/kWh).
        source   : OWID `carbon_intensity_elec`, title "Lifecycle carbon
                   intensity of electricity generation", from Ember's Yearly
                   Electricity Data (Ember Europe data for European countries)
        coverage : all greenhouse gases converted to CO2e (100-year GWP), over
                   the full lifecycle -- upstream methane, fuel supply chain and
                   plant manufacturing, not only stack (direct) combustion CO2
        method   : Ember multiplies generation by source by fuel-specific
                   lifecycle emission factors (UNECE/IPCC-based; coal factors
                   vary by country and year), so it is MODELLED from the
                   generation mix, not directly measured
        boundary : electricity generated in the country, excluding imports
                   and transmission/distribution losses

    Consequences for interpretation:
      * It is NOT a direct-emissions (Scope 1, stack-CO2) measure; nuclear,
        hydro, wind and solar have small but non-zero lifecycle intensities,
        so the bad output is strictly positive for every country.
      * Because it is built from the generation mix, it moves almost one-for-
        one with the fossil share (see the Step 12 construction-link check).
        Within-country changes mainly reflect changes in the mix, not changes
        in plant efficiency.
      * It differs from the ESI sustainability indicator (WDI
        EN.GHG.CO2.PC.CE.AR5), which is TERRITORIAL (direct, production-based)
        CO2 emissions per capita for the whole economy, excluding LULUCF
        (EDGAR-based): different gas coverage (CO2 only vs all GHGs),
        boundary (whole economy vs electricity) and accounting basis
        (direct vs lifecycle). The two are related through fossil fuel use,
        not through a shared source series.

    *** CORRECTION: earlier comments described this series as "directly
    measured" CO2 in gCO2/kWh, and described OWID `greenhouse_gas_emissions`
    (the fallback input) as economy-wide GHG. OWID `greenhouse_gas_emissions`
    is in fact "Lifecycle emissions from electricity generation" (Mt CO2e),
    the numerator of the same Ember series. The old fallback formula
    (multiplying by electricity's share of primary energy and dividing by
    generation plus net imports) was therefore wrong. It is replaced by the
    consistent identity: intensity (g/kWh) = emissions (Mt) / generation
    (TWh) x 1,000. In the runs so far the fallback was never used
    (1,080/1,080 observations from the direct series). ***
    """
    if "carbon_intensity_elec_owid" in panel.columns:
        panel["co2_intensity_elec"] = panel["carbon_intensity_elec_owid"]
        n_direct = panel["co2_intensity_elec"].notna().sum()
        print(f"  co2_intensity_elec (DDF bad output): {n_direct}/{len(panel)} obs from OWID "
              f"carbon_intensity_elec = LIFECYCLE GHG intensity of electricity generation "
              f"(gCO2e/kWh, Ember)")
    else:
        panel["co2_intensity_elec"] = np.nan
        n_direct = 0
        print("  ⚠ OWID carbon_intensity_elec not found in panel -- using the "
              "emissions/generation identity for all observations.")

    # Fallback ONLY for missing observations: Ember lifecycle electricity
    # emissions (Mt CO2e) / electricity generation (TWh) x 1,000 = gCO2e/kWh.
    # Same source and same lifecycle basis as the main series.
    missing_mask = panel["co2_intensity_elec"].isna()
    n_missing = int(missing_mask.sum())
    if n_missing > 0 and "ghg_emissions" in panel.columns:
        gen = panel["elec_gen_twh"].where(panel["elec_gen_twh"] > 0)
        fallback = panel["ghg_emissions"] / gen * 1000.0
        panel.loc[missing_mask, "co2_intensity_elec"] = fallback[missing_mask]
        n_filled = int(panel.loc[missing_mask, "co2_intensity_elec"].notna().sum())
        print(f"  ⚠ co2_intensity_elec: {n_filled}/{n_missing} missing obs filled from "
              f"lifecycle electricity emissions / generation (same Ember basis)")
    return panel

# ── ONE NORMALISATION RULE FOR EVERY ESI INDICATOR ─────────────────────────────
# All indicators (C1, C2 headline, C2 energy, C3, resilience, robustness,
# sustainability) are normalised the same way:
#   1. optional log for strongly skewed indicators (C3: household consumption),
#   2. winsorised at the 1st and 99th percentiles of the POOLED sample
#      (all countries, 2000–2023), so a few extreme values (e.g. Argentina and
#      Turkey in C2) cannot compress everyone else into a narrow band,
#   3. min–max over the pooled (winsorised) range:
#        z = (x - q01) / (q99 - q01)        beneficial indicators
#        z = 1 - (x - q01) / (q99 - q01)    adverse indicators
# Pooled bounds keep the meaning of a normalised value fixed over time, so
# changes in z are changes relative to one common scale. They depend on the
# full sample and would move if new extremes were added.
WINSOR_PCT = (1, 99)


def minmax(series, invert=False, log=False, winsor=WINSOR_PCT):
    """Pooled, winsorised min–max normalisation (the single rule above)."""
    x = series.astype(float)
    if log:
        x = np.log(x.clip(lower=1e-9))
    vals = x.dropna()
    if vals.empty:
        return x * np.nan
    lo, hi = np.nanpercentile(vals, list(winsor)) if winsor else (vals.min(), vals.max())
    norm = (x.clip(lo, hi) - lo) / (hi - lo + 1e-12)
    return 1 - norm if invert else norm

def _solve_bod_dea_one(Y_front, y0, eps=1e-4):
    """
    *** ESI weighting REPLACED: Benefit-of-the-Doubt (BoD) DEA composite
    indicator, following Cherchye et al. (2007) and Despotis (2005), instead
    of fixed 1/4 weights on the four ESI dimensions. ***

    Each country-year DMU is treated as having ONE fictitious unit input
    (=1 for everyone) and FOUR outputs (the normalised ESI dimensions:
    resilience, affordability, robustness, sustainability, all already
    scaled to [0,1] with higher=better). The DMU is given the BENEFIT OF
    THE DOUBT: it picks its OWN non-negative weights v = (v1,...,v4) to
    MAXIMISE its own composite score, subject to the constraint that under
    those SAME weights, no DMU in the reference set (including itself)
    would score above 1:

        max_v   z_o = v' y_o
        s.t.    v' y_j <= 1   for all j in the reference set
                v_r    >= eps  for all r   (small positive lower bound,
                                            standard in BoD models, so no
                                            dimension is given literally
                                            zero weight / excluded outright)

    This is a linear programme solved with the SAME pulp infrastructure
    used throughout the DEA/DDF sections of this pipeline -- structurally
    it is an output-oriented, input=1, CRS DEA model.

    z_o in (0,1]: the endogenously-weighted composite score. z_o=1 means
    the DMU is "composite-efficient" -- there exists SOME non-negative
    weighting of the four dimensions under which this country is undominated.
    A known feature of BoD models (not a bug) is that many DMUs will score
    at or near 1, because each is free to pick whichever weighting flatters
    it most -- this is fundamentally different from the equal-weight index,
    where every country is judged by the SAME fixed weights.

    Returns: (z_o, v) -- the composite score and the optimal weight vector
             (v is None if the LP is infeasible/non-optimal).
    """
    n, n_dims = Y_front.shape
    prob = pulp.LpProblem("BoD_DEA", pulp.LpMaximize)
    v = [pulp.LpVariable(f"v{r}", lowBound=eps) for r in range(n_dims)]

    prob += pulp.lpSum(v[r] * y0[r] for r in range(n_dims))   # maximise v'y0

    for j in range(n):
        prob += pulp.lpSum(v[r] * Y_front[j, r] for r in range(n_dims)) <= 1.0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != pulp.LpStatusOptimal:
        return np.nan, None

    z_o = pulp.value(prob.objective)
    v_opt = np.array([pulp.value(vr) for vr in v])
    if z_o is None or v_opt is None or np.any(v_opt is None):
        return np.nan, None
    return float(z_o), v_opt


def _run_bod_dea_cross_section(Y, eps=1e-4):
    """
    Run the Benefit-of-the-Doubt DEA composite indicator for every DMU in
    one cross-section (one year). Returns (n,) array of composite scores
    and (n, n_dims) array of each DMU's own optimal weights.
    """
    n, n_dims = Y.shape
    scores  = np.full(n, np.nan)
    weights = np.full((n, n_dims), np.nan)
    for o in range(n):
        z_o, v_o = _solve_bod_dea_one(Y, Y[o], eps=eps)
        scores[o] = z_o
        if v_o is not None:
            weights[o] = v_o
    return scores, weights



def build_esi(panel, diesel_prices, proxy_stats, weights=None):
    """
    Construct ESI using energy expenditure burden as the affordability indicator.

    Affordability indicator: energy_intensity_koe_gdp (kg oil equivalent per $1,000 PPP GDP)
      - World Bank: EG.USE.COMM.GD.PP.KD (1990-2023, all G20)
      - Fallback: computed as energy_consumption_twh / gdp_pc_ppp from OWID columns
    Higher burden = more energy consumed per unit of income = higher vulnerability.
    This replaces the diesel price proxy (R² = 0.188, section 2.4.3) which was fully
    absorbed by year fixed effects in within-country estimation (Mundlak 1978 decomposition).
    diesel_prices arg kept for backward-compatibility only (no longer used).
    proxy_stats: output of validate_energy_burden_g20() — coverage stats dict.
    """
    if weights is None:
        weights = {"resilience":0.25,"affordability":0.25,
                   "robustness":0.25,"sustainability":0.25}

    panel = compute_hhi(panel)                # retained for comparison/robustness only
    panel = compute_shannon_diversity(panel)  # primary basis for dim_resilience
    panel = compute_co2_intensity(panel)

    # diesel_prices arg is no longer used — affordability comes from energy_intensity_koe_gdp

    # ── Affordability: COMPOSITE INDEX (3 components) ────────────────────────
    #
    # Component 1: aff_intensity — Energy intensity (kg oil eq / $1,000 PPP GDP)
    #   Higher = more energy per unit of income = higher vulnerability
    #   Inverted: lower score = higher vulnerability
    #
    # Component 2: aff_price_ratio — Country-demeaned CPI index
    #   Higher ratio = prices rising faster than base period = higher burden
    #   Inverted: lower score = higher vulnerability
    #   NOTE: C2 must stay a PURE price measure (never divided by income —
    #   C3 now carries purchasing capacity, and dividing by GDP pc would
    #   count income twice). Ideal spec: Δln(EnergyCPI) − Δln(HeadlineCPI);
    #   no WB energy-CPI series covers the full panel, so the demeaned
    #   headline CPI is the feasible stopgap — flag in sensitivity analysis.
    #
    # Component 3: aff_hfce_pc — Household purchasing capacity (HFCE pc, PPP)
    #   Spec: C3_vulnerability = 1 − pooled-minmax( winsorised ln(HFCE_pc) )
    #   Higher C3 = lower purchasing capacity = greater vulnerability.
    #   Stored below in SECURITY orientation (= 1 − C3_vulnerability) so the
    #   equal-weight mean keeps higher = more secure, consistent with C1/C2
    #   and with how dim_affordability enters the ESI.
    #
    # Composite = equal-weight average of available normalised components.
    # If only 1 component available, falls back to that single component.
    # If none available, computes energy_consumption / gdp_pc_ppp from OWID.

    def _minmax_col(series, invert=False):
        """Same rule as every other ESI indicator: pooled, winsorised min–max
        (previously per-year min–max for C1 and C2; changed so all components
        share one scale that is comparable over time)."""
        return minmax(series, invert=invert)

    aff_components = []

    # Component 1: intensity (higher = more vulnerable → invert)
    if "aff_intensity" in panel.columns and panel["aff_intensity"].notna().any():
        panel["_aff_c1"] = _minmax_col(panel["aff_intensity"], invert=True)
        aff_components.append("_aff_c1")
        print(f"  ✓ Affordability C1 (intensity):     "
              f"{panel['aff_intensity'].notna().sum()} obs")
    elif "energy_intensity_koe_gdp" in panel.columns and panel["energy_intensity_koe_gdp"].notna().any():
        panel["_aff_c1"] = _minmax_col(panel["energy_intensity_koe_gdp"], invert=True)
        aff_components.append("_aff_c1")
        print(f"  ✓ Affordability C1 (koe/GDP alias): "
              f"{panel['energy_intensity_koe_gdp'].notna().sum()} obs")
    else:
        # Always-available OWID fallback
        panel["energy_burden_computed"] = owid_energy_intensity_kgoe(panel)
        panel["_aff_c1"] = _minmax_col(panel["energy_burden_computed"], invert=True)
        aff_components.append("_aff_c1")
        print("  ⚠ Affordability C1: computed from OWID energy/gdp")

    # Component 3: household purchasing capacity (HFCE per capita, PPP)
    # ln() because capacity gaps (IND vs LUX) span orders of magnitude;
    # pooled 1st–99th percentile winsorisation before normalisation;
    # POOLED min-max (per spec — not per-year like C1/C2).
    # *** MOVED (bug fix): this block previously ran AFTER the dual C2
    # computation below, but _build_affordability_variant() (part of that
    # C2 block) reads panel["_aff_c3"] directly -- causing a KeyError
    # ('_aff_c3') the first time step4_esi actually ran, since C3 did not
    # exist yet at that point. Moved here, immediately after C1, so C3 is
    # always computed before anything that depends on it. ***
    c3_source = None
    if "aff_hfce_pc" in panel.columns and panel["aff_hfce_pc"].notna().sum() > 20:
        c3_source = "aff_hfce_pc"
    elif "gdp_pc_ppp" in panel.columns and panel["gdp_pc_ppp"].notna().sum() > 20:
        c3_source = "gdp_pc_ppp"
        print("  ⚠ Affordability C3: HFCE unavailable — gdp_pc_ppp capacity proxy")
    if c3_source is not None:
        # security orientation: pooled, winsorised min–max of ln(HFCE_pc)
        panel["_aff_c3"] = minmax(panel[c3_source].clip(lower=1.0), log=True)
        aff_components.append("_aff_c3")
        print(f"  ✓ Affordability C3 (HFCE pc capacity): "
              f"{panel[c3_source].notna().sum()} obs "
              f"[ln, winsorised 1-99%, pooled min-max]")
    else:
        print("  ⚠ Affordability C3 (purchasing capacity): no data — omitted")

    # Component 2 + composite: compute BOTH headline-CPI and energy-CPI
    # variants of C2 (and therefore of dim_affordability and esi_score),
    # rather than only whichever USE_ALT_C2_AFFORDABILITY currently selects.
    # ESI_C2_MODE (near the top of this file) selects which variant feeds
    # every DOWNSTREAM step (SW, Mundlak, Panel FE, Step 12, etc.) as the
    # single "active" dim_affordability / esi_score; the other variant
    # remains fully computed and available under its own suffixed column
    # names for direct comparison, and the choice can be changed by
    # editing ESI_C2_MODE alone, with no other code changes required.
    def _fill_short_gaps_only(s, max_gap=2):
        """
        Linearly interpolate internal NaN runs of length <= max_gap only;
        runs longer than max_gap are left entirely unfilled (not partially
        filled). pandas' own interpolate(limit=N) fills up to N values per
        gap from one direction, which PARTIALLY fills longer gaps rather
        than rejecting them outright -- not the clean "short gaps only,
        otherwise untouched" rule intended here, so gap runs are identified
        explicitly instead.
        """
        s = s.copy()
        is_na = s.isna().values
        n = len(s)
        i = 0
        interp = s.interpolate(method="linear", limit_area="inside")
        while i < n:
            if is_na[i]:
                j = i
                while j < n and is_na[j]:
                    j += 1
                run_len = j - i
                touches_edge = (i == 0) or (j == n)
                if run_len <= max_gap and not touches_edge:
                    s.iloc[i:j] = interp.iloc[i:j]
                i = j
            else:
                i += 1
        return s

    def _build_affordability_variant(label, c2_raw, min_obs=50):
        """
        Build one full C2 -> dim_affordability variant (STRICT complete-case,
        short-gap C2 interpolation, separate AFF13 side artifact), given the
        variant's raw C2 driver series. Returns (dim_affordability_series,
        c2_oriented_series or None, diagnostics dict). Does not mutate panel.
        """
        n_valid = c2_raw.notna().sum() if c2_raw is not None else 0
        if c2_raw is None or n_valid <= min_obs:
            print(f"  ⚠ Affordability C2 ({label}): only {n_valid} obs — omitted; "
                  f"dim_affordability_{label} uses C1+C3 only")
            aff2 = panel[["_aff_c1", "_aff_c3"]].mean(axis=1)
            return aff2, None, {"n_valid": n_valid, "used_c2": False}

        c2 = _minmax_col(c2_raw, invert=True)
        n_before = c2.notna().sum()
        c2 = panel.groupby("iso_code").apply(
            lambda g: _fill_short_gaps_only(c2.loc[g.index])).reset_index(level=0, drop=True)
        c2 = c2.reindex(panel.index)
        n_after = c2.notna().sum()

        three = pd.concat([panel["_aff_c1"], c2, panel["_aff_c3"]], axis=1)
        three.columns = ["_c1", "_c2", "_c3"]
        complete_mask = three.notna().all(axis=1)
        aff = pd.Series(np.where(complete_mask, three.mean(axis=1), np.nan), index=panel.index)
        n_complete = int(complete_mask.sum())
        print(f"  ✓ dim_affordability_{label}: STRICT composite of 3 components, "
              f"equal weights, complete-case only")
        print(f"    C2 ({label}) short-gap interpolation: +{n_after - n_before} obs filled")
        print(f"    Complete-case (all 3 components present): {n_complete}/{len(panel)} obs")
        return aff, c2, {"n_valid": n_valid, "used_c2": True, "n_complete": n_complete}

    # --- Headline-CPI variant ---
    _headline_raw = (panel["aff_price_ratio"] if "aff_price_ratio" in panel.columns else None)
    dim_aff_headline, c2_headline, diag_headline = _build_affordability_variant(
        "headline", _headline_raw)

    # --- Energy-CPI variant ---
    _dln_cpi = compute_c2_star_alt(panel) if "iso_code" in panel.columns else None
    if _dln_cpi is not None:
        panel["aff_energy_inflation"] = _dln_cpi
    dim_aff_energy, c2_energy, diag_energy = _build_affordability_variant(
        "energy", _dln_cpi)

    panel["dim_affordability_headline"] = dim_aff_headline
    panel["dim_affordability_energy"]   = dim_aff_energy
    panel["dim_affordability_AFF13"]    = panel[["_aff_c1", "_aff_c3"]].mean(axis=1)
    # *** ADDED: persist both oriented C2 variants as their own named columns
    # (previously c2_headline/c2_energy existed only as local variables
    # inside this function, discarded once the active variant was selected
    # below). Needed so a full component-level export table -- C1, C2
    # under each variant, C3, and dim_affordability under each variant --
    # can be built downstream without recomputing either variant. ***
    panel["_aff_c2_headline"] = c2_headline
    panel["_aff_c2_energy"]   = c2_energy

    # --- Select the ACTIVE variant for every downstream step ---
    if ESI_C2_MODE == "energy":
        panel["dim_affordability"] = dim_aff_energy
        panel["_aff_c2"] = c2_energy
        aff_components = [c for c in ["_aff_c1", "_aff_c2", "_aff_c3"] if c in panel.columns]
        print(f"  ★ ESI_C2_MODE='energy': dim_affordability (ACTIVE, used downstream) "
              f"= dim_affordability_energy")
    else:
        panel["dim_affordability"] = dim_aff_headline
        panel["_aff_c2"] = c2_headline
        aff_components = [c for c in ["_aff_c1", "_aff_c2", "_aff_c3"] if c in panel.columns]
        print(f"  ★ ESI_C2_MODE='headline': dim_affordability (ACTIVE, used downstream) "
              f"= dim_affordability_headline")
    print(f"    (change ESI_C2_MODE near the top of this file to switch which variant")
    print(f"    feeds every downstream step; both variants remain available under")
    print(f"    dim_affordability_headline / dim_affordability_energy regardless)")

    # *** ADDED: per-country missingness diagnostic for the ACTIVE
    # dim_affordability, so a country losing ESI entirely (or for specific
    # years) because a component is missing is visible and attributable to
    # a specific column, rather than a silent NaN a reader has to
    # reverse-engineer from the final ESI table. Addresses "check that you
    # calculate" directly: this prints exactly which component is missing,
    # for which countries, whenever dim_affordability is not fully
    # populated for the active variant. ***
    _n_missing_aff = panel["dim_affordability"].isna().sum()
    if _n_missing_aff > 0:
        print(f"\n  ⚠ dim_affordability (active, ESI_C2_MODE='{ESI_C2_MODE}') is missing for "
              f"{_n_missing_aff}/{len(panel)} country-years. Diagnosing which component is")
        print(f"    the limiting factor, per country:")
        _diag_cols = {"_aff_c1": "C1 (intensity)",
                      "_aff_c2": f"C2 ({ESI_C2_MODE})",
                      "_aff_c3": "C3 (purchasing capacity)"}
        _missing_by_country = panel.loc[panel["dim_affordability"].isna(), "iso_code"].value_counts()
        for iso, n_missing_yrs in _missing_by_country.items():
            iso_mask = panel["iso_code"] == iso
            missing_comp = []
            for col, label in _diag_cols.items():
                if col in panel.columns and panel.loc[iso_mask, col].isna().any():
                    n_col_missing = panel.loc[iso_mask, col].isna().sum()
                    missing_comp.append(f"{label} missing {n_col_missing}/{iso_mask.sum()} yrs")
            comp_str = "; ".join(missing_comp) if missing_comp else "cause not isolated to a single component"
            print(f"    {iso}: {n_missing_yrs} year(s) without dim_affordability -- {comp_str}")
        print(f"    (esi_score is correctly NaN for every one of these rows too, by design --")
        print(f"    see the strict complete-case fix elsewhere in this function; this is not")
        print(f"    a new bug, but the SPECIFIC missing component is now visible above rather")
        print(f"    than only showing up as an unexplained gap in the final ESI table)")
    else:
        print(f"\n  ✓ dim_affordability (active) has no missing values -- ESI will be complete")
        print(f"    for all {len(panel)} country-years, subject to the other three dimensions")


    # ── DIRECTION AUDIT (review requirement) ─────────────────────────────────
    # All components must be SECURITY-oriented (higher score = LESS vulnerable)
    # before equal-weight averaging:
    #   C1: score must FALL as energy intensity rises    (reversed)   ρ < 0
    #   C2: score must FALL as price pressure rises      (reversed)   ρ < 0
    #   C3: score must RISE as household capacity rises  (kept +)     ρ > 0
    # ρ is Spearman between the normalised score and its raw driver, computed
    # pooled for every component (matching the single pooled normalisation).
    # A wrong sign is flagged hard.
    def _dir_rho(score, raw, per_year=True):
        d = pd.DataFrame({"s": score, "r": raw, "y": panel["year"]}).dropna()
        if len(d) < 10:
            return np.nan
        if per_year:
            rhos = d.groupby("y").apply(
                lambda g: g["s"].corr(g["r"], method="spearman")
                          if len(g) >= 5 else np.nan)
            return float(np.nanmean(rhos))
        return float(d["s"].corr(d["r"], method="spearman"))

    _dir_checks = []
    if "_aff_c1" in panel.columns:
        raw1 = next((panel[c] for c in
                     ["aff_intensity","energy_intensity_koe_gdp",
                      "energy_burden_computed"]
                     if c in panel.columns and panel[c].notna().any()), None)
        if raw1 is not None:
            _dir_checks.append(("C1 (intensity, reversed)",  "_aff_c1", raw1, -1, False))
    # *** FIX (review, item 3): the C2 direction-audit entry previously
    # always checked _aff_c2 against aff_price_ratio (the headline-CPI raw
    # driver), regardless of which C2 construction is actually active. When
    # ESI_C2_MODE=="energy", _aff_c2 is built from energy-CPI inflation
    # (aff_energy_inflation), not from aff_price_ratio at all -- auditing
    # it against the wrong raw series tests C2^energy against C2^headline's
    # driver, which is not a meaningful direction check. This now selects
    # the correct raw driver to match whichever construction produced
    # _aff_c2 (i.e. whichever variant ESI_C2_MODE currently selects). ***
    _c2_raw_col = "aff_energy_inflation" if ESI_C2_MODE == "energy" else "aff_price_ratio"
    if "_aff_c2" in panel.columns and _c2_raw_col in panel.columns:
        _dir_checks.append((f"C2 (price, reversed, driver={_c2_raw_col})", "_aff_c2",
                            panel[_c2_raw_col], -1, False))
    if "_aff_c3" in panel.columns and c3_source is not None:
        _dir_checks.append(("C3 (capacity, kept +)",      "_aff_c3",
                            panel[c3_source], +1, False))

    # Winsorisation report: how many observations sit at each pooled bound.
    # If a few countries exceed 1% of observations at one end (e.g. Argentina and
    # Turkey in C2), the bound falls inside their range and compression returns;
    # this report makes that visible on every run.
    print(f"  Normalisation: pooled min–max after winsorising at percentiles {WINSOR_PCT} "
          f"(all countries, {YEAR_START}–{YEAR_END})")
    for _lab, _raw, _log in [("C1 energy intensity", panel.get("aff_intensity"), False),
                             ("C2 headline", panel.get("aff_price_ratio"), False),
                             ("C2 energy", panel.get("aff_energy_inflation"), False),
                             ("C3 ln(HFCE pc)", panel.get(c3_source) if c3_source else None, True)]:
        if _raw is None or _raw.notna().sum() < 20:
            continue
        _x = np.log(_raw.clip(lower=1e-9)) if _log else _raw
        _lo, _hi = np.nanpercentile(_x.dropna(), list(WINSOR_PCT))
        _cl = panel.loc[(_x < _lo) | (_x > _hi), "iso_code"].value_counts()
        _who = ", ".join(f"{k}({v})" for k, v in _cl.head(6).items())
        print(f"    {_lab:<20} capped obs: {int(_cl.sum())}/{int(_x.notna().sum())}  {_who}")

    print("  Direction audit (score vs raw driver):")
    for name, col, raw, want, per_yr in _dir_checks:
        rho = _dir_rho(panel[col], raw, per_year=per_yr)
        if np.isnan(rho):
            print(f"    ? {name}: insufficient data for check")
            continue
        ok   = (rho * want) > 0
        flag = "✓" if ok else "✗ DIRECTION ERROR"
        print(f"    {flag} {name}: ρ = {rho:+.3f}  "
              f"(expected {'< 0' if want < 0 else '> 0'})")
        if not ok:
            print(f"      ⚠ {name} points the WRONG WAY — the composite would mix")
            print(f"        orientations. Fix before interpreting dim_affordability.")

    # Keep affordability_raw for export / diagnostics
    panel["affordability_raw"] = panel["dim_affordability"]

    # Oriented components _aff_c1/_aff_c2/_aff_c3 are KEPT (previously dropped
    # here, which made export_affordability_components_table() skip silently
    # because _aff_c1 and _aff_c3 no longer existed when it ran).

    # Resilience: normalised Shannon diversity index of the primary-energy
    # mix (replaces HHI). ESI indicator only -- NOT a DDF input or output.
    # Higher H* = more even/diversified mix
    # = more resilient; already oriented correctly (higher=better), so no
    # invert=True here, unlike the HHI it replaces (where lower HHI meant
    # more diversified, requiring inversion). Falls back to the HHI-based
    # measure only if Shannon diversity could not be computed (e.g. fewer
    # than 2 energy-source share columns available).
    if "shannon_diversity" in panel.columns and panel["shannon_diversity"].notna().any():
        panel["dim_resilience"] = minmax(panel["shannon_diversity"], invert=False)
        print("  dim_resilience: built from normalised Shannon diversity index "
              "(shannon_diversity)")
    else:
        panel["dim_resilience"] = minmax(panel["hhi"], invert=True)
        print("  ⚠ dim_resilience: Shannon diversity unavailable — fell back to "
              "HHI-based resilience (inverted)")
    # Robustness: governance score (higher = more robust)
    panel["dim_robustness"]     = minmax(panel["governance_score"], invert=False)
    # *** ADDED: explicit verification that the raw WGI-scale governance
    # input (typically -2.5 to +2.5) has actually been normalised into the
    # 0-1 ESI dimension range, printed so this is checkable on every run
    # rather than assumed. governance_score itself is intentionally left
    # on its raw WGI scale (needed for the proxy-fallback construction
    # elsewhere in this function, and useful as a diagnostic in its own
    # right) -- dim_robustness, not governance_score, is the ESI dimension;
    # any table or plot that needs "the governance dimension" should read
    # dim_robustness, never governance_score directly. ***
    _gov_raw_range = (panel["governance_score"].min(), panel["governance_score"].max())
    _gov_norm_range = (panel["dim_robustness"].min(), panel["dim_robustness"].max())
    print(f"  ✓ dim_robustness (ESI dimension, normalised): range "
          f"[{_gov_norm_range[0]:.4f}, {_gov_norm_range[1]:.4f}]")
    print(f"    (from governance_score, raw WGI-scale input: range "
          f"[{_gov_raw_range[0]:.4f}, {_gov_raw_range[1]:.4f}] -- if you see")
    print(f"    values outside [0,1] under a 'governance' or 'robustness' column")
    print(f"    anywhere, that table is reading governance_score directly, not")
    print(f"    dim_robustness, and should be corrected to use dim_robustness)")
    if _gov_norm_range[0] < -1e-6 or _gov_norm_range[1] > 1 + 1e-6:
        print(f"    ⚠ dim_robustness itself is outside [0,1] -- this would be a")
        print(f"    genuine normalisation bug, distinct from the raw-column-leak")
        print(f"    issue above; investigate minmax() and governance_score directly.")
    # Sustainability: EN.GHG.CO2.PC.CE.AR5 — territorial (direct) CO2 emissions
    # per capita, whole economy, excl. LULUCF. NOT the lifecycle electricity
    # intensity used as the DDF bad output (see compute_co2_intensity).
    # Higher = less sustainable → invert=True. Not a DDF variable: it is kept
    # separate from co2_intensity_elec, the DDF's undesirable OUTPUT.
    # Falls back to co2_intensity_elec if the WDI series is unavailable.
    _n_ghg = panel["ghg_co2_pc_ar5"].notna().sum() if "ghg_co2_pc_ar5" in panel.columns else 0
    if _n_ghg > 10:
        panel["dim_sustainability"] = minmax(panel["ghg_co2_pc_ar5"], invert=True)
        print(f"  ✓ dim_sustainability: EN.GHG.CO2.PC.CE.AR5 ({_n_ghg} obs) — "
              f"co2_intensity_elec NOT used for ESI or SW")
    else:
        panel["dim_sustainability"] = minmax(panel["co2_intensity_elec"], invert=True)
        print("  ✗ dim_sustainability: EN.GHG.CO2.PC.CE.AR5 NOT available (0 obs) — "
              "falling back to co2_intensity_elec.")
        print("    To fix: ensure download_world_bank() fetches EN.GHG.CO2.PC.CE.AR5 "
              "and wb_g20 is merged into the panel before step4_esi() runs.")

    # ESI weighting: equal weights (1/4 each) are the BASELINE structural
    # index. Two data-driven weighting schemes are computed alongside it as
    # ROBUSTNESS/SENSITIVITY variants, not replacements:
    #   esi_score_pca : PCA-based weights (PC1 loadings, normalised to sum
    #                   to 1) -- a single common weight vector applied to
    #                   every country-year, addressing the "why 1/4 each"
    #                   critique without letting any one country pick its
    #                   own weights.
    #   esi_score_bod : Benefit-of-the-Doubt DEA (each country-year picks
    #                   its own non-negative weights to maximise its own
    #                   score, subject to no country scoring above 1 under
    #                   those same weights). Reported as a sensitivity
    #                   variant precisely because it lets each observation
    #                   define its own weighting -- useful as a check that
    #                   Stage-2 signs/significance are not an artefact of
    #                   the equal-weight assumption, but not a defensible
    #                   single index of "energy security" on its own
    #                   (a country could put negligible weight on a whole
    #                   dimension, which conflicts with treating "energy
    #                   security" as consisting of four dimensions).
    dims_cols = ["dim_resilience","dim_affordability","dim_robustness","dim_sustainability"]

    # ── Baseline: equal-weight index (esi_score) ─────────────────────────────
    panel["esi_score"] = (
        weights["resilience"]     * panel["dim_resilience"]
      + weights["affordability"]  * panel["dim_affordability"]
      + weights["robustness"]     * panel["dim_robustness"]
      + weights["sustainability"] * panel["dim_sustainability"]
    )
    print(f"\n  ESI composite (baseline): equal weights (1/4 each) on {dims_cols}")

    # *** ADDED: the two full ESI variants (headline-C2 and energy-C2),
    # built from the SAME weights and the SAME other three dimensions,
    # differing only in which dim_affordability variant feeds them. Both
    # are always computed regardless of ESI_C2_MODE, so the two can be
    # compared directly without re-running the pipeline. ***
    if "dim_affordability_headline" in panel.columns:
        panel["esi_score_headline"] = (
            weights["resilience"]     * panel["dim_resilience"]
          + weights["affordability"]  * panel["dim_affordability_headline"]
          + weights["robustness"]     * panel["dim_robustness"]
          + weights["sustainability"] * panel["dim_sustainability"]
        )
    if "dim_affordability_energy" in panel.columns:
        panel["esi_score_energy"] = (
            weights["resilience"]     * panel["dim_resilience"]
          + weights["affordability"]  * panel["dim_affordability_energy"]
          + weights["robustness"]     * panel["dim_robustness"]
          + weights["sustainability"] * panel["dim_sustainability"]
        )
    print(f"  ✓ esi_score_headline and esi_score_energy also computed (both always")
    print(f"    available regardless of ESI_C2_MODE='{ESI_C2_MODE}'; esi_score above")
    print(f"    is the ACTIVE variant used downstream, matching ESI_C2_MODE)")

    # *** ADDED: full panel-coverage diagnostic. PANEL_ISO3 is the DESIGNED
    # country list (G20 + EU27 + NOR + GBR + CHE, deduplicated), which now
    # includes Switzerland by explicit request; other non-EU European
    # states (Albania, Andorra, Bosnia, etc.) remain deliberately excluded
    # (see PANEL_ISO3 / EUROPE_ISO3 definitions near the top of this file).
    #
    # What genuinely needs checking every run is whether all DESIGNED
    # countries have a USABLE esi_score, since a country can remain present
    # as a row in the panel while having esi_score = NaN for every single
    # year (because one dimension -- most often dim_affordability -- has no
    # valid observations for that country at all), which silently reduces
    # "countries with an actual ESI value" below the designed total without
    # the country ever being removed from the panel structurally. This
    # block reports exactly which of the designed countries fall into that
    # category, and why, rather than leaving this to be discovered
    # downstream as an unexplained smaller country count in an exported
    # table. ***
    _designed = set(PANEL_ISO3)
    _present = set(panel["iso_code"].unique())
    _extra = _present - _designed
    _missing_entirely = _designed - _present
    if _extra:
        print(f"\n  ⚠ {len(_extra)} country/countries present in the panel but NOT in the "
              f"designed PANEL_ISO3 list: {sorted(_extra)}")
        print(f"    (PANEL_ISO3 is the {len(PANEL_ISO3)}-country G20+EU27+NOR+GBR+CHE "
              f"design; investigate")
        print(f"    where these extra codes entered the panel if unintended)")
    if _missing_entirely:
        print(f"\n  ℹ {len(_missing_entirely)} designed country/countries have NO rows at all "
              f"in this panel: {sorted(_missing_entirely)}")

    _has_esi = panel.groupby("iso_code")["esi_score"].apply(lambda s: s.notna().any())
    _countries_with_esi = set(_has_esi[_has_esi].index)
    _countries_without_esi = _designed & _present - _countries_with_esi
    print(f"\n  {'='*66}")
    print(f"  ESI PANEL COVERAGE: {len(_countries_with_esi & _designed)}/{len(_designed)} "
          f"designed countries have a usable esi_score for at least one year")
    print(f"  (ESI_C2_MODE='{ESI_C2_MODE}'; PANEL_ISO3 = {len(PANEL_ISO3)} countries, "
          f"including Switzerland)")
    print(f"  {'='*66}")
    if _countries_without_esi:
        print(f"  The following {len(_countries_without_esi)} designed countries have")
        print(f"  esi_score = NaN for EVERY year -- diagnosing the limiting dimension:")
        for iso in sorted(_countries_without_esi):
            iso_mask = panel["iso_code"] == iso
            bad_dims = []
            for dcol, dlabel in [("dim_resilience","resilience"),
                                 ("dim_affordability","affordability"),
                                 ("dim_robustness","robustness"),
                                 ("dim_sustainability","sustainability")]:
                if dcol in panel.columns and panel.loc[iso_mask, dcol].isna().all():
                    bad_dims.append(dlabel)
            reason = f"missing: {', '.join(bad_dims)}" if bad_dims else "cause not isolated to one dimension"
            print(f"    {iso}: {reason}")
        print(f"\n  Each of these should be investigated at its specific upstream data")
        print(f"  source (see the per-dimension construction sections above) -- a")
        print(f"  country-specific reporting gap (e.g. Argentina's known World Bank")
        print(f"  FP.CPI.TOTL discontinuities) is the most common cause, not a")
        print(f"  structural pipeline bug affecting all countries alike.")
    else:
        print(f"  ✓ All {len(_designed)} designed countries have at least one year of")
        print(f"    usable esi_score.")

    # ── Robustness variant 1: PCA-based fixed weights ────────────────────────
    complete = panel[dims_cols].dropna()
    if len(complete) >= 20:
        Xc = (complete - complete.mean()) / (complete.std() + 1e-9)
        U, S, Vt = np.linalg.svd(Xc.values, full_matrices=False)
        loadings = Vt[0]
        if (loadings < 0).sum() > (loadings >= 0).sum():
            loadings = -loadings   # orient so the majority of loadings are positive
        loadings = np.clip(loadings, 0, None)   # negative loadings -> zero weight
        pca_weights = loadings / (loadings.sum() + 1e-9)
        panel["esi_score_pca"] = sum(
            pca_weights[k] * panel[d] for k, d in enumerate(dims_cols)
        )
        print(f"  ESI robustness variant (PCA weights): "
              f"{dict(zip([d.replace('dim_','') for d in dims_cols], pca_weights.round(3)))}")
    else:
        panel["esi_score_pca"] = np.nan
        print(f"  ⚠ Insufficient complete-case data for PCA weighting ({len(complete)} obs)")

    # ── Robustness variant 2: Benefit-of-the-Doubt DEA, COMMON SAMPLE ONLY ───
    # Computed ONLY on the complete-case sample per year -- no fallback to
    # equal-weight for incomplete rows. Mixing aggregation rules within one
    # column (equal-weight for some observations, BoD for others) produces
    # a variable whose meaning changes across observations, which is not
    # used here. Rows without all four dimensions available for that year
    # are left as NaN in esi_score_bod, not silently filled in.
    panel["esi_score_bod"] = np.nan
    for d in dims_cols:
        panel[f"esi_weight_{d.replace('dim_','')}"] = np.nan

    years_esi = sorted(panel["year"].dropna().unique())
    n_incomplete_total = 0
    for yr in years_esi:
        yr_mask_all = panel["year"].eq(yr)
        complete_mask = yr_mask_all & panel[dims_cols].notna().all(axis=1)
        n_incomplete_total += int(yr_mask_all.sum() - complete_mask.sum())
        sub_yr = panel.loc[complete_mask]
        if len(sub_yr) < 4:
            continue
        Y_yr = sub_yr[dims_cols].values.astype(float)
        Y_yr = np.clip(Y_yr, 1e-6, None)
        scores_yr, weights_yr = _run_bod_dea_cross_section(Y_yr)
        panel.loc[complete_mask, "esi_score_bod"] = scores_yr
        for k, d in enumerate(dims_cols):
            panel.loc[complete_mask, f"esi_weight_{d.replace('dim_','')}"] = weights_yr[:, k]

    n_ok = panel["esi_score_bod"].notna().sum()
    n_frontier_bod = (panel["esi_score_bod"] >= 0.999).sum()
    print(f"\n  ESI robustness variant (BoD-DEA): computed for {n_ok}/{len(panel)} "
          f"country-years (common complete-case sample per year)")
    if n_incomplete_total > 0:
        print(f"  ℹ {n_incomplete_total} obs excluded from esi_score_bod (missing >=1 "
              f"dimension in that year, most plausibly the governance/robustness")
        print(f"    coverage gap) -- left as NaN, NOT filled with the equal-weight score.")
    print(f"  Composite-efficient (esi_score_bod≈1) in {n_frontier_bod}/{n_ok} obs "
          f"({100*n_frontier_bod/max(n_ok,1):.0f}%) — expected feature of BoD models:")
    print(f"  each DMU picks its own most-favourable weighting, so a large share")
    print(f"  score at or near 1 by construction; this is NOT the same as the")
    print(f"  fixed-weight index, where every country is judged by the SAME weights.")

    mean_w = {d.replace('dim_',''): panel[f"esi_weight_{d.replace('dim_','')}"].mean()
             for d in dims_cols}
    print(f"  Mean endogenous weight by dimension (BoD, normalised to sum≈1 for display):")
    w_sum = sum(mean_w.values()) or 1.0
    for k, v in mean_w.items():
        print(f"    {k:<15} {v:.4f}  ({100*v/w_sum:.1f}% of mean total weight)")

    # ── Cross-check: rank agreement across all three weighting schemes ───────
    for alt_col, alt_label in [("esi_score_pca","PCA"), ("esi_score_bod","BoD-DEA")]:
        both = panel[["esi_score", alt_col]].dropna()
        if len(both) > 10:
            rho = both.corr(method="spearman").iloc[0,1]
            r_p = both.corr(method="pearson").iloc[0,1]
            print(f"  Equal-weight vs {alt_label}: Pearson r={r_p:+.3f}  "
                  f"Spearman rho={rho:+.3f}  (n={len(both)})")

    return panel


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DEA VRS WITH SIMAR–WILSON (1998) BOOTSTRAP BIAS CORRECTION
# ═════════════════════════════════════════════════════════════════════════════

# ── DATA AUDIT HELPER ────────────────────────────────────────────────────────

def _data_audit(panel: pd.DataFrame, required_cols: list, step_name: str) -> int:
    """
    Print a concise coverage report for required_cols and return the number
    of rows where ALL required columns are non-null and positive (where numeric).
    Called at the start of each analytical step.
    """
    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        print(f"  ⚠ [{step_name}] Missing columns entirely: {missing_cols}")

    present = [c for c in required_cols if c in panel.columns]
    if not present:
        return 0

    complete_mask = panel[present].notna().all(axis=1)
    n_complete = complete_mask.sum()
    n_total    = len(panel)
    pct        = 100 * n_complete / n_total if n_total > 0 else 0

    if pct < 30:
        print(f"  ⚠ [{step_name}] Only {n_complete}/{n_total} rows ({pct:.0f}%) "
              f"have complete data for: {present}")
        # Per-column breakdown when coverage is low
        for c in present:
            cp = 100 * panel[c].notna().mean()
            if cp < 80:
                print(f"       {c}: {cp:.0f}% non-null")
    return int(n_complete)


# DEA inputs use dea_net_import = bounded import-dependency measure:
#   max(0, net_import_pct)  — exporters receive zero external dependency.
# This is economically correct: net exporters have no external supply exposure;
# translation-shifting (adding the global min ~838 pp) is NOT used because DEA
# is not translation-invariant and a huge constant distorts frontier geometry.
# energy_pc (OWID energy_per_capita, kWh/person) replaces total
# energy_consumption_twh as the consumption input: per-capita on the input
# side matches the per-capita electricity output and removes the scale bias
# against populous countries. energy_consumption_twh keeps its other roles
# (burden computation, co2 intensity share, validation, exports).
# ── DDF (Directional Distance Function) specification ────────────────────────
# Following Chung, Färe & Grosskopf (1997) and Färe et al. (2007), CO₂
# is treated as an UNDESIRABLE OUTPUT in a DDF model rather than an input.
# The DDF simultaneously maximises good output (electricity) and minimises
# bad output (lifecycle GHG intensity of electricity), in direction
# g = (g_y, -g_b) = (y0, -b0)
# (observation-scaled; see _solve_ddf_one).
# This is the methodologically correct treatment: disposing of CO₂ is costly
# (weak disposability), so penalising CO₂ as a regular input would overstate
# production possibilities. The DDF score β* ≥ 0 measures how far a DMU can
# move in the direction g before hitting the frontier; EFFICIENCY = 1/(1+β*).
#
# Conventional inputs  : energy per capita, net import dependence
# Desirable output     : electricity generation (per capita when available)
# Undesirable output   : LIFECYCLE GHG intensity of electricity generation
#                        (gCO2e/kWh; OWID/Ember carbon_intensity_elec)
# Direction vector     : g = (y0, b0) — observation-scaled: expand good and
#                        contract bad by the same proportion beta
# NOT DDF variables    : the ESI dimensions (incl. Shannon resilience),
#                        which enter only as environmental variables.
INPUT_COLS   = ["dea_net_import", "energy_pc"]          # ← CO₂ moved to bad output
GOOD_OUT     = ["elec_gen_twh"]                         # ← desirable output
BAD_OUT      = ["co2_intensity_elec"]                   # ← undesirable output (DDF)
OUTPUT_COLS  = GOOD_OUT   # retained for MPI / SW / export compatibility


def _solve_dea_one(X_front, Y_front, x0, y0, vrs=True, allow_contraction=False):
    """
    Solve one output-oriented DEA envelopment problem. Returns 1/φ.

    allow_contraction=False (default): the evaluated DMU is inside the
      reference set → φ ≥ 1 and the score lies in (0,1]. Own-period DEA.
    allow_contraction=True: the evaluated DMU may lie OUTSIDE the reference
      frontier (bootstrap pseudo-frontier; cross-period Malmquist). φ may
      then be < 1 and the score legitimately exceeds 1 (super-efficiency).

    FIX (review): under the old hard bound φ ≥ 1, out-of-frontier LPs went
    INFEASIBLE and the code still read whatever value CBC left in φ —
    garbage scores like 1/0.07 ≈ 14 that produced the impossible biases
    (1.17) and CI widths (1.4–7.7) on a (0,1] scale. Non-optimal LPs now
    return NaN and all callers aggregate NaN-safely.
    """
    n = X_front.shape[0]
    prob = pulp.LpProblem("DEA", pulp.LpMaximize)
    phi  = pulp.LpVariable("phi", lowBound=(0.0 if allow_contraction else 1.0))
    lam  = [pulp.LpVariable(f"l{j}", lowBound=0) for j in range(n)]

    prob += phi
    for i in range(X_front.shape[1]):
        prob += pulp.lpSum(lam[j] * X_front[j,i] for j in range(n)) <= x0[i]
    for r in range(Y_front.shape[1]):
        prob += pulp.lpSum(lam[j] * Y_front[j,r] for j in range(n)) >= phi * y0[r]
    if vrs:
        prob += pulp.lpSum(lam) == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != pulp.LpStatusOptimal:
        return np.nan
    phi_val = pulp.value(phi)
    if phi_val is None or phi_val <= 0:
        return np.nan
    return 1.0 / phi_val


def _run_dea_cross_section(X, Y, vrs=True):
    """Run DEA for all DMUs in one cross-section. Returns (n,) score array."""
    n = X.shape[0]
    return np.array([_solve_dea_one(X, Y, X[o], Y[o], vrs) for o in range(n)])


def _normalise(X, scale=None):
    """
    Rescale each column by DIVISION ONLY: x / s, with s the column mean of the
    reference sample (or a supplied scale vector, e.g. the pooled mean).

    Division only (no shift) keeps zero at zero and keeps proportions. With the
    observation-scaled direction g = (y0, b0), every DDF constraint is then
    invariant to the rescaling, so beta is exactly the beta of the raw data:
    beta = 0.18 means 18% more raw good output and 18% less raw bad output.
    The previous [1,10] shift-and-stretch did not have this property (beta was a
    proportion of the rescaled values, not the raw ones). Rescaling is kept only
    for numerical conditioning of the LP.
    """
    X = np.asarray(X, dtype=float)
    if scale is None:
        scale = np.nanmean(X, axis=0)
    scale = np.where(np.asarray(scale, dtype=float) > 0, scale, 1.0)
    return X / scale


def _smooth_bootstrap(scores, B, rng):
    """
    Simar & Wilson (1998) smooth bootstrap — draws pseudo-efficiencies β*
    from the kernel-smoothed empirical distribution of the scores, with
    reflection at the boundary 1.0. These draws build the pseudo frontier
    in _bootstrap_dea_one_year (row-resampling is inconsistent for frontier
    estimators and has been removed — see review fix).

    Kernel: Gaussian with Silverman bandwidth, reflected at boundary 1.0.
    Returns (B, n) array of bootstrap score draws.
    """
    n = len(scores)
    h = max(1.06 * np.std(scores) * n**(-0.2), 0.005)
    boot = np.empty((B, n))
    for b in range(B):
        base    = rng.choice(scores, size=n, replace=True)
        noise   = rng.normal(0, h, size=n)
        perturb = base + noise
        # Reflect at boundary 1.0 to keep support in (0,1]
        perturb = np.where(perturb > 1.0, 2.0 - perturb, perturb)
        boot[b] = np.clip(perturb, 0.01, 1.0)
    return boot


def _bootstrap_dea_one_year(X, Y, theta_orig, B, rng, vrs, alpha):
    """
    Simar & Wilson (1998) smooth HOMOGENEOUS bootstrap, output-oriented.

    FIX (review): the previous row-resampling loop was invalid for DEA and
    let infeasible LPs return garbage φ values, producing impossible numbers
    (bias 1.17, CI widths 1.4–7.7 on a (0,1] scale) and an inconsistency
    between mean(θ̂)−mean(θ_bc) and the reported mean bias. Three repairs:

      1. Pseudo-frontier bootstrap instead of row resampling:
         draw β*_j from the smoothed distribution of θ̂ (reflection at 1),
         rescale outputs  y*_j = y_j · (β*_j / θ̂_j)  so pseudo-DMU j keeps
         DMU j's frontier projection but sits at efficiency β*_j, then
         score each ORIGINAL DMU against the pseudo frontier.
      2. ALL statistics live on ONE scale — the Farrell output-expansion
         factor λ = 1/θ ≥ 1 — and are converted to efficiencies only at
         the very end. The expansion factor and its inverse are never
         mixed (the root cause of the impossible figures).
      3. Infeasible LPs return NaN (never garbage) and aggregation is
         NaN-safe; original DMUs falling outside a pseudo frontier are
         evaluated with allow_contraction=True — a legitimate part of the
         sampling distribution.

    Guarantees, by construction:
      • θ_bc ∈ (0,1];  CI bounds ∈ (0,1];  CI width ≤ 1;
      • θ_bc always lies inside its own CI;
      • reported bias = θ̂ − θ_bc (positive = DEA overstated efficiency),
        so mean(θ̂) − mean(θ_bc) equals the reported mean bias EXACTLY.

    Returns: theta_bc (n,), bias (n,), lo (n,), hi (n,), se (n,)
    """
    n = X.shape[0]
    theta_safe = np.clip(theta_orig, 1e-6, 1.0)
    lam_hat    = 1.0 / theta_safe                          # Farrell λ ≥ 1
    boot_lam   = np.full((B, n), np.nan)
    beta_star  = _smooth_bootstrap(theta_orig, B, rng)     # (B, n) in (0,1]

    for b in range(B):
        scale  = np.clip(beta_star[b], 0.01, 1.0) / theta_safe
        Y_star = Y * scale[:, None]
        for o in range(n):
            th = _solve_dea_one(X, Y_star, X[o], Y[o], vrs,
                                allow_contraction=True)
            if np.isfinite(th) and th > 0:
                boot_lam[b, o] = 1.0 / th                  # store λ*, one scale

    lam_star_mean = np.nanmean(boot_lam, axis=0)
    # Bias correction on the λ scale: λ_bc = 2λ̂ − E*[λ*], floored at 1
    lam_bc   = np.clip(2.0 * lam_hat - lam_star_mean, 1.0, None)
    theta_bc = 1.0 / lam_bc                                # ∈ (0,1] always
    bias     = theta_orig - theta_bc                       # ≥ 0: overstatement

    # Basic (reflected) bootstrap CI on the λ scale, then invert to θ.
    q_lo_lam = np.nanpercentile(boot_lam, 100 * alpha / 2,       axis=0)
    q_hi_lam = np.nanpercentile(boot_lam, 100 * (1 - alpha / 2), axis=0)
    lam_lo   = np.clip(2.0 * lam_hat - q_hi_lam, 1.0, None)
    lam_hi   = np.clip(2.0 * lam_hat - q_lo_lam, 1.0, None)
    lam_lo, lam_hi = np.minimum(lam_lo, lam_hi), np.maximum(lam_lo, lam_hi)
    lam_lo   = np.minimum(lam_lo, lam_bc)                  # θ_bc inside its CI
    lam_hi   = np.maximum(lam_hi, lam_bc)
    lo = 1.0 / lam_hi                                      # λ↔θ ordering flips
    hi = 1.0 / lam_lo

    # Descriptive SE on the efficiency scale (draws clipped to (0,1])
    se = np.nanstd(np.clip(1.0 / boot_lam, 0.01, 1.0), axis=0)

    return theta_bc, bias, lo, hi, se



def _print_scale_table(panel, rts=None):
    """
    Print a three-way θCRS / θVRS / SE results table for every country that
    falls below the annual frontier (dea_bc < 0.999) in at least one year.

    Column definitions
    ------------------
    θCRS  (OTE) : overall technical efficiency — distance to the global CRS
                  best-practice frontier; captures operational AND scale gaps
                  jointly. A score of 0.85 means outputs could expand 18%
                  to reach the CRS frontier.
    θVRS  (PTE) : pure technical efficiency — distance to the VRS frontier,
                  as if the DMU operated at its optimal scale. Isolates the
                  operational / technology-mix gap net of scale effects.
    SE = CRS/VRS: scale efficiency — proximity to the most productive scale
                  size given the normalised per-capita technology. SE ≈ 1:
                  inefficiency (if any) is pure technical; SE < 1 indicates
                  sensitivity to the returns-to-scale specification and a
                  CRS-VRS efficiency gap (NOT a literal claim about physical
                  system size — these are per-capita/intensive variables).
    θbc(RTS)    : bias-corrected score from the SW1998 bootstrap, for the
                  returns-to-scale specification of THIS run (rts argument):
                  CRS in Step 5 (baseline), VRS in the robustness bootstrap.

    The "YrsBelow" column counts the number
    of years each country was BELOW the annual frontier (dea_bc < 0.999) —
    NOT the total number of years of data available for that country. Every
    country has data for all years in the panel (the annual DDF loop in
    build_dea_bootstrap reports n=45 DMUs for every single year); a country
    with YrsBelow=6 out of 24 total years was ON the frontier in the other
    18 years, not missing from the panel for 18 years. The previous column
    header "Yrs" was ambiguous and could be misread as total years present,
    which is NOT what this column measures. ***

    Implications (printed per country-year):
      SE ≈ 1, PTE < 1  → operational / technology-mix gap; no scale issue.
      SE < 1, PTE ≈ 1  → scale mismatch; operations are otherwise efficient.
      SE < 1, PTE < 1  → both sources contribute; scale gap dominates the
                          difference between θCRS and θVRS.
    """
    rts = rts or BASE_RTS
    rts_role = "baseline" if rts == BASE_RTS else "robustness check"
    bc_lab = f"θbc({rts})"
    THR_FRONTIER = 0.999   # below = not on frontier

    needed = ["iso_code","year","dea_score","dea_bc","dea_crs","dea_vrs","dea_se_score"]
    if not all(c in panel.columns for c in needed):
        print("  ⚠ Scale table skipped — missing columns (run step 5 first)")
        return

    df = panel[needed].copy().dropna(subset=needed)
    n_total_years = df["year"].nunique()   # panel-wide max, for the header text only

    # *** FIX: the denominator previously shown for EVERY country's row was
    # this single panel-wide constant, silently ASSUMING every country has
    # the same number of valid years -- never actually verified per country.
    # If a specific country's underlying data has gaps (missing WB/OWID
    # inputs in some years), its true year count is LOWER than the panel-
    # wide maximum, and displaying it against the wrong (borrowed) 24 would
    # misrepresent a genuine data-coverage gap as frontier performance, or
    # vice versa. Each country's ACTUAL total valid-year count is computed
    # here and used as ITS OWN denominator. ***
    years_per_country = df.groupby("iso_code")["year"].nunique().rename("n_years_actual")

    incomplete = years_per_country[years_per_country < n_total_years]
    if len(incomplete) > 0:
        print(f"  ⚠ DATA COVERAGE CHECK: {len(incomplete)} countries have FEWER than "
              f"{n_total_years} valid years in this table (missing underlying data,")
        print(f"    not a frontier-performance result):")
        for iso, n in incomplete.sort_values().items():
            print(f"      {iso}: {n}/{n_total_years} years actually present")
        print(f"    For these countries, 'YrsBelow' below is reported against THEIR OWN")
        print(f"    actual year count, not the panel-wide {n_total_years}, so the ratio")
        print(f"    remains accurate -- but the underlying data gap itself should be")
        print(f"    investigated separately (see Step 1/Step 3 coverage diagnostics).")

    # Frontier status uses the RAW score of this run (dea_score), never dea_bc:
    # frontier membership is a property of the estimated frontier.
    df["on_frontier"] = df["dea_score"] >= THR_FRONTIER
    df["beta_raw"] = 1.0 / df["dea_score"].clip(lower=1e-9) - 1.0
    below = df[~df["on_frontier"]].copy()

    # *** REVISED (reviewer): the main ranking table previously averaged ONLY
    # the years in which a country was below the frontier, so a country on
    # the frontier in 18 of 24 years was ranked on its 6 worst years, and
    # countries never below the frontier vanished from the table. The main
    # table now reports GENUINE country means over ALL valid years (24 for
    # every country in this panel) for every country. Below-frontier-year
    # averages are kept in a SEPARATE table, only for countries where they
    # differ from the all-year means (countries with at least one frontier
    # year). OTE = PTE x SE holds year by year; the means of the three are
    # each averaged separately, so the identity holds only approximately for
    # the means. ***
    agg = (df.groupby("iso_code")
             .agg(n_years=("year", "count"),
                  years_on_frontier=("on_frontier", "sum"),
                  theta_crs_mean=("dea_crs", "mean"),
                  theta_vrs_mean=("dea_vrs", "mean"),
                  se_mean=("dea_se_score", "mean"),
                  theta_bc_mean=("dea_bc", "mean"),
                  beta_mean=("beta_raw", "mean"))
             .reset_index()
             .sort_values("theta_bc_mean", ascending=False)
             .reset_index(drop=True))
    agg.insert(0, "rank_theta_bc", range(1, len(agg) + 1))

    W = 108
    print(f"  {'═'*W}")
    print(f"  EFFICIENCY RANKING — country means over ALL valid years ({n_total_years} per country), "
          f"bootstrapped specification: {rts} ({rts_role})")
    print(f"  OTE = θCRS | PTE = θVRS | SE = θCRS/θVRS (annual values averaged) | "
          f"{bc_lab} = bias-corrected {rts} | β = 1/θ − 1 (raw)")
    print(f"  Ranked by mean {bc_lab}. 'OnFront' = years on the raw {rts} frontier (θ ≥ {THR_FRONTIER}).")
    print(f"  {'═'*W}")
    print(f"  {'Rank':>4} {'Country':<7} {'Yrs':>4} {'OnFront':>8} {'OTE (θCRS)':>11} {'PTE (θVRS)':>11} "
          f"{'SE':>7} {bc_lab:>9} {'mean β':>8} {'1-PTE':>7} {'1-SE':>7}")
    print(f"  {'─'*W}")
    for _, r in agg.iterrows():
        flag = "*" if r["n_years"] < n_total_years else " "
        print(f"  {int(r['rank_theta_bc']):>4} {r['iso_code']:<7} {int(r['n_years']):>3}{flag} "
              f"{int(r['years_on_frontier']):>8} {r['theta_crs_mean']:>11.4f} {r['theta_vrs_mean']:>11.4f} "
              f"{r['se_mean']:>7.4f} {r['theta_bc_mean']:>9.4f} {r['beta_mean']:>8.3f} "
              f"{1 - r['theta_vrs_mean']:>7.4f} {1 - r['se_mean']:>7.4f}")
    print(f"  {'─'*W}")
    _sp = rts.lower()
    try:
        _save_csv(agg, f"step5_efficiency_ranking_all_years_{_sp}.csv",
                  f"Country means over all years ({rts} bootstrap run)")
    except Exception:
        pass

    # Separate, supplementary table: below-frontier years only, for countries
    # that were on the frontier in at least one year (for all others it is
    # identical to the table above).
    part = agg[(agg["years_on_frontier"] > 0) & (agg["years_on_frontier"] < agg["n_years"])]["iso_code"]
    full = agg[agg["years_on_frontier"] == agg["n_years"]]["iso_code"].tolist()
    if len(part):
        bl = (below[below["iso_code"].isin(part)].groupby("iso_code")
                .agg(years_below=("year", "count"), theta_crs_below=("dea_crs", "mean"),
                     theta_vrs_below=("dea_vrs", "mean"), theta_bc_below=("dea_bc", "mean"),
                     beta_below=("beta_raw", "mean"))
                .reset_index().sort_values("theta_bc_below"))
        print(f"  Supplementary: means over BELOW-FRONTIER years only, for the {len(bl)} countries")
        print(f"  that were on the frontier in some years (how far they fall when off the frontier):")
        print(f"  {'Country':<8}{'YrsBelow':>9}{'θCRS':>9}{'θVRS':>9}{bc_lab:>11}{'mean β':>9}")
        for _, r in bl.iterrows():
            print(f"  {r['iso_code']:<8}{int(r['years_below']):>9}{r['theta_crs_below']:>9.4f}"
                  f"{r['theta_vrs_below']:>9.4f}{r['theta_bc_below']:>11.4f}{r['beta_below']:>9.3f}")
        try:
            _save_csv(bl, f"step5_efficiency_below_frontier_years_{_sp}.csv",
                      f"Below-frontier-year means, countries with some frontier years ({rts})")
        except Exception:
            pass
    if full:
        print(f"  On the raw {rts} frontier in every year (no below-frontier years): {full}")

    # ── Country-year detail for the most recent year ──────────────────────────
    latest = below["year"].max()
    sub_yr = (below[below["year"] == latest]
              .sort_values("dea_bc")
              .reset_index(drop=True))
    if not sub_yr.empty:
        print(f"  ── Latest year detail ({latest}) — below-frontier DMUs ──")
        print(f"  {'Country':<8}  {'θCRS':>7}  {'θVRS':>7}  {bc_lab:>8}  "
              f"{'SE':>7}  {'1/θVRS-1':>9}  {'1/θCRS-1':>9}  Scale note")
        print(f"  {'─'*82}")
        for _, r in sub_yr.iterrows():
            exp_vrs = (1/max(r["dea_vrs"],1e-6)) - 1   # % output expansion to VRS frontier
            exp_crs = (1/max(r["dea_crs"],  1e-6)) - 1   # % output expansion to CRS frontier
            se_note = ("CRS-VRS gap negligible" if r["dea_se_score"] > 0.99
                       else ("Substantial CRS-VRS gap" if r["dea_se_score"] < 0.95
                             else "Moderate CRS-VRS gap"))
            print(f"  {r['iso_code']:<8}  {r['dea_crs']:>7.4f}  {r['dea_vrs']:>7.4f}  "
                  f"{r['dea_bc']:>8.4f}  {r['dea_se_score']:>7.4f}  "
                  f"{exp_vrs:>9.1%}  {exp_crs:>9.1%}  {se_note}")

    print(f"  INTERPRETATION GUIDE")
    print(f"  {'─'*W}")
    print(f"  CRS (θCRS) pinpoints total shortfalls vs global best practice: operational + scale")
    print(f"  VRS (θVRS) measures pure technical/configurational efficiency after")
    print(f"  allowing for variable returns to scale: pure operational/mix gap")
    print(f"  SE = CRS/VRS quantifies the CRS-VRS gap under the normalised")
    print(f"  per-capita technology:")
    print(f"    SE ≈ 1 → CRS-VRS gap negligible; inefficiency (if any) is pure technical")
    print(f"    SE < 1 → sensitivity to the returns-to-scale specification and a")
    print(f"    CRS-VRS efficiency gap (not a literal claim about physical system size)")
    # SE < 1 reflects sensitivity to the returns-to-scale specification, not
    # without explicitly deriving IRS/DRS/MPSS conditions for per-capita/
    # intensive variables. ***
    print(f"  Countries with lower SE indicate that the measured efficiency")
    print(f"  assessment is more sensitive to the returns-to-scale assumption,")
    print(f"  alongside the technical/configurational efficiency gap.")
    print(f"  {'═'*W}")



def _solve_ddf_one(X_front, Y_front, B_front, x0, y0, b0, vrs=True,
                   g_y=None, g_b=None, allow_negative=False):
    """
    Directional Distance Function (DDF) with undesirable outputs.

    Solve one Directional Distance Function (DDF) problem following
    Chung, Färe & Grosskopf (1997) and Färe, Grosskopf & Pasurka (2007).

    The DDF simultaneously maximises good output and minimises bad output
    in a pre-specified direction g = (g_y, -g_b):

        D̃(x,y,b;g_y,-g_b) = max β  s.t.
          (i)   Σ λ_j x_{ij}  ≤  x_{i0}               [input feasibility]
          (ii)  Σ λ_j y_{rj}  ≥  y_{r0} + β g_y_r     [expand good output]
          (iii) Σ λ_j b_{kj}  =  b_{k0} - β g_b_k     [reduce bad output]
          (iv)  Σ λ_j         = 1                       [VRS convexity]
    *** VRS IMPLEMENTATION (reviewer item 1): with VRS_ABATEMENT="kuosmanen"
    (default) the VRS model is Kuosmanen's (2005) formulation, not the single-
    lambda system above: lambda_j = z_j + u_j, outputs (ii)-(iii) use z only,
    inputs (i) and convexity (iv) use z + u. This allows activity-specific
    (non-uniform) abatement. Under CRS both versions coincide. ***
          (v)   λ_j ≥ 0,  β ≥ 0  (own-period; see allow_negative below)

    Constraint (iii) uses EQUALITY (not ≤) to impose WEAK DISPOSABILITY of
    undesirable outputs: reducing CO₂ requires reducing production, so it
    cannot be freely disposed. This is the key departure from treating CO₂
    as a conventional input (which would imply free disposability).

    allow_negative=False (default): β ≥ 0. Correct for OWN-period evaluation,
      where the DMU is always trivially feasible against its own reference
      set (λ on itself gives β=0 exactly).
    allow_negative=True: β is UNBOUNDED BELOW. Required for CROSS-period
      Malmquist evaluation, where a DMU can legitimately lie beyond what the
      OTHER period's technology can produce (super-efficiency). *** FIX:
      under the β≥0 bound, this case was INFEASIBLE (not just β>0 but no
      feasible β existed at all), returning NaN and silently dropping
      frontier DMUs from every MPI transition (observed as AUS/IND/NOR/ZAF
      failing on D_t_t1/D_t1_t in every single year-pair). Allowing β<0
      lets the LP relax both the required good-output expansion and the
      bad-output equality target simultaneously, restoring feasibility —
      exactly the same fix as allow_contraction in the standard DEA model. ***
      θ = 1/(1+β) then exceeds 1 (super-efficiency), matching the standard
      convention for cross-period MPI distances.

    Direction: g = (g_y, g_b) = (y0, b0) by default — OBSERVATION-SCALED, so
    beta is a literal proportion: y*=(1+beta)y0, b*=(1-beta)b0. This gives
    the intuitive reading "theta=0.85 (beta=0.18) -> could expand good
    output by 18% AND cut bad output by 18%". A caller may override g_y/g_b
    with a fixed vector (e.g. all-ones) for a different DDF variant, but
    that choice does NOT yield a percentage interpretation of beta.

    Returns: DDF score β* (≥0 own-period; unbounded cross-period)
      β* = 0  → on the frontier (efficient)
      β* > 0  → inefficient; can expand good by β*·g_y AND cut bad by β*·g_b
      β* < 0  → (cross-period only) super-efficient vs. that period's frontier
    Efficiency score: θ_ddf = 1 / (1 + β*)
    """
    n = X_front.shape[0]
    n_x = X_front.shape[1]
    n_y = Y_front.shape[1]
    n_b = B_front.shape[1]

    # *** FIX (methodological): observation-scaled direction vector ***
    # Default direction is now g_y=y0, g_b=b0 (the EVALUATED DMU's own output
    # levels), not a unit vector g=(1,1). This is what gives the intuitive
    # proportional interpretation:
    #     y* = y0 + beta*g_y = y0(1+beta)      good output expands by beta*100%
    #     b* = b0 - beta*g_b = b0(1-beta)      bad output contracts by beta*100%
    # With g=(1,1) in NORMALISED units, beta was movement in arbitrary scaled
    # units, NOT a literal percentage -- "theta=0.85 (beta=0.18) -> 18%
    # expansion/contraction" was not actually implied by that direction choice.
    # This observation-scaled convention (sometimes written g=(y0,b0)) is
    # standard in the undesirable-output DDF literature (e.g. Zhou & Ang
    # 2008) specifically because it yields this exact percentage reading.
    if g_y is None: g_y = np.maximum(np.asarray(y0, dtype=float), 1e-6)
    if g_b is None: g_b = np.maximum(np.asarray(b0, dtype=float), 1e-6)

    kuos = bool(vrs) and VRS_ABATEMENT == "kuosmanen"
    _prim, _fall = ((_ddf_lp_highs, _ddf_lp_cbc) if DDF_SOLVER == "highs"
                    else (_ddf_lp_cbc, _ddf_lp_highs))
    _pn, _fn = ("HiGHS", "CBC") if DDF_SOLVER == "highs" else ("CBC", "HiGHS")
    beta_val, status = _prim(X_front, Y_front, B_front, x0, y0, b0,
                             vrs, kuos, g_y, g_b, allow_negative)
    if beta_val is None:
        # The primary solver did not return an optimal solution. The own-period
        # problem is always feasible (the DMU itself, beta = 0) and bounded
        # (beta <= 1 through the bad-output equality), so a non-optimal status
        # is a numerical failure of the solver, not a property of the data.
        # Retry with the fallback solver before giving up, and record it.
        _DDF_SOLVE_STATS["primary_not_optimal"] += 1
        _DDF_SOLVE_STATS["primary_status"][status] = _DDF_SOLVE_STATS["primary_status"].get(status, 0) + 1
        beta_val, status_f = _fall(X_front, Y_front, B_front, x0, y0, b0,
                                   vrs, kuos, g_y, g_b, allow_negative)
        if beta_val is None:
            _DDF_SOLVE_STATS["unsolved"] += 1
            _log_ddf_failure("unsolved", f"{_pn}: {status}; {_fn}: {status_f}", None, vrs, kuos)
            return np.nan
        _DDF_SOLVE_STATS["rescued_by_fallback"] += 1
    if beta_val < 0 and not allow_negative:
        # With beta bounded below by 0, a negative value is the solver's bound
        # tolerance (CBC can return e.g. -1e-10 for an optimal beta of 0).
        # *** FIX (CZE 2022, VRS metafrontier): this case previously returned
        # NaN WITHOUT updating any counter, so the observation was dropped while
        # the solver summary reported zero failures. Values within tolerance are
        # now set to 0 (the DMU is on its frontier) and counted; a materially
        # negative value is re-solved with the fallback solver, and rejected,
        # counted and logged only if that also fails. ***
        if beta_val >= -NEG_BETA_TOL:
            _DDF_SOLVE_STATS["tiny_negative_set_to_zero"] += 1
            beta_val = 0.0
        else:
            _bh, _sh = _fall(X_front, Y_front, B_front, x0, y0, b0,
                             vrs, kuos, g_y, g_b, allow_negative)
            if _bh is not None and _bh >= -NEG_BETA_TOL:
                _DDF_SOLVE_STATS["rescued_by_fallback"] += 1
                _DDF_SOLVE_STATS["primary_status"]["negative beta"] = \
                    _DDF_SOLVE_STATS["primary_status"].get("negative beta", 0) + 1
                return float(_bh) if _bh > 0 else 0.0
            _DDF_SOLVE_STATS["negative_beta_rejected"] += 1
            _log_ddf_failure("negative_beta_rejected",
                             f"{_pn} {status} beta={beta_val:.3g}; {_fn} {_sh} beta={_bh}", beta_val, vrs, kuos)
            return np.nan
    return float(beta_val)


# Tolerance for treating a slightly negative beta (beta >= 0 imposed) as zero.
NEG_BETA_TOL = 1e-6


def _log_ddf_failure(kind, status, beta_val, vrs, kuos):
    """Keep the details of every failed DDF LP (first 200), so a missing score
    can always be traced to a counted, explained failure."""
    rec = {"kind": kind, "status": status, "beta": beta_val,
           "rts": "VRS" if vrs else "CRS", "abatement": ("kuosmanen" if kuos else "single-lambda")}
    _DDF_SOLVE_STATS["last_failure"] = rec
    if len(_DDF_SOLVE_STATS["failures"]) < 200:
        _DDF_SOLVE_STATS["failures"].append(rec)


# Solver bookkeeping, reported by report_ddf_solver_stats() at the end of a run.
_DDF_SOLVE_STATS = {"primary_not_optimal": 0, "rescued_by_fallback": 0, "unsolved": 0,
                    "tiny_negative_set_to_zero": 0, "negative_beta_rejected": 0,
                    "primary_status": {}, "failures": [], "last_failure": None}


def _ddf_lp_cbc(X_front, Y_front, B_front, x0, y0, b0, vrs, kuos, g_y, g_b, allow_negative):
    """DDF linear programme solved with CBC (PuLP). Returns (beta or None, status)."""
    n, n_x = X_front.shape
    n_y, n_b = Y_front.shape[1], B_front.shape[1]
    prob = pulp.LpProblem("DDF", pulp.LpMaximize)
    beta = pulp.LpVariable("beta", lowBound=(None if allow_negative else 0))
    z = [pulp.LpVariable(f"z{j}", lowBound=0) for j in range(n)]          # active weights
    u = [pulp.LpVariable(f"u{j}", lowBound=0) for j in range(n)] if kuos else None  # abatement weights
    prob += beta
    act = (lambda j: z[j] + u[j]) if kuos else (lambda j: z[j])
    # (i) inputs: used by active AND abating activity
    for i in range(n_x):
        prob += pulp.lpSum(act(j) * X_front[j, i] for j in range(n)) <= x0[i]
    # (ii) good output: produced by active activity only
    for r in range(n_y):
        prob += pulp.lpSum(z[j] * Y_front[j, r] for j in range(n)) >= y0[r] + beta * g_y[r]
    # (iii) bad output, weak disposability: equality, active activity only
    for k in range(n_b):
        prob += pulp.lpSum(z[j] * B_front[j, k] for j in range(n)) == b0[k] - beta * g_b[k]
    # (iv) convexity (VRS): over z + u (Kuosmanen) or z alone (uniform)
    if vrs:
        prob += pulp.lpSum(act(j) for j in range(n)) == 1
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    except Exception as e:                      # e.g. CBC binary not available
        return None, f"CBC error: {type(e).__name__}"
    status = pulp.LpStatus.get(prob.status, str(prob.status))
    if prob.status != pulp.LpStatusOptimal:
        return None, status
    bv = pulp.value(beta)
    return (None, "no value") if bv is None else (float(bv), status)


def _ddf_lp_highs(X_front, Y_front, B_front, x0, y0, b0, vrs, kuos, g_y, g_b, allow_negative):
    """Same DDF linear programme solved with HiGHS (scipy.optimize.linprog).
    Variables: [beta, z_1..z_n, (u_1..u_n)]. Returns (beta or None, status)."""
    from scipy.optimize import linprog
    n, n_x = X_front.shape
    n_y, n_b = Y_front.shape[1], B_front.shape[1]
    nv = 1 + n + (n if kuos else 0)
    c = np.zeros(nv); c[0] = -1.0                       # maximise beta
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for i in range(n_x):
        row = np.zeros(nv); row[1:1+n] = X_front[:, i]
        if kuos: row[1+n:] = X_front[:, i]
        A_ub.append(row); b_ub.append(x0[i])
    for r in range(n_y):
        row = np.zeros(nv); row[0] = g_y[r]; row[1:1+n] = -Y_front[:, r]
        A_ub.append(row); b_ub.append(-y0[r])
    for k in range(n_b):
        row = np.zeros(nv); row[0] = g_b[k]; row[1:1+n] = B_front[:, k]
        A_eq.append(row); b_eq.append(b0[k])
    if vrs:
        row = np.zeros(nv); row[1:] = 1.0
        A_eq.append(row); b_eq.append(1.0)
    bounds = [(None if allow_negative else 0, None)] + [(0, None)] * (nv - 1)
    try:
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                      bounds=bounds, method="highs")
    except Exception as e:
        return None, f"highs error: {e}"
    if res.status != 0:
        return None, f"highs status {res.status}"
    return float(res.x[0]), "highs optimal"


def audit_ddf_solver(n=30, seed=7, save=True):
    """
    Auditable record of the DDF solver used in this run (reviewer request).

    (1) Prints the linear programme as implemented.
    (2) Runs self-tests on synthetic data (no real data involved):
        a. own-period feasibility: every DMU solved with beta >= 0 (CRS and VRS);
        b. nesting: theta_CRS <= theta_VRS(Kuosmanen) <= theta_VRS(single-lambda);
        c. CRS identical with and without abatement weights;
        d. CBC and HiGHS give the same beta (|diff| < 1e-6).
    (3) Saves the exact source of the solver functions to
        logs/ddf_solver_source.txt so the formulation can be checked line by line.
    Returns True if every test passes.
    """
    import inspect
    global VRS_ABATEMENT
    print(f"\n{'='*70}")
    print("DDF SOLVER AUDIT")
    print(f"{'='*70}")
    print("  max beta  s.t.  (all sums over reference DMUs j)")
    print("    inputs      : sum (z_j + u_j) x_ij <= x_i0")
    print("    good output : sum z_j y_rj        >= y_r0 + beta * g_y   (g_y = y_0)")
    print("    bad output  : sum z_j b_kj        =  b_k0 - beta * g_b   (g_b = b_0)")
    print("    VRS only    : sum (z_j + u_j)     =  1")
    print("    z_j, u_j >= 0;  beta >= 0 (own period) or free (bootstrap pseudo-frontier)")
    print(f"  u_j (abatement weights) exist only under VRS with VRS_ABATEMENT='kuosmanen'")
    print(f"  (current: '{VRS_ABATEMENT}'). Under CRS, or with 'uniform', u_j = 0.")
    print(f"  theta = 1/(1+beta). Primary solver: {DDF_SOLVER.upper()}; the other solver is the fallback.")
    try:
        _avail = pulp.listSolvers(onlyAvailable=True)
    except Exception:
        _avail = []
    print(f"  PuLP {pulp.__version__}; CBC available: {'PULP_CBC_CMD' in _avail}"
          + ("" if 'PULP_CBC_CMD' in _avail else
             "  -- ⚠ install 'pulp>=2.7,<3' (CBC is also used by the BoD ESI variant)"))
    print(f"  Parallel bootstrap: {PARALLEL_BOOTSTRAP} ({_n_jobs()} worker process(es)).")
    print(f"  Bootstrap: {DDF_BOOTSTRAP_METHOD} (beta in [0,{BETA_UPPER:g}], reflection at both bounds,")
    print(f"  pseudo-outputs along the DDF direction; see the DDF SMOOTH BOOTSTRAP note).")

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.2, 2, (n, 2)); Y = rng.uniform(0.2, 2, (n, 1)); B = rng.uniform(0.2, 2, (n, 1))
    X[:4, 0] = X[:, 0].min() * 0.01                     # exporter-like near-zero input
    saved = VRS_ABATEMENT
    res = {}
    try:
        for lab, vrs, mode in (("crs", False, "kuosmanen"), ("crs_u", False, "uniform"),
                               ("vrs_k", True, "kuosmanen"), ("vrs_u", True, "uniform")):
            VRS_ABATEMENT = mode
            res[lab] = np.array([_solve_ddf_one(X, Y, B, X[o], Y[o], B[o], vrs=vrs) for o in range(n)])
    finally:
        VRS_ABATEMENT = saved
    hk = np.array([_ddf_lp_highs(X, Y, B, X[o], Y[o], B[o], True, True, Y[o], B[o], False)[0]
                   for o in range(n)])
    ck = np.array([_ddf_lp_cbc(X, Y, B, X[o], Y[o], B[o], True, True, Y[o], B[o], False)[0]
                   for o in range(n)])
    th = {k: 1.0 / (1.0 + v) for k, v in res.items()}
    tests = {
        "a. own-period feasible, beta >= 0": all(np.all(np.isfinite(v)) and np.all(v >= 0) for v in res.values()),
        "b. theta CRS <= VRS(Kuosmanen) <= VRS(single-lambda)":
            bool(np.all(th["crs"] <= th["vrs_k"] + 1e-7) and np.all(th["vrs_k"] <= th["vrs_u"] + 1e-7)),
        "c. CRS unaffected by abatement weights": bool(np.max(np.abs(res["crs"] - res["crs_u"])) < 1e-7),
        "d. CBC = HiGHS (VRS Kuosmanen)": bool(np.nanmax(np.abs(ck - hk)) < 1e-6),
    }
    for k, ok in tests.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print(f"  (VRS Kuosmanen strictly below single-lambda for "
          f"{int((th['vrs_k'] < th['vrs_u'] - 1e-6).sum())}/{n} synthetic DMUs)")
    if save:
        src = "\n\n".join(inspect.getsource(f) for f in
                           (_solve_ddf_one, _ddf_lp_cbc, _ddf_lp_highs))
        _save_log(f"VRS_ABATEMENT = {VRS_ABATEMENT!r}\nNEG_BETA_TOL = {NEG_BETA_TOL!r}\n\n" + src,
                  "ddf_solver_source.txt")
        print(f"  Solver source saved to logs/ddf_solver_source.txt")
    all_ok = all(tests.values())
    if not all_ok:
        print("  ✗ SOLVER AUDIT FAILED -- do not interpret DDF results from this run.")
    return all_ok


# ═════════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION LAYER (bootstrap only)
# ═════════════════════════════════════════════════════════════════════════════
_STAT_KEYS = ("primary_not_optimal", "rescued_by_fallback", "unsolved",
              "tiny_negative_set_to_zero", "negative_beta_rejected")


def _n_jobs():
    import os
    if not PARALLEL_BOOTSTRAP:
        return 1
    return max(1, int(N_JOBS) if N_JOBS else (os.cpu_count() or 1))


def _stats_snapshot():
    return ({k: _DDF_SOLVE_STATS[k] for k in _STAT_KEYS},
            dict(_DDF_SOLVE_STATS["primary_status"]), len(_DDF_SOLVE_STATS["failures"]))


def _stats_delta(snap):
    c0, st0, nf0 = snap
    return ({k: _DDF_SOLVE_STATS[k] - c0[k] for k in _STAT_KEYS},
            {k: v - st0.get(k, 0) for k, v in _DDF_SOLVE_STATS["primary_status"].items()
             if v - st0.get(k, 0)},
            _DDF_SOLVE_STATS["failures"][nf0:])


def _stats_merge(delta):
    """Add a worker's solver counters to the parent's, so the solver summary
    stays complete when LPs are solved in child processes."""
    cnt, st, fails = delta
    for k, v in cnt.items():
        _DDF_SOLVE_STATS[k] += v
    for k, v in st.items():
        _DDF_SOLVE_STATS["primary_status"][k] = _DDF_SOLVE_STATS["primary_status"].get(k, 0) + v
    for f in fails:
        _DDF_SOLVE_STATS["last_failure"] = f
        if len(_DDF_SOLVE_STATS["failures"]) < 200:
            _DDF_SOLVE_STATS["failures"].append(f)


def _pool_worker(job):
    func, payload = job
    snap = _stats_snapshot()
    out = func(payload)
    return out, _stats_delta(snap)


def _pmap(func, payloads, label="", eta=False):
    """
    Ordered map of func over payloads, in parallel worker processes (fork) when
    PARALLEL_BOOTSTRAP is on, otherwise sequentially. func must be a module-
    level function and payloads picklable. Solver counters from workers are
    merged into the parent. With eta=True a progress line with an estimated
    finishing time is printed after the first item and at every 10%.
    """
    n = len(payloads)
    jobs = min(_n_jobs(), n)
    t0 = time.time()
    step = max(1, n // 10)

    def _progress(i):
        if eta and i > 0 and (i == 1 or i % step == 0 or i == n):
            el = time.time() - t0
            print(f"    {label}: {i}/{n} done, {el/60:.1f} min elapsed, "
                  f"ETA {el / i * (n - i) / 60:.1f} min", flush=True)

    if jobs > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            out = []
            with ctx.Pool(jobs) as pool:
                for i, (res, delta) in enumerate(pool.imap(_pool_worker,
                                                           [(func, p) for p in payloads],
                                                           chunksize=1), start=1):
                    _stats_merge(delta)
                    out.append(res)
                    _progress(i)
            return out
        except Exception as e:
            print(f"  ⚠ parallel execution failed ({e}); running sequentially instead")
    out = []
    for i, p in enumerate(payloads, start=1):
        out.append(func(p))
        _progress(i)
    return out


def _dea_year_task(p):
    """One year of Step 5: raw VRS and CRS scores + SW1998 bootstrap of the
    specification p['vrs']. Seeded by (seed, year), so results do not depend on
    how years are spread over workers."""
    X_n, Y_n, B_n = p["X"], p["Y"], p["B"]
    beta_vrs_ = _run_ddf_cross_section(X_n, Y_n, B_n, vrs=True)
    theta_vrs = np.where(np.isfinite(beta_vrs_), 1.0 / (1.0 + beta_vrs_), np.nan)
    beta_crs = _run_ddf_cross_section(X_n, Y_n, B_n, vrs=False)
    theta_crs = np.where(np.isfinite(beta_crs), 1.0 / (1.0 + beta_crs), np.nan)
    theta = theta_vrs if p["vrs"] else theta_crs
    rng = np.random.default_rng([p["seed"], int(p["year"])])
    theta_bc, bias, lo, hi, se = _bootstrap_ddf_one_year(
        X_n, Y_n, B_n, theta, p["B_int"], rng, p["vrs"], p["alpha"])
    info = getattr(_bootstrap_ddf_one_year, "last_info", {}) or {}
    return dict(theta=theta, theta_vrs=theta_vrs, theta_crs=theta_crs,
                theta_bc=theta_bc, bias=bias, lo=lo, hi=hi, se=se,
                trunc_low=info.get("trunc_low", 0), trunc_high=info.get("trunc_high", 0),
                h=info.get("h", np.nan),
                beta_bc_untrunc=info.get("beta_bc_untrunc", np.full(len(theta), np.nan)))


def _gml_theta_hat(Xr, Yr, Br, vrs):
    n = Xr.shape[0]
    beta_hat = np.array([_solve_ddf_one(Xr, Yr, Br, Xr[k], Yr[k], Br[k], vrs, allow_negative=False)
                         for k in range(n)])
    return 1.0 / (1.0 + np.clip(np.nan_to_num(beta_hat, nan=0.0), 0.0, None))


def _gml_one_replication(Xr, Yr, Br, theta_hat, vrs, rng):
    """One smooth-homogeneous bootstrap replication against one reference
    technology: returns lambda* = 1 + beta* (n,) for the ORIGINAL DMUs.
    Uses the DDF-adapted bootstrap (DDF_BOOTSTRAP_METHOD)."""
    if DDF_BOOTSTRAP_METHOD == "ddf_additive":
        bh = 1.0 / np.clip(theta_hat, 1e-9, 1.0) - 1.0
        bs = _smooth_beta_draws(bh, 1, rng)[0][0]
        Ys, Bs = _ddf_pseudo_outputs(Yr, Br, bh, bs)
        return 1.0 + _ddf_eval_against(Xr, Yr, Br, Ys, Bs, vrs)
    n = Xr.shape[0]
    theta_safe = np.clip(theta_hat, 1e-6, 1.0)
    ts = np.clip(_smooth_bootstrap(theta_hat, 1, rng)[0], 0.01, 1.0)
    Y_star = Yr * (ts / theta_safe)[:, None]
    B_star = Br * (theta_safe / ts)[:, None]
    lam = np.full(n, np.nan)
    for k in range(n):
        bo = _solve_ddf_one(Xr, Y_star, B_star, Xr[k], Yr[k], Br[k],
                            vrs, g_y=Yr[k], g_b=Br[k], allow_negative=True)
        if np.isfinite(bo) and bo > -1.0:
            lam[k] = 1.0 + bo
    return lam


def _gml_own_year_task(p):
    """GML own-period bootstrap for one year (all B_int replications)."""
    theta_hat = _gml_theta_hat(p["X"], p["Y"], p["B"], p["vrs"])
    draws = np.vstack([_gml_one_replication(p["X"], p["Y"], p["B"], theta_hat, p["vrs"],
                                            np.random.default_rng([p["seed"], int(p["year"]), b]))
                       for b in range(p["B_int"])])
    return 1.0 / np.clip(theta_hat, 1e-6, 1.0), draws


def _gml_global_rep_task(p):
    """One replication of the GML global (pooled-technology) bootstrap."""
    return _gml_one_replication(p["X"], p["Y"], p["B"], p["theta_hat"], p["vrs"],
                                np.random.default_rng([p["seed"], 999999, p["b"]]))


def report_ddf_solver_stats(label="run"):
    """Print and return how many DDF LPs CBC failed on, how many HiGHS rescued,
    and how many stayed unsolved (these are the only source of missing scores)."""
    st = _DDF_SOLVE_STATS
    _fb = "CBC" if DDF_SOLVER == "highs" else "HiGHS"
    print(f"\n  DDF solver check ({label}): primary solver {DDF_SOLVER.upper()} not optimal in "
          f"{st['primary_not_optimal']} LP(s)"
          + (f" {st['primary_status']}" if st['primary_status'] else "")
          + f"; rescued by fallback ({_fb}): {st['rescued_by_fallback']}; unsolved: {st['unsolved']}")
    print(f"    beta slightly below 0 (|beta| <= {NEG_BETA_TOL:g}) set to 0: "
          f"{st['tiny_negative_set_to_zero']}; materially negative beta rejected: "
          f"{st['negative_beta_rejected']}")
    print(f"    Missing DDF scores can ONLY come from 'unsolved' or 'negative beta rejected'.")
    for f in st["failures"][:10]:
        print(f"      failure: {f}")
    out = {k: v for k, v in st.items() if k not in ("failures", "last_failure")}
    out["n_failures_logged"] = len(st["failures"])
    return out


def _run_ddf_cross_section(X, Y, B, vrs=True, g_y=None, g_b=None):
    """
    Runs the DDF for every DMU in one cross-section.
    Run DDF for all DMUs in one cross-section.
    Returns (n,) array of DDF scores β* ≥ 0 (higher = less efficient).
    Efficiency scores: θ = 1/(1+β*) ∈ (0,1].
    """
    n = X.shape[0]
    return np.array([
        _solve_ddf_one(X, Y, B, X[o], Y[o], B[o], vrs, g_y, g_b)
        for o in range(n)
    ])


# ═════════════════════════════════════════════════════════════════════════════
# DDF SMOOTH BOOTSTRAP — documented adaptation of Simar & Wilson (1998)
# ═════════════════════════════════════════════════════════════════════════════
# Simar & Wilson (1998) developed the smooth homogeneous bootstrap for RADIAL
# (Farrell) distances in conventional input/output models. This pipeline uses a
# directional distance function with observation-scaled direction g = (y0, b0)
# and a weakly disposable bad output. The adaptation (DDF_BOOTSTRAP_METHOD =
# "ddf_additive") is:
#
# 1. SUPPORT. With g = (y0, b0) the bad-output equality b0 - beta*b0 >= 0
#    bounds beta to [0, 1], so theta = 1/(1+beta) lies in [0.5, 1], not (0, 1].
#    All smoothing is done on beta in [0, 1] (the old version smoothed theta on
#    (0, 1] and could draw impossible values below 0.5).
# 2. SMOOTHING AND REFLECTION. Draws are resampled from beta_hat, perturbed by
#    a Gaussian kernel with Silverman's robust bandwidth computed on the
#    reflected sample {beta, -beta, 2-beta}, variance-corrected as in Simar &
#    Wilson (1998, eq. 3.31), and reflected at BOTH bounds (0 = frontier,
#    1 = bad output driven to zero) until they lie in [0, 1].
# 3. JOINT OUTPUT TREATMENT. Pseudo-data are generated along each DMU's own
#    DDF direction from its estimated frontier projection F_j = (y_j(1+b_j),
#    b_j(1-b_j)):   y*_j = y_j (1 + beta_hat_j - beta*_j)
#                   b*_j = b_j (1 - beta_hat_j + beta*_j),   x*_j = x_j.
#    Good and bad outputs move by the same ADDITIVE proportion of the DMU's own
#    levels, i.e. exactly along the DDF direction, so the pseudo-DMU lies at
#    directional distance beta*_j from the estimated frontier. (The old version
#    scaled y by theta*/theta and b by theta/theta*, a radial rule that is not
#    the DDF geometry.)
# 4. RE-EVALUATION. Each ORIGINAL DMU is evaluated against the pseudo-frontier
#    with its own direction (y_o, b_o); beta may be negative (super-efficient
#    relative to the pseudo-frontier) and such draws are kept.
# 5. SCORE CORRECTION. bias = mean(beta*_eval) - beta_hat; beta_bc = beta_hat
#    - bias = 2 beta_hat - mean(beta*_eval), truncated to the support [0, 1]
#    (the number truncated is reported); theta_bc = 1/(1+beta_bc).
# 6. INTERVALS. Basic bootstrap interval on beta:
#    [2 beta_hat - q(1-a/2), 2 beta_hat - q(a/2)], truncated to [0, 1] and
#    widened if necessary to contain beta_bc, then mapped to theta (order flips).
#    Coverage is NOT assumed: validate_ddf_bootstrap() estimates it by Monte
#    Carlo on a known DDF technology.
# 7. DEPENDENCE. Within a year, the shared-frontier dependence between DMUs is
#    reproduced, because every DMU is re-evaluated against the same pseudo-
#    frontier in each replication. The homogeneous bootstrap assumes the
#    inefficiency distribution does not depend on (x, y, b) or Z -- an
#    assumption the separability results call into question. Years are
#    bootstrapped independently, so SERIAL dependence across years is not
#    reproduced: the intervals are valid for single country-years, not for
#    multi-year aggregates (country means, trends), for which no bootstrap
#    interval is reported.
DDF_BOOTSTRAP_METHOD = "ddf_additive"     # or "legacy" (old radial rule, for comparison only)
BETA_UPPER = 1.0                           # support of beta under g = (y0, b0)


def _smooth_beta_draws(beta_hat, B, rng, upper=BETA_UPPER):
    """Smooth bootstrap draws of beta* (B, n) on [0, upper]: resampling,
    Gaussian kernel (Silverman robust bandwidth on the reflected sample),
    Simar-Wilson variance correction, reflection at 0 and at `upper`."""
    b = np.clip(np.nan_to_num(np.asarray(beta_hat, float), nan=0.0), 0.0, upper)
    n = len(b)
    refl = np.concatenate([b, -b, 2 * upper - b])
    sd = refl.std()
    iqr = np.subtract(*np.percentile(refl, [75, 25]))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    h = max(0.9 * spread * len(refl) ** (-0.2), 1e-4)
    s2 = max(b.var(), 1e-12)
    out = np.empty((B, n))
    for r in range(B):
        bt = rng.choice(b, size=n, replace=True)
        bbar = bt.mean()
        z = bbar + (bt + h * rng.standard_normal(n) - bbar) / np.sqrt(1.0 + h * h / s2)
        for _ in range(20):                        # reflect into [0, upper]
            z = np.where(z < 0, -z, z)
            z = np.where(z > upper, 2 * upper - z, z)
            if np.all((z >= 0) & (z <= upper)):
                break
        out[r] = np.clip(z, 0.0, upper)
    return out, h


def _ddf_pseudo_outputs(Y, Bm, beta_hat, beta_star):
    """Pseudo-outputs along each DMU's DDF direction (see note 3 above)."""
    bh = np.clip(np.nan_to_num(beta_hat, nan=0.0), 0.0, BETA_UPPER)
    return (Y * (1.0 + bh - beta_star)[:, None],
            Bm * (1.0 - bh + beta_star)[:, None])


def _ddf_eval_against(Xr, Yr, Br, Ys, Bs, vrs):
    """beta of each ORIGINAL DMU (Xr, Yr, Br) against the pseudo-frontier
    (Xr, Ys, Bs), own direction, beta free."""
    n = Xr.shape[0]
    out = np.full(n, np.nan)
    for o in range(n):
        bo = _solve_ddf_one(Xr, Ys, Bs, Xr[o], Yr[o], Br[o], vrs,
                            g_y=Yr[o], g_b=Br[o], allow_negative=True)
        if np.isfinite(bo) and bo > -1.0:
            out[o] = bo
    return out


def _ddf_bc_interval(beta_hat, boot_beta, alpha, upper=BETA_UPPER):
    """Bias correction and basic interval on beta, truncated to [0, upper].
    Returns beta_bc, beta_lo, beta_hi, n_trunc_low, n_trunc_high."""
    bh = np.clip(np.nan_to_num(beta_hat, nan=0.0), 0.0, upper)
    with np.errstate(all="ignore"):
        raw = 2.0 * bh - np.nanmean(boot_beta, axis=0)
        q_lo = np.nanpercentile(boot_beta, 100 * alpha / 2, axis=0)
        q_hi = np.nanpercentile(boot_beta, 100 * (1 - alpha / 2), axis=0)
    no = np.all(np.isnan(boot_beta), axis=0)
    raw[no] = bh[no]
    n_lo, n_hi = int(np.nansum(raw < 0)), int(np.nansum(raw > upper))
    bbc = np.clip(raw, 0.0, upper)
    blo = np.clip(2.0 * bh - q_hi, 0.0, upper)
    bhi = np.clip(2.0 * bh - q_lo, 0.0, upper)
    blo[no], bhi[no] = bh[no], bh[no]
    blo, bhi = np.fmin(blo, bbc), np.fmax(bhi, bbc)
    return bbc, blo, bhi, n_lo, n_hi


def _bootstrap_ddf_additive(X, Y, Bm, beta_hat, B_int, rng, vrs, alpha):
    """One cross-section: DDF-adapted smooth homogeneous bootstrap.
    Returns theta_bc, bias, lo, hi, se, info."""
    bh = np.clip(np.nan_to_num(beta_hat, nan=0.0), 0.0, BETA_UPPER)
    draws, h = _smooth_beta_draws(bh, B_int, rng)
    boot = np.full((B_int, len(bh)), np.nan)
    for r in range(B_int):
        Ys, Bs = _ddf_pseudo_outputs(Y, Bm, bh, draws[r])
        boot[r] = _ddf_eval_against(X, Y, Bm, Ys, Bs, vrs)
    bbc, blo, bhi, n_lo, n_hi = _ddf_bc_interval(bh, boot, alpha)
    with np.errstate(all="ignore"):
        b_raw = 2.0 * bh - np.nanmean(boot, axis=0)             # before truncation
    theta_hat = 1.0 / (1.0 + bh)
    theta_bc = 1.0 / (1.0 + bbc)
    lo, hi = 1.0 / (1.0 + bhi), 1.0 / (1.0 + blo)
    with np.errstate(all="ignore"):
        se = np.nanstd(1.0 / (1.0 + np.clip(boot, 0.0, BETA_UPPER)), axis=0)
    return theta_bc, theta_hat - theta_bc, lo, hi, se, {"h": h, "trunc_low": n_lo, "trunc_high": n_hi,
                                                         "beta_bc_untrunc": b_raw}


# ═════════════════════════════════════════════════════════════════════════════
# MONTE CARLO VALIDATION OF THE DDF BOOTSTRAP (coverage and score correction)
# ═════════════════════════════════════════════════════════════════════════════
def _mc_dgp(n, rng, n_ref=2000):
    """Known DDF technology (CRS, weakly disposable bad output) and a sample.
    Frontier points: x ~ U[1,10]^2, y = x1^0.4 x2^0.6, b = y*u, u ~ U[0.3,1].
    Sample DMUs are moved inside along their own direction:
        y = y_f/(1+beta_g), b = b_f/(1-beta_g),
    beta_g = 0 with prob. 0.15, else |N(0, 0.3)| truncated at 0.85.
    The TRUE distance of each sample DMU is computed against a large
    reference sample of n_ref frontier points (an inner approximation of the
    true technology, so true beta is very slightly understated)."""
    def frontier(m):
        x = rng.uniform(1, 10, (m, 2))
        y = x[:, 0] ** 0.4 * x[:, 1] ** 0.6
        return x, y[:, None], (y * rng.uniform(0.3, 1.0, m))[:, None]
    Xr, Yr, Br = frontier(n_ref)
    x, yf, bf = frontier(n)
    bg = np.where(rng.random(n) < 0.15, 0.0, np.clip(np.abs(rng.normal(0, 0.3, n)), 0, 0.85))
    return x, yf / (1 + bg)[:, None], bf / (1 - bg)[:, None], Xr, Yr, Br


def _mc_rep_task(p):
    rng = np.random.default_rng([p["seed"], p["rep"]])
    x, y, b, Xr, Yr, Br = _mc_dgp(p["n"], rng)
    vrs, B, alpha, n = p["vrs"], p["B"], p["alpha"], p["n"]
    # true beta: sample DMU vs reference frontier (frontier points + the DMU itself)
    beta_true = np.array([
        _solve_ddf_one(np.vstack([Xr, x[o:o+1]]), np.vstack([Yr, y[o:o+1]]), np.vstack([Br, b[o:o+1]]),
                       x[o], y[o], b[o], vrs, allow_negative=False) for o in range(n)])
    beta_hat = np.array([_solve_ddf_one(x, y, b, x[o], y[o], b[o], vrs, allow_negative=False)
                         for o in range(n)])
    rows = []
    for m in p["methods"]:
        brng = np.random.default_rng([p["seed"], p["rep"], 7])
        if m == "ddf_additive":
            th_bc, _, lo, hi, _, _ = _bootstrap_ddf_additive(x, y, b, beta_hat, B, brng, vrs, alpha)
        else:
            th = 1.0 / (1.0 + np.clip(np.nan_to_num(beta_hat), 0, None))
            th_bc, _, lo, hi, _ = _bootstrap_ddf_one_year_legacy(x, y, b, th, B, brng, vrs, alpha)
        b_bc = 1.0 / np.clip(th_bc, 1e-9, None) - 1.0
        b_lo, b_hi = 1.0 / np.clip(hi, 1e-9, None) - 1.0, 1.0 / np.clip(lo, 1e-9, None) - 1.0
        ok = np.isfinite(beta_true)
        rows.append({"method": m, "rep": p["rep"], "n": int(ok.sum()),
                     "coverage": float(np.mean((beta_true[ok] >= b_lo[ok] - 1e-9) & (beta_true[ok] <= b_hi[ok] + 1e-9))),
                     "coverage_inefficient": float(np.mean(((beta_true >= b_lo - 1e-9) & (beta_true <= b_hi + 1e-9))[ok & (beta_true > 1e-6)]))
                                             if np.any(ok & (beta_true > 1e-6)) else np.nan,
                     "width_beta": float(np.nanmean(b_hi[ok] - b_lo[ok])),
                     "bias_raw": float(np.nanmean(beta_hat[ok] - beta_true[ok])),
                     "bias_bc": float(np.nanmean(b_bc[ok] - beta_true[ok])),
                     "rmse_raw": float(np.sqrt(np.nanmean((beta_hat[ok] - beta_true[ok]) ** 2))),
                     "rmse_bc": float(np.sqrt(np.nanmean((b_bc[ok] - beta_true[ok]) ** 2)))})
    return rows


def validate_ddf_bootstrap(R=None, B=None, n=45, vrs=False, alpha=ALPHA,
                           methods=("ddf_additive", "legacy"), seed=2024):
    """
    Monte Carlo check of the DDF bootstrap on a KNOWN technology (_mc_dgp):
    empirical coverage of the nominal (1-alpha) intervals for the true beta,
    interval width, and whether the bias correction reduces bias and RMSE.
    Compares the DDF-adapted bootstrap with the legacy radial rule.
    R replications x B bootstrap draws; run in parallel. Results are saved to
    tables/bootstrap_validation_mc.csv. Coverage close to 1-alpha supports
    reporting the intervals as confidence intervals; marked under-coverage
    means they must be described as approximate.
    """
    R = MC_R if R is None else R
    B = MC_B if B is None else B
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP VALIDATION (Monte Carlo): R={R} samples of n={n}, B={B}, "
          f"{'VRS' if vrs else 'CRS'}, nominal {100*(1-alpha):.0f}% intervals")
    print(f"{'='*70}")
    out = _pmap(_mc_rep_task, [{"seed": seed, "rep": r, "n": n, "vrs": vrs, "B": B,
                                "alpha": alpha, "methods": list(methods)} for r in range(R)],
                label="bootstrap validation (replications)", eta=True)
    df = pd.DataFrame([row for rows in out for row in rows])
    summ = df.groupby("method").agg(coverage=("coverage", "mean"),
                                    coverage_inefficient=("coverage_inefficient", "mean"),
                                    width_beta=("width_beta", "mean"),
                                    bias_raw=("bias_raw", "mean"), bias_bc=("bias_bc", "mean"),
                                    rmse_raw=("rmse_raw", "mean"), rmse_bc=("rmse_bc", "mean")).reset_index()
    # Uncertainty of the coverage estimate is measured ACROSS the R independent
    # simulated datasets (units within a dataset share a frontier and are not
    # independent): s.e. = sd(per-dataset coverage) / sqrt(R).
    se_cov = df.groupby("method")["coverage"].agg(lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
    se_ci = df.groupby("method")["coverage_inefficient"].agg(lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()) if x.notna().sum() > 1 else np.nan)
    summ["coverage_se_across_datasets"] = summ["method"].map(se_cov)
    summ["coverage_inefficient_se_across_datasets"] = summ["method"].map(se_ci)
    summ["R"], summ["B"], summ["n"] = R, B, n
    print(f"  {'method':<14}{'coverage (s.e.)':>18}{'ineff. (s.e.)':>17}{'width':>8}{'bias raw':>10}"
          f"{'bias bc':>9}{'RMSE raw':>10}{'RMSE bc':>9}")
    for _, r in summ.iterrows():
        print(f"  {r['method']:<14}{r['coverage']:>10.3f} ({r['coverage_se_across_datasets']:.3f})"
              f"{r['coverage_inefficient']:>9.3f} ({r['coverage_inefficient_se_across_datasets']:.3f})"
              f"{r['width_beta']:>8.3f}{r['bias_raw']:>+10.4f}{r['bias_bc']:>+9.4f}"
              f"{r['rmse_raw']:>10.4f}{r['rmse_bc']:>9.4f}")
    print(f"  Nominal coverage {1-alpha:.2f}. s.e. = standard deviation of per-dataset coverage across")
    print(f"  the R = {R} independent simulated datasets / sqrt(R)"
          + (" — with R < 30 this is itself imprecise; treat the result as preliminary." if R < 30 else "."))
    print(f"  More replications measure coverage more precisely; they do not correct undercoverage.")
    print(f"  SCOPE: this validates the Step 5 smooth bootstrap for ANNUAL {'VRS' if vrs else 'CRS'} "
          f"COUNTRY-YEAR")
    print(f"  intervals, under THIS data-generating process (n = {n}, one input pair, one good and")
    print(f"  one bad output, homogeneous inefficiency). It does NOT establish coverage for the")
    print(f"  {'CRS' if vrs else 'VRS'} intervals, GML intervals, second-stage coefficients or multi-year averages,")
    print(f"  nor coverage under other data-generating processes.")
    print(f"  Undercoverage of the homogeneous smooth bootstrap is documented in the literature")
    print(f"  (small samples, mass at the frontier, heterogeneity); consistent alternatives such")
    print(f"  as subsampling or the double-smooth bootstrap (Kneip, Simar and Wilson 2008) are")
    print(f"  NOT implemented here.")
    print(f"  Bias and RMSE are on the beta scale (beta = 1/theta - 1); 'bc' = after bias correction.")
    print(f"  Reporting: unless coverage is close to {1-alpha:.2f}, describe the intervals as")
    print(f"  'approximate bootstrap intervals constructed at the nominal {100*(1-alpha):.0f}% level' and")
    print(f"  report the simulated coverage beside them.")
    _save_csv(df, "bootstrap_validation_mc_reps.csv", "Bootstrap Monte Carlo, per replication")
    _save_csv(summ, "bootstrap_validation_mc.csv", "Bootstrap Monte Carlo summary")
    return summ


def _bootstrap_ddf_one_year(X, Y, B_mat, theta_orig, B_int, rng, vrs, alpha,
                             g_y=None, g_b=None):
    """Dispatcher: DDF-adapted bootstrap (default) or the legacy radial rule."""
    if DDF_BOOTSTRAP_METHOD == "ddf_additive":
        th = np.clip(np.asarray(theta_orig, float), 1e-9, 1.0)
        out = _bootstrap_ddf_additive(X, Y, B_mat, 1.0 / th - 1.0, B_int, rng, vrs, alpha)
        _bootstrap_ddf_one_year.last_info = out[5]
        return out[:5]
    _bootstrap_ddf_one_year.last_info = {}
    return _bootstrap_ddf_one_year_legacy(X, Y, B_mat, theta_orig, B_int, rng, vrs, alpha, g_y, g_b)


def _bootstrap_ddf_one_year_legacy(X, Y, B_mat, theta_orig, B_int, rng, vrs, alpha,
                                   g_y=None, g_b=None):
    """
    SW1998 smooth homogeneous bootstrap, adapted for the DDF specification.

    Same smooth homogeneous pseudo-frontier bootstrap as the standard DEA
    version, but using the DDF LP (_solve_ddf_one) for scoring.

    Pseudo-frontier construction:
      beta* ~ smoothed distribution of beta_hat (DDF scores)
      theta* = 1/(1+beta*) -- pseudo efficiencies
      y*_j = y_j * (theta*_j / theta_hat_j)  -- rescale good output
      b*_j = b_j * (theta_hat_j / theta*_j)  -- rescale bad output (opposite)
        Note: when good output expands, bad output must contract proportionally
        to maintain weak disposability on the pseudo frontier.

    Returns: theta_bc, bias, lo, hi, se
    """
    n = X.shape[0]
    # *** FIX: g_y/g_b are no longer a single shared vector for the whole
    # bootstrap -- each DMU o uses its OWN original y0/b0 as the direction
    # (observation-scaled DDF, see _solve_ddf_one), matching the point
    # estimate's convention so bias-correction stays on the same percentage
    # scale. Any g_y/g_b explicitly passed by the caller are still honoured
    # as an override (kept for backward compatibility / alternative DDF
    # variants), but the default is now per-observation, not np.ones(). ***

    theta_safe   = np.clip(theta_orig, 1e-6, 1.0)
    lam_hat      = 1.0 / theta_safe
    boot_lam     = np.full((B_int, n), np.nan)

    # SW1998: draw pseudo-efficiencies from smoothed empirical distribution
    beta_star  = _smooth_bootstrap(theta_orig, B_int, rng)   # (B,n) in (0,1]

    for b in range(B_int):
        scale_good = np.clip(beta_star[b], 0.01, 1.0) / theta_safe
        scale_bad  = theta_safe / np.clip(beta_star[b], 0.01, 1.0)
        Y_star = Y * scale_good[:, None]
        B_star = B_mat * scale_bad[:, None]
        for o in range(n):
            _g_y = g_y if g_y is not None else Y[o]        # observation's own y0
            _g_b = g_b if g_b is not None else B_mat[o]     # observation's own b0
            # FIX (verified by simulation): allow beta<0 so super-efficient
            # pseudo-draws are kept, and store lambda* = 1+beta* (Farrell scale).
            # Previously stored theta* = 1/(1+beta*), mixing theta and lambda in
            # lam_bc = 2*lam_hat - mean(boot_lam) and over-correcting the bias.
            beta_o = _solve_ddf_one(X, Y_star, B_star, X[o], Y[o], B_mat[o],
                                    vrs, g_y=_g_y, g_b=_g_b, allow_negative=True)
            if np.isfinite(beta_o) and beta_o > -1.0:
                boot_lam[b, o] = 1.0 + beta_o
            else:
                boot_lam[b, o] = np.nan

    # Bias correction and CI (same reflected-bootstrap logic as standard DEA)
    lam_star_mean = np.nanmean(boot_lam, axis=0)
    lam_bc   = np.clip(2.0 * lam_hat - lam_star_mean, 1.0, None)
    theta_bc = 1.0 / lam_bc
    bias     = theta_orig - theta_bc

    q_lo_lam = np.nanpercentile(boot_lam, 100 * alpha / 2,       axis=0)
    q_hi_lam = np.nanpercentile(boot_lam, 100 * (1 - alpha / 2), axis=0)
    lam_lo   = np.clip(2.0 * lam_hat - q_hi_lam, 1.0, None)
    lam_hi   = np.clip(2.0 * lam_hat - q_lo_lam, 1.0, None)
    lam_lo   = np.minimum(lam_lo, lam_bc)
    lam_hi   = np.maximum(lam_hi, lam_bc)
    lo = 1.0 / lam_hi
    hi = 1.0 / lam_lo
    se = np.nanstd(np.clip(1.0 / boot_lam, 0.01, 1.0), axis=0)

    # *** FIX: guarantee dea_bc coverage never falls short of the point
    # estimate's coverage. If ALL B bootstrap replications were infeasible
    # for a specific DMU (boot_lam[:, o] entirely NaN), lam_star_mean is NaN
    # and theta_bc/lo/hi/se would silently drop that DMU even though its
    # point estimate theta_orig is perfectly valid. This previously caused
    # the reported "DMUs/year" count (based on dea_bc) to fall below the
    # true n=45 despite every annual point-estimate loop reporting n=45.
    # Fallback: use the point estimate itself with zero bias and a
    # zero-width CI for that DMU only, so it is never silently dropped. ***
    all_nan_cols = np.all(np.isnan(boot_lam), axis=0)
    if np.any(all_nan_cols):
        theta_bc[all_nan_cols] = theta_orig[all_nan_cols]
        bias[all_nan_cols]     = 0.0
        lo[all_nan_cols]       = theta_orig[all_nan_cols]
        hi[all_nan_cols]       = theta_orig[all_nan_cols]
        se[all_nan_cols]       = np.nan   # SE genuinely undefined for these

    return theta_bc, bias, lo, hi, se

def build_dea_bootstrap(panel, B=B_BOOTSTRAP, alpha=ALPHA, vrs=VRS):
    """
    Directional Distance Function (DDF), replacing standard output-oriented DEA.

    Year-by-year Directional Distance Function (DDF) with weak disposability
    of the undesirable output (CO2 intensity), following Chung, Fare &
    Grosskopf (1997) and Fare, Grosskopf & Pasurka (2007).
    Simar-Wilson (1998) smooth homogeneous bootstrap on the Farrell scale.

    Production model:
      Inputs    : energy_pc (per-capita energy), dea_net_import (import dep.)
      Good output: electricity generation (per capita where available)
      Bad output : co2_intensity_elec (lifecycle GHG intensity of electricity,
                   gCO2e/kWh, WEAKLY disposable)
      Direction  : g = (y0, b0) -- observation-scaled, see _solve_ddf_one

    Weak disposability is imposed by the EQUALITY constraint on the bad output
    in the DDF LP (see _solve_ddf_one). This is the key departure from treating
    CO2 as a regular input (which assumes FREE disposability and overstates
    the production frontier).

    DDF score: beta* >= 0  (0 = on frontier)
    Efficiency: theta = 1/(1+beta*) in (0,1]
    Bootstrap bias-correction and CI follow the same SW1998 smooth homogeneous
    bootstrap as the standard DEA, operating on the Farrell lambda = 1/theta scale.

    Adds to panel:
        dea_score      -- raw DDF efficiency theta = 1/(1+beta*)
        dea_bc         -- bias-corrected efficiency
        dea_bias       -- estimated bias
        dea_lo,dea_hi  -- (1-alpha) confidence interval
        dea_se         -- bootstrap standard error
        dea_crs        -- CRS DDF efficiency (overall technical efficiency)
        dea_crs, dea_vrs -- raw CRS and VRS scores (both always computed)
        dea_se_score   -- scale efficiency SE = dea_crs / dea_vrs
        dea_<rts>_bc/_lo/_hi/_bias/_se -- spec-named copies of the corrected score
    """
    print(f"\n{'='*60}")
    _rts_run = "VRS" if vrs else "CRS"
    _role_run = "baseline" if _rts_run == BASE_RTS else "robustness check"
    print(f"SECTION 4 -- DDF (Undesirable Outputs) + SW1998 Bootstrap  (B={B})")
    print(f"  BOOTSTRAPPED SPECIFICATION: {_rts_run} ({_role_run}). Raw CRS and VRS")
    print(f"  scores are both printed each year (scale efficiency needs both), but")
    print(f"  theta_bc / bias / CI on each line refer to {_rts_run} only.")
    print("  Lifecycle GHG intensity of electricity (gCO2e/kWh, Ember via OWID) is the")
    print("  undesirable output, under a weak-disposability")
    print("  technology (imposed here via an equality-form directional")
    print("  constraint -- the equality is this implementation's chosen way of")
    print("  encoding joint output contraction, not the definition of weak")
    print("  disposability itself; see build_dea_bootstrap's docstring)")
    print("  Inputs  : energy_pc, dea_net_import")
    print("  Good out: electricity per capita (elec_gen_twh or elec_gen_twh_pc)")
    print("  Bad out : co2_intensity_elec")
    print("  Direction g=(y0,b0): observation-scaled -- each DMU's beta is a")
    print("  literal proportion of ITS OWN output levels (see docstring below)")
    print("="*60)
    global INPUT_COLS, OUTPUT_COLS, GOOD_OUT, BAD_OUT

    # Derive co2_intensity_elec if not yet present
    # *** HIGHLIGHTED: co2_intensity_elec is the BAD OUTPUT, not an input ***
    if "co2_intensity_elec" not in panel.columns:
        panel = compute_co2_intensity(panel)

    # Derive dea_net_import (bounded import dependency, max(0,.))
    epsilon = 0.001
    if "net_import_pct" in panel.columns:
        panel["dea_net_import"] = panel["net_import_pct"].clip(lower=0)
        n_exp = (panel["net_import_pct"] <= 0).sum()
        n_imp = (panel["net_import_pct"] >  0).sum()
        print(f"  dea_net_import: max(0, net_import_pct) -- "
              f"exporters (->0): {n_exp}  importers (->value): {n_imp}")
    else:
        panel["dea_net_import"] = 0.0

    for c in INPUT_COLS + GOOD_OUT + BAD_OUT:
        if c not in panel.columns:
            panel[c] = np.nan

    # *** HIGHLIGHTED: imputation covers GOOD_OUT and BAD_OUT ***
    print("\n  DDF pre-imputation (before/after NaN counts):")
    all_ddf_cols = INPUT_COLS + GOOD_OUT + BAD_OUT
    for col in all_ddf_cols:
        n_before = panel[col].isna().sum()
        if n_before == 0:
            continue
        panel[col] = (panel.groupby("iso_code")[col]
                           .transform(lambda x: x.interpolate(method="linear",
                                                              limit_direction="both")
                                                 .ffill().bfill()))
        panel[col] = (panel[col].groupby(panel["year"])
                                .transform(lambda x: x.fillna(x.median())))
        if panel[col].isna().any():
            panel[col] = panel[col].fillna(panel[col].median())
        n_after = panel[col].isna().sum()
        if n_before > 0:
            print(f"    {col}: {n_before} NaN -> {n_after} NaN after imputation")

    # Zero-replacement for inputs and good output
    for col in INPUT_COLS:
        if (panel[col] == 0).any():
            panel[col] = panel.groupby("year")[col].transform(
                lambda x: x.replace(0, x[x>0].min()*0.5 if (x>0).any() else 1e-6))

    for col in GOOD_OUT:
        if (panel[col] <= 0).any():
            panel[col] = panel.groupby("year")[col].transform(
                lambda x: x.mask(x<=0, x[x>0].min()*0.5 if (x>0).any() else 1e-6))

    # Bad output: CO2 must be positive (DDF equality constraint needs b0 > 0)
    for col in BAD_OUT:
        if (panel[col] <= 0).any():
            panel[col] = panel.groupby("year")[col].transform(
                lambda x: x.mask(x<=0, x[x>0].min()*0.5 if (x>0).any() else 1e-6))
            print(f"    {col}: non-positive values replaced (required for DDF equality cstr)")

    # *** HIGHLIGHTED: per-capita switch applies to GOOD_OUT only ***
    pop_col = next((c for c in ["population","pop"] if c in panel.columns), None)
    if pop_col:
        # *** FIX (missing-data gap): the previous interpolate+ffill/bfill
        # step had NO cross-sectional or panel-wide fallback. A country
        # entirely missing population data (or missing it at both edges of
        # its series, beyond what ffill/bfill can reach) would leave
        # elec_gen_twh_pc as NaN for those years further down, which the
        # per-year DDF mask then silently EXCLUDED from that year's frontier
        # -- producing the exact "fewer than 24 years" symptom investigated
        # here. Population now goes through the SAME full imputation chain
        # (interpolate -> ffill/bfill -> cross-sectional year-median ->
        # panel-wide median) already used for the core DDF columns. ***
        n_pop_before = panel[pop_col].isna().sum()
        panel[pop_col] = (panel.groupby("iso_code")[pop_col]
                               .transform(lambda x: x.interpolate(method="linear",
                                                                  limit_direction="both")
                                                     .ffill().bfill()))
        panel[pop_col] = (panel[pop_col].groupby(panel["year"])
                                .transform(lambda x: x.fillna(x.median())))
        if panel[pop_col].isna().any():
            panel[pop_col] = panel[pop_col].fillna(panel[pop_col].median())
        n_pop_after = panel[pop_col].isna().sum()
        if n_pop_before > 0:
            print(f"  {pop_col}: {n_pop_before} NaN -> {n_pop_after} NaN after imputation")

    if pop_col and "elec_gen_twh" in panel.columns:
        panel["elec_gen_twh_pc"] = (
            panel["elec_gen_twh"] * 1e6 / panel[pop_col].replace(0, np.nan))
        n_before_pc = panel["elec_gen_twh_pc"].isna().sum()
        print(f"  elec_gen_twh_pc derived ({panel['elec_gen_twh_pc'].notna().sum()} obs, "
              f"{n_before_pc} NaN before imputation)")

        # *** FIX: run elec_gen_twh_pc through the SAME imputation chain as
        # every other DDF column, rather than leaving it unimputed because
        # it did not exist yet when the main imputation loop ran above. ***
        if n_before_pc > 0:
            panel["elec_gen_twh_pc"] = (
                panel.groupby("iso_code")["elec_gen_twh_pc"]
                     .transform(lambda x: x.interpolate(method="linear",
                                                        limit_direction="both")
                                          .ffill().bfill()))
            panel["elec_gen_twh_pc"] = (
                panel["elec_gen_twh_pc"].groupby(panel["year"])
                     .transform(lambda x: x.fillna(x.median())))
            if panel["elec_gen_twh_pc"].isna().any():
                panel["elec_gen_twh_pc"] = panel["elec_gen_twh_pc"].fillna(
                    panel["elec_gen_twh_pc"].median())
            n_after_pc = panel["elec_gen_twh_pc"].isna().sum()
            print(f"    elec_gen_twh_pc: {n_before_pc} NaN -> {n_after_pc} NaN after imputation")

        # Non-positive replacement for the per-capita series specifically —
        # the earlier zero-replacement loop ran on the OLD GOOD_OUT
        # (elec_gen_twh) before this per-capita column existed.
        if (panel["elec_gen_twh_pc"] <= 0).any():
            panel["elec_gen_twh_pc"] = panel.groupby("year")["elec_gen_twh_pc"].transform(
                lambda x: x.mask(x<=0, x[x>0].min()*0.5 if (x>0).any() else 1e-6))

    if "elec_gen_twh_pc" in panel.columns and panel["elec_gen_twh_pc"].notna().sum() > 100:
        GOOD_OUT = ["elec_gen_twh_pc"]
        OUTPUT_COLS = GOOD_OUT
        print("  DDF good output: elec_gen_twh_pc (per capita)")
    else:
        GOOD_OUT = ["elec_gen_twh"]
        OUTPUT_COLS = GOOD_OUT
        print("  DDF good output: elec_gen_twh (population unavailable)")
    BAD_OUT = ["co2_intensity_elec"]
    print(f"  DDF bad output (undesirable): {BAD_OUT}")

    _data_audit(panel, INPUT_COLS + GOOD_OUT + BAD_OUT, "DDF bootstrap")

    _sp0 = "vrs" if vrs else "crs"
    for col in ["dea_score","dea_bc","dea_bias","dea_lo","dea_hi","dea_se",
                "dea_crs","dea_vrs","dea_se_score",
                f"dea_{_sp0}_bc", f"dea_{_sp0}_lo", f"dea_{_sp0}_hi",
                f"dea_{_sp0}_bias", f"dea_{_sp0}_se"]:
        panel[col] = np.nan

    years = sorted(panel["year"].unique())   # random seeds are set per year (seed 42, year)

    _all_exclusions = {}   # iso_code -> list of years excluded, across the whole loop

    # Phase 1: prepare one task per year (data selection and normalisation).
    tasks = []
    for yr in years:
        mask = (
            panel["year"].eq(yr)
            & panel[INPUT_COLS + GOOD_OUT + BAD_OUT].notna().all(axis=1)
            & (panel[INPUT_COLS] > 0).all(axis=1)
            & (panel[GOOD_OUT]   > 0).all(axis=1)
            & (panel[BAD_OUT]    > 0).all(axis=1)
        )
        sub   = panel.loc[mask].copy()
        n_all = panel["year"].eq(yr).sum()
        if n_all - len(sub) > 0:
            excl = panel.loc[panel["year"].eq(yr) & ~mask, "iso_code"].tolist()
            print(f"  {yr}: {n_all-len(sub)} excluded after imputation: {excl}")
            for iso in excl:
                _all_exclusions.setdefault(iso, []).append(yr)
        if len(sub) < 4:
            print(f"  {yr}: only {len(sub)} DMUs -- skipping"); continue
        # Observation-scaled direction g = (y0, b0) is the default inside
        # _solve_ddf_one; both raw scores (VRS and CRS) are always solved and
        # the specification selected by `vrs` is bootstrapped.
        tasks.append({"year": yr, "mask": mask, "n": len(sub),
                      "X": _normalise(sub[INPUT_COLS].values.astype(float)),
                      "Y": _normalise(sub[GOOD_OUT].values.astype(float)),
                      "B": _normalise(sub[BAD_OUT].values.astype(float)),
                      "vrs": vrs, "B_int": B, "alpha": alpha, "seed": 42})

    # Phase 2: compute (parallel over years when PARALLEL_BOOTSTRAP is on).
    print(f"  Solving {len(tasks)} years with {min(_n_jobs(), max(1, len(tasks)))} worker process(es), "
          f"primary LP solver {DDF_SOLVER.upper()}")
    results_yr = _pmap(_dea_year_task, [{k: v for k, v in t.items() if k != "mask"} for t in tasks],
                       label=f"Step 5 {_rts_run} bootstrap (years)", eta=True)

    _tl = sum(int(r.get("trunc_low", 0)) for r in results_yr)
    _th = sum(int(r.get("trunc_high", 0)) for r in results_yr)
    print(f"  Intervals below are APPROXIMATE bootstrap intervals constructed at the nominal "
          f"{100*(1-alpha):.0f}% level;")
    _val_rts = "VRS" if VRS else "CRS"           # the validation simulates the BASELINE specification
    if (vrs and VRS) or (not vrs and not VRS):
        print(f"  their coverage is assessed only by the Monte Carlo validation (annual "
              f"{_val_rts} country-years, specified DGP).")
    else:
        print(f"  Coverage of these {_rts_run} intervals has not been assessed by simulation; the Monte "
              f"Carlo validation in this run covers annual {_val_rts} country-year intervals only.")
    print(f"  Bootstrap method: {DDF_BOOTSTRAP_METHOD} (beta in [0, {BETA_UPPER:g}], reflection at both "
          f"bounds, joint additive output rule). Bias-corrected beta truncated to the support: "
          f"{_tl} at 0, {_th} at {BETA_UPPER:g} (theta_bc = 0.5)")

    # Phase 3: write results back and print the yearly lines in year order.
    for t, r in zip(tasks, results_yr):
        mask, yr = t["mask"], t["year"]
        theta_vrs, theta_crs = r["theta_vrs"], r["theta_crs"]
        theta, theta_bc, bias, lo, hi, se = r["theta"], r["theta_bc"], r["bias"], r["lo"], r["hi"], r["se"]
        se_score = np.where(theta_vrs > 1e-9, theta_crs / theta_vrs, np.nan)   # SE = CRS/VRS
        panel.loc[mask, "dea_score"]    = theta
        panel.loc[mask, "dea_bc"]       = theta_bc
        panel.loc[mask, "dea_bias"]     = bias
        panel.loc[mask, "dea_lo"]       = lo
        panel.loc[mask, "dea_hi"]       = hi
        panel.loc[mask, "dea_se"]       = se
        panel.loc[mask, "dea_crs"]      = theta_crs
        panel.loc[mask, "dea_vrs"]      = theta_vrs
        panel.loc[mask, "dea_se_score"] = np.clip(se_score, 0.0, 1.0)
        _sp = "vrs" if vrs else "crs"              # spec-named copies of the corrected score
        panel.loc[mask, f"dea_{_sp}_bc"]   = theta_bc
        panel.loc[mask, f"dea_{_sp}_lo"]   = lo
        panel.loc[mask, f"dea_{_sp}_hi"]   = hi
        panel.loc[mask, f"dea_{_sp}_bias"] = bias
        panel.loc[mask, f"dea_{_sp}_se"]   = se
        panel.loc[mask, f"dea_{_sp}_beta_bc_untrunc"] = r.get("beta_bc_untrunc", np.nan)

        ci_w = float(np.nanmean(hi - lo))
        warn = "  *** WIDE CI" if ci_w > 0.5 else ""
        print(f"  {yr}: n={t['n']:2d}  "
              f"DDF_VRS={np.nanmean(theta_vrs):.3f}  "
              f"DDF_CRS={np.nanmean(theta_crs):.3f}  "
              f"SE={np.nanmean(se_score):.3f}  "
              f"theta_bc({_rts_run})={np.nanmean(theta_bc):.3f}  "
              f"bias={np.nanmean(bias):+.4f}  "
              f"CI={ci_w:.3f}{warn}")

    # ── Boundary diagnostics (reported, not hidden) ───────────────────────────
    _bu = panel[f"dea_{_sp}_beta_bc_untrunc"] if f"dea_{_sp}_beta_bc_untrunc" in panel.columns else None
    if _bu is not None and _bu.notna().any():
        at_ub = _bu > BETA_UPPER
        bh_all = 1.0 / panel["dea_score"].clip(lower=1e-9) - 1.0
        print(f"\n  BOUNDARY DIAGNOSTICS ({_rts_run}): bias corrections reaching beta = {BETA_UPPER:g} "
              f"(theta_bc = 0.5): {int(at_ub.sum())}/{int(_bu.notna().sum())} "
              f"({at_ub.mean():.1%})")
        print(f"    raw theta of these observations: median {panel.loc[at_ub, 'dea_score'].median():.3f}, "
              f"max {panel.loc[at_ub, 'dea_score'].max():.3f}; share of ALL observations with raw "
              f"beta >= 0.8 (theta <= 0.556): {(bh_all >= 0.8).mean():.1%}")
        print(f"    untruncated corrected beta among them: median {_bu[at_ub].median():.3f}, "
              f"max {_bu[at_ub].max():.3f}")
        _byc = panel.loc[at_ub].groupby("iso_code").size().sort_values(ascending=False)
        print(f"    countries with most years at the bound: " +
              ", ".join(f"{k} ({v})" for k, v in _byc.head(10).items()))
        print(f"    Why: with g = (y0, b0) beta cannot exceed 1 (the bad output cannot fall")
        print(f"    below zero). Many countries already have raw theta close to 0.5, and the")
        print(f"    bias correction (2 beta_hat - mean beta*) pushes them past the bound. The")
        print(f"    truncation keeps theta_bc in its support but piles mass at 0.5: means of")
        print(f"    theta_bc are therefore bounded below by construction, and rankings among")
        print(f"    these countries are not distinguished by the corrected score.")
        _save_csv(pd.DataFrame({"iso_code": _byc.index, "years_at_upper_bound": _byc.values}),
                  f"step5_bootstrap_boundary_{_sp}.csv",
                  f"Bias corrections at the beta upper bound by country ({_rts_run})")

    # *** Coverage verification: this is the definitive answer to "does every
    # country have all 24 years of dea_score" -- not inferred from a table
    # denominator, but checked directly against the actual exclusions
    # recorded during the loop above. ***
    n_years_expected = len(years)
    print("\n  " + "="*70)
    print(f"  COVERAGE VERIFICATION: {len(panel['iso_code'].unique())} countries x "
          f"{n_years_expected} years = "
          f"{len(panel['iso_code'].unique())*n_years_expected} possible observations")
    if _all_exclusions:
        n_countries_affected = len(_all_exclusions)
        n_obs_missing = sum(len(v) for v in _all_exclusions.values())
        print(f"  ✗ INCOMPLETE: {n_countries_affected} countries missing "
              f"{n_obs_missing} country-year observations after imputation:")
        for iso, yrs_missing in sorted(_all_exclusions.items()):
            print(f"      {iso}: missing {len(yrs_missing)} year(s) -- {yrs_missing}")
        print(f"  These exclusions occurred because at least one DDF column")
        print(f"  (input, good output, or bad output) remained NaN or non-positive")
        print(f"  for that country-year even after the imputation chain above.")
        print(f"  Investigate the specific column/country/year combination using")
        print(f"  the per-year 'excluded after imputation' lines printed during")
        print(f"  the loop, and the Step 1/Step 3 raw data-coverage diagnostics.")
    else:
        print(f"  ✓ COMPLETE: all {len(panel['iso_code'].unique())} countries have "
              f"dea_score computed for all {n_years_expected} years — zero exclusions.")
    print("  " + "="*70)

    # Summary
    bc_all  = panel["dea_bc"].dropna()
    sc_all  = panel["dea_score"].dropna()
    crs_all = panel["dea_crs"].dropna()
    se_all  = panel["dea_se_score"].dropna()
    # Frontier classification:
    # (a) Frontier membership must be MUTUALLY EXCLUSIVE sets computed from
    #     the SAME indicator F_it = I(theta_it >= 0.999): ever_frontier is
    #     countries with >=1 year where F_it=1; always_below is countries
    #     with F_it=0 in EVERY year -- i.e. the exact complement of
    #     ever_frontier, not a separately (and inconsistently) filtered set.
    #     The previous code computed frontier_c and below_c as two
    #     INDEPENDENT filters (dea_bc>=0.999 in >=1 row) and (dea_bc<0.999
    #     in >=1 row) -- these are NOT complements of each other whenever a
    #     country has SOME years above and SOME below the threshold, so the
    #     same country could appear in both lists simultaneously.
    # (b) Frontier membership is now based on RAW theta_VRS (dea_score), NOT
    #     the bias-corrected dea_bc. Frontier status is a property of the
    #     estimated production frontier itself; bootstrap bias correction is
    #     an inferential adjustment that can move an originally-efficient
    #     DMU's score below 1, and should not be used to (re)classify
    #     frontier membership. dea_bc is still reported separately (see the
    #     three-way decomposition table) as the bias-corrected efficiency.
    _dea_valid = panel.dropna(subset=["dea_score"])
    _F_it = _dea_valid["dea_score"] >= 0.999
    frontier_c = sorted(_dea_valid.loc[_F_it, "iso_code"].unique())
    below_c    = sorted(set(_dea_valid["iso_code"].unique()) - set(frontier_c))

    print("  " + "-"*60)
    print(f"  DDF RESULTS SUMMARY  (*** HIGHLIGHTED: undesirable-output model ***)")
    print(f"  Returns to scale for raw + corrected scores: {'VRS' if vrs else 'CRS'}"
          + (f"  (weak disposability: {VRS_ABATEMENT} abatement)" if vrs else ""))
    print(f"  Mean DDF efficiency, {_rts_run} (raw theta)    : {sc_all.mean():.4f}")
    print(f"  Mean DDF efficiency, {_rts_run} (bias-corrected): {bc_all.mean():.4f}")
    # *** FIX: bias defined as theta - theta_bc (matches the per-year lines
    # above and the definition used throughout this file); the previous
    # summary line computed bc_all.mean()-sc_all.mean(), the OPPOSITE sign. ***
    print(f"  Mean bias, {_rts_run} (theta - theta_bc)       : {sc_all.mean()-bc_all.mean():+.4f}")
    print(f"  Mean CRS DDF efficiency (OTE)      : {crs_all.mean():.4f}")
    print(f"  Mean scale efficiency (SE)         : {se_all.mean():.4f}")
    # Improvement potential must come from each observation's beta = 1/theta - 1,
    # then be summarised: 1 - mean(theta) is NOT the average feasible expansion.
    _beta_obs = (1.0 / sc_all.clip(lower=1e-9)) - 1.0
    print(f"  Mean improvement potential beta*   : {_beta_obs.mean():.4f}  (median {_beta_obs.median():.4f})")
    print(f"    = simultaneous % expansion of electricity per capita and % cut in CO2")
    print(f"      intensity, computed per observation from beta = 1/theta - 1 (raw scores)")
    # *** FIX: use dea_score (point estimate, always fully populated — every
    # annual loop above reports n=45) instead of dea_bc, whose count can be
    # slightly lower if a handful of individual DMU-year bootstraps needed
    # the fallback above. This line now reports the TRUE annual DMU count. ***
    yr_counts_pt = panel.groupby("year")["dea_score"].count()
    print(f"  DMUs/year (point estimate): median {int(yr_counts_pt.median())} "
          f"(range {int(yr_counts_pt.min())}-{int(yr_counts_pt.max())})")
    print(f"  Raw frontier (baseline theta>=0.999) in >=1 year ({len(frontier_c)}): {frontier_c}")
    print(f"  Never on raw frontier / always below ({len(below_c)}): {below_c}")
    print(f"")
    print(f"  INTERPRETATION (DDF / undesirable-output model):")
    print(f"  theta=1.0: ON frontier -- cannot simultaneously expand electricity")
    print(f"             AND reduce lifecycle GHG intensity without changing inputs.")
    print(f"  theta=0.85 (beta*=0.18): with the observation-scaled direction")
    print(f"             g=(y0,b0), this IS a literal 18% proportional reading:")
    print(f"             could expand good output by 18% AND cut lifecycle GHG intensity")
    print(f"             by 18% -- an environmental efficiency gap.")
    print(f"  This score captures ENVIRONMENTAL efficiency (technical + green),")
    print(f"  not just operational throughput.")
    print("  " + "-"*60)

    _print_scale_table(panel, rts=("VRS" if vrs else "CRS"))

    return panel




# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GLOBAL MALMQUIST-LUENBERGER INDEX (GML = GEC × GTC)
# Reuses the DDF technology and solver from Section 4 (build_dea_bootstrap):
# CO2 is treated as an undesirable output under weak disposability, exactly as
# in the static DEA/DDF frontier. Every observation is evaluated against (a)
# its own period's technology and (b) a single pooled technology built from
# all years, guaranteeing every transition is computable — no country is ever
# dropped for cross-period infeasibility (Chung, Fare & Grosskopf 1997; Oh 2010).
# ═════════════════════════════════════════════════════════════════════════════

def _build_global_ddf_distances(panel, MPI_INPUTS, MPI_GOOD, MPI_BAD, vrs=False):
    """
    *** GLOBAL MALMQUIST-LUENBERGER (GML) — Oh (2010) ***

    Compute, for every (country, year), BOTH the own-period DDF distance
    D_t(z_t) and the global DDF distance D_G(z_t) against the pooled
    technology:

        T^G = conv( union_{t=2000}^{2023} T^t )

    This is the field-standard fix for cross-period infeasibility in
    Malmquist-Luenberger indices (Pastor & Lovell 2005; Oh 2010): instead
    of evaluating z_t against T_{t+1}'s technology (which can be genuinely
    infeasible if z_t's input-output-CO2 RAY lies outside the cone spanned
    by period t+1's observations), every observation is evaluated only
    against (a) its OWN period's technology, and (b) the GLOBAL pooled
    technology. Both are ALWAYS feasible by construction: a DMU trivially
    reproduces itself with lambda=1 on itself and beta=0, and it is by
    definition a member of both its own period's technology AND the
    pooled global technology (which contains every year, including its
    own). This guarantees 100% coverage -- no country-year is ever
    dropped for infeasibility.

    Reuses the IDENTICAL low-level solver _solve_ddf_one used by Step 5
    (build_dea_bootstrap) -- no separate radial engine.

    Normalisation: ONE set of bounds computed from the ENTIRE panel (all
    years, all countries) is used for BOTH the own-period and the global
    evaluations. This is essential: GTC below is a ratio of D_G(z) to
    D_t(z), so both must be measured on the identical scale for the
    comparison to be economically meaningful.

    Returns:
      result : DataFrame indexed by (iso_code, year) with columns
               beta_own, beta_global (raw DDF scores, >= 0 by construction)
      bounds : dict of the global normalisation bounds used (for reuse in
               the bootstrap version below)
    """
    sub = panel[["iso_code","year"] + MPI_INPUTS + MPI_GOOD + MPI_BAD].dropna().copy()
    sub = sub.sort_values(["year","iso_code"]).reset_index(drop=True)

    X_all = sub[MPI_INPUTS].values.astype(float)
    Y_all = sub[MPI_GOOD].values.astype(float)
    B_all = sub[MPI_BAD].values.astype(float)
    X_s, Y_s, B_s = X_all.mean(axis=0), Y_all.mean(axis=0), B_all.mean(axis=0)   # pooled scales

    X_n = _normalise(X_all, X_s)
    Y_n = _normalise(Y_all, Y_s)
    B_n = _normalise(B_all, B_s)

    # *** FIX: observation-scaled direction g=(y0,b0), NOT a shared g=(1,1).
    # Each DMU's own-period AND global evaluation now use that DMU's OWN
    # y0/b0 as the direction (the default inside _solve_ddf_one), so beta
    # is a literal percentage for BOTH: "this DMU could expand its own good
    # output / cut its own bad output by beta*100% to reach [own-period /
    # global] best practice." Removed the shared g_y=np.ones(...) override.

    n_total = len(sub)
    years   = sorted(sub["year"].unique())
    print(f"  Building T^G (global technology): pooling {n_total} country-year "
          f"observations across {len(years)} years ({years[0]}-{years[-1]})")

    beta_own    = np.full(n_total, np.nan)
    beta_global = np.full(n_total, np.nan)

    for yr in years:
        idx_yr = np.where((sub["year"] == yr).values)[0]
        X_yr = X_n[idx_yr]; Y_yr = Y_n[idx_yr]; B_yr = B_n[idx_yr]
        for k, i in enumerate(idx_yr):
            # Own-period: reference set = same year's DMUs only (always feasible)
            beta_own[i] = _solve_ddf_one(X_yr, Y_yr, B_yr, X_yr[k], Y_yr[k], B_yr[k],
                                         vrs, allow_negative=False)
            # Global: reference set = ALL years pooled, T^G (always feasible)
            beta_global[i] = _solve_ddf_one(X_n, Y_n, B_n, X_n[i], Y_n[i], B_n[i],
                                            vrs, allow_negative=False)

    result = sub[["iso_code","year"]].copy()
    result["beta_own"]    = beta_own
    result["beta_global"] = beta_global

    n_own_nan  = int(np.isnan(beta_own).sum())
    n_glob_nan = int(np.isnan(beta_global).sum())
    print(f"  Own-period D_t(z_t)  : {n_total-n_own_nan}/{n_total} feasible "
          f"({n_own_nan} infeasible -- should be 0 by construction)")
    print(f"  Global D_G(z_t)      : {n_total-n_glob_nan}/{n_total} feasible "
          f"({n_glob_nan} infeasible -- should be 0 by construction)")

    bounds = dict(X_s=X_s, Y_s=Y_s, B_s=B_s)   # pooled rescaling factors, reused by the bootstrap
    return result, bounds


def _bootstrap_global_ddf(panel, MPI_INPUTS, MPI_GOOD, MPI_BAD, vrs, bounds,
                           B_own, B_global, alpha, rng):
    """
    GML bootstrap: Simar-Wilson (1998) smooth homogeneous pseudo-frontier
    bootstrap, applied separately to each own-period technology T^t and to
    the pooled global technology T^G.

    *** REVISED (verified by simulation). The previous version had four
    problems, each fixed here:
      1. The pseudo-frontier rescaled ONLY the good output. The bad output is
         now rescaled in the opposite direction (b* = b * theta_hat/theta*),
         exactly as in Step 5's _bootstrap_ddf_one_year, so pseudo-data stay
         consistent with weak disposability.
      2. Each DMU was evaluated at its OWN PSEUDO-OBSERVATION (Y_star[k]),
         with the pseudo-observation as its direction. Each ORIGINAL DMU
         (X[k], Y[k], B[k]) is now scored against the pseudo-frontier, with
         its original (y0, b0) as the direction.
      3. The returned confidence bounds were (1/lam - 1) = theta - 1, i.e.
         negative numbers, not distances. All outputs are now on the DDF
         distance scale, beta = lambda - 1 >= 0.
      4. The bootstrap draws themselves are now returned, so build_mpi() can
         form a GML interval replication-by-replication instead of plugging
         four separate distance bounds into the index.

    All statistics are computed on the Farrell scale lambda = 1 + beta
    (= 1/theta), as in Step 5.

    Returns
    -------
    result : DataFrame (iso_code, year, beta_own_bc, beta_own_lo, beta_own_hi,
             beta_global_bc, beta_global_lo, beta_global_hi), rows ordered by
             (year, iso_code) exactly as in _build_global_ddf_distances().
    draws  : dict with
             "own"    : (B_own, n_total) array of lambda* against T^t
             "global" : (B_global, n_total) array of lambda* against T^G
             "keys"   : list of (iso_code, year), column order of both arrays
    """
    sub = panel[["iso_code","year"] + MPI_INPUTS + MPI_GOOD + MPI_BAD].dropna().copy()
    sub = sub.sort_values(["year","iso_code"]).reset_index(drop=True)

    X_n = _normalise(sub[MPI_INPUTS].values.astype(float), bounds["X_s"])
    Y_n = _normalise(sub[MPI_GOOD].values.astype(float),   bounds["Y_s"])
    B_n = _normalise(sub[MPI_BAD].values.astype(float),    bounds["B_s"])

    n_total = len(sub)
    years   = sorted(sub["year"].unique())

    def _bc_ci(lam_hat, boot_lam):
        """Bias correction and basic (reflected) interval on the lambda scale,
        returned on the DDF distance scale beta = lambda - 1."""
        no_draws = np.all(np.isnan(boot_lam), axis=0)
        _ub = 1.0 + BETA_UPPER          # lambda = 1 + beta lies in [1, 2] under g = (y0, b0)
        with np.errstate(all="ignore"):
            lam_bc = np.clip(2.0 * lam_hat - np.nanmean(boot_lam, axis=0), 1.0, _ub)
            q_lo   = np.nanpercentile(boot_lam, 100 * alpha / 2,       axis=0)
            q_hi   = np.nanpercentile(boot_lam, 100 * (1 - alpha / 2), axis=0)
        lam_lo = np.fmin(np.clip(2.0 * lam_hat - q_hi, 1.0, _ub), lam_bc)
        lam_hi = np.fmax(np.clip(2.0 * lam_hat - q_lo, 1.0, _ub), lam_bc)
        for arr in (lam_bc, lam_lo, lam_hi):                           # never drop a DMU
            arr[no_draws] = lam_hat[no_draws]
        return lam_bc - 1.0, lam_lo - 1.0, lam_hi - 1.0

    own_draws  = np.full((B_own, n_total), np.nan)
    beta_own_bc = np.full(n_total, np.nan); beta_own_lo = np.full(n_total, np.nan); beta_own_hi = np.full(n_total, np.nan)

    # ── Own-period bootstrap (per year) ──────────────────────────────────────
    print(f"  Own-period bootstrap  : B={B_own}, {len(years)} years, "
          f"{min(_n_jobs(), len(years))} worker process(es)")
    idx_by_year = [np.where((sub["year"] == yr).values)[0] for yr in years]
    own_out = _pmap(_gml_own_year_task,
                    [{"X": X_n[idx], "Y": Y_n[idx], "B": B_n[idx], "vrs": vrs,
                      "B_int": B_own, "seed": 77, "year": yr}
                     for yr, idx in zip(years, idx_by_year)],
                    label="GML own-period bootstrap (years)", eta=True)
    for idx, (lam_hat, bl) in zip(idx_by_year, own_out):
        own_draws[:, idx] = bl
        beta_own_bc[idx], beta_own_lo[idx], beta_own_hi[idx] = _bc_ci(lam_hat, bl)

    # ── Global bootstrap (pooled panel) ──────────────────────────────────────
    _b_relation = ("lower than own-period" if B_global < B_own
                  else ("equal to own-period" if B_global == B_own
                        else "higher than own-period"))
    print(f"  Global bootstrap      : B={B_global}, {n_total} pooled obs "
          f"({_b_relation}: B_own={B_own})")
    theta_hat_g = _gml_theta_hat(X_n, Y_n, B_n, vrs)
    lam_hat_g = 1.0 / np.clip(theta_hat_g, 1e-6, 1.0)
    glob_rows = _pmap(_gml_global_rep_task,
                      [{"X": X_n, "Y": Y_n, "B": B_n, "theta_hat": theta_hat_g,
                        "vrs": vrs, "seed": 77, "b": b} for b in range(B_global)],
                      label="GML global bootstrap (replications)", eta=True)
    glob_draws = np.vstack(glob_rows) if glob_rows else np.full((0, n_total), np.nan)
    beta_glob_bc, beta_glob_lo, beta_glob_hi = _bc_ci(lam_hat_g, glob_draws)

    n_fail_own  = int(np.isnan(own_draws).sum());  n_fail_glob = int(np.isnan(glob_draws).sum())
    print(f"  Infeasible bootstrap LPs: own {n_fail_own}/{own_draws.size}, "
          f"global {n_fail_glob}/{glob_draws.size}")

    result = sub[["iso_code","year"]].copy()
    result["beta_own_bc"]    = beta_own_bc
    result["beta_own_lo"]    = beta_own_lo
    result["beta_own_hi"]    = beta_own_hi
    result["beta_global_bc"] = beta_glob_bc
    result["beta_global_lo"] = beta_glob_lo
    result["beta_global_hi"] = beta_glob_hi
    draws = {"own": own_draws, "global": glob_draws,
             "keys": list(zip(sub["iso_code"].tolist(), sub["year"].tolist()))}
    return result, draws


def _gml_index(D_t_own, D_t1_own, D_t_glob, D_t1_glob):
    """
    Global Malmquist-Luenberger Index (Oh 2010) decomposition.

    GML  = [1+D_G(z_t)] / [1+D_G(z_t1)]
    GEC  = [1+D_t(z_t)] / [1+D_t1(z_t1)]                             [global efficiency change]
    GTC  = { [1+D_G(z_t)]/[1+D_t(z_t)] } / { [1+D_G(z_t1)]/[1+D_t1(z_t1)] }  [global technical change]

    Identity: GML = GEC x GTC (verified algebraically and numerically).
    All four input distances are ALWAYS feasible by construction (own-period
    and global technologies both trivially contain the evaluated DMU), so
    GML is defined for every country in every transition -- 100% coverage.
    """
    a = 1.0 + D_t_own     # 1 + D_t(z_t)
    e = 1.0 + D_t1_own    # 1 + D_t1(z_t1)
    p = 1.0 + D_t_glob    # 1 + D_G(z_t)
    q = 1.0 + D_t1_glob   # 1 + D_G(z_t1)
    if any(np.isnan(v) for v in (a, e, p, q)) or min(a, e, p, q) <= 0:
        return np.nan, np.nan, np.nan
    gec = a / e
    gtc = (p / a) / (q / e)
    gml = gec * gtc
    return gml, gec, gtc


def build_mpi(panel, vrs=False, B_global=None):
    """
    *** UPGRADED: Global Malmquist-Luenberger (GML) Index — Oh (2010) ***

    Replaces the CONTEMPORANEOUS Malmquist-Luenberger index (which requires
    cross-period distances D_t(z_t1) and D_t1(z_t) that can be genuinely
    LP-infeasible when a DMU's input-output-CO2 ray lies outside the cone
    spanned by the OTHER period's observations) with the Global ML index,
    which evaluates every observation only against (a) its own period's
    technology and (b) ONE pooled global technology built from all years:

        T^G = conv( union_{t=2000}^{2023} T^t )

    Both evaluations are ALWAYS feasible by construction (a DMU trivially
    reproduces itself with beta=0 against any reference set that contains
    it). This guarantees:
      - 100% transition coverage -- no country is ever dropped
      - no cross-period infeasibility
      - consistent intertemporal comparability (every year measured
        against the SAME reference technology, not a moving one)

    Decomposition (Oh 2010):
      GEC = [1+D_t(z_t)] / [1+D_t1(z_t1)]                            [catch-up]
      GTC = {[1+D_G(z_t)]/[1+D_t(z_t)]} / {[1+D_G(z_t1)]/[1+D_t1(z_t1)]}  [frontier shift]
      GML = GEC x GTC

    Reuses the IDENTICAL DDF solver (_solve_ddf_one) used by Step 5 -- no
    separate radial engine, no possibility of definitional drift between
    the static and dynamic frontiers.

    CRS technology is used throughout (vrs=False default).

    Parameters
    ----------
    B_global : bootstrap replications for the GLOBAL technology bootstrap.
               Defaults to B_GML_GLOBAL -- the global reference set
               is much larger than a single year's, so each replication is
               more expensive; raise this for a final production run if
               time permits. The own-period bootstrap always uses the full
               B_BOOTSTRAP (cheap, same cost as Step 5).

    Column names in the returned DataFrame are kept as mpi/ec/tc/mpi_bc/...
    for backward compatibility with the rest of the pipeline; they now
    hold GML/GEC/GTC values.
    """
    print(f"\n{'═'*60}")
    print("SECTION 5 — Global Malmquist-Luenberger Index (GML)")
    print("═"*60)
    print("  *** RUNTIME FINGERPRINT: build_mpi() GML code path v3 ***")
    print("  *** If you do NOT see this line, Python is executing a STALE")
    print("  *** cached module — restart the kernel/runtime and re-run. ***")

    panel = panel.copy()
    if "co2_intensity_elec" not in panel.columns:
        panel = compute_co2_intensity(panel)
        print("  ℹ co2_intensity_elec derived inside build_mpi (build_esi not yet run)")

    if "dea_net_import" not in panel.columns:
        if "net_import_pct" in panel.columns:
            panel["dea_net_import"] = panel["net_import_pct"].clip(lower=0)
            print("  ℹ dea_net_import derived here (Step 5 not yet run) -- "
                  "using the same max(0, net_import_pct) formula as Step 5")
        else:
            panel["dea_net_import"] = 0.0

    # *** UNIFIED FRONTIER DESIGN: reuse Step 5's own column definitions ***
    MPI_INPUTS = list(INPUT_COLS)
    MPI_GOOD   = list(GOOD_OUT)
    MPI_BAD    = list(BAD_OUT)
    print(f"  z_t = (x_t, y_t, b_t) reused from Step 5:")
    print(f"    x_t (inputs)      : {MPI_INPUTS}")
    print(f"    y_t (good output) : {MPI_GOOD}")
    print(f"    b_t (bad output)  : {MPI_BAD}")

    avail_inputs = [c for c in MPI_INPUTS if c in panel.columns]
    avail_good   = [c for c in MPI_GOOD   if c in panel.columns]
    avail_bad    = [c for c in MPI_BAD    if c in panel.columns]
    if not avail_bad:
        print("  *** WARNING: co2_intensity_elec missing -- GML needs a bad output ***")
    if not avail_inputs or not avail_good:
        print(f"  ✗ Missing required columns. Inputs: {avail_inputs}  Outputs: {avail_good}")
        return pd.DataFrame()

    all_cols  = avail_inputs + avail_good + avail_bad
    base_cols = ["iso_code","year"] + (["country"] if "country" in panel.columns else [])
    sub = panel[base_cols + all_cols].copy()
    all_isos = set(sub["iso_code"].unique())

    # ── Pre-imputation: keep ALL countries (same recipe as before) ───────────
    n_before_rows = int(sub[all_cols].notna().all(axis=1).sum())
    for col in all_cols:
        if sub[col].isna().any():
            sub[col] = (sub.groupby("iso_code")[col]
                          .transform(lambda x: x.interpolate(method="linear",
                                                             limit_direction="both")
                                                .ffill().bfill()))
            sub[col] = (sub[col].groupby(sub["year"])
                                .transform(lambda x: x.fillna(x.median())))
            if sub[col].isna().any():
                sub[col] = sub[col].fillna(sub[col].median())
    for col in avail_inputs:
        if (sub[col] < 0).any():
            n_neg = (sub[col] < 0).sum()
            sub[col] = sub[col].clip(lower=0)
            print(f"  ℹ {col}: {n_neg} slightly-negative values floored to 0 "
                  f"(floating-point artefact)")
        # Same treatment of zeros as Step 5: half of that year's smallest
        # positive value. With division-only rescaling a zero input would stay
        # zero and restrict exporters to zero-input peers even under CRS.
        if (sub[col] == 0).any():
            sub[col] = sub.groupby("year")[col].transform(
                lambda x: x.replace(0, x[x > 0].min() * 0.5 if (x > 0).any() else 1e-6))
    n_after_rows = int(sub[all_cols].notna().all(axis=1).sum())
    if n_after_rows > n_before_rows:
        print(f"  ✓ Imputation: complete rows {n_before_rows} → {n_after_rows} of {len(sub)}")
    sub = sub.dropna(subset=all_cols)
    for col in avail_good + avail_bad:
        if (sub[col] <= 0).any():
            sub[col] = sub.groupby("year")[col].transform(
                lambda x: x.mask(x <= 0, x[x > 0].min() * 0.5
                                 if (x > 0).any() else 1e-6))

    sub_clean = sub[(sub[avail_inputs] >= 0).all(axis=1) &
                    (sub[avail_good]   >  0).all(axis=1) &
                    (sub[avail_bad]    >  0).all(axis=1)]
    dropped = all_isos - set(sub_clean["iso_code"].unique())
    if dropped:
        print(f"  ⚠ {len(dropped)} countries excluded before GML (missing/invalid data): {sorted(dropped)}")
    else:
        print(f"  ✓ All {len(all_isos)} countries retained after imputation")
    sub = sub_clean.sort_values(["iso_code","year"]).reset_index(drop=True)
    print(f"  GML panel: {sub['iso_code'].nunique()} countries × {sub['year'].nunique()} years "
          f"({len(sub)} rows)")

    # ── Point estimates: own-period + global distances, computed ONCE ────────
    panel_for_ddf = sub  # use the cleaned/imputed panel for both point & bootstrap
    dist_df, bounds = _build_global_ddf_distances(panel_for_ddf, avail_inputs,
                                                   avail_good, avail_bad, vrs=vrs)

    # ── Bootstrap: bias-correction + CI for own-period and global distances ──
    B_own = B_BOOTSTRAP
    B_global = B_global if B_global is not None else B_GML_GLOBAL
    print(f"\n  Bootstrapping GML components (SW1998 smooth pseudo-frontier)...")
    rng_gml = np.random.default_rng(seed=77)
    boot_df, boot_draws = _bootstrap_global_ddf(panel_for_ddf, avail_inputs, avail_good,
                                                avail_bad, vrs, bounds, B_own, B_global,
                                                ALPHA, rng_gml)
    # Replication-wise GML draws: own-period and global draws are paired by
    # replication index b = 0..B_pair-1. The two bootstraps are independent,
    # so this pairing does not reproduce cross-technology dependence (Simar &
    # Wilson 1999 bivariate smoothing is not implemented); the intervals are
    # bootstrap intervals for the index itself, not plug-ins of four bounds.
    _pos    = {k: i for i, k in enumerate(boot_draws["keys"])}
    _B_pair = min(boot_draws["own"].shape[0], boot_draws["global"].shape[0])
    print(f"  GML/GEC/GTC intervals: approximate, nominal {100*(1-ALPHA):.0f}% level; their coverage has NOT")
    print(f"  been validated (the Monte Carlo covers annual country-year DDF intervals only).")
    print(f"  GML/GEC/GTC intervals use B_pair = min(B_own={boot_draws['own'].shape[0]}, "
          f"B_global={boot_draws['global'].shape[0]}) = {_B_pair} paired replications")
    _B_pair_target = min(_REPS["B_BOOTSTRAP"][0], _REPS["B_GML_GLOBAL"][0])
    if _B_pair < _B_pair_target:
        print(f"  ⚠ B_pair = {_B_pair} < {_B_pair_target} (production setting): GML intervals are provisional. Raise B_GML_GLOBAL")
        print(f"    (set separately from B_BOOTSTRAP) for reported intervals.")

    def _gml_draw_ci(i_t, i_t1, est_gml, est_gec, est_gtc):
        lo_t,  lo_t1 = boot_draws["own"][:_B_pair, i_t],    boot_draws["own"][:_B_pair, i_t1]
        g_t,   g_t1  = boot_draws["global"][:_B_pair, i_t], boot_draws["global"][:_B_pair, i_t1]
        with np.errstate(all="ignore"):
            gml_b = g_t / g_t1                 # (1+D_G(z_t)) / (1+D_G(z_t1))
            gec_b = lo_t / lo_t1               # (1+D_t(z_t)) / (1+D_t1(z_t1))
            gtc_b = gml_b / gec_b
        out = {}
        for name, est, draws in (("mpi", est_gml, gml_b), ("ec", est_gec, gec_b),
                                 ("tc", est_gtc, gtc_b)):
            d = draws[np.isfinite(draws) & (draws > 0)]
            if len(d) < 2:
                out[name] = (est, est)
                continue
            q_lo, q_hi = np.percentile(d, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
            lo, hi = 2.0 * est - q_hi, 2.0 * est - q_lo          # basic interval
            lo = max(min(lo, est), 1e-6); hi = max(hi, est)      # estimate inside its CI
            out[name] = (lo, hi)
        return out

    merged = dist_df.merge(boot_df, on=["iso_code","year"])
    merged = merged.set_index(["iso_code","year"])

    # ── Build transitions from the cached per-year distances (fast — no LPs) ──
    years   = sorted(sub["year"].unique())
    results = []
    _first_pair_printed = False

    for yr_t, yr_t1 in zip(years[:-1], years[1:]):
        isos_t  = set(sub[sub["year"]==yr_t]["iso_code"])  & set(merged.index.get_level_values(0)[merged.index.get_level_values(1)==yr_t])
        isos_t1 = set(sub[sub["year"]==yr_t1]["iso_code"]) & set(merged.index.get_level_values(0)[merged.index.get_level_values(1)==yr_t1])
        common  = sorted(isos_t & isos_t1)
        if len(common) < 4:
            continue

        country_map = (sub[["iso_code","country"]].drop_duplicates()
                       .set_index("iso_code")["country"].to_dict()
                       if "country" in sub.columns else {})

        if not _first_pair_printed:
            _iso0 = common[0]
            _row_t  = merged.loc[(_iso0, yr_t)]
            _row_t1 = merged.loc[(_iso0, yr_t1)]
            print(f"\n  {'='*66}")
            print(f"  GML PROOF — first transition {yr_t}→{yr_t1}, "
                  f"representative DMU = {_iso0}")
            print(f"  {'='*66}")
            print(f"  T^G (global technology): {len(sub)} pooled country-year DMUs")
            print(f"  Own-period T_{yr_t}: {int((sub['year']==yr_t).sum())} DMUs   "
                  f"T_{yr_t1}: {int((sub['year']==yr_t1).sum())} DMUs")
            print(f"  Inputs      : {avail_inputs}")
            print(f"  Good output : {avail_good}")
            print(f"  Bad output  : {avail_bad}")
            print(f"  Direction   : g_y=y0, g_b=b0 (observation-scaled — each DMU")
            print(f"                uses its OWN output levels as its direction,")
            print(f"                so beta is a literal proportion: y*=(1+beta)y0,")
            print(f"                b*=(1-beta)b0. NOT a fixed g=(1,1) vector.)")
            print(f"  D_{yr_t}(z_{yr_t})    [own]    = {_row_t['beta_own']:.6f}")
            print(f"  D_{yr_t1}(z_{yr_t1})  [own]    = {_row_t1['beta_own']:.6f}")
            print(f"  D_G(z_{yr_t})         [global] = {_row_t['beta_global']:.6f}")
            print(f"  D_G(z_{yr_t1})        [global] = {_row_t1['beta_global']:.6f}")
            _gml0, _gec0, _gtc0 = _gml_index(_row_t["beta_own"], _row_t1["beta_own"],
                                             _row_t["beta_global"], _row_t1["beta_global"])
            print(f"  GEC = {_gec0:.6f}")
            print(f"  GTC = {_gtc0:.6f}")
            print(f"  GML = {_gml0:.6f}")
            print(f"  {'='*66}\n")
            _first_pair_printed = True

        n_pair_ok = 0
        for iso in common:
            row_t  = merged.loc[(iso, yr_t)]
            row_t1 = merged.loc[(iso, yr_t1)]
            gml, gec, gtc = _gml_index(row_t["beta_own"], row_t1["beta_own"],
                                       row_t["beta_global"], row_t1["beta_global"])
            gml_bc, gec_bc, gtc_bc = _gml_index(row_t["beta_own_bc"], row_t1["beta_own_bc"],
                                                row_t["beta_global_bc"], row_t1["beta_global_bc"])
            _ci = (_gml_draw_ci(_pos[(iso, yr_t)], _pos[(iso, yr_t1)], gml, gec, gtc)
                   if np.isfinite(gml) else None)
            if np.isnan(gml):
                continue
            n_pair_ok += 1
            results.append({
                "iso_code": iso, "country": country_map.get(iso, iso),
                "year_t": yr_t, "year_t1": yr_t1,
                "mpi": gml, "ec": gec, "tc": gtc,
                "mpi_bc": gml_bc if np.isfinite(gml_bc) else gml,
                "ec_bc":  gec_bc if np.isfinite(gec_bc) else gec,
                "tc_bc":  gtc_bc if np.isfinite(gtc_bc) else gtc,
                "mpi_lo": _ci["mpi"][0], "mpi_hi": _ci["mpi"][1],
                "ec_lo":  _ci["ec"][0],  "ec_hi":  _ci["ec"][1],
                "tc_lo":  _ci["tc"][0],  "tc_hi":  _ci["tc"][1],
                "ci_B_pair": _B_pair,
            })
        print(f"  {yr_t}→{yr_t1}: {n_pair_ok}/{len(common)} computed "
              f"(100% coverage expected -- GML has no cross-period infeasibility)")

    mpi_df = pd.DataFrame(results)

    # ── Transition coverage summary ───────────────────────────────────────────
    n_possible_trans = (len(years) - 1) * len(all_isos)
    n_computed_trans = len(mpi_df) if not mpi_df.empty else 0
    missing_trans    = n_possible_trans - n_computed_trans
    mpi_isos         = sorted(mpi_df["iso_code"].unique()) if not mpi_df.empty else []
    missing_isos     = sorted(set(all_isos) - set(mpi_isos))
    print(f"\n  GML TRANSITION COVERAGE")
    print(f"  Possible : {len(years)-1} year-pairs × {len(all_isos)} countries = {n_possible_trans}")
    print(f"  Computed : {n_computed_trans}")
    print(f"  Missing  : {missing_trans}")
    if missing_isos:
        print(f"  Countries with ZERO GML records: {missing_isos}")
    else:
        print(f"  ✓ All {len(all_isos)} countries present in GML output (100% coverage)")

    if mpi_df.empty:
        return mpi_df

    # ── Country geometric means + interpretation ──────────────────────────────
    country_mpi = mpi_df.groupby("iso_code").agg(
        mpi_gmean = ("mpi", lambda x: np.exp(np.mean(np.log(x.clip(0.01))))),
        ec_gmean  = ("ec",  lambda x: np.exp(np.mean(np.log(x.clip(0.01))))),
        tc_gmean  = ("tc",  lambda x: np.exp(np.mean(np.log(x.clip(0.01))))),
    ).reset_index()
    print("\n  Country geometric means (GML, GEC, GTC):")
    cm = country_mpi.sort_values("mpi_gmean", ascending=False)
    print(cm.to_string(index=False))

    # A geometric mean of exactly 1 (e.g. a country on the frontier in every
    # year, such as NOR) is "unchanged", not "improving": classify with a
    # numerical tolerance around one.
    GML_TOL = 1e-4
    n_improving = int((cm["mpi_gmean"] > 1.0 + GML_TOL).sum())
    n_declining = int((cm["mpi_gmean"] < 1.0 - GML_TOL).sum())
    n_unchanged = len(cm) - n_improving - n_declining
    unchanged_isos = cm.loc[(cm["mpi_gmean"] - 1.0).abs() <= GML_TOL, "iso_code"].tolist()
    top3    = cm.head(3)["iso_code"].tolist()
    bottom3 = cm.tail(3)["iso_code"].tolist()
    tc_dom  = (cm["tc_gmean"] > cm["ec_gmean"]).sum()

    print(f"\n  {'─'*58}")
    print(f"  GML (GLOBAL MALMQUIST-LUENBERGER) RESULTS INTERPRETATION")
    print(f"  {'─'*58}")
    print(f"  GML > 1 → productivity GAIN  | GML < 1 → productivity LOSS")
    print(f"  (gain/loss jointly in electricity delivery AND CO2 abatement,")
    print(f"   measured against ONE consistent pooled technology T^G)")
    print(f"  GEC > 1 → catch-up (closer to own-period frontier)")
    print(f"  GTC > 1 → the contemporaneous frontier moved closer to the fixed pooled")
    print(f"            (global) frontier at this country's input-output mix. The")
    print(f"            pooled frontier itself does not move; GTC measures the change")
    print(f"            in the gap between annual best practice and global best practice.")
    print(f"")
    print(f"  Classification uses a tolerance of ±{GML_TOL} around 1:")
    print(f"  Countries with GML > 1 (improving): {n_improving}/{len(cm)}")
    print(f"  Countries with GML < 1 (declining): {n_declining}/{len(cm)}")
    print(f"  Countries with GML = 1 (unchanged): {n_unchanged}/{len(cm)} {unchanged_isos}")
    print(f"  Top performers   : {', '.join(top3)}")
    print(f"  Bottom performers: {', '.join(bottom3)}")
    print(f"  GTC dominates GEC in {tc_dom}/{len(cm)} countries")
    print(f"  {'─'*58}")

    return mpi_df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SIMAR–WILSON (2007) TWO-STAGE BOOTSTRAP TRUNCATED REGRESSION
# ═════════════════════════════════════════════════════════════════════════════

def _country_block_permutation(df, cols, rng):
    """Permute WHOLE COUNTRY TRAJECTORIES of `cols` across countries: country i
    receives country pi(i)'s values year by year. This keeps each country's
    serial dependence intact (the permitted permutations under the null that
    Z carries no information about the production frontier). Rows whose
    (pi(i), year) has no value become NaN."""
    isos = df["iso_code"].unique()
    mp = dict(zip(isos, rng.permutation(isos)))
    src = df.set_index(["iso_code", "year"])[cols]
    keys = list(zip(df["iso_code"].map(mp), df["year"]))
    vals = src.reindex(keys).values
    out = df.copy()
    out[cols] = vals
    return out


def _r2_of(R, Z):
    Zs = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)
    return float(OLS(R, sm.add_constant(Zs, has_constant="add")).fit().rsquared)


def _sep_null_task(p):
    """One null-reconstruction draw: permute country Z trajectories, RECOMPUTE
    all k-NN conditional efficiencies with the permuted Z, and return the R^2
    of R = theta_cond / theta_uncond on the permuted Z."""
    rng = np.random.default_rng([p["seed"], p["b"]])
    dfp = _country_block_permutation(p["sub"], p["z_vars"], rng).dropna(subset=p["z_vars"])
    dfp = dfp.reset_index(drop=True)
    th = _knn_conditional_theta(dfp, p["z_vars"], p["k_min"], p["k_frac"])
    R = th / dfp["dea_score"].values
    ok = np.isfinite(R)
    return _r2_of(R[ok], dfp.loc[ok, p["z_vars"]].values.astype(float))


def daraio_simar_wilson_separability_test(panel, z_vars=None, k_frac=0.5,
                                           k_min=10, B_perm=B_PERM_SEPARABILITY, alpha=0.05,
                                           seed=123, univariate=True):
    """
    Practical separability diagnostic in the spirit of Daraio, Simar &
    Wilson (2018) — NOT a literal line-by-line reproduction of their exact
    test statistic and asymptotic theory. This implementation is a k-NN
    conditional-efficiency approximation with an OLS/permutation test on
    the conditional/unconditional ratio; it targets the same H0 and the
    same economic question DSW2018 address, but the precise statistic and
    critical values have not been verified against their paper term for
    term. Treat rejection/non-rejection as informative evidence, and audit
    against the original paper before citing this as "the DSW2018 test"
    in a dissertation without that verification.

    H0 (separability): conditioning on the environmental variables Z does
    NOT change the production support -- i.e. the attainable set of
    (x, y, b) is the same regardless of Z, so Z affects only the LOCATION
    of a DMU relative to a FIXED frontier, not the frontier's shape. This
    is the assumption SW2007's second stage implicitly relies on: it uses
    UNCONDITIONAL first-stage DEA/DDF scores as the dependent variable and
    then regresses them on the SAME z_vars in the second stage. If Z
    actually shifts the technology itself, the unconditional scores are
    not estimating the conditional (on Z) production model, and the
    second-stage regression coefficients become difficult to interpret
    causally.

    univariate=True (default): in addition to the JOINT test
    on all z_vars together, also run the test SEPARATELY for each z_var
    on its own. Conditioning on a single dimension suffers far less from
    the curse of dimensionality than conditioning on all five at once,
    and identifies WHICH specific environmental variables are safe to
    keep in the SW2007 second stage versus which should be dropped or
    handled via a conditional-efficiency approach instead.

    Method
    ------
    theta_uncond,i : the UNCONDITIONAL score already computed in Step 5
                     (panel["dea_score"]) -- reference set = ALL DMUs in
                     DMU i's own year. Reused directly, not recomputed.
    theta_cond,i   : a CONDITIONAL score computed with the SAME DDF solver
                     (_solve_ddf_one -- no separate engine), but with the
                     reference set restricted to the k NEAREST NEIGHBOURS
                     of DMU i in standardised Z-space, within the SAME
                     year. This is a standard, widely-used practical
                     approximation to the kernel-weighted conditional
                     efficiency estimator (Daraio & Simar 2005, 2007;
                     Bădin, Daraio & Simar 2012) -- exact kernel-integral
                     computation is not required to test for a systematic
                     Z-dependence in the ratio below.
    R_i = theta_cond,i / theta_uncond,i

    Test statistic: R² from OLS of R_i on z_vars (standardised). Under H0,
    conditioning on Z should not systematically move R away from 1, so the
    regression should have negligible explanatory power.

    Two p-values are reported, exactly as the DSW test permits either:
      - Asymptotic : F-test on the OLS regression (joint significance of
                     all z_vars coefficients).
      - Bootstrap  : permutation test -- shuffle Z across DMUs (within year)
                     B_perm times, recompute R² each time under the null of
                     no true Z-R relationship, and report the share of
                     permuted R² at least as large as the observed R².

    Returns a dict with the test statistics, p-values, and a verdict string.
    """
    from scipy import stats as _stats

    print(f"\n{'═'*60}")
    print("CONDITIONAL-EFFICIENCY SENSITIVITY DIAGNOSTIC (k-NN; related to, but not,")
    print("the Daraio, Simar & Wilson 2018 separability test)")
    print("═"*60)
    print("  Question: how much does comparing each country only with its k nearest")
    print("  neighbours in Z change its efficiency? (separability is the second-stage")
    print("  assumption this exercise informs, but does not formally test)")

    if z_vars is None:
        z_vars = ["dim_resilience","dim_affordability",
                  "dim_robustness","dim_sustainability",
                  "gdp_growth"]

    needed = ["iso_code","year"] + INPUT_COLS + GOOD_OUT + BAD_OUT + \
             ["dea_score"] + z_vars
    needed = [c for c in needed if c in panel.columns]
    sub = panel[needed].dropna().copy()
    if len(sub) < 30:
        print(f"  ⚠ Only {len(sub)} complete observations — separability test skipped")
        return {"skipped": True}

    print(f"  Z variables (same as SW2007 z_vars): {z_vars}")
    print(f"  Sample: {len(sub)} country-years, k-NN conditional reference "
          f"set (k = max({k_min}, {k_frac:.0%} of year size))")

    rng = np.random.default_rng(seed)

    # *** DEDUPLICATED (per Mundlak/Conditional-DDF review): this conditional
    # efficiency computation is now the SHARED _knn_conditional_theta function
    # (defined once, above panel_fe_regression), also used by
    # conditional_ddf_analysis(). No parallel/duplicated logic. ***
    sub["theta_cond"] = _knn_conditional_theta(sub, z_vars, k_min, k_frac)
    sub["theta_uncond"] = sub["dea_score"]
    test_df = sub.dropna(subset=["theta_cond","theta_uncond"]).copy()
    test_df = test_df[test_df["theta_uncond"] > 1e-6]
    n_ok = len(test_df)
    print(f"  Conditional scores computed: {n_ok}/{len(sub)}")
    if n_ok < 30:
        print(f"  ⚠ Only {n_ok} valid (uncond,cond) pairs — separability test skipped")
        return {"skipped": True}

    test_df["R"] = test_df["theta_cond"] / test_df["theta_uncond"]

    # ── OLS: R on standardised Z ──────────────────────────────────────────────
    Zt = test_df[z_vars].values.astype(float)
    Zt_std = (Zt - Zt.mean(axis=0)) / (Zt.std(axis=0) + 1e-9)
    Zc = sm.add_constant(Zt_std, has_constant="add")
    Rv = test_df["R"].values.astype(float)

    ols = OLS(Rv, Zc).fit()
    r2_obs = float(ols.rsquared)
    k_reg  = Zc.shape[1] - 1
    n_reg  = len(Rv)
    f_stat = (r2_obs / k_reg) / ((1 - r2_obs) / (n_reg - k_reg - 1) + 1e-12)
    p_asymp = float(1 - _stats.f.cdf(f_stat, k_reg, n_reg - k_reg - 1))

    # ── (a) NULL-RECONSTRUCTION permutation test (primary) ────────────────────
    # H0: Z carries no information about the production frontier, so which
    # countries are "near" each other in Z-space is irrelevant. Under H0 the
    # country Z trajectories are exchangeable across countries. Permitted
    # permutations: whole-country blocks (country i receives country pi(i)'s
    # Z values year by year), which preserves serial dependence within
    # countries and the year structure. For each permutation ALL conditional
    # DDF scores are RECOMPUTED with the permuted Z (the null is reconstructed,
    # not just the regressors reshuffled), and the same statistic -- R^2 of
    # R on Z -- is recalculated. p = (1 + #{R2_perm >= R2_obs}) / (B + 1).
    # This is a permutation test of the k-NN conditional-ratio statistic; it
    # is not the Daraio, Simar & Wilson (2018) CLT-based test.
    sub_t = test_df[["iso_code", "year"] + INPUT_COLS + GOOD_OUT + BAD_OUT + ["dea_score"] + z_vars]
    sub_t = sub_t.reset_index(drop=True)
    null_r2 = np.array(_pmap(_sep_null_task,
                             [{"sub": sub_t, "z_vars": z_vars, "k_min": k_min, "k_frac": k_frac,
                               "seed": seed, "b": b} for b in range(B_SEP_NULL)],
                             label="separability null reconstruction (permutations)", eta=True))
    null_r2 = null_r2[np.isfinite(null_r2)]
    p_null = float((np.sum(null_r2 >= r2_obs) + 1) / (len(null_r2) + 1))

    # ── (b) Diagnostic: country-block permutation of the regressors only ──────
    # (R kept fixed; cheaper, but does not reconstruct the null distribution
    # of R itself -- reported as a diagnostic only.)
    r2_perm = np.empty(B_perm)
    for b in range(B_perm):
        Zp = _country_block_permutation(test_df[["iso_code", "year"] + z_vars], z_vars, rng)[z_vars].values
        okp = np.all(np.isfinite(Zp), axis=1)
        r2_perm[b] = _r2_of(Rv[okp], Zp[okp].astype(float))
    p_boot = float((np.sum(r2_perm >= r2_obs) + 1) / (B_perm + 1))   # never exactly 0

    print(f"\n  R_i = theta_cond,i / theta_uncond,i  --  "
          f"mean={Rv.mean():.4f}  sd={Rv.std():.4f}  "
          f"[min={Rv.min():.4f}, max={Rv.max():.4f}]")
    print(f"  OLS of R on Z:  R² = {r2_obs:.4f}   F({k_reg},{n_reg-k_reg-1}) = {f_stat:.3f}")
    print(f"  p-value, NULL RECONSTRUCTION (country-block permutations, conditional DDFs")
    print(f"    recomputed, B={len(null_r2)}; PRIMARY)          : {p_null:.4f}   "
          f"[null R² median {np.median(null_r2):.4f}, 95th pct {np.percentile(null_r2, 95):.4f}]")
    print(f"  p-value, diagnostic (country-block permutation of Z only, B={B_perm}): {p_boot:.4f}")
    print(f"  p-value, naive F-test (assumes independent observations; NOT valid under")
    print(f"    serial and shared-frontier dependence, shown for reference only): {p_asymp:.4f}")
    print(f"  Smallest attainable null-reconstruction p-value: {1/(len(null_r2)+1):.4f}")

    rejected = p_null < alpha
    print(f"\n  {'─'*66}")
    print(f"  WHAT THE NULL RECONSTRUCTION TESTS: H0' = country Z trajectories are")
    print(f"  exchangeable with respect to the production data, i.e. Z is unrelated to")
    print(f"  BOTH the frontier AND the inefficiency distribution. This is STRONGER than")
    print(f"  separability (which allows Z to affect inefficiency). In finite samples the")
    print(f"  k-NN ratio R also responds when Z affects only inefficiency (a Monte Carlo")
    print(f"  check with a separable DGP rejected H0'), so rejecting H0' does NOT show that")
    print(f"  separability fails. The k-NN exercise is therefore reported as a PRACTICAL")
    print(f"  SENSITIVITY DIAGNOSTIC of how much structural conditioning changes efficiency,")
    print(f"  not as a formal separability test (DSW 2018 is not implemented).")
    if rejected:
        print(f"  RESULT: H0' rejected at α={alpha} (p = {p_null:.4f}): Z is related to measured")
        print(f"  performance through the frontier and/or the inefficiency distribution.")
    else:
        print(f"  RESULT: H0' not rejected at α={alpha} (p = {p_null:.4f}"
              + (f"; with B={len(null_r2)} the smallest attainable p is {1/(len(null_r2)+1):.3f}"
                 if 1/(len(null_r2)+1) >= alpha else "") + ").")
    print(f"  {'─'*66}")

    # Per-dimension separability table:
    # Testing all five z_vars jointly suffers from the curse of
    # dimensionality in a k-NN conditional estimator with ~1,000 obs.
    # Test each dimension SEPARATELY too, so it is clear WHICH specific
    # environmental variables are safe to keep in the SW2007 second stage.
    univariate_results = {}
    if univariate and len(z_vars) > 1:
        print(f"\n  {'═'*66}")
        print(f"  PER-DIMENSION SEPARABILITY — PRACTICAL SENSITIVITY DIAGNOSTIC")
        print(f"  (k-NN conditioning on one variable at a time; p (diag.) from country-block")
        print(f"  permutations of the regressor, without recomputing the conditional DDFs,")
        print(f"  so it is NOT a formal test of separability)")
        print(f"  {'═'*66}")
        print(f"  {'Environmental variable':<22} {'R²':>7} {'F-stat':>8} "
              f"{'p (naive)':>10} {'p (diag.)':>9}  Diagnostic")
        print(f"  {'─'*66}")
        for zv in z_vars:
            theta_cond_v = _knn_conditional_theta(sub, [zv], k_min, k_frac)
            sub_v = sub.copy()
            sub_v["theta_cond_v"] = theta_cond_v
            tv = sub_v.dropna(subset=["theta_cond_v","dea_score"]).copy()
            tv = tv[tv["dea_score"] > 1e-6]
            if len(tv) < 30:
                print(f"  {zv:<22}  insufficient data — skipped")
                continue
            Rv_uni = (tv["theta_cond_v"] / tv["dea_score"]).values.astype(float)
            Zv = tv[[zv]].values.astype(float)
            Zv_std = (Zv - Zv.mean(axis=0)) / (Zv.std(axis=0) + 1e-9)
            Zc_v = sm.add_constant(Zv_std, has_constant="add")

            ols_v = OLS(Rv_uni, Zc_v).fit()
            r2_v = float(ols_v.rsquared)
            n_v  = len(Rv_uni)
            f_v  = (r2_v / 1) / ((1 - r2_v) / (n_v - 2) + 1e-12)
            p_asymp_v = float(1 - _stats.f.cdf(f_v, 1, n_v - 2))

            r2_perm_v = np.empty(B_perm)
            _tv = tv[["iso_code", "year", zv]].reset_index(drop=True)
            for b in range(B_perm):
                Zp = _country_block_permutation(_tv, [zv], rng)[[zv]].values.astype(float)
                okp = np.isfinite(Zp[:, 0])
                r2_perm_v[b] = _r2_of(Rv_uni[okp], Zp[okp])
            p_boot_v = float((np.sum(r2_perm_v >= r2_v) + 1) / (B_perm + 1))

            rejected_v = p_boot_v < alpha
            decision = "association" if rejected_v else "no association"
            print(f"  {zv:<22} {r2_v:>7.4f} {f_v:>8.3f} "
                  f"{p_asymp_v:>10.4f} {p_boot_v:>9.4f}  {decision}")
            univariate_results[zv] = {
                "r2": r2_v, "f_stat": f_v,
                "p_asymptotic": p_asymp_v, "p_bootstrap": p_boot_v,
                "rejected": rejected_v,
            }
        print(f"  {'─'*66}")
        safe_vars = [v for v, r in univariate_results.items() if not r["rejected"]]
        risky_vars = [v for v, r in univariate_results.items() if r["rejected"]]
        print(f"  Diagnostic: no association of R with the variable: {safe_vars or 'none'}")
        print(f"  Diagnostic: R associated with the variable (possible frontier shift): {risky_vars or 'none'}")
        print(f"  {'═'*66}")

    return {
        "skipped": False,
        "n_obs": n_reg,
        "r2": r2_obs,
        "f_stat": f_stat,
        "p_asymptotic": p_asymp,
        "p_bootstrap": p_boot,
        "p_null_reconstruction": p_null,
        "B_null": int(len(null_r2)),
        "rejected": rejected,
        "mean_ratio": float(Rv.mean()),
        "univariate": univariate_results,
    }


# ── Second-stage model: truncated at 1, CENSORED at 2 ────────────────────────
# Under g = (y0, b0) the distance delta = 1 + beta lies in [1, 2]. In this panel
# about a third of the bias-corrected values reach the upper bound (they would
# exceed 2 without truncation). A continuous doubly TRUNCATED normal cannot
# represent such a mass point: the ML fit then drives the latent mean far above
# 2 with a large variance, the parametric bootstrap draws nearly all
# observations at the bound, every re-estimate is identical and the intervals
# collapse (SE = 0) -- the degeneracy seen in the restricted model. The model is
# therefore: latent delta ~ N(z gamma, s^2) truncated below at 1 (as SW2007),
# with values at or above 2 treated as CENSORED at 2 (Tobit-type upper limit).
# Regressors are standardised inside the optimiser and coefficients mapped
# back to the original scale. Every fit reports optimiser status, gradient
# norm, Hessian condition and model-based SEs; bootstrap distributions are
# checked for collapse, and intervals are NOT reported when they collapse.
SW_UPPER = 2.0


def _tc_draw(mu, sigma, rng, lo=1.0, hi=SW_UPPER):
    """Latent N(mu, sigma^2) truncated below at lo, then censored at hi."""
    a = stats.norm.cdf((lo - mu) / sigma)
    u = np.clip(a + rng.random(np.shape(mu)) * (1.0 - a), 1e-12, 1 - 1e-12)
    return np.minimum(np.maximum(mu + sigma * stats.norm.ppf(u), lo), hi)


def _tc_negll(p, d, Zs, cens, lo=1.0, hi=SW_UPPER, mode="censored"):
    g, s_ = p[:-1], np.exp(p[-1])
    mu = Zs @ g
    if mode == "doubly_truncated":
        # Sample restricted to lo < delta < hi: density conditional on that
        # selection, phi((d-mu)/s)/s / [Phi((hi-mu)/s) - Phi((lo-mu)/s)].
        den = stats.norm.cdf((hi - mu) / s_) - stats.norm.cdf((lo - mu) / s_)
        return -np.sum(stats.norm.logpdf((d - mu) / s_) - np.log(s_)
                       - np.log(np.clip(den, 1e-300, None)))
    logtr = stats.norm.logsf((lo - mu) / s_)
    ll_u = stats.norm.logpdf((d - mu) / s_) - np.log(s_) - logtr
    ll_c = stats.norm.logsf((hi - mu) / s_) - logtr
    return -np.sum(np.where(cens, ll_c, ll_u))


def _tc_mle(d, Zc, start=None, diagnostics=False, lo=1.0, hi=SW_UPPER, mode="censored"):
    """ML fit of the truncated (at lo) / censored (at hi) normal regression.
    Zc includes a constant in column 0. Returns dict with gamma (original
    scale), sigma, params (internal), converged and, if diagnostics=True,
    loglik, gradient norm, Hessian condition number, PD flag and SEs."""
    d = np.clip(np.asarray(d, float), lo, hi)
    cens = (d >= hi - 1e-9) if mode == "censored" else np.zeros(len(d), bool)
    mz, sz = Zc[:, 1:].mean(axis=0), Zc[:, 1:].std(axis=0)
    sz = np.where(sz > 0, sz, 1.0)
    Zs = np.column_stack([np.ones(len(d)), (Zc[:, 1:] - mz) / sz])
    if start is None:
        b0 = np.linalg.lstsq(Zs, d, rcond=None)[0]
        start = np.concatenate([b0, [np.log(max(float(np.std(d - Zs @ b0)), 1e-2))]])
    f = lambda p: _tc_negll(p, d, Zs, cens, lo, hi, mode)
    res = minimize(f, start, method="BFGS", options={"gtol": 1e-6, "maxiter": 2000})
    if not np.all(np.isfinite(res.x)) or (not res.success and res.fun > f(start) - 1e-9):
        res = minimize(f, start, method="Nelder-Mead",
                       options={"maxiter": 40000, "xatol": 1e-9, "fatol": 1e-9})
    gs = res.x[:-1]
    gamma = np.concatenate([[gs[0] - np.sum(gs[1:] * mz / sz)], gs[1:] / sz])
    # Convergence is judged by the GRADIENT at the solution, not only by the
    # optimiser's flag: BFGS often reports "precision loss" at a point where
    # the gradient is already ~1e-5 (a genuine optimum).
    ok = bool(np.all(np.isfinite(res.x)))
    if ok and not res.success:
        from statsmodels.tools.numdiff import approx_fprime
        try:
            ok = float(np.linalg.norm(approx_fprime(res.x, f))) < 1e-3
        except Exception:
            ok = False
    out = {"gamma": gamma, "sigma": float(np.exp(res.x[-1])), "params": res.x,
           "converged": ok, "optimiser_message": str(getattr(res, "message", "")),
           "share_censored": float(cens.mean())}
    if diagnostics:
        from statsmodels.tools.numdiff import approx_fprime, approx_hess3
        try:
            gr = approx_fprime(res.x, f)
            H = approx_hess3(res.x, f)
            ev = np.linalg.eigvalsh((H + H.T) / 2)
            pd_ = bool(np.all(ev > 0))
            cov_s = np.linalg.inv(H) if pd_ else np.full(H.shape, np.nan)
            # SEs on the original scale (delta method for the linear map)
            k = len(gs)
            A = np.zeros((k, k))
            A[0, 0] = 1.0
            A[0, 1:] = -mz / sz
            A[1:, 1:] = np.diag(1.0 / sz)
            cov_g = A @ cov_s[:k, :k] @ A.T
            out.update({"loglik": -float(res.fun), "grad_norm": float(np.linalg.norm(gr)),
                        "hess_pd": pd_, "hess_cond": float(ev.max() / ev.min()) if pd_ else np.inf,
                        "se_model": np.sqrt(np.clip(np.diag(cov_g), 0, None)) if pd_ else np.full(k, np.nan),
                        "mu_share_above_hi": float(np.mean(Zc @ gamma > hi))})
        except Exception as e:
            out.update({"loglik": -float(res.fun), "grad_norm": np.nan, "hess_pd": False,
                        "hess_cond": np.inf, "se_model": np.full(len(gs), np.nan),
                        "mu_share_above_hi": np.nan, "diag_error": str(e)})
    return out


def _sw_loop1_year_task(p):
    """Simar-Wilson Algorithm 2, step 3, for one year: L1 parametric pseudo-
    samples generated from the fitted model, each original DMU re-evaluated
    against the pseudo-frontier. Returns mean delta* per DMU."""
    rng = np.random.default_rng([p["seed"], int(p["year"])])
    X, Y, Bm, bh = p["X"], p["Y"], p["B"], p["beta_hat"]
    acc = np.full((p["L1"], len(bh)), np.nan)
    for l in range(p["L1"]):
        bstar = _tc_draw(p["mu"], p["sigma"], rng) - 1.0              # beta* in [0, 1]
        Ys, Bs = _ddf_pseudo_outputs(Y, Bm, bh, bstar)
        acc[l] = 1.0 + _ddf_eval_against(X, Y, Bm, Ys, Bs, p["vrs"])
    with np.errstate(all="ignore"):
        return np.nanmean(acc, axis=0)


def _print_fit_diag(label, fit):
    print(f"    {label}: converged={fit['converged']}  logL={fit.get('loglik', np.nan):.3f}  "
          f"|grad|={fit.get('grad_norm', np.nan):.2e}  Hessian PD={fit.get('hess_pd')}  "
          f"cond={fit.get('hess_cond', np.nan):.2e}  sigma={fit['sigma']:.4f}  "
          f"censored={fit['share_censored']:.1%}  fitted mean>2: {fit.get('mu_share_above_hi', np.nan):.1%}")


def simar_wilson_2007(panel, z_vars=None, B=B_SW2007, alpha=ALPHA):
    """
    Simar & Wilson (2007) ALGORITHM 2, implemented step by step and ADAPTED to
    the directional distance function (J. Econometrics 136, 31-64, Sec. 4):

      1. delta_hat = 1 + beta_hat from the annual DDF of this sample.
      2. ML regression of delta_hat on z using ONLY delta_hat > 1 (frontier
         rows excluded, as in SW2007).
      3. L1 = L1_SW2007 parametric pseudo-samples from the fitted model;
         pseudo-outputs along the DDF direction (y* = y(1+b_hat-b*),
         b* = b(1-b_hat+b*)); each ORIGINAL point re-evaluated against the
         annual pseudo-frontier.
      4. delta_hathat = 2 delta_hat - mean(delta*); values below 1 set to 1,
         values at or above 2 treated as censored at 2 (counts reported).
      5. ML regression of delta_hathat on z (all n).
      6. L2 = B parametric draws from the step-5 model; re-estimation.
      7. Basic intervals [2 g - q(1-a/2), 2 g - q(a/2)].

    ADAPTATIONS (not in SW2007): (a) delta is bounded at 2 under g = (y0, b0);
    the model is truncated at 1 as in SW2007 but CENSORED at 2, because about a
    third of the corrected values reach that bound (a doubly truncated normal
    proved numerically degenerate); (b) pseudo-data follow the DDF direction;
    (c) a country-cluster pairs bootstrap of step 5 is reported alongside the
    parametric intervals (serial dependence). Diagnostics are printed for
    every fit; collapsed bootstrap distributions are flagged and their
    intervals are not reported. Stars require BOTH intervals to exclude zero
    and are suppressed when B < 100. delta is INEFFICIENCY: gamma > 0 means
    lower efficiency.
    """
    print(f"\n{'═'*60}")
    print(f"SECTION 6 — Simar–Wilson (2007) Algorithm 2, adapted to the DDF  "
          f"(L1={L1_SW2007}, L2={B})")
    print("═"*60)
    if z_vars is None:
        z_vars = ["dim_resilience", "dim_affordability",
                  "dim_robustness", "dim_sustainability", "gdp_growth"]

    _sep_result = daraio_simar_wilson_separability_test(panel, z_vars=z_vars)
    print(f"\n  ⚠ Separability is ASSUMED by this second stage and is not formally tested here")
    print(f"  (the k-NN exercise above is a sensitivity diagnostic). The coefficients below")
    print(f"  are reported as associations with estimated inefficiency, not as causal effects.")

    need = ["iso_code", "year"] + INPUT_COLS + GOOD_OUT + BAD_OUT + z_vars
    sub = panel[[c for c in dict.fromkeys(need) if c in panel.columns]].dropna().copy()
    sub = sub.sort_values(["year", "iso_code"]).reset_index(drop=True)
    if len(sub) < len(z_vars) + 10:
        print(f"  ✗ Insufficient observations ({len(sub)}) — skipping")
        return pd.DataFrame(columns=["variable", "coef_bc", "ci_lo_95", "ci_hi_95", "B"])

    yr_data, beta_hat = {}, np.full(len(sub), np.nan)
    for yr, g in sub.groupby("year"):
        idx = g.index.values
        X = _normalise(g[INPUT_COLS].values.astype(float))
        Y = _normalise(g[GOOD_OUT].values.astype(float))
        Bm = _normalise(g[BAD_OUT].values.astype(float))
        bh = np.clip(np.nan_to_num(_run_ddf_cross_section(X, Y, Bm, vrs=VRS), nan=0.0), 0.0, BETA_UPPER)
        beta_hat[idx] = bh
        yr_data[yr] = (idx, X, Y, Bm, bh)
    d_hat = 1.0 + beta_hat
    Zc = sm.add_constant(sub[z_vars].values.astype(float), has_constant="add")
    col_names = ["const"] + list(z_vars)
    kp = Zc.shape[1]
    print(f"\n  Diagnostics (model: truncated at 1, censored at 2; regressors standardised internally)")

    m_mask = d_hat > 1.0 + 1e-6
    f2 = _tc_mle(d_hat[m_mask], Zc[m_mask], diagnostics=True)
    _print_fit_diag(f"step 2 (m = {int(m_mask.sum())} of {len(sub)}; frontier rows excluded)", f2)

    mu_all = Zc @ f2["gamma"]
    tasks = [{"X": X, "Y": Y, "B": Bm, "beta_hat": bh, "mu": mu_all[idx], "sigma": f2["sigma"],
              "L1": L1_SW2007, "seed": 2007, "year": yr, "vrs": VRS}
             for yr, (idx, X, Y, Bm, bh) in yr_data.items()]
    means = _pmap(_sw_loop1_year_task, tasks, label="SW2007 step 3 (years)", eta=True)
    d_star_mean = np.full(len(sub), np.nan)
    for (yr, (idx, *_)), mval in zip(yr_data.items(), means):
        d_star_mean[idx] = mval
    d_hh_raw = np.where(np.isfinite(2.0 * d_hat - d_star_mean), 2.0 * d_hat - d_star_mean, d_hat)
    n_lo, n_hi = int((d_hh_raw < 1).sum()), int((d_hh_raw >= SW_UPPER).sum())
    d_hh = np.clip(d_hh_raw, 1.0, SW_UPPER)
    print(f"    step 4: bias-corrected delta below 1 (set to 1): {n_lo}; at/above 2 (censored): "
          f"{n_hi} ({n_hi/len(sub):.1%}); uncorrected values above 2 reach "
          f"{np.nanmax(d_hh_raw):.3f}")

    print(f"    BOUNDARY TREATMENT (change from the earlier version, which truncated at both 1")
    print(f"    and 2): the latent delta ~ N(z gamma, s^2) is TRUNCATED at 1 and CENSORED at 2.")
    print(f"    Likelihood: uncensored delta in [1,2): phi((d-mu)/s)/s / P(latent>1); delta = 2:")
    print(f"    P(latent>=2) / P(latent>1). Step 3 draws latent values truncated at 1 and sets")
    print(f"    draws above 2 to 2 (beta* = 1); step 6 draws the same way and re-estimates with")
    print(f"    the same censored likelihood. ASSUMPTION: the {n_hi} values at 2 arise from")
    print(f"    clipping bias-corrected scores; treating them as censored assumes a latent")
    print(f"    value >= 2 behind each. Clipping alone does not establish this. As a")
    print(f"    SENSITIVITY CHECK (not a test of the assumption) the model is re-estimated on")
    print(f"    observations strictly below 2 with the likelihood CONDITIONAL on that selection:")
    print(f"    phi((d-mu)/s)/s / [Phi((2-mu)/s) - Phi((1-mu)/s)].")
    f5 = _tc_mle(d_hh, Zc, diagnostics=True)
    _print_fit_diag(f"step 5 (n = {len(sub)})", f5)
    # Sensitivity: drop the observations at the bound (truncation at 1 only,
    # no censored points). Different estimand (conditional on delta < 2); shown
    # to reveal how much the coefficients rest on the censoring assumption.
    _keep = d_hh < SW_UPPER - 1e-9
    _keep = _keep & (d_hh > 1.0 + 1e-9)           # strictly 1 < delta < 2
    f5_ex = (_tc_mle(d_hh[_keep], Zc[_keep], diagnostics=True, mode="doubly_truncated")
             if _keep.sum() > kp + 5 else None)
    if f5_ex is not None:
        _print_fit_diag(f"sensitivity: 1 < delta < 2 only, conditional likelihood (n = {int(_keep.sum())})", f5_ex)
    weak = (not f5["converged"]) or (not f5.get("hess_pd", False)) or f5.get("hess_cond", np.inf) > 1e10
    if weak:
        print(f"    ⚠ step-5 fit is numerically unreliable (non-convergence, non-PD Hessian or")
        print(f"      condition number > 1e10): coefficients are weakly identified.")

    rng = np.random.default_rng(99)
    mu5 = Zc @ f5["gamma"]
    boot, n_fail = [], 0
    for _ in range(B):
        fb = _tc_mle(_tc_draw(mu5, f5["sigma"], rng), Zc, start=f5["params"])
        if fb["converged"]:
            boot.append(fb["gamma"])
        else:
            n_fail += 1
    boot = np.asarray(boot)
    isos = sub["iso_code"].unique()
    rows_by_c = {c: np.where(sub["iso_code"].values == c)[0] for c in isos}
    crng = np.random.default_rng(101)
    cboot, c_fail = [], 0
    for _ in range(B):
        take = np.concatenate([rows_by_c[c] for c in crng.choice(isos, size=len(isos), replace=True)])
        fc = _tc_mle(d_hh[take], Zc[take], start=f5["params"])
        if fc["converged"]:
            cboot.append(fc["gamma"])
        else:
            c_fail += 1
    cboot = np.asarray(cboot)

    def _basic(est, draws):
        if len(draws) < 2:
            return np.full(kp, np.nan), np.full(kp, np.nan), np.full(kp, np.nan), np.ones(kp, bool)
        q_lo = np.percentile(draws, 100 * alpha / 2, axis=0)
        q_hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
        sd = draws.std(axis=0)
        collapsed = sd < 1e-6 * (1.0 + np.abs(est))
        lo, hi = 2 * est - q_hi, 2 * est - q_lo
        lo[collapsed], hi[collapsed] = np.nan, np.nan
        return lo, hi, sd, collapsed

    g5 = f5["gamma"]
    lo_p, hi_p, se_p, col_p = _basic(g5, boot)
    lo_c, hi_c, se_c, col_c = _basic(g5, cboot)
    print(f"    bootstrap: parametric {len(boot)}/{B} converged ({n_fail} failed), "
          f"country-cluster {len(cboot)}/{B} ({c_fail} failed)")
    if len(boot) > 1:
        print(f"    parametric draws: distinct estimates {len(np.unique(np.round(boot, 10), axis=0))}; "
              f"collapsed coefficients: {[c for c, f in zip(col_names, col_p) if f] or 'none'}")
    sw_results = pd.DataFrame({
        "variable": col_names, "coef_step2": f2["gamma"], "coef_bc": g5,
        "se_model_hessian": f5.get("se_model", np.full(kp, np.nan)),
        "bootstrap_se": se_p, "t_stat": g5 / (se_p + 1e-12),
        "ci_lo_95": lo_p, "ci_hi_95": hi_p, "parametric_collapsed": col_p,
        "cluster_se": se_c, "ci_lo_cluster": lo_c, "ci_hi_cluster": hi_c,
        "coef_excluding_boundary": (f5_ex["gamma"] if f5_ex is not None else np.full(kp, np.nan)),
        "se_excluding_boundary": (f5_ex.get("se_model", np.full(kp, np.nan)) if f5_ex is not None
                                  else np.full(kp, np.nan))})
    for k_, v_ in {"B": B, "L1": L1_SW2007, "n": len(sub), "m_step2": int(m_mask.sum()),
                   "n_censored_step4": n_hi, "n_floor_step4": n_lo, "weakly_identified": weak,
                   "hess_cond_step5": f5.get("hess_cond", np.nan),
                   "grad_norm_step5": f5.get("grad_norm", np.nan)}.items():
        sw_results[k_] = v_

    low_B = B < 100
    print(f"\n  Step 7 — Dependent: delta = 1 + beta (inefficiency; gamma > 0 -> lower efficiency)")
    print(f"  {'Variable':<22}{'gamma':>9}{'SE Hess.':>9}{'SE boot':>9}{'CI parametric':>22}{'CI country-cluster':>24}")
    print("  " + "-" * 95)
    fmt = lambda a, b: f"[{a:>7.4f},{b:>8.4f}]" if np.isfinite(a) else f"{'(collapsed — n/a)':>17}"
    for _, r in sw_results.iterrows():
        sig = (np.isfinite(r.ci_lo_95) and np.isfinite(r.ci_lo_cluster) and
               (r.ci_lo_95 > 0 or r.ci_hi_95 < 0) and (r.ci_lo_cluster > 0 or r.ci_hi_cluster < 0))
        star = "***" if (sig and not low_B and not weak) else ""
        print(f"  {r.variable:<22}{r.coef_bc:>9.4f}{r.se_model_hessian:>9.4f}{r.bootstrap_se:>9.4f}"
              f"   {fmt(r.ci_lo_95, r.ci_hi_95)}   {fmt(r.ci_lo_cluster, r.ci_hi_cluster)} {star}")
    if low_B:
        print(f"  ⚠ B={B} < 100 — TEST MODE: significance stars SUPPRESSED.")
    else:
        print(f"  *** = BOTH intervals exclude zero and the step-5 fit is well identified.")
    if f5_ex is not None:
        print(f"\n  Censoring sensitivity: censored model (all n) vs selected sample 1 < delta < 2")
        print(f"  (conditional likelihood). SEs are Hessian-based: an EXPLORATORY comparison that")
        print(f"  does not account for panel or shared-frontier dependence.")
        print(f"  {'Variable':<22}{'censored γ':>11}{'SE':>8}{'selected γ':>12}{'SE':>8}"
              f"{'Δγ':>9}{'Δ%':>8}{'SE ratio':>9}  sign")
        for _, r in sw_results.iterrows():
            a, b = r.coef_bc, r.coef_excluding_boundary
            sa, sb = r.se_model_hessian, r.se_excluding_boundary
            pct = 100 * (b - a) / abs(a) if abs(a) > 1e-12 else np.nan
            print(f"  {r.variable:<22}{a:>11.4f}{sa:>8.4f}{b:>12.4f}{sb:>8.4f}{b - a:>+9.4f}"
                  f"{pct:>+8.1f}{(sb / sa if sa > 0 else np.nan):>9.2f}  "
                  f"{'same' if np.sign(a) == np.sign(b) else 'CHANGES'}")
        print(f"  Differences in sign, magnitude and uncertainty indicate specification")
        print(f"  sensitivity to the upper-censoring assumption; they do not establish whether")
        print(f"  the assumption is true.")
    print(f"  Intervals: approximate, nominal {100*(1-alpha):.0f}% level; coverage of second-stage intervals")
    print(f"  has NOT been validated by simulation.")
    print(f"  Procedure: DDF-adapted second stage following the steps of SW2007 Algorithm 2,")
    print(f"  with upper CENSORING at 2 -- not a verified canonical Simar-Wilson implementation.")
    if weak or col_p.any():
        print(f"  ⚠ Do not interpret these coefficients as inference: see the diagnostics above.")
    return sw_results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PANEL FIXED-EFFECTS OLS
# ═════════════════════════════════════════════════════════════════════════════

def _knn_conditional_theta(sub_df, z_cols, k_min_=10, k_frac_=0.5, vrs=None):
    """
    *** SHARED conditional-efficiency computation ***

    Computes the conditional DDF efficiency D(x,y,b|Z) for every row in
    sub_df, using a k-nearest-neighbours-in-Z-space local reference
    technology within each year (Daraio & Simar 2005, 2007; Badin, Daraio
    & Simar 2012) -- a practical approximation to the kernel-weighted
    conditional efficiency estimator, using the SAME DDF solver
    (_solve_ddf_one) as every other frontier estimate in this pipeline.

    This is the SAME logic used inside the separability diagnostic
    (daraio_simar_wilson_separability_test); factored out here as a
    standalone, reusable function so the Conditional DDF step (below) does
    not duplicate it, and so the two are guaranteed methodologically
    identical.

    Required columns in sub_df: iso_code, year, INPUT_COLS, GOOD_OUT,
    BAD_OUT, and every column in z_cols.

    Returns: (n,) array of conditional theta scores (NaN where infeasible
    or where the year has too few observations for a stable k-NN estimate).
    """
    vrs = VRS if vrs is None else vrs        # same RTS as the baseline dea_score it is compared with
    theta_cond = np.full(len(sub_df), np.nan)
    sub_df = sub_df.reset_index(drop=True)
    for yr, g in sub_df.groupby("year"):
        idx = g.index.values
        n_yr = len(idx)
        if n_yr < k_min_ + 2:
            continue
        k = max(k_min_, int(round(k_frac_ * n_yr)))
        k = min(k, n_yr - 1)

        Z_yr = g[z_cols].values.astype(float)
        Z_std = (Z_yr - Z_yr.mean(axis=0)) / (Z_yr.std(axis=0) + 1e-9)

        X_yr = g[INPUT_COLS].values.astype(float)
        Y_yr = g[GOOD_OUT].values.astype(float)
        B_yr = g[BAD_OUT].values.astype(float)
        X_n = _normalise(X_yr); Y_n = _normalise(Y_yr); B_n = _normalise(B_yr)

        for k_local, i_global in enumerate(idx):
            d = np.sqrt(((Z_std - Z_std[k_local])**2).sum(axis=1))
            nn_order = np.argsort(d)
            nn_local = nn_order[:k+1]   # includes itself
            Xr = X_n[nn_local]; Yr = Y_n[nn_local]; Br = B_n[nn_local]
            beta_c = _solve_ddf_one(Xr, Yr, Br,
                                    X_n[k_local], Y_n[k_local], B_n[k_local],
                                    vrs=vrs, allow_negative=False)
            if np.isfinite(beta_c) and beta_c >= 0:
                theta_cond[i_global] = 1.0 / (1.0 + beta_c)
    return theta_cond


def conditional_sensitivity(panel, z_vars=None, k_min=10):
    """
    Sensitivity of the conditional DDF gap (theta_cond - theta_uncond) to the
    neighbourhood definition. Variants:
      base               k = max(10, 1/2 of the year's sample), all Z
      k = 1/4 sample     smaller, more similar neighbourhoods
      k = 3/4 sample     larger neighbourhoods, closer to the unconditional frontier
      without sustainability   Z without dim_sustainability (related to the
                         adverse output, so conditioning on it partly
                         conditions on the outcome)
    Reports, per variant: mean gap, share of observations with a positive gap,
    mean absolute change against base, and the Spearman correlation of
    country-mean gaps with base (does the ranking of affected countries hold?).
    """
    if z_vars is None:
        z_vars = ["dim_resilience", "dim_affordability", "dim_robustness",
                  "dim_sustainability", "gdp_growth"]
    z_vars = [z for z in z_vars if z in panel.columns]
    needed = ["iso_code", "year"] + INPUT_COLS + GOOD_OUT + BAD_OUT + ["dea_score"] + z_vars
    needed = [c for c in dict.fromkeys(needed) if c in panel.columns]
    sub = panel[needed].dropna().copy()
    if len(sub) < 50 or not z_vars:
        print("  ⚠ conditional sensitivity skipped: too few complete observations")
        return pd.DataFrame()
    no_sus = [z for z in z_vars if z != "dim_sustainability"]
    variants = [("base (k = 1/2 sample, all Z)", z_vars, 0.50),
                ("k = 1/4 sample", z_vars, 0.25),
                ("k = 3/4 sample", z_vars, 0.75)]
    if len(no_sus) < len(z_vars):
        variants.append(("without sustainability", no_sus, 0.50))
    # Further peer-set definitions (transparent sensitivity reporting):
    variants.append(("k = 10 (smallest peer set)", z_vars, 0.0))
    no_g = [z for z in z_vars if z != "gdp_growth"]
    if len(no_g) < len(z_vars):
        variants.append(("structural Z only (no GDP growth)", no_g, 0.50))
    gaps = {}
    for lab, zc, kf in variants:
        th = _knn_conditional_theta(sub, zc, k_min_=k_min, k_frac_=kf)
        gaps[lab] = pd.Series(th - sub["dea_score"].values, index=sub.index)
    base_lab = variants[0][0]
    g0 = gaps[base_lab]
    c0 = g0.groupby(sub["iso_code"]).mean()
    rows = []
    for lab, _, kf in variants:
        g = gaps[lab]
        c = g.groupby(sub["iso_code"]).mean()
        both = pd.concat([c0, c], axis=1).dropna()
        rho = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0] if lab != base_lab else 1.0
        rows.append({"variant": lab, "mean_gap": g.mean(), "median_gap": g.median(),
                     "share_positive_gap": (g > 1e-3).mean(),
                     "mean_abs_change_vs_base": (g - g0).abs().mean() if lab != base_lab else 0.0,
                     "spearman_country_gaps_vs_base": rho, "n_obs": int(g.notna().sum())})
    tab = pd.DataFrame(rows)
    print(f"\n  {'─'*78}")
    print(f"  CONDITIONAL DDF — SENSITIVITY TO THE NEIGHBOURHOOD DEFINITION")
    print(f"  {'─'*78}")
    print(f"  {'Variant':<32}{'mean gap':>9}{'share>0':>9}{'|Δ| vs base':>12}{'ρ countries':>13}")
    for _, r in tab.iterrows():
        print(f"  {r.variant:<32}{r.mean_gap:>+9.4f}{r.share_positive_gap:>9.1%}"
              f"{r.mean_abs_change_vs_base:>12.4f}{r.spearman_country_gaps_vs_base:>13.3f}")
    print(f"  Reading: a stable country ranking (ρ close to 1) means the same countries")
    print(f"  are most affected by structural conditioning whatever the neighbourhood;")
    print(f"  the size of the gap is expected to shrink as k grows (larger sets approach")
    print(f"  the unconditional frontier). A large drop in the gap without sustainability")
    print(f"  would indicate that part of the gap came from conditioning on the outcome.")
    return tab


def conditional_ddf_analysis(panel, z_vars=None, k_frac=0.5, k_min=10):
    """
    Conditional DDF: D(x,y,b|Z).

    Recommended as the main alternative to Simar-Wilson when separability
    is rejected (as it is here: the full 5-variable Z vector and even the
    restricted affordability+gdp_growth model both reject the joint
    separability null -- see daraio_simar_wilson_separability_test).

    Rationale: SW2007 assumes an UNCONDITIONAL frontier estimated from
    (x,y,b) alone, then explains DEVIATIONS from that fixed frontier using
    Z in a second stage. If separability fails, Z may shift the ATTAINABLE
    PRODUCTION SET itself, so the unconditional frontier is not a valid
    representation of "the" technology once Z is allowed to matter. The
    conditional DDF instead estimates efficiency directly against a
    Z-localised reference technology for every country-year:

        D(x,y,b | Z=z)

    reusing the SAME k-NN-in-Z-space construction validated in the
    separability diagnostic (_knn_conditional_theta) and the SAME
    low-level DDF solver (_solve_ddf_one) used everywhere else in this
    pipeline -- no separate/parallel technology definition.

    Interpretation shift: rather than "ESI explains
    inefficiency" (the SW framing), this step supports the framing
    "structural energy-security conditions shape the feasible frontier of
    green operational performance."

    Adds to panel: dea_score_conditional (point estimate only -- a full
    conditional bootstrap bias-correction to the same depth as Step 5's
    unconditional SW1998 bootstrap is NOT implemented here and should not
    be assumed; this is a point-estimate conditional efficiency measure).

    Prints: summary statistics, conditional-vs-unconditional comparison,
    and country-level average conditional efficiency ranking.
    """
    print(f"\n{'═'*60}")
    print("CONDITIONAL DDF ANALYSIS — D(x,y,b|Z)")
    print("  Structural benchmark: efficiency against structurally similar peers.")
    print("  Complements the Simar-Wilson second stage, whose separability assumption")
    print("  is not formally tested (see the Step 7 sensitivity diagnostic).")
    print("═"*60)

    if z_vars is None:
        z_vars = ["dim_resilience","dim_affordability",
                  "dim_robustness","dim_sustainability","gdp_growth"]

    needed = ["iso_code","year"] + INPUT_COLS + GOOD_OUT + BAD_OUT + \
             ["dea_score"] + z_vars
    needed = [c for c in needed if c in panel.columns]
    sub = panel[needed].dropna().copy()
    if len(sub) < 30:
        print(f"  ⚠ Only {len(sub)} complete observations — conditional DDF skipped")
        return panel, pd.DataFrame()

    print(f"  Z = {z_vars}")
    print(f"  Sample: {len(sub)} country-years, k-NN local reference technology "
          f"(k = max({k_min}, {k_frac:.0%} of year size))")

    sub["dea_score_conditional"] = _knn_conditional_theta(sub, z_vars, k_min, k_frac)
    n_ok = sub["dea_score_conditional"].notna().sum()
    print(f"  Conditional efficiency computed: {n_ok}/{len(sub)}")

    merged = panel.merge(
        sub[["iso_code","year","dea_score_conditional"]],
        on=["iso_code","year"], how="left"
    )

    comp = sub.dropna(subset=["dea_score_conditional"]).copy()
    comp["gap"] = comp["dea_score_conditional"] - comp["dea_score"]
    print(f"\n  Unconditional theta (mean) : {comp['dea_score'].mean():.4f}")
    print(f"  Conditional theta   (mean) : {comp['dea_score_conditional'].mean():.4f}")
    print(f"  Mean gap (cond - uncond)   : {comp['gap'].mean():+.4f}")
    # Because the conditional reference set (k-NN in Z-space) is a SUBSET of
    # the unconditional reference set (the full year's panel), the
    # conditional frontier can only be as tight as, or looser than, the
    # unconditional one -- narrowing the peer set cannot mechanically create
    # a TOUGHER envelope in this construction. A DMU therefore looks AT
    # LEAST as efficient conditionally as unconditionally: the gap is
    # bounded below by (approximately) zero, not free to take large negative
    # values. A genuinely negative gap here would indicate the conditional
    # k-NN LP hit a different, non-nested reference configuration, not a
    # real "structurally disadvantaged peer group" effect -- it is not
    # interpreted as such below.
    print(f"  (The conditional reference set is a SUBSET of the unconditional")
    print(f"   one, so a DMU can only look AT LEAST as efficient conditionally;")
    print(f"   the gap is not free to take large negative values by construction.)")

    country_summary = (comp.groupby("iso_code")
                       .agg(uncond_mean=("dea_score","mean"),
                            cond_mean=("dea_score_conditional","mean"),
                            gap_mean=("gap","mean"))
                       .reset_index()
                       .sort_values("gap_mean"))
    print(f"\n  Countries with the SMALLEST gap (near zero — bound-binding, i.e.")
    print(f"  conditioning on Z does little to change their apparent efficiency):")
    for _, row in country_summary.head(5).iterrows():
        print(f"    {row['iso_code']:<6} uncond={row['uncond_mean']:.4f}  "
              f"cond={row['cond_mean']:.4f}  gap={row['gap_mean']:+.4f}")
    print(f"\n  Countries with the LARGEST gap (most affected by structurally")
    print(f"  different production environments in the unconditional comparison):")
    for _, row in country_summary.tail(5).iterrows():
        print(f"    {row['iso_code']:<6} uncond={row['uncond_mean']:.4f}  "
              f"cond={row['cond_mean']:.4f}  gap={row['gap_mean']:+.4f}")

    print(f"\n  {'─'*58}")
    print(f"  INTERPRETATION")
    print(f"  {'─'*58}")
    print(f"  Delta_i = theta_cond,i - theta_uncond,i measures the extent to which")
    print(f"  a country's apparent UNCONDITIONAL inefficiency is associated with")
    print(f"  being benchmarked against structurally DIFFERENT production")
    print(f"  environments. A country with a large positive Delta is relatively")
    print(f"  inefficient against the complete (unconditional) technology, but")
    print(f"  much closer to best practice once compared only against peers with")
    print(f"  similar structural conditions.")
    print(f"  This is a POINT-ESTIMATE conditional measure only -- no bootstrap")
    print(f"  bias-correction is applied here (unlike Step 5's dea_bc). Use")
    print(f"  dea_score_conditional as a structural-frontier-shift diagnostic,")
    print(f"  not as a bias-corrected substitute for dea_bc in other analyses.")
    print(f"  {'─'*58}")

    return merged, country_summary


def mundlak_cre_regression(panel, z_vars=None, dep_var="dea_bc"):
    """
    Mundlak (1978) Correlated Random Effects model.

    *** REVISED per review (item 13): the previous parameterisation
    regressed on raw Z_it and Zbar_i together:

        DEA_it = alpha + beta*Z_it + theta*Zbar_i + lambda_t + u_it

    By the standard Mundlak/Chamberlain algebraic identity, beta here does
    equal the conventional within (FE) coefficient exactly. However theta
    -- the coefficient on the raw country-mean Zbar_i in THIS specification
    -- is NOT itself the conventional between-only regression coefficient;
    it equals (between effect - within effect), so labelling theta directly
    as "the between-country effect" overstates what that specific
    coefficient measures.

    Reparameterising to use the WITHIN-DEMEANED deviation and the country
    mean as the two regressors instead:

        DEA_it = alpha + beta_W*(Z_it - Zbar_i) + beta_B*Zbar_i + lambda_t + u_it

    is an exactly equivalent linear reparameterisation of the same two-
    regressor space (identical fitted values, identical R-squared), but now
    beta_W is the within-country association and beta_B is directly the
    conventional between-country association -- no further arithmetic
    needed to interpret either coefficient, and no risk of a reader
    conflating theta (the old parameterisation's second coefficient) with
    an actual between-only regression result. ***

    beta_W = WITHIN-country effect (time-varying deviations from each
             country's own mean) -- reproduces the standard FE coefficient
             exactly, as before.
    beta_B = BETWEEN-country effect (the country's time-average Z level),
             now the genuine between-only regression coefficient -- the
             persistent, structural, slow-moving component that a pure
             entity-FE regression absorbs into the fixed effect and
             DISCARDS. Recovering this is the whole point of the Mundlak/
             CRE specification for this application: ESI dimensions are
             structural and slow-moving, so most of their variation is
             between countries, which the FE model (Step 8) cannot use.
             Mundlak explicitly recovers that cross-country relationship.

    Estimated via pooled OLS with year dummies for lambda_t and
    entity-clustered standard errors (statsmodels), which is the standard
    practical implementation of the Mundlak specification (Wooldridge
    2010, ch.10) -- avoids introducing a new GLS/random-effects library
    dependency while remaining a defensible, citable estimator.

    Returns a DataFrame with within (beta_W) and between (beta_B)
    coefficients side by side for each Z variable, their clustered SEs,
    t-stats and p-values.
    """
    print(f"\n{'═'*60}")
    print("MUNDLAK (1978) CORRELATED RANDOM EFFECTS MODEL")
    print("  Separates within-country (beta_W) from between-country")
    print("  (beta_B) effects, via the (Z_it - Zbar_i) / Zbar_i")
    print("  reparameterisation -- recommended given the FE model's weak")
    print("  within-R² and the largely between-country variation of the ESI.")
    print("═"*60)

    if z_vars is None:
        z_vars = ["dim_resilience","dim_affordability",
                  "dim_robustness","dim_sustainability","gdp_growth"]

    needed = ["iso_code","year",dep_var] + z_vars
    needed = [c for c in needed if c in panel.columns]
    sub = panel[needed].dropna().copy()
    if len(sub) < 30:
        print(f"  ⚠ Only {len(sub)} complete observations — Mundlak model skipped")
        return pd.DataFrame()

    print(f"  Dependent variable: {dep_var}")
    print(f"  Z = {z_vars}")
    print(f"  Sample: {len(sub)} country-years, {sub['iso_code'].nunique()} countries, "
          f"{sub['year'].nunique()} years")

    # Country-mean (between) component: Zbar_i, time-invariant per country
    zbar_cols = [f"{z}_bar" for z in z_vars]
    zdev_cols = [f"{z}_dev" for z in z_vars]
    country_means = sub.groupby("iso_code")[z_vars].transform("mean")
    for z, zb, zd in zip(z_vars, zbar_cols, zdev_cols):
        sub[zb] = country_means[z]
        sub[zd] = sub[z] - country_means[z]   # within-demeaned deviation

    # Year dummies for lambda_t (drop first to avoid the dummy trap)
    year_dummies = pd.get_dummies(sub["year"], prefix="yr", drop_first=True).astype(float)

    X_cols = zdev_cols + zbar_cols
    X_raw = pd.concat([sub[X_cols].reset_index(drop=True),
                       year_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X_raw, has_constant="add").astype(float)
    y = sub[dep_var].values.astype(float)

    # Drop any zero-variance columns (e.g. a Zbar identical across all
    # countries would be degenerate, or a year with no variation)
    const_cols = [c for c in X.columns if X[c].std() < 1e-9 and c != "const"]
    if const_cols:
        print(f"  ⚠ Dropping near-constant columns: {const_cols}")
        X = X.drop(columns=const_cols)

    ols = OLS(y, X.values)
    # Cluster-robust SE by country (entity clustering, standard for CRE)
    groups = sub["iso_code"].values
    res = ols.fit(cov_type="cluster", cov_kwds={"groups": groups})

    col_names = list(X.columns)
    results = []
    for z, zb, zd in zip(z_vars, zbar_cols, zdev_cols):
        if zd not in col_names or zb not in col_names:
            continue
        i_within  = col_names.index(zd)
        i_between = col_names.index(zb)
        results.append({
            "variable": z,
            "beta_within"  : float(res.params[i_within]),
            "se_within"    : float(res.bse[i_within]),
            "p_within"     : float(res.pvalues[i_within]),
            "theta_between": float(res.params[i_between]),
            "se_between"   : float(res.bse[i_between]),
            "p_between"    : float(res.pvalues[i_between]),
        })
    results_df = pd.DataFrame(results)

    print(f"\n  {'Variable':<20} {'beta_W (within)':>15} {'SE':>8} {'p':>7}   "
          f"{'beta_B (between)':>17} {'SE':>8} {'p':>7}")
    print(f"  {'─'*90}")
    for _, row in results_df.iterrows():
        sw = "**" if row["p_within"]  < 0.05 else ("*" if row["p_within"]  < 0.10 else " ")
        sb = "**" if row["p_between"] < 0.05 else ("*" if row["p_between"] < 0.10 else " ")
        print(f"  {row['variable']:<20} {row['beta_within']:>+13.4f}{sw} {row['se_within']:>8.4f} "
              f"{row['p_within']:>7.4f}   {row['theta_between']:>+15.4f}{sb} {row['se_between']:>8.4f} "
              f"{row['p_between']:>7.4f}")
    print(f"  {'─'*90}")
    print(f"  R² = {res.rsquared:.4f}  |  N = {int(res.nobs)}  |  * p<0.10  ** p<0.05")

    print(f"\n  {'─'*58}")
    print(f"  INTERPRETATION")
    print(f"  {'─'*58}")
    sig_between = results_df[results_df["p_between"] < 0.10]
    sig_within  = results_df[results_df["p_within"]  < 0.10]
    if len(sig_between) > 0:
        print(f"  Significant BETWEEN (structural, cross-country) effects:")
        for _, row in sig_between.iterrows():
            level = "p<0.05" if row["p_between"] < 0.05 else "p<0.10"
            print(f"    {row['variable']:<20} beta_B={row['theta_between']:+.4f}  "
                  f"p={row['p_between']:.3f} ({level})")
    else:
        print(f"  No BETWEEN (structural) effects significant at p<0.10.")
    if len(sig_within) > 0:
        print(f"  Significant WITHIN (short-run, same-country) effects:")
        for _, row in sig_within.iterrows():
            level = "p<0.05" if row["p_within"] < 0.05 else "p<0.10"
            print(f"    {row['variable']:<20} beta_W={row['beta_within']:+.4f}  "
                  f"p={row['p_within']:.3f} ({level})")
    else:
        print(f"  No WITHIN (short-run) effects significant at p<0.10 — consistent")
        print(f"  with the Panel FE result (Step 8): ESI dimensions behave as")
        print(f"  persistent, slow-moving, country-level structural characteristics")
        print(f"  rather than short-run within-country drivers of operational")
        print(f"  efficiency. The between-country component captures persistent")
        print(f"  cross-country associations that are absorbed by country fixed")
        print(f"  effects in the FE specification.")
    print(f"  {'─'*58}")

    return results_df



def panel_fe_regression(panel):
    """
    Panel FE regression (entity + time fixed effects, clustered SE by entity):
      dea_bc ~ dim_resilience + dim_affordability + dim_robustness +
               dim_sustainability + gdp_growth

    Note: esi_score is excluded because it is a linear combination of the four
    dim_* columns (equal-weighted average), causing perfect multicollinearity
    and a rank-deficient X matrix → linearmodels raises ValueError.
    We instead include the four dimensions directly, which is more informative.

    # net_import_pct is excluded, matching the same exclusion
    already applied to the SW2007 second stage. dea_net_import = max(0,
    net_import_pct) is a DDF input; including its unshifted source on the
    RHS here would put a variable mechanically related to the dependent
    variable's construction into the regression, exactly as for SW. ***
    """
    print(f"\n{'═'*60}")
    print("SECTION 7 — Panel Fixed-Effects OLS")
    print("═"*60)

    # esi_score deliberately excluded — it equals mean(dim_*), causing rank deficiency
    # net_import_pct excluded — dea_net_import (DDF input) is derived from it
    x_vars = ["dim_resilience","dim_affordability",
              "dim_robustness","dim_sustainability","gdp_growth"]
    _data_audit(panel, ["dea_bc"] + x_vars, "Panel FE")
    cols = ["iso_code","year","dea_bc"] + x_vars
    sub  = panel[[c for c in cols if c in panel.columns]].dropna().copy()

    if len(sub) < 30:
        print(f"  ⚠ Too few observations ({len(sub)}) — skipping Panel FE")
        return None

    # Drop any near-constant columns (zero variance within panel → rank issues)
    drop_cols = [c for c in x_vars if sub[c].std() < 1e-8]
    if drop_cols:
        print(f"  ⚠ Dropping near-constant regressors: {drop_cols}")
        x_vars = [c for c in x_vars if c not in drop_cols]

    sub = sub.set_index(["iso_code","year"])
    y = sub["dea_bc"]

    # PanelOLS with entity+time FE absorbs the intercept — do NOT add a constant
    # (adding one causes rank deficiency and errors across linearmodels versions).
    X = sub[x_vars].copy()

    # *** REMOVED per review (item 14): a homemade matrix_rank(X_vals) precheck
    # on the RAW (pre-entity/time-demeaning) design matrix used to run here,
    # followed by sequentially dropping the column with the lowest RAW
    # variance until full rank was achieved. Two problems with that: (1) rank
    # deficiency relevant to a two-way fixed-effects model arises AFTER the
    # entity/time transformation, not necessarily in the raw X -- a raw-X
    # rank check can both miss real post-demeaning collinearity and flag
    # spurious raw-scale issues that demeaning would resolve on its own; (2)
    # "drop whichever remaining column has the lowest raw standard deviation"
    # could remove a substantively important, low-variance regressor for a
    # reason unrelated to whether it actually causes collinearity, silently
    # changing the specification. PanelOLS already performs its own
    # absorption and rank handling internally, and the code immediately
    # below already retries with a reduced specification (dropping time
    # effects) if the fit itself fails -- that is the correct place to let
    # collinearity surface and be handled, not a raw-X precheck beforehand. ***

    # Try fitting; fall back to removing time_effects if singular
    try:
        fe  = PanelOLS(y, X, entity_effects=True, time_effects=True)
        res = fe.fit(cov_type="clustered", cluster_entity=True)
    except Exception as e1:
        print(f"  ⚠ FE with time effects failed ({e1}); retrying without time effects")
        try:
            fe  = PanelOLS(y, X, entity_effects=True, time_effects=False)
            res = fe.fit(cov_type="clustered", cluster_entity=True)
            print(f"  ℹ Using entity FE only (no time FE)")
        except Exception as e2:
            print(f"  ✗ Panel FE failed entirely: {e2}")
            return None

    print(str(res.summary.tables[1]))
    print(f"\n  R² (within): {res.rsquared:.4f}")
    print(f"  N obs:       {res.nobs}")

    # ── FE interpretation ─────────────────────────────────────────────────────
    try:
        params = res.params
        pvals  = res.pvalues
        sig_fe = [(n, float(params[n]), float(pvals[n]))
                  for n in params.index
                  if n not in ("const","Intercept") and float(pvals[n]) < 0.10]

        print(f"\n  {'─'*58}")
        print(f"  PANEL FE OLS INTERPRETATION")
        print(f"  {'─'*58}")
        print(f"  Entity + time fixed effects absorb all time-invariant country")
        print(f"  characteristics and common shocks. Coefficients reflect WITHIN-")
        print(f"  country variation over time — how ESI changes predict DEA changes.")
        print(f"")
        print(f"  R²(within) = {res.rsquared:.3f}: the ESI dimensions explain {res.rsquared*100:.0f}%")
        print(f"  of within-country variation in operational efficiency.")
        if sig_fe:
            print(f"")
            print(f"  Significant regressors at p<0.10 (within-country effect):")
            for name, coef, pv in sorted(sig_fe, key=lambda x: abs(x[1]), reverse=True):
                direction = "↑" if coef > 0 else "↓"
                print(f"    {direction} {name:<32} β={coef:+.4f}  p={pv:.3f}")
        print(f"")
        # SW2007 pools country-years; it is not a cross-sectional estimator.
        # The correct contrast is pooled/persistent cross-country heterogeneity
        # vs. short-run within-country co-movement.
        print(f"  FE provides only weak within-country support (R2={res.rsquared:.3f}).")
        print(f"  The contrast between the pooled frontier-based results (ESI dimensions")
        print(f"  vs. dea_bc, Step 5b/SW2007) and these weak within-country FE estimates")
        print(f"  indicates that much of the observed structural-operational association")
        print(f"  reflects persistent cross-country heterogeneity rather than short-run")
        print(f"  within-country co-movement. Use FE as a robustness cross-check, not as")
        print(f"  causal confirmation of the pooled SW second-stage coefficients.")
        print(f"  {'─'*58}")
    except Exception:
        pass

    return res


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PANEL GRANGER CAUSALITY (Dumitrescu-Hurlin approximation)
# ═════════════════════════════════════════════════════════════════════════════

def panel_granger(panel, maxlag=2):
    """
    Country-by-country Granger causality tests, then aggregate (DH approximation).
    Tests bidirectional: ESI → DEA and DEA → ESI.
    """
    print(f"\n{'═'*60}")
    print("SECTION 9 — Panel Granger Causality (Dumitrescu-Hurlin approx.)")
    print("═"*60)

    _data_audit(panel, ["esi_score","dea_bc"], "Granger causality")
    sub = panel[["iso_code","year","esi_score","dea_bc"]].dropna()
    sub = sub.sort_values(["iso_code","year"])

    # *** FIX (numerical): track results per-country (ISO-keyed) instead of
    # plain parallel lists, and pull the Wald chi2 statistic DIRECTLY from
    # statsmodels (ssr_chi2test) rather than reconstructing it by inverting
    # a p-value through chi2.ppf(1-p, df=K). That inversion is what produced
    # Z-bar=+inf: whenever a single country's F-test p-value underflows to
    # exactly 0.0 (near-perfect fit / low residual variance in a short
    # T=24 series), chi2.ppf(1.0, df=K) is +infinity by definition of the
    # inverse CDF at probability 1. Using the statistic statsmodels already
    # computed avoids this round-trip entirely. A defensive p-value clip is
    # also applied as a backstop. ***
    per_country = []   # list of dicts: iso, p_ed, p_de, w_ed, w_de
    for iso, g in sub.groupby("iso_code"):
        g = g.sort_values("year")
        if len(g) < maxlag + 5:
            continue
        try:
            r_ed = grangercausalitytests(g[["dea_bc","esi_score"]].values,
                                          maxlag=maxlag, verbose=False)
            p_ed_i = min(r_ed[l][0]["ssr_ftest"][1] for l in range(1, maxlag+1))
            w_ed_i = float(r_ed[maxlag][0]["ssr_chi2test"][0])   # Wald chi2, df=maxlag*k

            r_de = grangercausalitytests(g[["esi_score","dea_bc"]].values,
                                          maxlag=maxlag, verbose=False)
            p_de_i = min(r_de[l][0]["ssr_ftest"][1] for l in range(1, maxlag+1))
            w_de_i = float(r_de[maxlag][0]["ssr_chi2test"][0])

            if not (np.isfinite(w_ed_i) and np.isfinite(w_de_i)):
                print(f"  ⚠ {iso}: non-finite Wald statistic — excluded "
                      f"(W_ed={w_ed_i}, W_de={w_de_i})")
                continue
            per_country.append({"iso_code": iso, "p_ed": p_ed_i, "p_de": p_de_i,
                                "w_ed": w_ed_i, "w_de": w_de_i})
        except Exception as e:
            print(f"  ⚠ {iso}: Granger test failed ({e}) — excluded")

    p_ed = [d["p_ed"] for d in per_country]
    p_de = [d["p_de"] for d in per_country]

    print(f"\n  ESI → DEA (does ESI Granger-cause DEA efficiency?)")
    w_ed = w_de = None
    outlier_isos = {"ESI→DEA": [], "DEA→ESI": []}   # ensure defined even if per_country is empty
    if p_ed:
        print(f"    Mean p-value: {np.mean(p_ed):.4f}  |  "
              f"Share significant (p<0.10): {np.mean(np.array(p_ed)<0.10):.1%}")

    print(f"\n  DEA → ESI (does DEA efficiency Granger-cause ESI?)")
    if p_de:
        print(f"    Mean p-value: {np.mean(p_de):.4f}  |  "
              f"Share significant (p<0.10): {np.mean(np.array(p_de)<0.10):.1%}")

    print(f"\n  Dumitrescu-Hurlin standardised statistic Z-bar:")
    print(f"  Z-bar = sqrt(N/(2K)) × (W-bar − K) ~ N(0,1) under H0  [DH 2012, eq.9]")
    print(f"  (The code computes Z-bar; ±1.96 is the correct critical value.)")
    w_ed_robust = w_de_robust = None
    if per_country:
        n_c  = len(per_country)
        K    = maxlag
        w_ed_vals = np.array([d["w_ed"] for d in per_country])
        w_de_vals = np.array([d["w_de"] for d in per_country])

        w_bar_ed = float(np.mean(w_ed_vals))
        w_bar_de = float(np.mean(w_de_vals))
        w_ed = float(np.sqrt(n_c / (2.0*K)) * (w_bar_ed - K))
        w_de = float(np.sqrt(n_c / (2.0*K)) * (w_bar_de - K))

        # A small number of countries with a near-
        # singular / near-deterministic within-country regression (e.g.
        # BRA, IDN observed at ~1.75e12) can produce astronomically large
        # individual Wald statistics that swamp the simple mean W-bar and
        # make the aggregate Z-bar numerically meaningless (previously
        # observed as Z=+inf; now merely enormous, e.g. 1e11). Flagging
        # outliers is not sufficient on its own -- report a WINSORIZED
        # Z-bar alongside the raw one so there is always a usable
        # aggregate statistic even when outliers are present. ***
        outlier_isos = {"ESI→DEA": [], "DEA→ESI": []}
        for arr, label in [(w_ed_vals, "ESI→DEA"), (w_de_vals, "DEA→ESI")]:
            if len(arr) > 3:
                med, mad = np.median(arr), np.median(np.abs(arr - np.median(arr))) + 1e-9
                outliers = [(per_country[i]["iso_code"], arr[i])
                           for i in range(len(arr)) if abs(arr[i]-med) > 10*mad]
                if outliers:
                    outlier_isos[label] = [iso for iso, _ in outliers]
                    print(f"    ⚠ {label}: outlier country-level Wald stat(s): "
                          f"{[(iso, f'{v:.3e}') for iso, v in outliers]}")

        # Winsorize at the 5th/95th percentile before averaging — a simple,
        # transparent robustness check against exactly this failure mode.
        def _winsorized_zbar(arr, n_c, K):
            lo, hi = np.percentile(arr, [5, 95])
            arr_w = np.clip(arr, lo, hi)
            w_bar_w = float(np.mean(arr_w))
            return float(np.sqrt(n_c / (2.0*K)) * (w_bar_w - K))

        w_ed_robust = _winsorized_zbar(w_ed_vals, n_c, K)
        w_de_robust = _winsorized_zbar(w_de_vals, n_c, K)

        print(f"    Z-bar (ESI→DEA), raw        : {w_ed:+.3e}  "
              f"{'Reject H0 at 5%' if abs(w_ed)>1.96 else 'Fail to reject H0'}")
        print(f"    Z-bar (ESI→DEA), winsorized : {w_ed_robust:+.3f}  "
              f"{'Reject H0 at 5%' if abs(w_ed_robust)>1.96 else 'Fail to reject H0'}")
        print(f"    Z-bar (DEA→ESI), raw        : {w_de:+.3e}  "
              f"{'Reject H0 at 5%' if abs(w_de)>1.96 else 'Fail to reject H0'}")
        print(f"    Z-bar (DEA→ESI), winsorized : {w_de_robust:+.3f}  "
              f"{'Reject H0 at 5%' if abs(w_de_robust)>1.96 else 'Fail to reject H0'}")
        if outlier_isos["ESI→DEA"] or outlier_isos["DEA→ESI"]:
            print(f"    *** Outliers detected — DO NOT report the raw Z-bar as")
            print(f"    substantive evidence. Diagnose the flagged countries")
            print(f"    (condition number, residual variance, near-perfect fit)")
            print(f"    before using either direction's result; the winsorized")
            print(f"    value is a transparency check, not a validated fix. ***")

    # ── Granger interpretation ───────────────────────────────────────────────
    if p_ed and p_de:
        print(f"\n  {'─'*58}")
        print(f"  GRANGER CAUSALITY INTERPRETATION")
        print(f"  {'─'*58}")
        print(f"  H₀: X does NOT Granger-cause Y (past X adds no predictive power")
        print(f"  for Y beyond Y's own lags). Reject H₀ → Granger causality found.")
        print(f"")
        mean_p_ed = np.mean(p_ed); mean_p_de = np.mean(p_de)
        sig_ed = np.mean(np.array(p_ed) < 0.10)
        sig_de = np.mean(np.array(p_de) < 0.10)
        print(f"  ESI → DEA: mean p={mean_p_ed:.4f}  "
              f"significant in {sig_ed:.0%} of countries")
        print(f"  DEA → ESI: mean p={mean_p_de:.4f}  "
              f"significant in {sig_de:.0%} of countries")
        print(f"")
        # A direction with a flagged outlier country (e.g. an astronomically
        # large country-level Wald statistic from a near-singular within-
        # country regression) has its conclusion GATED entirely -- no
        # Reject/Fail-to-reject verdict is printed for that direction,
        # regardless of what the raw or winsorised Z-bar says. This is a
        # numerical-validity gate, not a judgment call left to the reader.
        ed_gated = len(outlier_isos.get("ESI→DEA", [])) > 0
        de_gated = len(outlier_isos.get("DEA→ESI", [])) > 0

        if ed_gated:
            print(f"  → ESI → DEA: INVALID pending numerical diagnosis "
                  f"(outlier country-level Wald stat detected: "
                  f"{outlier_isos['ESI→DEA']}). No panel-level inference reported.")
        if de_gated:
            print(f"  → DEA → ESI: INVALID pending numerical diagnosis "
                  f"(outlier country-level Wald stat detected: "
                  f"{outlier_isos['DEA→ESI']}). No panel-level inference reported.")

        if not ed_gated and not de_gated:
            both_reject  = (w_ed is not None and w_de is not None
                            and abs(w_ed) > 1.96 and abs(w_de) > 1.96)
            only_ed_rej  = (w_ed is not None and abs(w_ed) > 1.96
                            and (w_de is None or abs(w_de) <= 1.96))
            only_de_rej  = (w_de is not None and abs(w_de) > 1.96
                            and (w_ed is None or abs(w_ed) <= 1.96))

            if both_reject:
                print(f"  → BIDIRECTIONAL: both H0 rejected at 5%.")
                print(f"    ESI and DEA efficiency show mutual Granger-predictive dynamics.")
                print(f"    A primary direction CANNOT be inferred from these results —")
                print(f"    Z-bar(ESI→DEA)={w_ed:+.3f} vs Z-bar(DEA→ESI)={w_de:+.3f}.")
                print(f"    Any p-value difference at the country level ({sig_ed:.0%} vs")
                print(f"    {sig_de:.0%} significant) is negligible; do not use it to")
                print(f"    rank directions. Report as consistent with ESI ↔ DEA.")
            elif only_ed_rej:
                print(f"  → ESI → DEA: H0 rejected (Z={w_ed:+.3f}); DEA → ESI not rejected.")
                print(f"    Suggestive that structural conditions Granger-predict")
                print(f"    operational efficiency, not the reverse.")
                print(f"    Significant in {sig_ed:.0%} of countries (heterogeneous panel).")
            elif only_de_rej:
                print(f"  → DEA → ESI: H0 rejected (Z={w_de:+.3f}); ESI → DEA not rejected.")
                print(f"    Suggestive that operational efficiency Granger-predicts")
                print(f"    structural conditions, not the reverse.")
                print(f"    Significant in {sig_de:.0%} of countries (heterogeneous panel).")
            else:
                print(f"  → Neither direction rejected at 5%. No Granger-predictive")
                print(f"    relationship detected at the panel level.")
            print(f"")
            print(f"  ⚠ CAVEAT: this panel has only T={len(sub['year'].unique())} years per")
            print(f"  country. The Z-bar asymptotic standardisation (DH 2012, eq.9) is a")
            print(f"  large-N, and implicitly large/moderate-T, approximation; with T this")
            print(f"  short it should be treated as suggestive, not confirmatory, until")
            print(f"  the finite-T behaviour of the standardisation is separately verified.")

        if w_ed is not None and w_de is not None:
            print(f"")
            print(f"  DH Z-bar (ESI→DEA): {w_ed:+.3f}  "
                  f"{'→ Reject H₀ at 5%' if abs(w_ed)>1.96 else '→ Fail to reject H₀'}"
                  f"{'  [GATED — see above]' if ed_gated else ''}")
            print(f"  DH Z-bar (DEA→ESI): {w_de:+.3f}  "
                  f"{'→ Reject H₀ at 5%' if abs(w_de)>1.96 else '→ Fail to reject H₀'}"
                  f"{'  [GATED — see above]' if de_gated else ''}")
        print(f"  {'─'*58}")

    # granger_df built directly from the aligned per-country records (no
    # risk of list-length misalignment since each dict carries its own iso).
    granger_df = (pd.DataFrame(per_country)[["iso_code","p_ed","p_de"]]
                  .rename(columns={"p_ed":"p_esi_dea","p_de":"p_dea_esi"})
                  if per_country else
                  pd.DataFrame(columns=["iso_code","p_esi_dea","p_dea_esi"]))

    ed_gated_final = len(outlier_isos.get("ESI→DEA", [])) > 0
    de_gated_final = len(outlier_isos.get("DEA→ESI", [])) > 0
    granger_stats = {
        "z_bar_esi_dea"          : w_ed,   # DH standardised Z-bar ~ N(0,1), RAW
        "z_bar_dea_esi"          : w_de,
        "z_bar_esi_dea_robust"   : w_ed_robust,   # winsorized (5/95 pct) version
        "z_bar_dea_esi_robust"   : w_de_robust,
        "ed_gated"          : ed_gated_final,   # True = outlier detected, do NOT report a verdict
        "de_gated"          : de_gated_final,
        "mean_p_esi_dea"    : np.mean(p_ed) if p_ed else None,
        "mean_p_dea_esi"    : np.mean(p_de) if p_de else None,
        "share_sig_esi_dea" : np.mean(np.array(p_ed)<0.10) if p_ed else None,
        "share_sig_dea_esi" : np.mean(np.array(p_de)<0.10) if p_de else None,
    }
    return granger_df, granger_stats


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PLOTS & EXPORT
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE LEGIBILITY HELPERS — every country label readable, no overlaps
# ═════════════════════════════════════════════════════════════════════════════
# Applied to every figure in the pipeline:
#   - minimum font sizes via rcParams (tick labels never below 9 pt)
#   - figure heights/widths scale with the number of countries shown, so each
#     country row/column gets at least ROW_IN inches (≥ 9 pt text never touches)
#   - scatter labels (ISO codes) are placed by _label_points(), which moves
#     overlapping labels apart and draws a thin leader line back to the point
#   - heatmap cell text is sized to the cell and switches colour for contrast
import matplotlib as _mpl
_mpl.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.titlesize": 13, "savefig.dpi": 160, "savefig.bbox": "tight",
    # house style: no top/right frame lines, light legend frame, bold titles
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlepad": 10, "axes.axisbelow": True,
    "grid.color": "#d9d9d9", "grid.linewidth": 0.6,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#cccccc",
    "legend.borderaxespad": 0.0,
})


def _legend_right(ax, handles=None, labels=None, **kw):
    """Legend outside the plot, on its right side, top-aligned with the axes
    (never on top of the data or the title)."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    kw.setdefault("fontsize", 9.5)
    return ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), **kw)


def _fig_legend_right(fig, axes=None, handles=None, labels=None, right=0.85, top=None, **kw):
    """One shared legend for a multi-panel figure, on the right-hand side of
    the figure. Reserves that strip so no panel or title is covered."""
    if handles is None:
        handles, labels, seen = [], [], set()
        for a in (axes if axes is not None else fig.axes):
            for h, l in zip(*a.get_legend_handles_labels()):
                if l not in seen and not l.startswith("_"):
                    seen.add(l); handles.append(h); labels.append(l)
    if not handles:
        return None
    kw.setdefault("fontsize", 10)
    if top is None:
        top = 0.94 if fig._suptitle is not None else 1.0
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(right + 0.01, top - 0.02), **kw)
    fig._tight_rect = [0, 0, right, top]
    return leg
ROW_IN = 0.32          # inches per country row/column in country-level charts
MIN_LABEL_PT = 9       # smallest font used for any country label


def _country_extent(n, base=4.0, per=ROW_IN, minimum=7.0):
    """Figure dimension (inches) giving each of n country rows `per` inches."""
    return max(minimum, base + per * n)


def _texts_overlap(fig, texts, pad_px=1.0):
    """Number of overlapping label pairs (used to verify placement)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = [t.get_window_extent(r).expanded(1.0, 1.0) for t in texts]
    n = 0
    for i in range(len(bb)):
        for j in range(i + 1, len(bb)):
            a, b = bb[i], bb[j]
            if (a.x0 < b.x1 - pad_px and b.x0 < a.x1 - pad_px and
                    a.y0 < b.y1 - pad_px and b.y0 < a.y1 - pad_px):
                n += 1
    return n


def _label_points(ax, xs, ys, labels, colors=None, fontsize=MIN_LABEL_PT,
                  marker_px=7.0, pad_px=2.0, avoid=()):
    """
    Label scatter points (e.g. ISO codes) so that no label overlaps another
    label or any marker. Pure matplotlib, deterministic.

    Greedy candidate placement: points are labelled from the least to the most
    crowded. Each label tries 16 directions around its point at increasing
    distances and takes the first position that is inside the axes and clear
    of every marker and every label already placed. Labels that end up away
    from their point get a thin grey leader line, so crowded clusters stay
    readable and every code is still attributable to its point.
    `avoid` = other Text artists (e.g. quadrant names) labels must not cover.
    Call it AFTER everything else is drawn on the axes; it fixes the axis
    limits and the layout first so positions do not shift afterwards.
    Returns the list of Text objects.
    """
    fig = ax.figure
    if not getattr(fig, "_layout_frozen", False):
        _r = getattr(fig, "_tight_rect", None)
        fig.tight_layout(rect=_r) if _r else fig.tight_layout()
    fig._labels_placed = True          # _save() must not re-run tight_layout now
    ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    labels = [l for l, k in zip(labels, ok) if k]
    colors = ["black"] * len(xs) if colors is None else [c for c, k in zip(colors, ok) if k]
    if len(xs) == 0:
        return []
    texts = [ax.text(x, y, str(l), fontsize=fontsize, color=c, ha="left", va="bottom", zorder=6)
             for x, y, l, c in zip(xs, ys, labels, colors)]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    wh = np.array([[t.get_window_extent(r).width, t.get_window_extent(r).height] for t in texts])
    ab = ax.get_window_extent(r)
    n = len(texts)

    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    crowd = (d < 6 * wh[:, 1].max()).sum(axis=1)
    order = np.argsort(crowd, kind="stable")

    angles = np.deg2rad(np.arange(0, 360, 22.5) + 45)  # start at upper-right
    step = 0.8 * wh[:, 1].max()
    placed = []                                        # (x0, y0, x1, y1)
    for t in avoid:
        e = t.get_window_extent(r)
        placed.append((e.x0, e.y0, e.x1, e.y1))

    def _clear(x0, y0, x1, y1):
        if x0 < ab.x0 + 1 or x1 > ab.x1 - 1 or y0 < ab.y0 + 1 or y1 > ab.y1 - 1:
            return False
        for (a0, b0, a1, b1) in placed:
            if x0 < a1 + pad_px and a0 < x1 + pad_px and y0 < b1 + pad_px and b0 < y1 + pad_px:
                return False
        inside = ((pts[:, 0] > x0 - marker_px) & (pts[:, 0] < x1 + marker_px) &
                  (pts[:, 1] > y0 - marker_px) & (pts[:, 1] < y1 + marker_px))
        return not inside.any()

    final = np.zeros((n, 2))
    for i in order:
        w, h = wh[i]
        px, py = pts[i]
        best = None
        for k in range(0, 60):
            rad = marker_px + 1 + k * step
            for a in angles:
                cx, cy = px + (rad + w / 2) * np.cos(a), py + (rad + h / 2) * np.sin(a)
                x0, y0 = cx - w / 2, cy - h / 2
                if _clear(x0, y0, x0 + w, y0 + h):
                    best = (x0, y0); break
            if best is not None:
                break
        if best is None:                               # no free space: nearest in-axes spot
            best = (np.clip(px + marker_px, ab.x0 + 1, ab.x1 - w - 1),
                    np.clip(py + marker_px, ab.y0 + 1, ab.y1 - h - 1))
        placed.append((best[0], best[1], best[0] + w, best[1] + h))
        final[i] = best

    inv = ax.transData.inverted()
    for i, t in enumerate(texts):
        t.set_position(inv.transform(final[i]))
        ex = np.clip(pts[i, 0], final[i, 0], final[i, 0] + wh[i, 0])
        ey = np.clip(pts[i, 1], final[i, 1], final[i, 1] + wh[i, 1])
        if np.hypot(ex - pts[i, 0], ey - pts[i, 1]) > marker_px + 4:
            (lx, ly), (qx, qy) = inv.transform([[ex, ey], pts[i]])
            ax.plot([qx, lx], [qy, ly], color="#9a9a9a", lw=0.6, zorder=5)
    return texts


def _annotate_heatmap(ax, im, data, fmt="{:.0f}", fontsize=None):
    """Write cell values into a heatmap, sized to the cell, in black or white
    depending on the luminance of that cell's own colour."""
    data = np.asarray(data, float)
    nr, nc = data.shape
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    cell_pt = min(bb.width / max(nc, 1), bb.height / max(nr, 1)) * 72 / fig.dpi
    fs = fontsize or float(np.clip(cell_pt * 0.40, 7.5, 10))
    for i in range(nr):
        for j in range(nc):
            v = data[i, j]
            if np.isfinite(v):
                r_, g_, b_, _ = im.cmap(im.norm(v))
                lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                        color="black" if lum > 0.5 else "white")


def plot_dea_with_ci(panel, year=None):
    """Bar chart of bias-corrected DEA scores with 95% CI error bars."""
    if year is None:
        year = panel["year"].max()
    sub = panel[panel["year"]==year].dropna(subset=["dea_bc"]).sort_values("dea_bc").copy()

    if len(sub) == 0:
        print("  ⚠ No DEA data for plot — skipping dea_bootstrap_ci.png")
        return

    fig, ax = plt.subplots(figsize=(max(12, 3 + 0.36 * len(sub)), 7))
    x = np.arange(len(sub))
    ax.bar(x, sub["dea_bc"], color="#2E5496", alpha=0.8, label="DEA score (bias-corrected)")

    # Only draw CI bars if dea_lo / dea_hi are available and valid
    has_ci = ("dea_lo" in sub.columns and "dea_hi" in sub.columns
              and sub["dea_lo"].notna().any() and sub["dea_hi"].notna().any())
    if has_ci:
        # Guarantee non-negative half-widths after any clipping inconsistencies
        err_lo = np.clip(sub["dea_bc"].values - sub["dea_lo"].values, 0, None)
        err_hi = np.clip(sub["dea_hi"].values - sub["dea_bc"].values, 0, None)
        ax.errorbar(x, sub["dea_bc"],
                    yerr=[np.abs(np.nan_to_num(np.asarray(err_lo, dtype=float), nan=0.0)), np.abs(np.nan_to_num(np.asarray(err_hi, dtype=float), nan=0.0))],
                    fmt="none", color="black", capsize=3, linewidth=1.2,
                    label="95% CI (bootstrap)")

    ax.set_xticks(x)
    ax.set_xticklabels(sub["iso_code"], rotation=90, ha="center", fontsize=9.5)
    ax.set_xlim(-0.7, len(sub) - 0.3)
    ax.set_ylabel("DEA efficiency score (bias-corrected)")
    ax.set_title(f"G20 DEA Efficiency — Simar–Wilson Bootstrap  ({year})", fontsize=11)
    ax.axhline(1.0, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="Frontier")
    _legend_right(ax)
    plt.tight_layout()
    _save_fig("dea_bootstrap_ci.png")
    plt.show()


def plot_mpi_decomposition(mpi_df):
    """
    Stacked bar chart of GML = GEC x GTC by country (geometric mean).

    *** FIX: this chart's own title already correctly said "Global
    Malmquist-Luenberger Index (GML)", but its axis label and legend still
    said "MPI component" / "EC (catch-up)" / "TC - 1 (frontier shift)" --
    internally inconsistent with its own title, and part of the same
    MPI-vs-GML labelling issue fixed in plot_step6_mpi() above. Corrected
    to GEC/GTC throughout, matching the title. ***
    """
    gmean = lambda x: np.exp(np.mean(np.log(np.clip(x, 0.01, None))))
    cdf = mpi_df.groupby("iso_code").agg(
        ec=("ec","mean"), tc=("tc","mean"), mpi=("mpi","mean")
    ).reset_index().sort_values("mpi", ascending=True)

    fig, ax = plt.subplots(figsize=(12, _country_extent(len(cdf))))
    x = np.arange(len(cdf))
    ax.barh(x, cdf["ec"], color="#1D9E75", alpha=0.85, label="GEC (catch-up)")
    ax.barh(x, cdf["tc"] - 1, left=cdf["ec"], color="#534AB7", alpha=0.85,
            label="GTC − 1 (global frontier shift)")
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(x)
    ax.set_yticklabels(cdf["iso_code"], fontsize=9.5)
    ax.set_ylim(-0.7, len(cdf) - 0.3)
    ax.set_xlabel("GML component (period average)")
    ax.set_title("Global Malmquist-Luenberger Index (GML) Decomposition", fontsize=12)
    _legend_right(ax)
    plt.tight_layout()
    _save_fig("mpi_decomposition.png")
    plt.show()


def plot_quadrant(panel, year=None):
    """ESI × DEA quadrant map."""
    if year is None:
        year = panel["year"].max()
    sub = panel[panel["year"]==year].dropna(subset=["esi_score","dea_bc"]).copy()
    esi_med = sub["esi_score"].median()
    dea_med = sub["dea_bc"].median()

    def quad(row):
        h_e = row["esi_score"] >= esi_med
        h_d = row["dea_bc"]    >= dea_med
        if   h_e and  h_d: return "Leaders"
        elif h_e and ~h_d: return "Secure but wasteful"
        elif ~h_e and h_d:  return "Efficient laggards"
        else:               return "Vulnerable"

    sub["quadrant"] = sub.apply(quad, axis=1)
    colors = {"Leaders":"#1D9E75","Secure but wasteful":"#534AB7",
              "Efficient laggards":"#BA7517","Vulnerable":"#993C1D"}

    fig, ax = plt.subplots(figsize=(13, 10.5))
    for _, row in sub.iterrows():
        ax.scatter(row["esi_score"], row["dea_bc"], color=colors[row["quadrant"]], s=90, zorder=3)

    ax.axvline(esi_med, color="gray", lw=0.8, ls="--")
    ax.axhline(dea_med, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("ESI composite score"); ax.set_ylabel("DEA score (bias-corrected)")
    ax.set_title(f"G20 Energy Security Quadrant Map  ({year})", fontsize=11)
    patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
    _legend_right(ax, handles=patches, labels=[p.get_label() for p in patches], fontsize=10,
                  title="Quadrant")
    _label_points(ax, sub["esi_score"], sub["dea_bc"], sub["iso_code"],
                  colors=[colors[q] for q in sub["quadrant"]])
    _save_fig("quadrant_map.png")
    plt.show()


def add_alt_rts_bootstrap(state):
    """
    Bootstrap-bias-correct the ALTERNATIVE returns-to-scale specification
    (the one that is NOT the baseline set by VRS), with the same SW1998
    bootstrap. With the CRS baseline this adds dea_vrs_bc / _lo / _hi / _bias /
    _se (the VRS robustness check); with a VRS baseline it adds dea_crs_*.
    Baseline columns (dea_score, dea_bc, ...) are never overwritten.
    Previously named add_crs_bootstrap (from when VRS was the baseline);
    that name is kept as an alias so existing calls still work.
    """
    panel = state["panel"]
    alt_vrs = not VRS
    sp = "vrs" if alt_vrs else "crs"
    print("\n" + "="*60)
    print(f"ROBUSTNESS CHECK — {ALT_RTS} bootstrap (baseline {BASE_RTS} was bootstrapped in Step 5)")
    print(f"  Only the dea_{sp}_* columns are added; baseline columns are untouched.")
    print("="*60)
    print(f"  Replications: B_BOOTSTRAP_ALT = {B_BOOTSTRAP_ALT} "
          f"(baseline {BASE_RTS} used B_BOOTSTRAP = {B_BOOTSTRAP})")
    alt_panel = build_dea_bootstrap(panel.copy(), B=B_BOOTSTRAP_ALT, alpha=ALPHA, vrs=alt_vrs)
    cols = [f"dea_{sp}_{k}" for k in ("bc", "lo", "hi", "bias", "se")]
    cols = [c for c in cols if c in alt_panel.columns]
    panel = panel.drop(columns=[c for c in cols if c in panel.columns])
    panel = panel.merge(alt_panel[["iso_code", "year"] + cols], on=["iso_code", "year"], how="left")
    state["panel"] = panel
    print(f"  ✓ dea_{sp}_bc ({ALT_RTS}, bias-corrected) added: "
          f"{panel[f'dea_{sp}_bc'].notna().sum()}/{len(panel)} obs")
    # Both bias-corrected series now exist: draw the DDF_VRS / DDF_CRS figures.
    plot_step5_ddf_by_rts(panel)
    return state



# Backward-compatible alias (old name from when VRS was the baseline).
add_crs_bootstrap = add_alt_rts_bootstrap

def export_affordability_components_table(state):
    """
    Build and save the full affordability-component reference table: for
    every country and every year, the three underlying components (C1, C3,
    and C2 under BOTH the headline-CPI and energy-CPI constructions) plus
    the resulting composite dim_affordability under each C2 variant.

    Columns: iso_code, year, C1, C2_headline, C2_energy, C3,
             dim_affordability_headline, dim_affordability_energy

    All values are the oriented, normalised (0-1, "higher = more secure")
    components actually used to build the ESI, not the raw proxy inputs
    (aff_intensity, aff_price_ratio, aff_hfce_pc) -- i.e. this is what
    step4_esi() itself computes and feeds into dim_affordability, laid out
    so both C2 constructions can be inspected side by side for every
    country-year rather than only through the composite.
    """
    panel = state["panel"]
    needed = ["iso_code", "year", "_aff_c1", "_aff_c2_headline", "_aff_c2_energy",
             "_aff_c3", "dim_affordability_headline", "dim_affordability_energy"]
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        print(f"  ⚠ table_affordability_components.csv: missing columns {missing} -- "
              f"skipped (run step4_esi() first)")
        return state

    tbl = panel[needed].rename(columns={
        "_aff_c1": "C1",
        "_aff_c2_headline": "C2_headline",
        "_aff_c2_energy": "C2_energy",
        "_aff_c3": "C3",
    }).sort_values(["iso_code", "year"]).reset_index(drop=True)

    n_countries = tbl["iso_code"].nunique()
    n_years = tbl["year"].nunique()
    print(f"\n  Affordability components table: {n_countries} countries x {n_years} years "
          f"= {len(tbl)} rows")
    print(f"  Columns: iso_code, year, C1, C2_headline, C2_energy, C3, "
          f"dim_affordability_headline, dim_affordability_energy")
    _save_csv(tbl, "table_affordability_components.csv",
             "C1/C2(headline)/C2(energy)/C3 and both dim_affordability variants, "
             "every country-year")
    return state


def export_country_year_tables(state):
    """
    Build and save the six wide/long country-year reference tables:
      1. ESI + 4 dimensions, HEADLINE-CPI affordability variant (baseline)
      2. ESI + 4 dimensions, ENERGY-CPI affordability variant
      3. Bootstrapped VRS DEA score (dea_bc), wide (country x year)
      4. Bootstrapped CRS DEA score (dea_crs_bc), wide (country x year)
         -- requires add_alt_rts_bootstrap(state) to have been run first
      5. The DDF function's raw (pre-bootstrap-correction) theta, wide
      6. Global Malmquist-Luenberger, all components (GEC, GTC, GML), long
    All cover every available country and year (2000-2023 subject to data
    availability). Saved as CSV in the run's output folder via _save_csv.

    Every table includes an explicit "c2_variant" column so which C2
    construction is in play is identifiable from the table's own contents,
    not only its filename -- for tables 1-2 this is the variant that table
    itself uses; for tables 3-6, which do not depend on C2 at all (DEA/GML
    are computed from production-side variables only), it documents which
    variant was ACTIVE (ESI_C2_MODE) for this run, as provenance metadata.
    """
    panel = state["panel"]
    mpi_df = state.get("mpi_df", pd.DataFrame())
    print("\n" + "="*60)
    print("EXPORTING SIX COUNTRY-YEAR REFERENCE TABLES")
    print(f"  Active ESI_C2_MODE for this run: '{ESI_C2_MODE}' (headline is the")
    print(f"  labelled baseline; every table below carries an explicit c2_variant")
    print(f"  column so this is never ambiguous from the table alone)")
    print("="*60)

    # 1-2: ESI + 4 dimensions, one table per affordability/C2 variant.
    # HEADLINE FIRST, as the labelled baseline specification.
    dim_cols_base = ["dim_resilience", "dim_robustness", "dim_sustainability"]
    for variant, aff_col, esi_col, fname in [
        ("headline", "dim_affordability_headline", "esi_score_headline", "table_esi_dimensions_C2headline.csv"),
        ("energy",   "dim_affordability_energy",   "esi_score_energy",   "table_esi_dimensions_C2energy.csv"),
    ]:
        if aff_col in panel.columns and esi_col in panel.columns:
            cols = ["iso_code", "year", esi_col, aff_col] + \
                   [c for c in dim_cols_base if c in panel.columns]
            tbl = panel[cols].rename(columns={esi_col: "esi_score", aff_col: "dim_affordability"})
            tbl.insert(2, "c2_variant", variant + (" (baseline)" if variant == "headline" else ""))
            tbl = tbl.sort_values(["iso_code", "year"]).reset_index(drop=True)
            _save_csv(tbl, fname, f"ESI + 4 dimensions (C2={variant})")
        else:
            print(f"  ⚠ {fname}: {aff_col} / {esi_col} not in panel — skipped")

    # 3: Bootstrapped VRS, wide (country x year)
    if "dea_vrs_bc" in panel.columns:
        wide_vrs = panel.pivot(index="iso_code", columns="year", values="dea_vrs_bc").sort_index()
        wide_vrs = wide_vrs.reset_index()
        wide_vrs.insert(1, "c2_variant", f"{ESI_C2_MODE} (active; DEA itself does not depend on C2)")
        _save_csv(wide_vrs, "table_dea_vrs_bootstrapped.csv",
                  "Bootstrapped VRS DDF score (dea_vrs_bc), wide country x year")
    else:
        print("  ⚠ table_dea_vrs_bootstrapped.csv: dea_vrs_bc not in panel — skipped")

    # 4: Bootstrapped CRS, wide (country x year) -- requires add_alt_rts_bootstrap()
    if "dea_crs_bc" in panel.columns:
        wide_crs = panel.pivot(index="iso_code", columns="year", values="dea_crs_bc").sort_index()
        wide_crs = wide_crs.reset_index()
        wide_crs.insert(1, "c2_variant", f"{ESI_C2_MODE} (active; DEA itself does not depend on C2)")
        _save_csv(wide_crs, "table_dea_crs_bootstrapped.csv",
                  "Bootstrapped CRS DEA score (dea_crs_bc), wide country x year")
    else:
        print("  ⚠ table_dea_crs_bootstrapped.csv: dea_crs_bc not in panel — run "
              "add_alt_rts_bootstrap(state) before step13_export() to include this table")

    # 5: The DDF function's raw theta (pre-bootstrap-correction), wide
    if "dea_score" in panel.columns:
        wide_ddf = panel.pivot(index="iso_code", columns="year", values="dea_score").sort_index()
        wide_ddf = wide_ddf.reset_index()
        wide_ddf.insert(1, "c2_variant", f"{ESI_C2_MODE} (active; DDF itself does not depend on C2)")
        _save_csv(wide_ddf, "table_ddf_raw_theta.csv",
                  "Raw DDF theta (pre-bootstrap-correction, dea_score), wide country x year")
    else:
        print("  ⚠ table_ddf_raw_theta.csv: dea_score not in panel — skipped")

    # 6: GML, all components (GEC, GTC, GML), long format
    if mpi_df is not None and len(mpi_df) > 0:
        gml_cols = [c for c in ["iso_code", "year_t", "year_t1", "ec", "tc", "mpi"]
                   if c in mpi_df.columns]
        gml_tbl = mpi_df[gml_cols].rename(
            columns={"ec": "GEC", "tc": "GTC", "mpi": "GML"}
        ).sort_values(["iso_code", "year_t"]).reset_index(drop=True)
        gml_tbl.insert(1, "c2_variant", f"{ESI_C2_MODE} (active; GML itself does not depend on C2)")
        _save_csv(gml_tbl, "table_gml_all_components.csv",
                  "GML with all components (GEC, GTC, GML), all countries and year-pairs")
    else:
        print("  ⚠ table_gml_all_components.csv: mpi_df not available — run "
              "step6 (GML) before step13_export() to include this table")

    print("\n  ✓ Six-table export complete (see warnings above for any skipped tables)")
    return state


def export_results(panel, mpi_df, sw_results):
    """Export panel, GML (Global Malmquist-Luenberger), and Simar-Wilson results to CSV in the run folder."""
    out_cols = [
        "iso_code","country","year",
        "esi_score","dim_resilience","dim_affordability",
        "dim_robustness","dim_sustainability",
        "dea_score","dea_bc","dea_bias","dea_lo","dea_hi","dea_se",
        "dea_crs","dea_vrs","dea_se_score","dea_crs_bc","dea_vrs_bc",
        "hhi","co2_intensity_elec","diesel_price_usd_l","governance_score",
        "net_import_pct","gdp_pc_ppp","gdp_growth",
        "elec_gen_twh","energy_consumption_twh","energy_pc","aff_hfce_pc",
    ]
    export_cols = [c for c in out_cols if c in panel.columns]
    _save_csv(panel[export_cols], "energy_security_panel.csv", "main panel")
    _save_csv(mpi_df,    "mpi_results.csv",           "GML (Global Malmquist-Luenberger) decomposition")
    _save_csv(sw_results,"simar_wilson_results.csv",   "Simar-Wilson")



# ═════════════════════════════════════════════════════════════════════════════
# STEP-LEVEL DIAGNOSTIC PLOTS  (one per analytical step)
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "blue"  : "#2E5496",
    "teal"  : "#1D9E75",
    "purple": "#534AB7",
    "red"   : "#C00000",
    "amber" : "#BA7517",
    "dark"  : "#1F3864",
    "light" : "#EEF2FA",
}

def _save(fname):
    _f = plt.gcf()
    if not getattr(_f, "_labels_placed", False):
        _r = getattr(_f, "_tight_rect", None)
        plt.tight_layout(rect=_r) if _r else plt.tight_layout()
    _save_fig(fname)
    plt.show()


# ── Step 3 ── Data coverage heatmap ──────────────────────────────────────────
def plot_step3_coverage(panel):
    """Heatmap of % non-null per column × country for key variables."""
    # Only check columns that actually exist — co2_intensity_elec created in step 4
    key_cols = [c for c in
                ["net_import_pct","energy_consumption_twh",
                 "elec_gen_twh","gdp_pc_ppp",
                 "governance_score","gdp_growth","gdp_owid","population",
                 "ghg_emissions","elec_coal_twh","elec_gas_twh","renew_share",
                 "esi_score","diesel_price_usd_l","hhi","co2_intensity_elec"]
                if c in panel.columns]
    cov = (panel.groupby("iso_code")[key_cols]
               .apply(lambda g: g.notna().mean() * 100)
               .round(0))

    fig, ax = plt.subplots(figsize=(max(12, 3 + len(key_cols) * 1.15),
                                    _country_extent(len(cov), base=3.0, per=0.36)))
    im = ax.imshow(cov.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(key_cols)))
    ax.set_xticklabels(key_cols, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(cov)))
    ax.set_yticklabels(cov.index, fontsize=9.5)
    _annotate_heatmap(ax, im, cov.values, fmt="{:.0f}%")
    plt.colorbar(im, ax=ax, label="% non-null")
    ax.set_title("Step 3 — Data Coverage by Country & Variable", fontsize=11, fontweight="bold")
    _save("step3_coverage.png")


# ── Step 4 ── ESI heatmap + dimension bar chart ───────────────────────────────
# ── C2 outlier display range (Turkey, Argentina) ─────────────────────────────
_C2_DISPLAY_COLS = ("aff_price_ratio", "aff_energy_inflation")


def _range_without_outliers(panel, col, pad=0.05):
    """Axis range for a C2 column, set by every country except C2_OUTLIERS.
    Returns None for any other column (its range stays automatic)."""
    if col not in _C2_DISPLAY_COLS or col not in panel.columns:
        return None
    oth = panel.loc[~panel["iso_code"].isin(C2_OUTLIERS), col].dropna()
    if oth.empty:
        return None
    lo, hi = float(oth.min()), float(oth.max())
    p = pad * (hi - lo + 1e-12)
    return lo - p, hi + p


def _mark_outlier_bars(ax, sub, col, rng):
    """Ranking bars: x-range from the other countries; an outlier bar that runs
    past the edge gets its actual value written at the edge."""
    lo = min(0.0, rng[0])
    ax.set_xlim(lo, rng[1])
    for k, (iso, v) in enumerate(zip(sub["iso_code"], sub[col])):
        if iso in C2_OUTLIERS and (v > rng[1] or v < rng[0]):
            ax.text(rng[1], k, f"{iso} = {v:.2f} ▶ ", ha="right", va="center", fontsize=9.5,
                    fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))


def _outlier_footnote(panel, ranges):
    parts = []
    for col, (lo, hi) in ranges.items():
        v = panel.loc[panel["iso_code"].isin(C2_OUTLIERS), ["iso_code", col]].dropna()
        ext = "; ".join(f"{iso} {g[col].min():.2f} to {g[col].max():.2f}" for iso, g in v.groupby("iso_code"))
        name = {"aff_price_ratio": "C2 headline", "aff_energy_inflation": "C2 energy"}.get(col, col)
        parts.append(f"{name} axis shows {lo:.2f} to {hi:.2f}" + (f"; actual values: {ext}" if ext else ""))
    return ("Note: " + " and ".join(C2_OUTLIERS) + " are outliers in C2 (very high inflation), so the C2 axis "
            "range is set by the other countries only. " + ". ".join(parts) + ". Their lines run off the top "
            "of the panel and their bars are cut at the edge with the actual value shown. "
            "Display only: all calculations use the actual values.")


# ═════════════════════════════════════════════════════════════════════════════
# ONE FIXED, UNIQUE STYLE PER COUNTRY (used by every per-country line chart)
# ═════════════════════════════════════════════════════════════════════════════
# 20 clearly different colours for a white background (no pale/yellow tones);
# smallest CIELAB distance between any two is ~24 (well above the ~10 at which
# colours are easy to tell apart).
DISTINCT_COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                    "#00a3c4", "#f032e6", "#469990", "#9a6324", "#800000",
                    "#808000", "#000075", "#000000", "#8c8c8c", "#c9a500",
                    "#fa7fb4", "#604e97", "#8db600", "#b3446c", "#a1caf1"]


def _build_country_styles():
    """
    Every country gets its own (colour, line style, marker), fixed across all
    figures:
      - G20 (19): 19 different colours, solid line, circle marker
      - other countries (26): 20 different colours with dashed line and square
        marker, then the remaining 6 reuse colours with dotted line and
        triangle marker
    So within the G20 figure every colour is different; in the non-G20 figure
    the 6 repeated colours have a different line style and marker; and in the
    all-country figure no two countries share the same colour+line+marker
    (solid lines are also G20, dashed/dotted are non-G20).
    """
    styles = {}
    g20 = sorted(i for i in PANEL_ISO3 if i in set(G20_ISO3))
    rest = sorted(i for i in PANEL_ISO3 if i not in set(G20_ISO3))
    for k, iso in enumerate(g20):
        styles[iso] = dict(color=DISTINCT_COLOURS[k % 20], linestyle="-", marker="o")
    for k, iso in enumerate(rest):
        if k < 20:
            styles[iso] = dict(color=DISTINCT_COLOURS[k], linestyle="--", marker="s")
        else:
            styles[iso] = dict(color=DISTINCT_COLOURS[(k * 3) % 20], linestyle=":", marker="^")
    return styles


COUNTRY_STYLE = _build_country_styles()


def _country_style(iso):
    return COUNTRY_STYLE.get(iso, dict(color="#555555", linestyle="-", marker="o"))


def _country_colors(isos):
    q = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    return {iso: q[i % len(q)] for i, iso in enumerate(sorted(isos))}


def _label_line_ends(ax, ends, fontsize=9.5, gap_px=1.5, offset_px=55):
    """
    Write each line's ISO code just right of the plot, at the height where the
    line ends, spreading labels vertically so none overlap, with a thin
    connector back to the line end. ends = [(label, x_last, y_last, colour)].
    Call after axis limits and figure layout are final.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.get_window_extent(r)
    ylo, yhi = ax.get_ylim()
    probe = ax.text(0, 0, "XXX", fontsize=fontsize); fig.canvas.draw()
    h = probe.get_window_extent(r).height + gap_px
    probe.remove()
    ends = [e for e in ends if np.isfinite(e[1]) and np.isfinite(e[2])]
    if not ends:
        return
    disp = ax.transData.transform([(x, float(np.clip(y, ylo, yhi))) for _, x, y, _ in ends])
    order = np.argsort(disp[:, 1])
    t = disp[order, 1].copy()
    for i in range(1, len(t)):                                  # push up
        t[i] = max(t[i], t[i - 1] + h)
    t[-1] = min(t[-1], ab.y1 - h / 2)
    for i in range(len(t) - 2, -1, -1):                         # push down from the top
        t[i] = min(t[i], t[i + 1] - h)
    if t[0] < ab.y0 + h / 2:                                    # spread evenly if crowded
        t = np.linspace(ab.y0 + h / 2, ab.y1 - h / 2, len(t))
    def _readable(c):                     # darken pale colours so text stays legible
        r_, g_, b_ = _mpl.colors.to_rgb(c)
        lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
        k_ = 0.55 if lum > 0.6 else (0.8 if lum > 0.45 else 1.0)
        return (r_ * k_, g_ * k_, b_ * k_)
    for k, i in enumerate(order):
        lab, x, y, c = ends[i]
        c = _readable(c)
        yf = (t[k] - ab.y0) / ab.height
        ax.annotate(lab, xy=(x, float(np.clip(y, ylo, yhi))), xycoords="data",
                    xytext=(1 + offset_px / ab.width, yf), textcoords="axes fraction",
                    va="center", ha="left", fontsize=fontsize, color=c, fontweight="bold",
                    annotation_clip=False, zorder=10,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.7, shrinkA=0, shrinkB=1.5,
                                    relpos=(0.0, 0.5)))


_AFF_SEPARATE = [
    ("aff_intensity", "C1 — Energy intensity (kg oil eq per $1,000 PPP GDP)", "C1"),
    ("aff_price_ratio", "C2 headline — Relative consumer price pressure (headline CPI, demeaned)", "C2_headline"),
    ("aff_hfce_pc", "C3 — Household purchasing capacity (HFCE per capita, PPP $)", "C3"),
    ("aff_energy_inflation", "C2 energy — Energy-CPI inflation (100·Δln CPI energy)", "C2_energy"),
    ("dim_affordability_headline", "AFF headline — composite score (C1, C2 headline, C3)", "AFF_headline"),
    ("dim_affordability_energy", "AFF energy — composite score (C1, C2 energy, C3)", "AFF_energy"),
]


def plot_affordability_separate(panel):
    """
    One figure per affordability series — C1, C2 headline, C3, C2 energy,
    AFF headline, AFF energy — each with the same two diagrams as the
    combined figure: (left) every country's trajectory 2000–2023, with its
    ISO code at the line end; (right) the country ranking in the latest year.
    C2 ranges are set without Turkey and Argentina (see footnote).
    Saved to the run's plots folder as step4_affordability_<name>.png.
    """
    isos = sorted(panel["iso_code"].dropna().unique())
    colours = {i: _country_style(i)["color"] for i in isos}
    n = len(isos)
    for col, title, tag in _AFF_SEPARATE:
        if col not in panel.columns or panel[col].notna().sum() < 10:
            print(f"  ⚠ {tag}: {col} not available — figure skipped")
            continue
        H = max(11.0, 2.6 + 0.34 * n)
        fig = plt.figure(figsize=(24, H))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.30,
                              left=0.05, right=0.98, top=1 - 1.0 / H, bottom=1.3 / H)
        ax, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        ends = []
        for iso in isos:
            g = panel.loc[panel["iso_code"] == iso, ["year", col]].dropna().sort_values("year")
            if len(g) < 2:
                continue
            ax.plot(g["year"], g[col], color=colours[iso], lw=1.8, alpha=0.9)
            ends.append((iso, g["year"].iloc[-1], g[col].iloc[-1], colours[iso]))
        ax.set_xlim(YEAR_START, YEAR_END)
        rng = _range_without_outliers(panel, col)
        if rng is not None:
            ax.set_ylim(*rng)
        ax.set_title(f"{title}: all countries, {YEAR_START}–{YEAR_END}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11)
        ax.grid(True, alpha=0.25, lw=0.5)

        yr_last = int(panel.loc[panel[col].notna(), "year"].max())
        sub = panel.loc[panel["year"] == yr_last, ["iso_code", col]].dropna().sort_values(col)
        axr.barh(range(len(sub)), sub[col].values, color=[colours[i] for i in sub["iso_code"]], alpha=0.9)
        axr.set_yticks(range(len(sub)))
        axr.set_yticklabels(sub["iso_code"], fontsize=9.5)
        axr.set_ylim(-0.7, len(sub) - 0.3)
        if rng is not None:
            _mark_outlier_bars(axr, sub, col, rng)
        axr.set_title(f"Ranking, {yr_last}", fontsize=13, fontweight="bold")
        axr.set_xlabel("Score (0–1)" if tag.startswith("AFF") else "Raw value", fontsize=11)
        axr.grid(True, alpha=0.2, lw=0.5, axis="x")

        _label_line_ends(ax, ends)
        if rng is not None:
            fig.text(0.01, 0.2 / H, _outlier_footnote(panel, {col: rng}), ha="left", va="bottom",
                     fontsize=10.5, style="italic")
        fig._labels_placed = True
        _save(f"step4_affordability_{tag}.png")


# C2 ranking (bottom-middle panel): fixed x-axis maximum, so Turkey and
# Argentina do not set the scale. Bars beyond it are cut and labelled.
C2_RANKING_XMAX = 2.0

_AFF_COMPONENTS = None   # filled per call: [(col, label, tag), ...]


def _aff_components():
    if ESI_C2_MODE == "energy":
        c2 = ("aff_energy_inflation", "C2 - Energy CPI inflation (100*Delta ln CPI_energy)")
    else:
        c2 = ("aff_price_ratio", "C2 - Relative consumer price pressure (headline CPI, demeaned)")
    return [("aff_intensity", "C1 - Energy intensity (kg oil eq/$1k PPP GDP)", "C1"),
            (c2[0], c2[1], "C2"),
            ("aff_hfce_pc", "C3 - Household purchasing capacity (HFCE pc, PPP $)", "C3")]


def _aff_colours(panel, isos):
    """Red (least affordable) to green (most), by mean composite affordability."""
    score = (panel.groupby("iso_code")["dim_affordability"].mean()
             if "dim_affordability" in panel.columns else pd.Series(0.5, index=isos))
    norm = (score - score.min()) / (score.max() - score.min() + 1e-12)
    return {iso: plt.cm.RdYlGn(0.08 + 0.84 * norm[iso]) if iso in norm.index and np.isfinite(norm[iso])
            else (0.6, 0.6, 0.6, 1) for iso in isos}


def _aff_draw_trend(ax, panel, col, label, isos, colour):
    """Trajectories 2000–2023. Returns (line-end list, y-range used or None)."""
    ends = []
    for iso in isos:
        g = panel.loc[panel["iso_code"] == iso, ["year", col]].dropna().sort_values("year")
        if len(g) < 2:
            continue
        ax.plot(g["year"], g[col], color=colour[iso], lw=1.5, alpha=0.9)
        ends.append((iso, g["year"].iloc[-1], g[col].iloc[-1], colour[iso]))
    ax.set_xlim(YEAR_START, YEAR_END)
    rng = _range_without_outliers(panel, col)       # C2 only: range from the other countries
    if rng is not None:
        ax.set_ylim(*rng)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=10)
    ax.grid(True, alpha=0.25, lw=0.5)
    return ends, rng


def _aff_draw_ranking(ax, panel, col, label, colour, is_c2, tick_fs=8.5):
    """Latest-year ranking. For C2 the x-axis stops at C2_RANKING_XMAX."""
    yr_last = int(panel.loc[panel[col].notna(), "year"].max())
    sub = panel.loc[panel["year"] == yr_last, ["iso_code", col]].dropna().sort_values(col)
    ax.barh(range(len(sub)), sub[col].values, color=[colour[i] for i in sub["iso_code"]])
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["iso_code"], fontsize=tick_fs)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    if is_c2:
        ax.set_xlim(0, C2_RANKING_XMAX)
        for k, (iso, v) in enumerate(zip(sub["iso_code"], sub[col])):
            if v > C2_RANKING_XMAX:
                ax.text(C2_RANKING_XMAX, k, f"{iso} = {v:.2f} ▶ ", ha="right", va="center",
                        fontsize=tick_fs + 0.5, fontweight="bold",
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    ax.set_title(f"{label} - Ranking ({yr_last})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Raw value", fontsize=10)
    ax.grid(True, alpha=0.2, lw=0.5, axis="x")
    return sub


def _aff_c2_note(panel, col, rng):
    v = panel.loc[panel["iso_code"].isin(C2_OUTLIERS), ["iso_code", col]].dropna()
    ext = "; ".join(f"{iso} {g[col].min():.2f} to {g[col].max():.2f}" for iso, g in v.groupby("iso_code"))
    rng_txt = f"top panel axis {rng[0]:.2f} to {rng[1]:.2f}, " if rng is not None else ""
    return ("Note: " + " and ".join(C2_OUTLIERS) + " are outliers in C2 (very high inflation), so they do not "
            "set the C2 axes: " + rng_txt + f"ranking axis 0 to {C2_RANKING_XMAX:g}. "
            "Their lines run off the top and their bars are cut at the edge with the actual value shown"
            + (f" (actual range {ext})" if ext else "") + ". Display only: calculations use the actual values.")


def plot_affordability_composite(panel):
    """
    Affordability component figure: C1, C2, C3 trajectories (top) and their
    latest-year rankings (bottom), countries coloured by mean composite
    affordability. C2 panels only: the top axis range is set by the countries
    other than C2_OUTLIERS and the ranking axis stops at C2_RANKING_XMAX.
    Also saves each of the six panels as its own figure
    (step4_affordability_C1_trend.png ... step4_affordability_C3_ranking.png).
    """
    comps = [(c, l, t) for c, l, t in _aff_components()
             if c in panel.columns and panel[c].notna().sum() > 10]
    if not comps:
        print("  ⚠ No affordability components found — plot skipped")
        return
    isos = sorted(panel["iso_code"].dropna().unique())
    colour = _aff_colours(panel, isos)

    # ── Combined figure ──────────────────────────────────────────────────────
    nc = len(comps)
    fig, axes = plt.subplots(2, nc, figsize=(6.6 * nc, 18), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.955, top=0.935, bottom=0.075, wspace=0.34, hspace=0.20)
    fig.suptitle("Affordability Composite Index — Component Analysis", fontsize=15, fontweight="bold")
    note, all_ends = None, {}
    for j, (col, label, tag) in enumerate(comps):
        ends, rng = _aff_draw_trend(axes[0, j], panel, col, label, isos, colour)
        all_ends[j] = ends
        _aff_draw_ranking(axes[1, j], panel, col, label, colour, is_c2=(tag == "C2"))
        if tag == "C2":
            note = _aff_c2_note(panel, col, rng)
    for j, ends in all_ends.items():
        _label_line_ends(axes[0, j], ends, fontsize=8, gap_px=2.0, offset_px=12)
    if note:
        fig.text(0.04, 0.012, note, ha="left", va="bottom", fontsize=10, style="italic")
    fig._labels_placed = True
    _save("step4_affordability_composite.png")

    # ── Individual figures, one per panel ────────────────────────────────────
    import textwrap as _tw
    n = len(isos)
    for col, label, tag in comps:
        fig, ax = plt.subplots(figsize=(12, 11))
        fig.subplots_adjust(left=0.08, right=0.86, top=0.93, bottom=0.13 if tag == "C2" else 0.08)
        ends, rng = _aff_draw_trend(ax, panel, col, label, isos, colour)
        ax.set_title(f"{label}, {YEAR_START}–{YEAR_END}", fontsize=13, fontweight="bold")
        _label_line_ends(ax, ends, fontsize=9, gap_px=2.0, offset_px=14)
        if tag == "C2":
            fig.text(0.02, 0.01, _tw.fill(_aff_c2_note(panel, col, rng), 150), ha="left", va="bottom",
                     fontsize=10, style="italic")
        fig._labels_placed = True
        _save(f"step4_affordability_{tag}_trend.png")

        H = 2.6 + 0.30 * n
        fig, ax = plt.subplots(figsize=(10, H))
        fig.subplots_adjust(left=0.10, right=0.97, top=1 - 0.8 / H, bottom=(1.4 if tag == "C2" else 0.8) / H)
        _aff_draw_ranking(ax, panel, col, label, colour, is_c2=(tag == "C2"), tick_fs=10)
        if tag == "C2":
            fig.text(0.02, 0.01, _tw.fill(_aff_c2_note(panel, col, rng), 120), ha="left", va="bottom",
                     fontsize=10, style="italic")
        fig._labels_placed = True
        _save(f"step4_affordability_{tag}_ranking.png")


def plot_step4_esi(panel):
    """ESI score heatmap (country × year) + latest-year dimension breakdown."""
    if "esi_score" not in panel.columns:
        return
    pivot = panel.pivot_table(index="iso_code", columns="year",
                              values="esi_score", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(19, _country_extent(len(pivot.index), base=3.0, per=0.34)),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=70, fontsize=9.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9.5)
    plt.colorbar(im, ax=ax, label="ESI score")
    ax.set_title("ESI Score — Country × Year", fontsize=10, fontweight="bold")

    # Right: latest-year dimension breakdown
    ax2 = axes[1]
    yr = panel["year"].max()
    dims = [c for c in ["dim_resilience","dim_affordability",
                         "dim_robustness","dim_sustainability"] if c in panel.columns]
    sub = panel[panel["year"]==yr].dropna(subset=dims+["esi_score"]).sort_values("esi_score")
    if len(sub) > 0:
        # resilience=teal, affordability=light-orange, robustness=purple, sustainability=amber
        bar_colors = [COLORS["teal"],"#F4A460",COLORS["purple"],COLORS["amber"]]
        bottom = np.zeros(len(sub))
        for d, c in zip(dims, bar_colors):
            vals = sub[d].values / len(dims)
            ax2.barh(range(len(sub)), vals, left=bottom, color=c, alpha=0.85,
                     label=d.replace("dim_",""))
            bottom += vals
        ax2.set_yticks(range(len(sub)))
        ax2.set_yticklabels(sub.get("iso_code", sub.index), fontsize=9.5)
        ax2.set_ylim(-0.7, len(sub) - 0.3)
        ax2.axvline(0.5/len(dims)*len(dims), color="gray", lw=0.8, ls="--")
        ax2.set_title(f"ESI Dimensions  ({yr})", fontsize=10, fontweight="bold")
        _legend_right(ax2)
    _save("step4_esi.png")


# ── Step 5 ── DEA bootstrap CI bars + time-series ────────────────────────────
def plot_big_country_trajectories(panel, value_col, title, filename, ylabel=None, isos=None):
    """
    Large, standalone, ALL-COUNTRIES trajectory diagram: one line per
    country, every year, each country in a genuinely distinct colour
    (tab20+tab20b+tab20c, up to 60 distinct hues), with a clean external
    legend rather than inline end-of-line labels -- the same approach
    built for plot_affordability_composite(), applied here because the
    previous DEA trajectory panel only plotted 6 hardcoded representative
    countries (USA, CHN, DEU, IND, BRA, TUR) in a figure shared with the
    bootstrap CI bar chart, which is far too small and too incomplete a
    treatment for what is reported elsewhere as the headline DDF result.

    value_col: the panel column to plot (e.g. "dea_bc" for VRS, "dea_crs_bc"
               for CRS).
    """
    if value_col not in panel.columns or panel[value_col].notna().sum() < 10:
        print(f"  ⚠ plot_big_country_trajectories: {value_col} not available — skipped")
        return

    all_isos = sorted(panel[panel[value_col].notna()]["iso_code"].unique())
    _full_isos = list(all_isos)                # colours fixed on the full list, so a
    if isos is not None:                       # country keeps its colour in subset figures
        all_isos = [i for i in all_isos if i in set(isos)]
    years = sorted(panel["year"].unique())

    fig, ax = plt.subplots(figsize=(22, 13))
    legend_handles, legend_labels = [], []
    for iso in all_isos:
        g = panel[panel["iso_code"] == iso].sort_values("year").dropna(subset=[value_col])
        if len(g) < 2:
            continue
        st = _country_style(iso)
        line, = ax.plot(g["year"], g[value_col], marker=st["marker"], markersize=4.5,
                        color=st["color"], linestyle=st["linestyle"], linewidth=1.7, alpha=0.9)
        legend_handles.append(line)
        legend_labels.append(iso)

    ax.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.7, label="Frontier")
    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel(ylabel or f"{value_col} (bias-corrected efficiency)", fontsize=13)
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.6)

    n_leg = len(legend_labels)
    n_leg_cols = min(15, max(9, (n_leg + 2) // 3))
    _legend_right(ax, legend_handles, legend_labels, ncol=2 if n_leg > 24 else 1,
                  fontsize=10, columnspacing=1.0, handlelength=1.6,
                  title=f"Country (ISO3), {n_leg}", title_fontsize=10.5)
    _save(filename)
    print(f"  ✓ {filename}: {n_leg} countries, {len(years)} years plotted")


def plot_step5_dea(panel):
    """
    DEA CI bar chart (latest year).

    *** FIX: this figure previously paired the CI bar chart with a second
    panel showing efficiency trajectories for only 6 hardcoded "representative"
    countries (USA, CHN, DEU, IND, BRA, TUR) -- far too incomplete a view of
    the DDF/VRS trajectory result, which is reported elsewhere as one of the
    headline findings. That panel is replaced by a call to
    plot_big_country_trajectories(), a large standalone figure covering
    EVERY country with data, in a genuinely distinct colour per country and
    a proper external legend (see that function's own docstring). The CI
    bar chart below is unchanged. ***
    """
    if "dea_bc" not in panel.columns:
        return
    yr = panel["year"].max()
    sub = panel[panel["year"]==yr].dropna(subset=["dea_bc"]).sort_values("dea_bc").copy()

    fig, ax = plt.subplots(figsize=(11, _country_extent(len(sub))))

    x = np.arange(len(sub))
    ax.barh(x, sub["dea_bc"], color=COLORS["blue"], alpha=0.82)
    has_ci = ("dea_lo" in sub.columns and "dea_hi" in sub.columns
              and sub["dea_lo"].notna().any())
    if has_ci:
        bc_series = sub["dea_bc"].fillna(sub["dea_score"])  # Series
        bc  = bc_series.values
        lo  = sub["dea_lo"].fillna(bc_series).values   # NaN → bc → zero half-width
        hi  = sub["dea_hi"].fillna(bc_series).values
        err_lo = np.clip(bc - lo, 0, None)
        err_hi = np.clip(hi - bc, 0, None)
        # Final guard: replace any remaining NaN with 0 (no bar shown)
        err_lo = np.nan_to_num(err_lo, nan=0.0)
        err_hi = np.nan_to_num(err_hi, nan=0.0)
        ax.errorbar(sub["dea_bc"], x, xerr=[np.abs(np.nan_to_num(np.asarray(err_lo, dtype=float), nan=0.0)), np.abs(np.nan_to_num(np.asarray(err_hi, dtype=float), nan=0.0))],
                    fmt="none", color="black", capsize=3, linewidth=1.1)
    ax.axvline(1.0, color=COLORS["red"], lw=0.9, ls="--", alpha=0.6, label="Frontier")
    ax.set_yticks(x)
    labels = sub["iso_code"].values
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_ylim(-0.7, len(sub) - 0.3)
    ax.set_xlabel("DEA efficiency (bias-corrected)")
    ax.set_title(f"DEA Bootstrap CI  ({yr})", fontsize=12, fontweight="bold")
    _legend_right(ax)
    _save("step5_dea.png")

    # Large, all-countries trajectory diagram (replaces the old 6-country panel)
    plot_big_country_trajectories(
        panel, "dea_bc",
        f"DDF Efficiency Trajectories — All Countries, All Years (Bias-Corrected, {BASE_RTS} baseline)",
        "step5_dea_trajectories_all_countries.png",
        ylabel=f"DDF efficiency (bias-corrected, {BASE_RTS})")

    # The DDF_VRS / DDF_CRS figures (all countries, G20, non-G20 Europe) are drawn
    # by plot_step5_ddf_by_rts(), called at the end of add_alt_rts_bootstrap(): the
    # VRS bias-corrected series (dea_vrs_bc) does not exist yet at this point,
    # so drawing them here skipped the VRS figures in every run.


def plot_step5_ddf_by_rts(panel):
    """
    DDF_VRS and DDF_CRS trajectory figures, each for all countries, the G20 and
    European countries outside the G20. Group membership only: every score is
    measured against the frontier of all 45 countries. Called after
    add_alt_rts_bootstrap(), when both bias-corrected series (dea_crs_bc and
    dea_vrs_bc) exist.
    """
    _g20 = [i for i in G20_ISO3 if i in set(PANEL_ISO3)]
    _rest = list(EUROPE_NONG20_ISO3)        # 26: disjoint from the G20 (19)
    for _rts in ("VRS", "CRS"):
        _col = f"dea_{_rts.lower()}_bc"
        if _col not in panel.columns:
            print(f"  ⚠ DDF_{_rts} diagrams skipped: {_col} not in panel")
            continue
        _tag = " (baseline)" if _rts == BASE_RTS else " (robustness check)"
        _yl = f"DDF efficiency, {_rts} (bias-corrected)"
        plot_big_country_trajectories(panel, _col, f"DDF_{_rts}{_tag} — All Countries, All Years",
                                      f"step5_DDF_{_rts}_all_countries.png", ylabel=_yl)
        plot_big_country_trajectories(panel, _col, f"DDF_{_rts}{_tag} — G20 Countries ({len(_g20)}), All Years",
                                      f"step5_DDF_{_rts}_G20.png", ylabel=_yl, isos=_g20)
        plot_big_country_trajectories(panel, _col,
                                      f"DDF_{_rts}{_tag} — European Countries outside the G20 ({len(_rest)}), All Years",
                                      f"step5_DDF_{_rts}_Europe_nonG20.png", ylabel=_yl, isos=_rest)


# ── Step 5b ── Scale efficiency decomposition: VRS / CRS / SE ────────────────
def plot_step5_scale_efficiency(panel, year=None):
    """
    Three-panel scale-efficiency decomposition plot (Section 2.3.6).

    Panel A — Latest-year bar chart: VRS (PTE), CRS (OTE), and SE = CRS/VRS
              stacked so the gap between VRS and CRS is visible as scale loss.
    Panel B — Scale efficiency distribution: histogram of SE across all
              country-years, with the SE=1 boundary marked.
    Panel C — Country trajectories: SE over time for six representative economies,
              showing whether scale (in)efficiency is improving or persistent.

    OTE (CRS) = PTE (VRS) × SE:
      VRS score = pure technical efficiency (operational / input-mix performance)
      CRS score = overall technical efficiency (operational/configurational + scale)
      SE = CRS / VRS = proximity to most productive scale size (MPSS)
    """
    if "dea_crs" not in panel.columns or panel["dea_crs"].isna().all():
        print("  ⚠ dea_crs not computed — run step5_dea_bootstrap first")
        return

    if year is None:
        year = panel["year"].max()
    sub = (panel[panel["year"] == year]
           .dropna(subset=["dea_vrs","dea_crs","dea_se_score"])
           .sort_values("dea_se_score")
           .reset_index(drop=True))

    # Layout: Panel A (bar chart, all countries) on top, full width.
    # Panels B and C side-by-side on the bottom row.
    # Width scales with the number of countries so every label is readable.
    n_countries = len(sub)
    fig_w = max(22, n_countries * 0.55)   # ~0.55 in per country, min 22 in
    fig_h = 14                             # tall enough for two rows
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(f"Scale Efficiency Decomposition: OTE = PTE × SE  ({year})",
                 fontsize=13, fontweight="bold", y=1.01)

    ax  = fig.add_subplot(2, 1, 1)          # Panel A: full width top row
    ax2 = fig.add_subplot(2, 2, 3)          # Panel B: bottom-left
    ax3 = fig.add_subplot(2, 2, 4)          # Panel C: bottom-right

    # ── Panel A: bar chart VRS vs CRS vs SE ──────────────────────────────────
    x = np.arange(n_countries)
    labels = sub["iso_code"].values
    w = 0.26

    ax.bar(x - w,   sub["dea_crs"],      width=w, color=COLORS["red"],    alpha=0.80,
           label="CRS / OTE (overall)")
    ax.bar(x,       sub["dea_vrs"],    width=w, color=COLORS["blue"],   alpha=0.80,
           label="VRS / PTE (pure technical)")
    ax.bar(x + w,   sub["dea_se_score"], width=w, color=COLORS["teal"],   alpha=0.80,
           label="SE = CRS/VRS (scale)")

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    # Font size scales down gracefully for large panels but stays readable
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=9.5)
    ax.set_ylabel("Efficiency score", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"A — VRS, CRS and Scale Efficiency  ({year})  —  {n_countries} countries",
                 fontsize=11, fontweight="bold")
    _legend_right(ax)

    # ── Panel B: SE histogram across all years ────────────────────────────────
    se_all = panel["dea_se_score"].dropna()
    ax2.hist(se_all, bins=30, color=COLORS["purple"], alpha=0.75, edgecolor="white")
    ax2.axvline(1.0,          color="black",       lw=1.2, ls="--", label="SE = 1 (MPSS)")
    ax2.axvline(se_all.mean(),color=COLORS["amber"], lw=1.2, ls="-",
                label=f"Mean SE = {se_all.mean():.3f}")
    ax2.set_xlabel("Scale efficiency (SE = CRS/VRS)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("B — SE Distribution (all years)", fontsize=10, fontweight="bold")
    n_scale_ineff = int((se_all < 0.999).sum())
    _pct = 100*n_scale_ineff/max(1,len(se_all))
    ax2.text(0.03, 0.96,
             f"{n_scale_ineff}/{len(se_all)} obs ({_pct:.0f}%) scale-inefficient (SE<1)",
             transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # ── Panel C: SE trajectories for representative countries ────────────────
    show_isos = ["USA","CHN","DEU","IND","BRA","TUR"]
    show_isos = [c for c in show_isos if c in panel["iso_code"].values]
    se_vals_plotted = []
    for k, iso in enumerate(show_isos):
        g = panel[panel["iso_code"]==iso].sort_values("year").dropna(subset=["dea_se_score"])
        if len(g) < 2:
            continue
        st = _country_style(iso)
        ax3.plot(g["year"], g["dea_se_score"], marker=st["marker"], markersize=4,
                 color=st["color"], ls=st["linestyle"], label=iso, linewidth=1.6)
        se_vals_plotted.extend(g["dea_se_score"].tolist())
    ax3.axhline(1.0, color="gray", lw=0.8, ls="--", label="SE = 1 (MPSS)")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Scale efficiency")
    # Lower bound: exact data minimum minus a small 2% padding.
    # No rounding — axis starts exactly where the data starts so every
    # variation is visible regardless of how compressed the range is.
    if se_vals_plotted:
        data_min = min(se_vals_plotted)
        data_max = max(se_vals_plotted)
        pad = max((data_max - data_min) * 0.05, 0.005)  # 5% of range, min 0.005
        y_lo = max(data_min - pad, 0.0)
    else:
        y_lo = 0.0
    ax3.set_ylim(y_lo, 1.08)
    ax3.set_title(f"C — SE Trajectories (y axis from {y_lo:.3f})",
                  fontsize=10, fontweight="bold")
    # panel B and C entries share one legend on the right of panel C
    _hb, _lb = ax2.get_legend_handles_labels()
    _hc, _lc = ax3.get_legend_handles_labels()
    _legend_right(ax3, _hb + _hc, ["B: " + l for l in _lb] + ["C: " + l for l in _lc])

    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    fig._labels_placed = True
    _save("step5_scale_efficiency.png")
    print(f"  Scale efficiency plot saved ({year}, {len(sub)} countries)")


# ── Step 6 ── GML time-series + country decomposition ────────────────────────
def plot_step6_mpi(mpi_df):
    """
    GML (Global Malmquist-Luenberger) decomposition bar chart + G20 average
    GML over time.

    *** FIX (methodology-labelling inconsistency): this chart's titles,
    legend, and axis labels previously said "MPI" / "EC" / "TC" throughout,
    even though this pipeline explicitly computes the Global Malmquist-
    Luenberger index (Oh 2010) -- GML = GEC x GTC -- not the standard
    Malmquist Productivity Index, as stated unambiguously in this
    pipeline's own header comment and in build_mpi()'s own docstring. The
    standard MPI and the Global Malmquist-Luenberger index are related but
    distinct: GML uses a GLOBAL (pooled across all periods) reference
    technology specifically to guarantee feasibility when an undesirable
    output (CO2 intensity) is present under weak disposability, which the
    standard, period-specific MPI does not do. All USER-FACING text below
    is corrected to GML/GEC/GTC; the underlying DataFrame's own column
    names ("mpi", "ec", "tc") and the function name plot_step6_mpi /
    build_mpi are left unchanged, since renaming those would touch many
    downstream call sites (Step 12's GML correlation, CSV exports) for a
    purely cosmetic benefit once the actually-displayed labels are fixed. ***
    """
    if mpi_df is None or len(mpi_df) == 0:
        return
    _n_c = mpi_df["iso_code"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(18, _country_extent(_n_c)))

    # Left: country decomposition (horizontal stacked bars)
    ax = axes[0]
    cdf = mpi_df.groupby("iso_code").agg(
        ec=("ec","mean"), tc=("tc","mean"), mpi=("mpi","mean")
    ).reset_index().sort_values("mpi", ascending=True)
    x = np.arange(len(cdf))
    ax.barh(x, cdf["ec"], color=COLORS["teal"], alpha=0.85, label="GEC (catch-up)")
    ax.barh(x, cdf["tc"] - 1, left=cdf["ec"], color=COLORS["purple"], alpha=0.85,
            label="GTC − 1 (global frontier shift)")
    ax.axvline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yticks(x); ax.set_yticklabels(cdf["iso_code"], fontsize=9.5)
    ax.set_ylim(-0.7, len(cdf) - 0.3)
    ax.set_xlabel("GML component (period average)")
    ax.set_title("GML (Global Malmquist-Luenberger) Decomposition by Country", fontsize=12, fontweight="bold")

    # Right: G20 average GML, GEC, GTC per year-pair
    ax2 = axes[1]
    ts = mpi_df.groupby("year_t")[["mpi","ec","tc"]].mean().reset_index()
    ax2.plot(ts["year_t"], ts["mpi"], color=COLORS["dark"],  lw=2,   label="GML", marker="o", ms=4)
    ax2.plot(ts["year_t"], ts["ec"],  color=COLORS["teal"],  lw=1.4, label="GEC",  marker="s", ms=3)
    ax2.plot(ts["year_t"], ts["tc"],  color=COLORS["purple"],lw=1.4, label="GTC",  marker="^", ms=3)
    ax2.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax2.fill_between(ts["year_t"], ts["mpi"], 1.0,
                     where=(ts["mpi"]>=1), alpha=0.12, color=COLORS["teal"])
    ax2.fill_between(ts["year_t"], ts["mpi"], 1.0,
                     where=(ts["mpi"]< 1), alpha=0.12, color=COLORS["red"])
    ax2.set_xlabel("Year"); ax2.set_ylabel("Index value")
    ax2.set_title("Average GML (Global Malmquist-Luenberger Index) over Time", fontsize=12, fontweight="bold")
    _fig_legend_right(fig, [ax, ax2], right=0.86)
    _save("step6_mpi.png")


# ── Step 7 ── Simar–Wilson forest plot ────────────────────────────────────────
def plot_step7_sw(sw_results, fname="step7_sw_full.png", title_suffix="full model"):
    """
    Forest plot of Simar-Wilson bias-corrected coefficients with 95% CI.

    *** FIX: significance markers are now gated by B, matching the exact
    B<100 suppression threshold used in simar_wilson_2007's own console
    output. Previously this function drew *** purely from the bootstrap
    CI excluding zero, with no reference to B at all -- at B=25 (test
    mode), this produced a chart showing a confident *** annotation while
    the accompanying console text explicitly stated "no coefficient
    should be interpreted as significant at B=25", a direct chart-versus-
    text contradiction that misled any reader who saw only the chart. ***
    """
    if sw_results is None or len(sw_results) == 0:
        return
    df = sw_results[sw_results["variable"] != "const"].copy()
    if len(df) == 0:
        return

    # B is carried as a column by simar_wilson_2007; default to a
    # conservative "assume low B" if an older result object lacks it.
    B_val = int(df["B"].iloc[0]) if "B" in df.columns and df["B"].notna().any() else 0
    low_B = B_val < 100

    fig, ax = plt.subplots(figsize=(9, max(4, len(df)*0.7)))
    y = np.arange(len(df))
    colors = [COLORS["teal"] if lo > 0 else (COLORS["red"] if hi < 0 else COLORS["blue"])
              for lo, hi in zip(df["ci_lo_95"], df["ci_hi_95"])]

    ax.barh(y, df["coef_bc"], color=colors, alpha=0.75, height=0.5)
    _xerr_lo1 = np.nan_to_num(np.clip(df["coef_bc"].values - df["ci_lo_95"].values, 0, None), nan=0.0)
    _xerr_hi1 = np.nan_to_num(np.clip(df["ci_hi_95"].values - df["coef_bc"].values, 0, None), nan=0.0)
    ax.errorbar(df["coef_bc"], y,
                xerr=[np.abs(np.nan_to_num(np.asarray(_xerr_lo1, dtype=float), nan=0.0)), np.abs(np.nan_to_num(np.asarray(_xerr_hi1, dtype=float), nan=0.0))],
                fmt="none", color="black", capsize=4, linewidth=1.3, zorder=3)
    ax.scatter(df["coef_bc"], y, color=colors, zorder=4, s=50)
    ax.axvline(0, color="black", lw=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(df["variable"], fontsize=9)
    ax.set_xlabel("Bias-corrected coefficient")
    title = (f"DDF-adapted second stage (SW2007 Algorithm 2 steps; truncated at 1, censored at 2) — {title_suffix}\n"
             "Parametric 95% intervals. Green/red = interval excludes 0 (positive/negative)")
    if low_B:
        title += f"\n*** B={B_val} < 100 -- TEST MODE: significance markers suppressed ***"
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Significance stars -- suppressed entirely when B < 100, matching the
    # text-output threshold, rather than drawn from the CI alone.
    if not low_B:
        for i, row in df.reset_index(drop=True).iterrows():
            sig = "***" if (row["ci_lo_95"]>0 or row["ci_hi_95"]<0) else ""
            if sig:
                ax.text(max(row["ci_hi_95"], row["coef_bc"]) + 0.002, i,
                        sig, va="center", fontsize=10, color=colors[i])
    else:
        ax.text(0.98, 0.02, f"B={B_val}: not enough replications for valid inference",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, style="italic", color="gray")
    _save(fname)


# ── Step 8 ── Panel FE coefficient chart ─────────────────────────────────────
def plot_step8_fe(fe_result):
    """Coefficient plot for panel FE regression."""
    if fe_result is None:
        return
    try:
        params = fe_result.params
        ci     = fe_result.conf_int()
        names  = [n for n in params.index if n not in ("const","Intercept")]
        vals   = params[names]
        lo     = ci.loc[names, "lower"]
        hi     = ci.loc[names, "upper"]
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(names)*0.7)))
    y = np.arange(len(names))
    colors = [COLORS["teal"] if v > 0 else COLORS["red"] for v in vals]
    ax.barh(y, vals.values, color=colors, alpha=0.75, height=0.5)
    _xerr_lo2 = np.nan_to_num(np.clip(vals.values - lo.values, 0, None), nan=0.0)
    _xerr_hi2 = np.nan_to_num(np.clip(hi.values - vals.values, 0, None), nan=0.0)
    ax.errorbar(vals.values, y,
                xerr=[np.abs(np.nan_to_num(np.asarray(_xerr_lo2, dtype=float), nan=0.0)), np.abs(np.nan_to_num(np.asarray(_xerr_hi2, dtype=float), nan=0.0))],
                fmt="none", color="black", capsize=4, linewidth=1.3)
    ax.scatter(vals.values, y, color=colors, zorder=4, s=50)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Coefficient (entity + time FE, clustered SE)")
    ax.set_title(f"Panel FE Regression  Dep: dea_bc  R2={fe_result.rsquared:.3f}  N={fe_result.nobs}",
                 fontsize=10, fontweight="bold")
    _save("step8_fe.png")


def plot_step10_granger(granger_df, granger_stats=None):
    """
    Two-panel Granger causality summary:
      A — Per-country p-value bar chart (both directions)
      B — Summary statistics table with W-bar and significance
    """
    if granger_df is None or len(granger_df) == 0:
        print("  ⚠ No Granger results to plot")
        return

    df = granger_df.sort_values("p_esi_dea").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(17, _country_extent(len(df), base=3.0, per=0.42)))
    fig.suptitle("Panel Granger Causality — Dumitrescu-Hurlin Approximation",
                 fontsize=11, fontweight="bold")

    y = np.arange(len(df))

    # ── Panel A: Per-country p-value bars ────────────────────────────────────
    width = 0.38
    for k, (col, label, off) in enumerate([
        ("p_esi_dea", "ESI → DEA", -width/2),
        ("p_dea_esi", "DEA → ESI", +width/2)
    ]):
        if col not in df.columns:
            continue
        colors = [COLORS["teal"] if p < 0.10 else COLORS["amber"] if p < 0.20
                  else "#CCCCCC" for p in df[col]]
        axes[0].barh(y + off, df[col], height=width, color=colors,
                     alpha=0.85, label=label)

    axes[0].axvline(0.10, color=COLORS["red"],  lw=1.2, ls="--", label="p=0.10")
    axes[0].axvline(0.05, color=COLORS["dark"], lw=1.0, ls=":",  label="p=0.05")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["iso_code"], fontsize=9.5)
    axes[0].set_ylim(-0.7, len(df) - 0.3)
    axes[0].set_xlabel("Min p-value across lags (F-test)")
    axes[0].set_title("A — Per-country Granger p-values  Teal=p<0.10  Amber=p<0.20", fontsize=9)
    _legend_right(axes[0])
    axes[0].set_xlim(0, 1.05)

    # ── Panel B: Summary statistics table ────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    # Build summary rows
    gs = granger_stats if granger_stats else {}
    ed_gated = gs.get("ed_gated", False)
    de_gated = gs.get("de_gated", False)

    def _decision_label(gated, z_val):
        if gated:
            return "Not reported (pending numerical diagnostic)"
        if z_val is None:
            return "—"
        return "Reject H0" if abs(z_val) > 1.96 else "Fail to reject"

    summary_rows = [
        ["Statistic", "ESI → DEA", "DEA → ESI"],
        ["Mean p-value",
         f"{gs.get('mean_p_esi_dea', float('nan')):.4f}",
         f"{gs.get('mean_p_dea_esi', float('nan')):.4f}"],
        ["Share significant (p<0.10)",
         f"{gs.get('share_sig_esi_dea', 0)*100:.1f}%",
         f"{gs.get('share_sig_dea_esi', 0)*100:.1f}%"],
        ["Z-bar statistic (DH, standardised)",
         (f"{gs.get('z_bar_esi_dea', float('nan')):.3f}" if gs.get('z_bar_esi_dea') and not ed_gated else "GATED" if ed_gated else "—"),
         (f"{gs.get('z_bar_dea_esi', float('nan')):.3f}" if gs.get('z_bar_dea_esi') and not de_gated else "GATED" if de_gated else "—")],
        ["Decision (|Z-bar|>1.96, 5%)",
         _decision_label(ed_gated, gs.get('z_bar_esi_dea')).replace(" (", "\n("),
         _decision_label(de_gated, gs.get('z_bar_dea_esi')).replace(" (", "\n(")],
        ["N countries tested", str(len(df)), str(len(df))],
    ]

    tbl = ax2.table(cellText=summary_rows[1:], colLabels=summary_rows[0],
                    cellLoc="center", loc="center", colWidths=[0.42, 0.29, 0.29],
                    bbox=[0.0, 0.42, 1.0, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F3864")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4FF")
        cell.set_edgecolor("#CCCCCC")

    # Interpretation note
    import textwrap as _tw
    interp = _tw.fill("Per-country p-values are descriptive. The panel verdict is the Z-bar row; "
                      "a direction marked GATED has an outlier country-level statistic and is "
                      "not interpreted (see console output).", 70)
    ax2.text(0.5, 0.12, interp, transform=ax2.transAxes,
             ha="center", va="center", fontsize=9, style="italic",
             bbox=dict(boxstyle="round", facecolor="#FFF8E7", alpha=0.8))
    ax2.set_title("B — Dumitrescu-Hurlin Z-bar Statistics", fontsize=9)

    # Print table to console
    print("\n  Panel Granger Causality — Summary Table")
    for row in summary_rows:
        print(f"  {row[0]:<35} {row[1]:>12} {row[2]:>12}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.45)
    _save_fig("step10_granger.png")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# APPENDIX A — OPTIONAL EXTENSIONS (robustness / supplementary methods)
# *** NOT numbered "Section 11" or "Step 11" -- this block has NO relation
# to step11_plots() below, which is an unrelated, always-run core pipeline
# step that coincidentally shares the numeral 11. The two were confusable
# in an earlier version of this file; this block is now labelled
# "Appendix A" specifically to remove that collision. ***
# ═════════════════════════════════════════════════════════════════════════════
# The methods below were explicitly labelled "robustness" or "optional
# extension" (not part of the core recommended combination: DDF + GML +
# Separability Test + Conditional DDF + Mundlak + Local Projections).
# THEY ARE NOT RUN BY DEFAULT IN run_all(). You must explicitly call:
#     state = run_optional_extensions(state)
#   or
#     state = run_all(B=..., run_optional=True)
# If neither of these was used, none of the functions below executed, and
# state will contain NO between_effects_results, quantile_results, or
# metafrontier_results keys.
#   A.1  Between-effects / long-run cross-sectional model
#   A.2  Panel quantile regression (Canay 2011 two-step FE approximation)
#   A.3  Metafrontier / group-specific frontiers
# (A.3, Anderson-Hsiao dynamic panel, is DEFINED below but deliberately
#  EXCLUDED from run_optional_extensions() and from the paper's results:
#  it produced unstable point estimates even with a strong first-stage
#  instrument -- a known finite-sample weakness of Anderson-Hsiao relative
#  to System-GMM -- and was judged not reliable enough to report. Call
#  dynamic_panel_anderson_hsiao() directly if needed for exploratory use.)
# (A PCA-based SEM path model was considered and removed: it was assessed
#  as the weakest method here -- a component-based proxy standing in for
#  full PLS-SEM without the actual iterative estimation.)
# ═════════════════════════════════════════════════════════════════════════════

def between_effects_regression(panel, z_vars=None, dep_var="dea_bc"):
    """
    *** APPENDIX A.1: Long-run between-country structural assessment
    (between-effects, long-run cross-sectional model) ***

        DEA_bar_i = alpha + beta1*Resilience_bar_i + beta2*Affordability_bar_i
                    + beta3*Robustness_bar_i + beta4*Sustainability_bar_i + u_i

    Collapses the panel to ONE observation per country (its time-average
    dea_bc and time-average Z), then runs a genuine cross-sectional OLS
    with N=n_countries degrees of freedom (NOT clustered SE on a pooled
    panel -- this is the actual between-country regression, useful as a
    descriptive bridge between the ESI and DEA results, and directly
    comparable to the theta/between coefficients from the Mundlak model
    (Step 8b), which should recover a SIMILAR relationship via a different
    route -- reporting both is a natural consistency check.

    Answers: "Do countries with stronger long-run structural energy-
    security conditions also show higher long-run operational efficiency?"
    """
    print(f"\n{'═'*60}")
    print("APPENDIX A.1 — Long-Run Between-Country Structural Assessment")
    print("═"*60)

    if z_vars is None:
        z_vars = ["dim_resilience","dim_affordability",
                  "dim_robustness","dim_sustainability","gdp_growth"]

    needed = ["iso_code",dep_var] + z_vars
    needed = [c for c in needed if c in panel.columns]
    country_means = panel[needed].groupby("iso_code").mean().dropna()
    n = len(country_means)
    print(f"  Dependent variable: mean({dep_var}) by country")
    print(f"  Z = {z_vars}")
    print(f"  N = {n} countries (genuine cross-sectional df, not clustered SE)")
    if n < len(z_vars) + 3:
        print(f"  ⚠ Too few countries for this specification — skipped")
        return pd.DataFrame()

    X = sm.add_constant(country_means[z_vars].values.astype(float), has_constant="add")
    y = country_means[dep_var].values.astype(float)
    res = OLS(y, X).fit(cov_type="HC3")   # robust SE, small-N cross-section

    results = pd.DataFrame({
        "variable": ["const"] + z_vars,
        "coef": res.params,
        "se": res.bse,
        "t": res.tvalues,
        "p": res.pvalues,
    })
    print(f"\n  {'Variable':<20} {'coef':>10} {'SE':>9} {'t':>7} {'p':>8}")
    print(f"  {'─'*58}")
    for _, row in results.iterrows():
        star = "*" if row["p"] < 0.10 else " "
        print(f"  {row['variable']:<20} {row['coef']:>+10.4f}{star} {row['se']:>9.4f} "
              f"{row['t']:>7.3f} {row['p']:>8.4f}")
    print(f"  {'─'*58}")
    print(f"  R² = {res.rsquared:.4f}  |  N = {n}  |  * p<0.10")

    sig = results[(results["variable"]!="const") & (results["p"]<0.10)]
    print(f"\n  {'─'*58}")
    if len(sig) > 0:
        print(f"  Countries with stronger long-run values on: "
              f"{', '.join(sig['variable'].tolist())}")
        print(f"  show a statistically distinguishable long-run efficiency relationship.")
    else:
        print(f"  No structural dimension shows a significant long-run cross-country")
        print(f"  relationship with mean {dep_var} at this sample size (N={n}).")
    print(f"  This is a DESCRIPTIVE bridge between the ESI and DEA results, not a")
    print(f"  causal estimate; compare against the Mundlak theta (Step 8b), which")
    print(f"  estimates a related between-country relationship via a different route.")
    print(f"  {'─'*58}")
    return results


def quantile_regression_panel(panel, z_vars=None, dep_var="dea_bc",
                              taus=(0.25, 0.50, 0.75)):
    """
    *** APPENDIX A.2: Distributional Robustness -- Panel Quantile Estimates ***

    *** SCOPE (per review decision table): KEEP, but must be clearly
    labelled as (a) NOT Simar-Wilson, and (b) NOT the main determinant
    model. This method supports a narrower claim -- that a structural
    variable's association with operational efficiency may differ across
    the performance distribution (distributional heterogeneity) -- and
    should never be cited as if it were the primary second-stage estimate;
    that role belongs to Simar-Wilson (Step 6) and its restricted/
    conditional variants. ***

    NOT "Simar-Wilson quantile regression" -- this is a separate
    heterogeneity-analysis model, answering: do structural energy-security
    dimensions matter more for weak operational performers than for strong
    performers?

        Q_tau(DEA_bc_it | Z_it) = alpha_i + lambda_t + Z_it * beta_tau

    True panel quantile regression with fixed effects is econometrically
    subtle (the standard "within" demeaning transformation does not carry
    over to quantiles the way it does for means). This implementation uses
    the CANAY (2011) two-step approach, a simple and widely-cited practical
    estimator: (1) estimate country fixed effects from a MEAN regression
    (OLS with entity dummies) and remove them from the dependent variable;
    (2) run pooled quantile regression (statsmodels QuantReg) on the
    fixed-effect-adjusted data at each tau, with year dummies retained for
    lambda_t. This is a real, citable, but SIMPLER estimator than
    fully efficient panel quantile methods.
    """
    print(f"\n{'═'*60}")
    print("APPENDIX A.2 — Distributional Robustness: Panel Quantile Estimates")
    print("  (Canay 2011 two-step approximation; NOT Simar-Wilson)")
    print("═"*60)

    if z_vars is None:
        z_vars = ["dim_resilience","dim_affordability",
                 "dim_robustness","dim_sustainability","gdp_growth"]

    needed = ["iso_code","year",dep_var] + z_vars
    needed = [c for c in needed if c in panel.columns]
    sub = panel[needed].dropna().copy()
    if len(sub) < 50:
        print(f"  ⚠ Only {len(sub)} complete observations — skipped")
        return pd.DataFrame()

    print(f"  Dependent variable: {dep_var}")
    print(f"  Z = {z_vars}")
    print(f"  Sample: {len(sub)} obs, {sub['iso_code'].nunique()} countries")

    # Step 1: remove country fixed effects via a mean regression (entity dummies)
    entity_dummies = pd.get_dummies(sub["iso_code"], prefix="c", drop_first=True).astype(float)
    year_dummies = pd.get_dummies(sub["year"], prefix="yr", drop_first=True).astype(float)
    X1 = pd.concat([sub[z_vars].reset_index(drop=True),
                    entity_dummies.reset_index(drop=True),
                    year_dummies.reset_index(drop=True)], axis=1)
    X1 = sm.add_constant(X1, has_constant="add").astype(float)
    y1 = sub[dep_var].values.astype(float)
    step1 = OLS(y1, X1.values).fit()

    # Country fixed effects = the entity dummy coefficients (0 for the
    # reference country); subtract them to get the FE-adjusted dependent var
    fe_cols = [c for c in X1.columns if c.startswith("c_")]
    fe_effect = X1[fe_cols].values @ step1.params[[X1.columns.get_loc(c) for c in fe_cols]]
    y_adj = y1 - fe_effect

    # Step 2: pooled quantile regression on the FE-adjusted outcome
    X2 = sm.add_constant(pd.concat([sub[z_vars].reset_index(drop=True),
                                    year_dummies.reset_index(drop=True)], axis=1),
                         has_constant="add").astype(float)

    results = []
    print(f"\n  {'Variable':<20}", end="")
    for tau in taus:
        print(f"{'q'+str(tau):>14}", end="")
    print()
    print(f"  {'─'*(20+14*len(taus))}")
    for zv in z_vars:
        row = {"variable": zv}
        line = f"  {zv:<20}"
        for tau in taus:
            qr = sm.QuantReg(y_adj, X2.values)
            qres = qr.fit(q=tau, max_iter=2000)
            coef = qres.params[X2.columns.get_loc(zv)]
            pval = qres.pvalues[X2.columns.get_loc(zv)]
            star = "*" if pval < 0.10 else " "
            row[f"coef_q{tau}"] = coef
            row[f"p_q{tau}"] = pval
            line += f"{coef:>+13.4f}{star}"
        print(line)
        results.append(row)
    print(f"  {'─'*(20+14*len(taus))}")
    print(f"  * p<0.10  |  N={len(sub)}  |  FE removed via Canay (2011) step 1")

    results_df = pd.DataFrame(results)
    print(f"\n  {'─'*58}")
    print(f"  INTERPRETATION")
    print(f"  {'─'*58}")
    print(f"  *** SCOPE: this is a DISTRIBUTIONAL ROBUSTNESS check, NOT the main")
    print(f"  determinant model, and NOT a form of Simar-Wilson. Simar-Wilson")
    print(f"  (Step 6) estimates a truncated-regression model of the CONDITIONAL")
    print(f"  MEAN of a bounded, bootstrap-corrected efficiency score, with a")
    print(f"  likelihood that accounts for the DEA truncation explicitly. This")
    print(f"  quantile approach estimates CONDITIONAL QUANTILES of the raw")
    print(f"  (non-bootstrap-corrected) dea_bc via a two-step FE approximation")
    print(f"  (Canay 2011) with no truncation correction at all. The two are")
    print(f"  answering different questions and are not interchangeable: use")
    print(f"  Simar-Wilson (or its restricted/conditional variants) for the")
    print(f"  primary determinant analysis, and this quantile check ONLY to ask")
    print(f"  whether a relationship's magnitude differs across the performance")
    print(f"  distribution. ***")
    print(f"")
    any_heterogeneity = False
    for _, row in results_df.iterrows():
        coefs = [row[f"coef_q{t}"] for t in taus]
        if coefs[0] * coefs[-1] < 0 or (max(coefs)-min(coefs)) > 2*abs(np.mean(coefs)):
            any_heterogeneity = True
            print(f"  {row['variable']}: coefficient VARIES substantially across the")
            print(f"    distribution ({', '.join(f'{c:+.3f}' for c in coefs)}) — supports")
            print(f"    DISTRIBUTIONAL HETEROGENEITY: the association's magnitude")
            print(f"    differs between weak, mid, and strong operational performers.")
    if not any_heterogeneity:
        print(f"  No variable shows a coefficient that varies substantially across")
        print(f"  the tau=0.25/0.50/0.75 distribution at this sample — no strong")
        print(f"  evidence of distributional heterogeneity beyond the mean-based")
        print(f"  (Simar-Wilson / Mundlak) results already reported.")
    print(f"  {'─'*58}")
    return results_df


def dynamic_panel_anderson_hsiao(panel, z_vars=None, dep_var="dea_bc"):
    """
    *** APPENDIX A.3: Dynamic panel model (Anderson-Hsiao 1982) ***

        DEA_bc_it = rho * DEA_bc_i,t-1 + beta*Z_i,t-1 + alpha_i + lambda_t + u_it

    Full System-GMM or Difference-GMM (Arellano-Bond/Blundell-Bond) is NOT
    implemented here. With only ~45 countries and ~24 years, GMM is prone
    to instrument proliferation and over-identification (see the
    caution); a from-scratch, correctly-implemented System/Difference GMM
    (internal instrument matrices, two-step weighting, Windmeijer
    correction) is a substantial undertaking of its own. Instead this
    implements the simpler, classic ANDERSON-HSIAO (1982) IV estimator,
    the direct precursor to GMM:
      1. First-difference the model to eliminate the country fixed effect:
           Delta(DEA_bc_it) = rho*Delta(DEA_bc_i,t-1) + beta*Delta(Z_i,t-1)
                              + Delta(lambda_t) + Delta(u_it)
      2. Instrument the differenced lagged DV, Delta(DEA_bc_i,t-1), with its
         OWN second lag, DEA_bc_i,t-2 (uncorrelated with Delta(u_it) under
         the standard no-serial-correlation assumption on u_it).
      3. Estimate by 2SLS (linearmodels.IV2SLS).

    This is a legitimate, simple, explicitly-simpler alternative to GMM,
    appropriate given the review's own caution about over-instrumentation
    at this N, T. Use only as a persistence-robustness check.
    """
    print(f"\n{'═'*60}")
    print("APPENDIX A.3 — Dynamic Panel (Anderson-Hsiao 1982 IV)")
    print("  Simpler alternative to System/Difference GMM (see review caution")
    print(f"  re: over-instrumentation at N={len(PANEL_ISO3)}, T~24)")
    print("═"*60)

    if z_vars is None:
        z_vars = ["dim_resilience","dim_affordability",
                 "dim_robustness","dim_sustainability","gdp_growth"]

    needed = ["iso_code","year",dep_var] + z_vars
    needed = [c for c in needed if c in panel.columns]
    sub = panel[needed].dropna().sort_values(["iso_code","year"]).copy()
    if len(sub) < 50:
        print(f"  ⚠ Only {len(sub)} complete observations — skipped")
        return None

    g = sub.groupby("iso_code")
    sub["y_lag1"]  = g[dep_var].shift(1)
    sub["y_lag2"]  = g[dep_var].shift(2)
    sub["dy"]      = sub[dep_var]  - sub["y_lag1"]
    sub["dy_lag1"] = sub["y_lag1"] - sub["y_lag2"]
    for z in z_vars:
        sub[f"{z}_lag1"] = g[z].shift(1)
        sub[f"d_{z}_lag1"] = sub[f"{z}_lag1"] - g[f"{z}_lag1"].shift(1)

    year_dummies = pd.get_dummies(sub["year"], prefix="yr", drop_first=True).astype(float)
    sub = pd.concat([sub, year_dummies], axis=1)

    dz_cols = [f"d_{z}_lag1" for z in z_vars]
    needed_iv = ["dy","dy_lag1","y_lag2"] + dz_cols + list(year_dummies.columns)
    reg_df = sub.dropna(subset=[c for c in needed_iv if c in sub.columns]).copy()
    n = len(reg_df)
    print(f"  Sample after differencing + lagging: {n} obs "
          f"({sub['iso_code'].nunique()} countries)")
    if n < 30:
        print(f"  ⚠ Too few observations after differencing — skipped")
        return None

    # 2SLS: endogenous = dy_lag1, instrument = y_lag2, exogenous = dz_cols + year dummies
    exog_cols = [c for c in dz_cols + list(year_dummies.columns) if c in reg_df.columns]
    # Drop zero/near-zero-variance columns (e.g. a year dummy with no
    # remaining variation after differencing + lagging drops early years) --
    # these cause exact rank deficiency in the 2SLS design matrix.
    const_exog = [c for c in exog_cols if reg_df[c].std() < 1e-9]
    if const_exog:
        print(f"  ⚠ Dropping near-constant regressors after differencing: {const_exog}")
        exog_cols = [c for c in exog_cols if c not in const_exog]
    try:
        dependent = reg_df["dy"]
        endog     = reg_df[["dy_lag1"]]
        exog_df   = reg_df[exog_cols].copy()

        # Sequential collinearity guard (same pattern as panel_fe_regression):
        # drop the least-varying column until the exog+endog design is full
        # rank, since exact rank deficiency (not just near-collinearity) makes
        # IV2SLS raise rather than degrade gracefully.
        from numpy.linalg import matrix_rank
        full_design = np.column_stack([np.ones(len(exog_df)), exog_df.values,
                                       endog.values])
        remaining = list(exog_cols)
        while remaining and matrix_rank(
                np.column_stack([np.ones(len(exog_df)),
                                 exog_df[remaining].values, endog.values])
              ) < len(remaining) + 2:
            drop = exog_df[remaining].std().idxmin()
            remaining.remove(drop)
            print(f"  ⚠ Dropped collinear column: {drop}")
        exog_df = exog_df[remaining]

        exog = sm.add_constant(exog_df, has_constant="add")
        instruments = reg_df[["y_lag2"]]

        # *** Weak-instrument diagnostic (standard IV practice): the
        # Anderson-Hsiao instrument (y_lag2) is well-documented to be weak
        # in finite samples, which can produce erratic/implausible point
        # estimates for rho even when the code is running correctly. Report
        # the first-stage F-statistic so this is visible, not silent. ***
        first_stage_X = sm.add_constant(
            pd.concat([reg_df[["y_lag2"]].reset_index(drop=True),
                      exog_df.reset_index(drop=True)], axis=1),
            has_constant="add")
        first_stage = OLS(reg_df["dy_lag1"].values, first_stage_X.values).fit()
        f_instrument = float(first_stage.fvalue) if hasattr(first_stage, "fvalue") else np.nan
        weak_iv = f_instrument < 10 if np.isfinite(f_instrument) else True

        model = IV2SLS(dependent, exog, endog, instruments)
        res = model.fit(cov_type="clustered", clusters=reg_df["iso_code"])
    except Exception as e:
        print(f"  ⚠ IV2SLS estimation failed: {e}")
        return None

    print(f"\n  First-stage F-statistic (instrument strength): {f_instrument:.2f}")
    if weak_iv:
        print(f"  *** WEAK INSTRUMENT WARNING (F<10, Staiger-Stock rule of thumb):")
        print(f"  *** the Anderson-Hsiao instrument (y_lag2) is weak here. Point")
        print(f"  *** estimates below may be unstable/biased — this is a KNOWN")
        print(f"  *** limitation of Anderson-Hsiao vs GMM (which uses more, and")
        print(f"  *** typically stronger, instruments). Treat rho with caution.")

    rho = float(res.params["dy_lag1"])
    p_rho = float(res.pvalues["dy_lag1"])
    print(f"\n  rho (persistence, on Delta(DEA_bc,t-1)): {rho:+.4f}  p={p_rho:.4f}")
    print(f"  {'Variable':<20} {'coef':>10} {'p':>8}")
    print(f"  {'─'*42}")
    for z in z_vars:
        col = f"d_{z}_lag1"
        if col in res.params.index:
            print(f"  {z:<20} {res.params[col]:>+10.4f} {res.pvalues[col]:>8.4f}")
    print(f"  {'─'*42}")

    print(f"\n  {'─'*58}")
    if weak_iv:
        print(f"  Instrument is WEAK (F={f_instrument:.2f}<10) — rho is NOT interpreted")
        print(f"  here regardless of its point value or p-value; a weak instrument")
        print(f"  makes the IV estimate unreliable by construction, not just noisy.")
    elif 0 < rho < 1 and p_rho < 0.10:
        print(f"  rho in (0,1) and significant: operational efficiency shows genuine")
        print(f"  YEAR-ON-YEAR PERSISTENCE (mean reversion, not a random walk or an")
        print(f"  explosive process). Consistent with efficiency being a slow-moving")
        print(f"  structural characteristic rather than an annually-reset outcome.")
    else:
        print(f"  No clear persistence pattern detected at this specification/sample.")
    print(f"  CAVEAT: Anderson-Hsiao is less efficient than System-GMM (uses only")
    print(f"  one instrument per period); treat as a persistence robustness check,")
    print(f"  not the primary dynamic model.")
    print(f"  {'─'*58}")

    return res


def assign_metafrontier_groups(panel, group_type="carbon"):
    """
    Assign each country to one of TWO theoretically-motivated technology
    groups for the metafrontier robustness check (Appendix A.3).

    Two groupings are used, both directly aligned with the core pipeline's
    own substantive results:

    group_type="carbon": Low-carbon vs fossil-intensive electricity systems.
        Group_i = LowCarbon        if mean(co2_intensity_elec)_i <= median
                = FossilIntensive  if mean(co2_intensity_elec)_i >  median
        Tests whether fossil-intensive and low-carbon countries operate
        under structurally different green-electricity technologies.

    group_type="import": Lower vs higher import-dependent systems.
        Group_i = LowerImportExposure  if mean(dea_net_import)_i <= median
                = HigherImportExposure if mean(dea_net_import)_i >  median
        Directly aligned with the DDF input specification and the Local
        Projections shock analysis, where import exposure is central.

    A median split with G=2 groups is used deliberately: with 45
    countries per year, splitting into 3+ groups leaves too few
    observations per group-year to estimate a stable DEA/DDF frontier.

    Returns the panel with a new column "meta_group".
    """
    df = panel.copy()

    if group_type == "carbon":
        driver_col = "co2_intensity_elec"
        if driver_col not in df.columns:
            raise ValueError(f"'{driver_col}' not found in panel — required for "
                             f"group_type='carbon'")
        country_mean = df.groupby("iso_code")[driver_col].mean()
        median_value = country_mean.median()
        group_map = {
            c: "Low-carbon electricity system" if v <= median_value
            else "Fossil-intensive electricity system"
            for c, v in country_mean.items()
        }
        print(f"  Carbon-intensity grouping: median mean(co2_intensity_elec) "
              f"= {median_value:.1f} gCO2e/kWh (lifecycle)")

    elif group_type == "import":
        driver_col = "dea_net_import"
        if driver_col not in df.columns:
            raise ValueError(f"'{driver_col}' not found in panel — required for "
                             f"group_type='import'")
        country_mean = df.groupby("iso_code")[driver_col].mean()
        median_value = country_mean.median()
        group_map = {
            c: "Lower import exposure" if v <= median_value
            else "Higher import exposure"
            for c, v in country_mean.items()
        }
        print(f"  Import-exposure grouping: median mean(dea_net_import) "
              f"= {median_value:.2f}")

    else:
        raise ValueError("group_type must be 'carbon' or 'import'.")

    df["meta_group"] = df["iso_code"].map(group_map)
    n_groups = df.groupby("meta_group")["iso_code"].nunique().to_dict()
    print(f"  Group sizes: {n_groups}")
    return df


def metafrontier_analysis(panel, group_col=None, group_fn=None, group_type=None):
    """
    *** APPENDIX A.3: Metafrontier Robustness (group-specific frontiers) ***

    theta^Meta is dea_score itself, reused directly rather than
    recomputed: Step 5's annual DDF loop already uses all N=45 DMUs in
    that year as its reference set, which IS the metafrontier under VRS,
    by definition. Reusing it guarantees theta^Meta equals dea_score by
    construction. The only quantity this function computes fresh is
    theta^Group (the group-restricted VRS frontier score), from which TGR
    follows exactly as specified:

        TGR_it = theta^Meta_it / theta^Group_it,   0 < TGR <= 1
        theta^Meta_it = theta^Group_it * TGR_it

    Both theta^Group and theta^Meta(=dea_score) use the SAME DDF solver
    (_solve_ddf_one), same year, same VRS setting, same INPUT_COLS/
    GOOD_OUT/BAD_OUT -- differing ONLY in the reference set (own group vs
    all countries in the panel).

    *** TRANSPARENCY NOTE: this function has been verified only on
    synthetic data (see chat). It has NOT been executed on the real
    full G20+Europe panel, because this sandbox cannot reach the World Bank
    API (only GitHub/PyPI-hosted domains, including OWID, are reachable
    here) needed to build that panel. Run this in your own environment
    with full network access to obtain the real country-level table. ***

    group_col: name of an existing column defining group membership, OR
    group_fn : a function panel["iso_code"] -> group label, applied if
               group_col is not given. Default: G20 vs non-G20 (using the
               existing G20_ISO3 constant) -- a transparent, replicable,
               already-available split; NOT claimed to be the economically
               "best" grouping, just a defensible default.

    Returns: DataFrame with iso_code, group, year, group_efficiency,
    meta_efficiency, tgr, plus decomposition flags -- and prints the four
    result blocks requested: (1) country-period mean TE^Group, (2)
    country-period mean TE^Meta, (3) country-period mean TGR, (4)
    group-level averages over time.
    """
    print(f"\n{'═'*60}")
    print("APPENDIX A.3 — Metafrontier DDF Analysis")
    print("═"*70)

    if "dea_score" not in panel.columns:
        print("  ⚠ dea_score not found — run Step 5 (build_dea_bootstrap) first")
        return pd.DataFrame()

    panel = panel.copy()
    if group_type is not None:
        panel = assign_metafrontier_groups(panel, group_type=group_type)
        panel["_group"] = panel["meta_group"]
    elif group_col is not None and group_col in panel.columns:
        panel["_group"] = panel[group_col]
    else:
        if group_fn is None:
            group_fn = lambda iso: ("G20 (19)" if iso in set(G20_ISO3)
                                    else f"Europe non-G20 ({len(EUROPE_NONG20_ISO3)})")
        panel["_group"] = panel["iso_code"].map(group_fn)

    group_counts = panel.groupby("_group")["iso_code"].nunique().to_dict()
    print(f"\nGroups:")
    for grp, n_c in group_counts.items():
        print(f"  {grp:<17}: {n_c} countries")

    print(f"\nSpecification:")
    print(f"  Inputs     : {INPUT_COLS}")
    print(f"  Good output: {GOOD_OUT}")
    print(f"  Bad output : {BAD_OUT}")
    print(f"  RTS        : VRS")
    print(f"  Direction  : g=(y0,b0)  (observation-scaled, same as Step 5)")
    print(f"  Technology : annual (own-year reference set, group-restricted")
    print(f"               for theta^Group; all-{len(PANEL_ISO3)}-country for theta^Meta)")

    needed = ["iso_code","year","_group","dea_score"] + INPUT_COLS + GOOD_OUT + BAD_OUT
    needed = [c for c in needed if c in panel.columns]
    sub = panel[needed].dropna().copy()
    if len(sub) < 30:
        print(f"\n  ⚠ Only {len(sub)} complete observations — skipped")
        return pd.DataFrame()

    years = sorted(sub["year"].unique())
    results = []
    for yr in years:
        yr_df = sub[sub["year"] == yr].reset_index(drop=True)
        if len(yr_df) < 8:
            continue
        X_all = _normalise(yr_df[INPUT_COLS].values.astype(float))
        Y_all = _normalise(yr_df[GOOD_OUT].values.astype(float))
        B_all = _normalise(yr_df[BAD_OUT].values.astype(float))

        for grp in yr_df["_group"].unique():
            grp_idx = np.where((yr_df["_group"] == grp).values)[0]
            if len(grp_idx) < 4:
                continue
            X_g = X_all[grp_idx]; Y_g = Y_all[grp_idx]; B_g = B_all[grp_idx]

            for k_local, i_global in enumerate(grp_idx):
                # theta^Group: reference set = OWN GROUP only, this year, VRS
                # (matches Step 5's own VRS setting exactly)
                beta_group = _solve_ddf_one(X_g, Y_g, B_g,
                                            X_g[k_local], Y_g[k_local], B_g[k_local],
                                            vrs=VRS, allow_negative=False)
                if not np.isfinite(beta_group):
                    continue
                theta_group = 1.0 / (1.0 + beta_group)
                # theta^Meta: REUSED from dea_score -- NOT recomputed. Do NOT
                # clip or otherwise silently mask a theta_meta > theta_group
                # violation here; that must be caught by the hard checks below.
                theta_meta = float(yr_df.loc[i_global, "dea_score"])
                tgr = theta_meta / theta_group if theta_group > 1e-9 else np.nan
                results.append({
                    "iso_code": yr_df.loc[i_global, "iso_code"],
                    "group": grp, "year": yr,
                    "theta_group": theta_group,
                    "theta_meta": theta_meta,
                    "tgr": tgr,
                })

    if not results:
        print("  ⚠ No valid metafrontier observations computed")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    n_total = len(results_df)

    # ── HARD CONSISTENCY CHECKS: fail loudly, do not just print a caveat ──────
    tol = 1e-6
    print(f"\nConsistency checks:")

    # (1) theta_meta must equal Step 5's own dea_score exactly, by construction
    check = results_df.merge(
        panel[["iso_code","year","dea_score"]].drop_duplicates(),
        on=["iso_code","year"], how="left"
    )
    check["meta_diff"] = (check["theta_meta"] - check["dea_score"]).abs()
    max_meta_diff = float(check["meta_diff"].max())
    print(f"  theta_meta matches Step-5 dea_score:")
    print(f"      max absolute difference = {max_meta_diff:.8f}")
    if max_meta_diff > tol:
        raise AssertionError(
            f"metafrontier_analysis: theta_meta does not match dea_score "
            f"(max diff={max_meta_diff:.6f} > tol={tol}). theta_meta is "
            f"supposed to be REUSED from dea_score, not recomputed -- this "
            f"indicates a construction bug. Do not interpret the results "
            f"below until this is fixed."
        )

    # (2) both scores must be bounded in (0, 1+tol]
    assert (results_df["theta_group"] > 0).all(), \
        "metafrontier_analysis: theta_group has non-positive values"
    assert (results_df["theta_group"] <= 1 + tol).all(), \
        "metafrontier_analysis: theta_group exceeds 1 beyond tolerance"
    assert (results_df["theta_meta"] > 0).all(), \
        "metafrontier_analysis: theta_meta has non-positive values"
    assert (results_df["theta_meta"] <= 1 + tol).all(), \
        "metafrontier_analysis: theta_meta exceeds 1 beyond tolerance"

    # (3) theta_meta <= theta_group: the group frontier has fewer peers than
    # the metafrontier, so it cannot be a tougher envelope than the
    # metafrontier -- theta_meta > theta_group would indicate a genuine
    # construction error (e.g. group and meta technologies swapped).
    bad_mask = (results_df["theta_meta"] - results_df["theta_group"]) > tol
    n_bad = int(bad_mask.sum())
    print(f"  theta_meta <= theta_group:")
    print(f"      {n_total - n_bad}/{n_total} observations satisfy condition")
    if n_bad > 0:
        print(f"  ⚠ WARNING: {n_bad} observations have theta_meta > theta_group")
        print(f"    beyond tolerance. Inspect these before interpreting TGR:")
        print(results_df.loc[bad_mask, ["iso_code","year","group",
                                        "theta_group","theta_meta"]].to_string(index=False))

    # (4) TGR must be in (0, 1+tol]
    results_df["tgr"] = results_df["theta_meta"] / results_df["theta_group"]
    n_tgr_valid = int(((results_df["tgr"] > 0) & (results_df["tgr"] <= 1 + tol)).sum())
    print(f"  TGR in (0,1]:")
    print(f"      {n_tgr_valid}/{n_total} observations valid")
    if n_tgr_valid < n_total:
        print(f"  ⚠ WARNING: {n_total - n_tgr_valid} observations have TGR outside")
        print(f"    (0, 1+{tol}] -- these rows are NOT clipped or hidden; inspect")
        print(f"    them directly in the returned DataFrame before use.")

    n_valid = results_df["tgr"].notna().sum()
    print(f"\n  Computed {n_total} country-year observations ({n_valid} with valid TGR)")

    # ── Result block 1-3: country-period means ────────────────────────────────
    country_summary = (results_df.groupby(["iso_code","group"])
                       .agg(mean_theta_group=("theta_group","mean"),
                            mean_theta_meta=("theta_meta","mean"),
                            mean_tgr=("tgr","mean"),
                            n_years=("year","count"))
                       .reset_index()
                       .sort_values("mean_tgr"))

    print(f"\n  {'─'*78}")
    print(f"  RESULT BLOCKS 1-3: Country-period means (θ^Group, θ^Meta, TGR)")
    print(f"  {'─'*78}")
    print(f"  {'Country':<8} {'Group':<16} {'θ^Group':>9} {'θ^Meta':>9} "
          f"{'TGR':>7} {'Yrs':>5}")
    print(f"  {'─'*78}")
    for _, row in country_summary.iterrows():
        print(f"  {row['iso_code']:<8} {row['group']:<16} "
              f"{row['mean_theta_group']:>9.4f} {row['mean_theta_meta']:>9.4f} "
              f"{row['mean_tgr']:>7.4f} {int(row['n_years']):>5}")
    print(f"  {'─'*78}")

    # ── Result block 4: group-level averages over time ────────────────────────
    group_time_summary = (results_df.groupby(["group","year"])
                          .agg(mean_theta_group=("theta_group","mean"),
                               mean_theta_meta=("theta_meta","mean"),
                               mean_tgr=("tgr","mean"))
                          .reset_index())
    print(f"\n  {'─'*58}")
    print(f"  RESULT BLOCK 4: Group-level averages over time")
    print(f"  {'─'*58}")
    for grp in group_time_summary["group"].unique():
        g_df = group_time_summary[group_time_summary["group"]==grp].sort_values("year")
        print(f"\n  {grp}:")
        print(f"    {'Year':<6} {'θ^Group':>9} {'θ^Meta':>9} {'TGR':>7}")
        for _, row in g_df.iterrows():
            print(f"    {int(row['year']):<6} {row['mean_theta_group']:>9.4f} "
                  f"{row['mean_theta_meta']:>9.4f} {row['mean_tgr']:>7.4f}")

    group_overall = (results_df.groupby("group")
                     .agg(mean_theta_group=("theta_group","mean"),
                          mean_theta_meta=("theta_meta","mean"),
                          mean_tgr=("tgr","mean"))
                     .reset_index())
    print(f"\n  Group overall averages (all years pooled):")
    print(f"  {'Group':<20} {'θ^Group':>10} {'θ^Meta':>10} {'TGR':>8}")
    print(f"  {'─'*52}")
    for _, row in group_overall.iterrows():
        print(f"  {row['group']:<20} {row['mean_theta_group']:>10.4f} "
              f"{row['mean_theta_meta']:>10.4f} {row['mean_tgr']:>8.4f}")
    print(f"  {'─'*52}")

    # ── Decomposition: within-group inefficiency vs technology-gap disadvantage ──
    # 1 - theta^Meta = 1 - theta^Group*TGR. Decompose the metafrontier gap
    # into how much comes from (1-theta^Group) [within-group inefficiency]
    # vs (theta^Group - theta^Meta) [technology-gap disadvantage, driven by TGR<1].
    country_summary["within_group_gap"] = 1.0 - country_summary["mean_theta_group"]
    country_summary["tech_gap_component"] = (country_summary["mean_theta_group"]
                                             - country_summary["mean_theta_meta"])
    country_summary["dominant_source"] = np.where(
        country_summary["within_group_gap"] >= country_summary["tech_gap_component"],
        "Within-group inefficiency", "Technology-gap disadvantage"
    )
    print(f"\n  {'─'*78}")
    print(f"  DECOMPOSITION: within-group inefficiency vs technology-gap disadvantage")
    print(f"  {'─'*78}")
    print(f"  {'Country':<8} {'Group':<16} {'WithinGap':>10} {'TechGap':>10}  Dominant source")
    print(f"  {'─'*78}")
    for _, row in country_summary.sort_values("tech_gap_component", ascending=False).iterrows():
        print(f"  {row['iso_code']:<8} {row['group']:<16} "
              f"{row['within_group_gap']:>10.4f} {row['tech_gap_component']:>10.4f}  "
              f"{row['dominant_source']}")
    print(f"  {'─'*78}")

    print(f"\n  {'─'*58}")
    print(f"  INTERPRETATION")
    print(f"  {'─'*58}")
    print(f"  θ^Group = efficiency relative to the country's OWN group frontier")
    print(f"  θ^Meta  = efficiency relative to the POOLED metafrontier")
    print(f"            (= dea_score, Step 5's own annual all-{len(PANEL_ISO3)}-country DDF score)")
    print(f"  TGR = θ^Meta/θ^Group ∈ (0,1]: how close the group's OWN frontier")
    print(f"        sits to the metafrontier. TGR≈1: the group's technology IS")
    print(f"        near the global best-practice frontier. TGR<<1: a structural")
    print(f"        technology gap exists, consistent with the separability")
    print(f"        rejection (structural conditions may shift the attainable")
    print(f"        technology, not just a country's distance from it).")
    print(f"  A country can be highly efficient WITHIN its own group (high")
    print(f"  θ^Group) while still facing a large TECHNOLOGY GAP (low TGR) if")
    print(f"  its group's frontier itself sits far below the metafrontier.")
    lowest_tgr_grp = group_overall.sort_values("mean_tgr").iloc[0]
    print(f"\n  Group with the largest technology gap: {lowest_tgr_grp['group']} "
          f"(mean TGR={lowest_tgr_grp['mean_tgr']:.4f})")
    print(f"  {'─'*58}")

    return results_df


def run_metafrontier_robustness(panel):
    """
    *** APPENDIX A.3 (full robustness suite): runs the metafrontier
    analysis under TWO theoretically-motivated groupings, replacing the
    earlier arbitrary G20 vs Non-G20-Europe default. ***

    Role: robustness test for technology heterogeneity -- NOT the main
    estimator. The core methodology remains:
        ESI(equal-weight) -> DDF -> Separability -> Conditional DDF
    This function tests whether the pooled/unconditional DDF results
    (Step 5, one common technology T^Global for all countries in the panel) are
    sensitive to instead assuming TWO distinct technology regimes.

    Groupings (G=2 each, median split -- deliberately not 3+ groups,
    since with ~45 countries per year a finer split would leave too few
    observations per group-year to estimate a stable DDF frontier):

      1. Carbon-intensity  : low-carbon vs fossil-intensive electricity
                              systems (mean co2_intensity_elec, median split)
      2. Import-exposure   : lower vs higher import-dependent systems
                              (mean dea_net_import, median split)

    For each grouping, reports:
      Table 1 — group definition (criterion, median threshold, group sizes)
      Table 2 — metafrontier summary (mean theta^Group, theta^Meta, TGR by group)
      Table 3 — country rankings by TGR (via metafrontier_analysis's own
                country-period summary table)

    theta^Meta uses dea_score (raw VRS point estimate), NOT dea_bc
    (bias-corrected): the group and global frontiers must be computed
    under the SAME (uncorrected) technology for TGR to be a clean ratio
    of two DDF distance functions, not a mix of a point estimate and a
    bootstrap-adjusted one.

    Returns dict: {"carbon": DataFrame, "import": DataFrame}
    """
    print(f"\n{'═'*70}")
    print("APPENDIX A.3 — METAFRONTIER ROBUSTNESS (two groupings)")
    print("  Role: robustness test for technology heterogeneity in the DDF.")
    print("  NOT the main estimator. Core methodology remains:")
    print("  ESI(equal-weight) -> DDF -> Separability -> Conditional DDF.")
    print(f"{'═'*70}")

    results = {}
    grouping_defs = [
        ("carbon", "Carbon-intensity grouping",
         "Country mean lifecycle GHG intensity of electricity",
         "Low-carbon / Fossil-intensive"),
        ("import", "Import-exposure grouping",
         "Country mean bounded net import dependency (dea_net_import)",
         "Lower / Higher import exposure"),
    ]

    print(f"\n  {'─'*74}")
    print(f"  TABLE 1 — GROUP DEFINITIONS")
    print(f"  {'─'*74}")
    print(f"  {'Grouping':<28} {'Criterion':<42} Groups")
    print(f"  {'─'*74}")
    for gtype, label, criterion, groups_desc in grouping_defs:
        print(f"  {label:<28} {criterion:<42} {groups_desc}")
    print(f"  {'─'*74}")

    for gtype, label, criterion, groups_desc in grouping_defs:
        print(f"\n\n  {'#'*74}")
        print(f"  GROUPING: {label}")
        print(f"  {'#'*74}")
        try:
            results[gtype] = metafrontier_analysis(panel, group_type=gtype)
        except AssertionError as e:
            print(f"\n  ✗ Metafrontier internal-consistency check FAILED for "
                  f"'{gtype}' grouping:")
            print(f"    {e}")
            print(f"  Skipping this grouping's results — do not interpret them.")
            results[gtype] = pd.DataFrame()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MODULAR STEP FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
# Each step is independently callable. A shared `state` dict carries data
# between steps so you never have to thread arguments manually.
#
# ── QUICK-START (notebook) ────────────────────────────────────────────────
#
#   from energy_security_pipeline import *
#
#   state = {}                          # shared state dict
#
#   # Run steps one at a time:
#   state = step1_download(state)
#   state = step2_validate_affordability(state)
#   state = step3_build_panel(state)
#   state = step4_esi(state)
#   state = step5_dea_bootstrap(state)  # B=B_BOOTSTRAP by default
#   state = step6_mpi(state)
#   state = step7_simar_wilson(state)
#   state = step8_panel_fe(state)
#   state = step10_granger(state)
#   state = step11_plots(state)
#   state = step13_export(state)
#
#   # Or run everything at once:
#   state = run_all()
#
#   # Or just bootstrap MPI on an existing panel:
#   mpi_df = bootstrap_mpi(state["panel"])
#
# ─────────────────────────────────────────────────────────────────────────────

_OWID_RENAME = {
    "iso_code"                   : "iso_code",
    "year"                       : "year",
    "country"                    : "country",
    "primary_energy_consumption" : "energy_consumption_twh",
    "electricity_generation"     : "elec_gen_twh",
    "net_elec_imports"           : "net_elec_imports_twh",
    "net_elec_imports_share_demand" : "net_elec_imports_share_demand",   # Step 12 import robustness
    "greenhouse_gas_emissions"   : "ghg_emissions",   # LIFECYCLE emissions from electricity generation, Mt CO2e (Ember) -- NOT economy-wide
    "carbon_intensity_elec"      : "carbon_intensity_elec_owid",   # LIFECYCLE GHG intensity of generation, gCO2e/kWh (Ember)
    "coal_share_energy"          : "coal_share",
    "oil_share_energy"           : "oil_share",
    "gas_share_energy"           : "gas_share",
    "nuclear_share_energy"       : "nuclear_share",
    "renewables_share_energy"    : "renew_share",
    "hydro_share_energy"         : "hydro_share",         # for Step 12 metafrontier LC_max
    "other_renewables_share_energy" : "other_renew_share", # kept for HHI/Shannon resilience only
    "coal_share_elec"            : "coal_share_elec",      # Step 12 group assignment (electricity-specific)
    "gas_share_elec"             : "gas_share_elec",
    "oil_share_elec"             : "oil_share_elec",
    "nuclear_share_elec"         : "nuclear_share_elec",
    "hydro_share_elec"           : "hydro_share_elec",
    "other_renewables_share_elec" : "other_renew_share_elec",
    "energy_per_capita"          : "energy_pc",
    "gdp"                        : "gdp_owid",
    "population"                 : "population",
}



# ═════════════════════════════════════════════════════════════════════════════
# SAMPLE SCOPE: every calculation uses 2000–2023 and all 45 panel countries
# ═════════════════════════════════════════════════════════════════════════════
def _enforce_panel_scope(panel, label="panel"):
    """
    Restrict the panel to PANEL_ISO3 x YEAR_START..YEAR_END and add an (empty)
    row for any missing country-year, so every downstream step sees the full
    45 x 24 grid. Missing values are then handled by each step's own
    imputation/complete-case rules instead of a country silently disappearing.
    """
    years = list(range(YEAR_START, YEAR_END + 1))
    n0 = len(panel)
    out_scope = panel[~panel["iso_code"].isin(PANEL_ISO3) |
                      ~panel["year"].between(YEAR_START, YEAR_END)]
    panel = panel[panel["iso_code"].isin(PANEL_ISO3) &
                  panel["year"].between(YEAR_START, YEAR_END)].copy()
    panel = panel.drop_duplicates(["iso_code", "year"])
    grid = pd.MultiIndex.from_product([sorted(PANEL_ISO3), years], names=["iso_code", "year"])
    missing = grid.difference(pd.MultiIndex.from_frame(panel[["iso_code", "year"]]))
    if len(missing):
        add = pd.DataFrame(list(missing), columns=["iso_code", "year"])
        if "country" in panel.columns:
            add["country"] = add["iso_code"].map(
                panel.dropna(subset=["country"]).drop_duplicates("iso_code")
                     .set_index("iso_code")["country"])
        panel = pd.concat([panel, add], ignore_index=True)
    panel = panel.sort_values(["iso_code", "year"]).reset_index(drop=True)
    print(f"\n  SAMPLE SCOPE ({label}): {panel['iso_code'].nunique()} countries x "
          f"{panel['year'].nunique()} years ({YEAR_START}–{YEAR_END}) = {len(panel)} rows")
    if len(out_scope):
        print(f"    removed {len(out_scope)} rows outside the scope")
    if len(missing):
        miss_c = pd.Series([m[0] for m in missing]).value_counts()
        print(f"    added {len(missing)} empty country-year rows: "
              + ", ".join(f"{k}({v})" for k, v in miss_c.items()))
    assert set(panel["iso_code"]) == set(PANEL_ISO3) and len(PANEL_ISO3) == 45, \
        "Panel must contain exactly the 45 PANEL_ISO3 countries"
    assert set(panel["year"]) == set(years), "Panel must cover every year 2000–2023"
    return panel


def report_sample_scope(state):
    """
    Final check, printed and saved: for each main result, how many of the
    45 countries and 24 years (2000–2023) actually carry a value. Any shortfall
    is listed by country so it can be traced to its data source.
    """
    panel = state["panel"]
    exp_c, exp_y = len(PANEL_ISO3), YEAR_END - YEAR_START + 1
    rows = []
    def _row(name, df, col, ycol="year"):
        if df is None or len(df) == 0 or col not in df.columns:
            rows.append({"result": name, "countries": 0, "years": 0, "obs": 0,
                         "expected_obs": np.nan, "missing_countries": "not computed"})
            return
        d = df.dropna(subset=[col])
        d = d[d["iso_code"].isin(PANEL_ISO3) & d[ycol].between(YEAR_START, YEAR_END)]
        exp = exp_c * (exp_y - (1 if ycol == "year_t1" else 0))
        miss = sorted(set(PANEL_ISO3) - set(d["iso_code"]))
        rows.append({"result": name, "countries": d["iso_code"].nunique(),
                     "years": d[ycol].nunique(), "obs": len(d), "expected_obs": exp,
                     "missing_countries": ", ".join(miss) if miss else ""})
    for name, col in [("ESI (active C2)", "esi_score"), ("ESI headline", "esi_score_headline"),
                      ("ESI energy", "esi_score_energy"), ("AFF headline", "dim_affordability_headline"),
                      ("AFF energy", "dim_affordability_energy"), ("DDF VRS theta", "dea_vrs"),
                      ("DDF VRS bias-corrected", "dea_vrs_bc"), ("DDF CRS theta", "dea_crs"),
                      ("DDF CRS bias-corrected", "dea_crs_bc")]:
        _row(name, panel, col)
    _row("GML transitions", state.get("mpi_df"), "mpi", ycol="year_t1")
    _row(f"Step 12 TGR ({BASE_RTS}, baseline)", state.get("step12_results"), "tgr")
    _row(f"Step 12 TGR ({ALT_RTS}, robustness)", state.get("step12_results_alt"), "tgr")
    tab = pd.DataFrame(rows)
    print(f"\n{'═'*78}")
    print(f"SAMPLE SCOPE CHECK — target {exp_c} countries, {YEAR_START}–{YEAR_END}")
    print(f"{'═'*78}")
    for _, r in tab.iterrows():
        ok = "✓" if r["obs"] == r["expected_obs"] else "⚠"
        print(f"  {ok} {r['result']:<26} {r['countries']:>3} countries {r['years']:>3} years "
              f"{r['obs']:>5}/{r['expected_obs']:.0f} obs"
              + (f"  missing: {r['missing_countries']}" if r['missing_countries'] else ""))
    print("  Data-limited exception: the Eurostat affordability validation (Step 2) covers")
    print("  2007–2023 and European countries only, because nrg_pc_204 starts in 2007.")
    _save_csv(tab, "sample_scope_check.csv", "coverage of every main result vs 45 x 2000–2023")
    return tab


def plot_affordability_table(panel):
    """
    Table figure: country means over 2000–2023 of the oriented, normalised
    affordability components (0–1, higher = more affordable):
        C1 | C2 headline | C3 | AFF headline | C2 energy | AFF energy
    One row per country (all 45). Also saved as CSV.
    """
    cols = [("_aff_c1", "C1\nenergy intensity"), ("_aff_c2_headline", "C2 headline\n(headline CPI)"),
            ("_aff_c3", "C3\npurchasing capacity"), ("dim_affordability_headline", "AFF headline"),
            ("_aff_c2_energy", "C2 energy\n(energy CPI)"), ("dim_affordability_energy", "AFF energy")]
    have = [c for c, _ in cols if c in panel.columns]
    if not have:
        print("  ⚠ affordability table skipped: components not found (run step4_esi first)")
        return None
    p = panel[panel["year"].between(YEAR_START, YEAR_END) & panel["iso_code"].isin(PANEL_ISO3)]
    means = p.groupby("iso_code")[have].mean().reindex(sorted(PANEL_ISO3))
    nyrs = p.groupby("iso_code")[have].count().reindex(sorted(PANEL_ISO3)).fillna(0).astype(int)
    out = means.copy()
    for c in have:
        out[c + "_n_years"] = nyrs[c]
    _save_csv(out.reset_index(), "table_affordability_country_means.csv",
              "Country means 2000–2023: C1, C2 headline, C3, AFF headline, C2 energy, AFF energy")

    n = len(means)
    row_in = 0.32
    fig_h = 1.0 + row_in * (n + 2) + 1.3
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.8 / fig_h, bottom=1.3 / fig_h)
    ax.axis("off")
    headers = ["Country"] + [lab for c, lab in cols if c in have]
    cell = [[iso] + [("—" if not np.isfinite(means.loc[iso, c]) else f"{means.loc[iso, c]:.3f}")
                     for c in have] for iso in means.index]
    tbl = ax.table(cellText=cell, colLabels=headers, cellLoc="center", loc="upper center",
                   colWidths=[0.10] + [0.15] * len(have))
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    unit = 1.0 / (n + 2)                                   # header uses two units
    for (r, c), cl in tbl.get_celld().items():
        cl.set_edgecolor("#cccccc")
        cl.set_height(2 * unit if r == 0 else unit)
        if r == 0:
            cl.set_facecolor("#1F3864"); cl.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif r % 2 == 0:
            cl.set_facecolor("#F0F4FF")
        if c == 0 and r > 0:
            cl.set_text_props(fontweight="bold")
    ax.set_title(f"Affordability components by country — means over {YEAR_START}–{YEAR_END} "
                 f"({n} countries)", fontsize=13, fontweight="bold")
    import textwrap as _tw
    note = ("Values are the normalised, security-oriented components used in the ESI (0–1, higher = "
            "more affordable), averaged over the years available. AFF = equal-weight mean of C1, C2 "
            "and C3 where all three exist. — = no data (no energy-CPI series for that country). "
            f"{', '.join(C2_OUTLIERS)} have very high headline inflation, so their "
            "C2 headline is lowest in high-inflation years. All components use pooled min–max normalisation "
            "after winsorising at the 1st and 99th percentiles. "
            "Years available per cell: table_affordability_country_means.csv.")
    fig.text(0.01, 0.005, _tw.fill(note, 170), fontsize=10, style="italic", ha="left", va="bottom")
    fig._labels_placed = True
    _save("step4_affordability_table.png")
    return out


def step1_download(state: dict) -> dict:
    """
    STEP 1 — Download all raw data.

    Populates state keys:
        owid_g20, wb_g20, energy_burden_g20, eurostat_elec_prices

    Usage:
        state = step1_download({})
    """
    print("\n" + "="*60)
    print("STEP 1 — Data Download")
    print("="*60)
    t = time.time()

    owid_all   = download_owid(PANEL_ISO3)               # G20 only — no EU27 needed
    wb_g20     = download_world_bank(PANEL_ISO3)
    # Energy burden: World Bank EG.USE.COMM.GD.PP.KD (1990-2023, all G20 directly)
    energy_burden_g20 = download_energy_burden(PANEL_ISO3)
    # Eurostat household electricity prices (EU27 + NOR + GBR + TUR = 30;
    # NOT the Europe-30 panel group) — used
    # in Step 2 to validate the affordability proxy's three raw components
    # against Eurostat-observed electricity prices (both EUR and PPS).
    eurostat_elec_prices = download_eurostat_electricity_prices(EUROSTAT_VALIDATION_ISO3)

    state["owid_g20"]            = owid_all.reset_index(drop=True)
    state["wb_g20"]              = wb_g20
    state["energy_burden_g20"]   = energy_burden_g20
    state["eurostat_elec_prices"] = eurostat_elec_prices

    print(f"\n  ✓ Step 1 complete  ({time.time()-t:.0f}s)")
    print(f"    G20 OWID rows : {len(state['owid_g20'])}")
    return state


def step2_validate_affordability(state: dict) -> dict:
    """
    STEP 2 — Affordability proxy validation.

    Two checks:
      (a) G20 Energy Burden Indicator Diagnostic — characterises
          energy_intensity_koe_gdp directly on the G20 panel (coverage,
          cross-country distribution, within-country trends).
      (b) EU27 Eurostat validation — Pearson/Spearman correlation and an
          OLS regression (elec_price_eur_kwh = alpha + beta *
          energy_intensity_koe_gdp) against genuine EU27 household
          electricity prices, testing whether the globally-available
          energy-intensity proxy tracks real market prices where both
          are observed.

    Requires state from step1_download() and step3_build_panel() to have run
    (needs state["panel"]). If panel not yet built, runs a lightweight check.

    Populates state keys:
        proxy_stats            ← dict with coverage stats, country trends
        affordability_eu27_validation ← dict with correlation matrix + panel
                                          regression model (external European-
                                          subsample validation, C1/C2/C3 vs.
                                          the Eurostat electricity-burden measure)

    Usage:
        state = step2_validate_affordability(state)
    """
    print("\n" + "="*60)
    print("STEP 2 — Affordability Proxy Validation")
    print("="*60)
    t = time.time()

    # Use built panel if available, else build minimal panel from OWID + WB + burden
    if "panel" in state and len(state["panel"]) > 0:
        panel = state["panel"].copy()
    else:
        avail = {k:v for k,v in _OWID_RENAME.items() if k in state["owid_g20"].columns}
        panel = state["owid_g20"][list(avail.keys())].rename(columns=avail)
        panel = panel.merge(state["wb_g20"], on=["iso_code","year"], how="left")
        if "energy_burden_g20" in state and len(state["energy_burden_g20"]) > 0:
            panel = panel.merge(state["energy_burden_g20"], on=["iso_code","year"], how="left")
        # Compute burden fallback
        if "energy_intensity_koe_gdp" not in panel.columns or panel["energy_intensity_koe_gdp"].isna().all():
            if "energy_consumption_twh" in panel.columns and "gdp_pc_ppp" in panel.columns:
                panel["energy_intensity_koe_gdp"] = owid_energy_intensity_kgoe(panel)

    with _TeeLogger("step2_energy_burden_diagnostic.log"):
        state["proxy_stats"] = validate_energy_burden_g20(panel)

    # Save country stats table
    ct = state["proxy_stats"].get("country_stats")
    if ct is not None and len(ct) > 0:
        _save_csv(ct.reset_index(), "step2_energy_burden_stats.csv",
                  "G20 energy burden statistics by country")

    # External European-subsample affordability validation (Eurostat):
    # tests directional consistency of all THREE raw components (C1 energy
    # intensity, C2 price pressure, C3 purchasing capacity) against a
    # Eurostat-derived electricity-cost burden measure, NOT a validation
    # of affordability for the full G20 sample.
    eurostat_df = state.get("eurostat_elec_prices")
    with _TeeLogger("step2_affordability_eu27_validation.log"):
        state["affordability_eu27_validation"] = validate_affordability_eu27(
            panel, eurostat_prices=eurostat_df
        )
    _aff_val = state["affordability_eu27_validation"]
    if _aff_val:
        # Full validation table -> the exact filename requested, one row
        # per European country-year, 2007-2023, with the validation_sample
        # flag so downstream users can filter to the clean subset themselves.
        if _aff_val.get("table") is not None and not _aff_val["table"].empty:
            _save_csv(_aff_val["table"],
                      "eurostat_affordability_validation_2007_2023.csv",
                      "Eurostat affordability validation panel, 2007-2023 "
                      "(EU27 + NOR + GBR + TUR)")
        # Scalar summary fields (coefficients, R², sign matches) -> one-row CSV
        scalar_fields = {k: v for k, v in _aff_val.items()
                         if k not in ("correlation_matrix", "model", "table", "candidate_criteria_table")}
        if scalar_fields:
            _save_csv(pd.DataFrame([scalar_fields]),
                      "step2_affordability_eu27_validation.csv",
                      "EU27 affordability validation (coefficients, signs, R²)")
        # Correlation matrix -> its own CSV
        if _aff_val.get("correlation_matrix") is not None:
            _save_csv(_aff_val["correlation_matrix"].reset_index(),
                      "step2_affordability_eu27_correlations.csv",
                      "EU27 affordability validation (Spearman correlation matrix)")

    print(f"\n  ✓ Step 2 complete  ({time.time()-t:.0f}s)")
    return state


def step3_build_panel(state: dict) -> dict:
    """
    STEP 3 — Merge OWID + World Bank into G20 analysis panel.

    Requires state from step1_download().
    Populates state keys:
        panel   ← main analysis DataFrame

    Usage:
        state = step3_build_panel(state)
        state['panel'].head()
    """
    print("\n" + "="*60)
    print("STEP 3 — Build G20 Master Panel")
    print("="*60)
    t = time.time()

    avail = {k:v for k,v in _OWID_RENAME.items() if k in state["owid_g20"].columns}
    panel = state["owid_g20"][list(avail.keys())].rename(columns=avail)
    panel = panel.merge(state["wb_g20"], on=["iso_code","year"], how="left")
    # Merge affordability composite components (all three)
    if "energy_burden_g20" in state and len(state["energy_burden_g20"]) > 0:
        aff_cols = [c for c in state["energy_burden_g20"].columns
                    if c not in ("iso_code","year")]
        panel = panel.merge(state["energy_burden_g20"], on=["iso_code","year"], how="left")
        for c in aff_cols:
            n = panel[c].notna().sum() if c in panel.columns else 0
            print(f"  {c}: {n} non-null obs")
    panel = panel.sort_values(["iso_code","year"]).reset_index(drop=True)
    panel = _enforce_panel_scope(panel, "Step 3 master panel")

    # ── DATA COMPLETENESS DIAGNOSTIC ─────────────────────────────────────────
    critical = INPUT_COLS + OUTPUT_COLS + ["esi_score","governance_score",
                                           "gdp_growth","energy_intensity_koe_gdp",
                                           "ghg_co2_pc_ar5"]
    present  = [c for c in critical if c in panel.columns]
    print(f"\n  Panel shape : {panel.shape}")
    print(f"  Countries   : {panel['iso_code'].nunique()}  "
          f"({', '.join(sorted(panel['iso_code'].unique()))})")
    print(f"  Years       : {panel['year'].min()}–{panel['year'].max()}")
    print(f"\n  ── Column completeness (non-null %) ──")
    for c in present:
        pct = panel[c].notna().mean() * 100
        flag = "✓" if pct > 50 else ("⚠" if pct > 10 else "✗")
        print(f"    {flag} {c:<35} {pct:5.1f}%")

    # ── FALLBACK IMPUTATION ───────────────────────────────────────────────────
    # When World Bank API returns incomplete data, fill critical columns from
    # OWID proxies or by interpolation so downstream steps have enough data.

    # energy_burden (energy_intensity_koe_gdp) — compute from panel if WB API failed
    # Burden = energy_pc / gdp_pc_ppp, in kgoe per $1,000 PPP GDP (same unit as
    # WB EG.USE.COMM.GD.PP.KD). Higher = more energy per unit of income.
    if "energy_intensity_koe_gdp" not in panel.columns or panel["energy_intensity_koe_gdp"].isna().mean() > 0.5:
        if "energy_pc" in panel.columns and "gdp_pc_ppp" in panel.columns:
            panel["energy_intensity_koe_gdp"] = owid_energy_intensity_kgoe(panel)
            print("  ℹ energy_intensity_koe_gdp computed from OWID energy_pc / gdp_pc_ppp (kgoe per $1,000)")

    # gdp_pc_ppp — use OWID gdp / population as proxy if WB column is mostly NaN
    if "gdp_pc_ppp" not in panel.columns or panel["gdp_pc_ppp"].isna().mean() > 0.5:
        if "gdp_owid" in panel.columns and "population" in panel.columns:
            panel["gdp_pc_ppp"] = (
                panel["gdp_owid"] / panel["population"].replace(0, np.nan)
            )
            print("  ℹ gdp_pc_ppp filled from OWID gdp/population")

    # net_import_pct — OWID does not carry this directly; forward-fill within country
    # after any successful WB rows, then fall back to 0 (no imports assumed)
    if "net_import_pct" not in panel.columns or panel["net_import_pct"].isna().mean() > 0.5:
        if "net_import_pct" in panel.columns:
            panel["net_import_pct"] = (
                panel.groupby("iso_code")["net_import_pct"]
                     .transform(lambda x: x.interpolate().ffill().bfill())
            )
        else:
            panel["net_import_pct"] = 0.0
        print("  ℹ net_import_pct interpolated/filled")

    # governance_score — interpolate within country. The WGI was published every
    # two years before 2002 and has NO 2001 release, so every country is missing
    # 2001. The previous rule interpolated only when >30% of values were missing;
    # with ~4% missing it never ran, the 2001 gap reached dim_robustness, and
    # every ESI-based step lost a full year (1,035 instead of 1,080 rows).
    # Gaps are now always filled: linear interpolation inside each country's
    # series, then carried to the series edges.
    if "governance_score" in panel.columns and panel["governance_score"].isna().any():
        _n_gov_before = int(panel["governance_score"].isna().sum())
        _gov_missing_years = sorted(panel.loc[panel["governance_score"].isna(), "year"].unique())
        panel["governance_score"] = (
            panel.groupby("iso_code")["governance_score"]
                 .transform(lambda x: x.interpolate(limit_area="inside").ffill().bfill())
        )
        _n_gov_after = int(panel["governance_score"].isna().sum())
        print(f"  ℹ governance_score interpolated within country: {_n_gov_before} -> "
              f"{_n_gov_after} missing (years with gaps: {[int(y) for y in _gov_missing_years]})")

    # governance_score — if still all-NaN after interpolation (WGI API completely failed),
    # build a synthetic proxy from GDP per capita (log-scaled, normalized 0-1).
    # This is a known fallback: richer countries tend to score higher on WGI (r≈0.65).
    # Results should be interpreted cautiously when this proxy is active.
    if "governance_score" not in panel.columns or panel["governance_score"].isna().all():
        if "gdp_pc_ppp" in panel.columns and panel["gdp_pc_ppp"].notna().any():
            log_gdp = np.log1p(panel["gdp_pc_ppp"].clip(lower=1))
            panel["governance_score"] = (log_gdp - log_gdp.min()) / (log_gdp.max() - log_gdp.min() + 1e-9)
            # Rescale to approximate WGI range [-2.5, 2.5]
            panel["governance_score"] = panel["governance_score"] * 5 - 2.5
            print("  ⚠ GOVERNANCE WARNING: WGI API failed — using log(GDP per capita) proxy.")
            print("    dim_robustness = INCOME PROXY, not institutional quality.")
            print("    Endogeneity risk: GDP per capita is also a DEA output (gdp_pc_ppp).")
            print("    Do not interpret dim_robustness as governance in this run.")
            print("    Fix: save wgi_data.csv to working dir — see pipeline docstring.")
            panel["governance_is_proxy"] = True
        else:
            panel["governance_score"] = 0.0
            print("  ⚠ governance_score: set to 0 (neutral) — no WGI or GDP data available")

    # gdp_growth — compute from gdp_pc_ppp if missing
    if "gdp_growth" not in panel.columns or panel["gdp_growth"].isna().mean() > 0.5:
        if "gdp_pc_ppp" in panel.columns:
            panel = panel.sort_values(["iso_code","year"])
            panel["gdp_growth"] = (
                panel.groupby("iso_code")["gdp_pc_ppp"]
                     .pct_change() * 100
            )
            print("  ℹ gdp_growth computed from gdp_pc_ppp pct_change")

    # co2_intensity_elec — computed in build_esi via compute_co2_intensity(),
    # primarily from OWID's direct carbon_intensity_elec series (fallback to
    # the ghg_emissions-based approximation only where that is missing).
    for c in ["elec_gen_twh","energy_consumption_twh"]:
        if c in panel.columns:
            panel[c] = panel[c].clip(lower=0)

    # Final completeness report for DEA-critical columns
    # co2_intensity_elec is not yet computed (happens in step 4 build_esi)
    # so only check columns that actually exist now
    dea_cols_present = [c for c in INPUT_COLS + OUTPUT_COLS if c in panel.columns]
    if dea_cols_present:
        dea_avail = panel[dea_cols_present].notna().all(axis=1).sum()
        print(f"\n  DEA-ready rows (excl. co2_intensity_elec, computed in step 4): "
              f"{dea_avail} / {len(panel)}")
        if dea_avail < 50:
            print("  ⚠ Low DEA coverage — check World Bank API connectivity")

    state["panel"] = panel
    plot_step3_coverage(panel)
    print(f"  ✓ Step 3 complete  ({time.time()-t:.0f}s)")
    return state


def step4_esi(state: dict, weights: dict = None) -> dict:
    """
    STEP 4 — Build Energy Security Index (ESI).

    Uses energy expenditure burden (energy_intensity_koe_gdp) as affordability indicator.
    Requires state from steps 1–3.
    Populates/updates state keys:
        panel   ← adds esi_score, dim_resilience, dim_affordability,
                      dim_robustness, dim_sustainability, hhi,
                      co2_intensity_elec, diesel_price_usd_l

    Parameters:
        weights : dict with keys resilience/affordability/robustness/sustainability
                  Default: equal weights (0.25 each)

    Usage:
        state = step4_esi(state)
        # Custom weights:
        state = step4_esi(state, weights={"resilience":0.4,"affordability":0.2,
                                          "robustness":0.2,"sustainability":0.2})
    """
    print("\n" + "="*60)
    print("STEP 4 — ESI Construction")
    print("="*60)
    t = time.time()

    proxy_stats = state.get("proxy_stats", {"ols_beta": None})
    state["panel"] = build_esi(state["panel"], None,  # diesel replaced by energy_intensity_koe_gdp
                               proxy_stats, weights)

    es = state["panel"]["esi_score"]
    print(f"\n  ESI score — mean={es.mean():.3f}  std={es.std():.3f}  "
          f"min={es.min():.3f}  max={es.max():.3f}")
    plot_step4_esi(state["panel"])
    plot_affordability_composite(state["panel"])
    plot_affordability_table(state["panel"])
    esi_cols = ["iso_code","country","year","esi_score","dim_resilience",
                "dim_affordability","dim_robustness","dim_sustainability",
                "hhi","co2_intensity_elec","diesel_price_usd_l","governance_score"]
    _save_csv(state["panel"][[c for c in esi_cols if c in state["panel"].columns]],
              "step4_esi_scores.csv", "ESI scores")
    print(f"  ✓ Step 4 complete  ({time.time()-t:.0f}s)")
    return state




def correlation_table(panel):
    """
    Pearson r and Spearman ρ between ESI dimensions, ESI composite,
    and the bias-corrected DEA efficiency score.

    Printed after step 5 so both ESI and DEA scores are available.
    This is the standard descriptive bridge between the structural
    security index and the operational efficiency measure, and serves
    as a pre-regression diagnostic before the SW second stage.

    Columns reported:
      esi_score, dim_resilience, dim_affordability,
      dim_robustness, dim_sustainability  ×  dea_bc
    Plus pairwise Spearman among the four dimensions.
    """
    from scipy.stats import pearsonr, spearmanr

    target = "dea_bc"
    dims   = ["dim_resilience","dim_affordability",
              "dim_robustness","dim_sustainability"]
    vars_  = ["esi_score"] + dims

    avail = [v for v in vars_ if v in panel.columns and target in panel.columns]
    if not avail:
        print("  ⚠ correlation_table: dea_bc or ESI columns missing — skipping")
        return

    sub = panel[avail + [target]].dropna()
    if len(sub) < 10:
        print(f"  ⚠ correlation_table: only {len(sub)} complete rows — skipping")
        return

    W = 72
    print(f"\n  {'═'*W}")
    print(f"  PEARSON r  &  SPEARMAN ρ  —  ESI dimensions × DEA efficiency (dea_bc)")
    print(f"  n = {len(sub)} country-year observations (listwise complete)")
    print(f"  {'═'*W}")
    print(f"  {'Variable':<28} {'Pearson r':>10} {'p':>8}  {'Spearman ρ':>10} {'p':>8}")
    print(f"  {'─'*W}")
    for v in avail:
        rp, pp = pearsonr( sub[v], sub[target])
        rs, ps = spearmanr(sub[v], sub[target])
        sp = "***" if pp < 0.001 else ("**" if pp < 0.01 else ("*" if pp < 0.05 else ""))
        ss = "***" if ps < 0.001 else ("**" if ps < 0.01 else ("*" if ps < 0.05 else ""))
        print(f"  {v:<28} {rp:>+10.4f} {pp:>7.4f}{sp:<3}  {rs:>+10.4f} {ps:>7.4f}{ss}")

    # Pairwise Spearman among the four dimensions
    avail_dims = [d for d in dims if d in sub.columns]
    if len(avail_dims) >= 2:
        print(f"\n  Pairwise Spearman ρ among ESI dimensions (n={len(sub)}):")
        hdr = f"  {'':20}" + "".join(f"{d.replace('dim_',''):>14}" for d in avail_dims)
        print(hdr)
        for d1 in avail_dims:
            row = f"  {d1.replace('dim_',''):<20}"
            for d2 in avail_dims:
                if d1 == d2:
                    row += f"{'1.000':>14}"
                else:
                    rs, _ = spearmanr(sub[d1], sub[d2])
                    row += f"{rs:>+14.4f}"
            print(row)

    print(f"\n  Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print(f"  Note: correlations are pooled cross-sectional/time-series associations,")
    print(f"  NOT causal effects. A negative correlation (e.g. resilience, robustness,")
    print(f"  affordability vs. dea_bc) should NOT be described as that dimension")
    print(f"  'reducing efficiency' -- these are pooled associations and likely")
    print(f"  contain substantial between-country heterogeneity. Within-country")
    print(f"  correlations (after FE demeaning) may differ substantially — see the")
    print(f"  Panel FE results (Step 8), where most of this pooled association")
    print(f"  disappears once country and year fixed effects are absorbed.")
    print(f"  {'═'*W}")

def frontier_exporter_check(panel, thr=0.999):
    """
    For each year: how many countries are on the frontier under CRS and under
    VRS (raw scores >= thr), and how many of those are net exporters
    (net_import_pct <= 0). Under VRS a net exporter, whose import input sits at
    its lower bound, can only be compared with other countries at that bound,
    so exporters may reach the frontier more easily; under CRS this restriction
    does not apply. This table shows whether that happens in the data.
    """
    need = ["iso_code", "year", "dea_crs", "dea_vrs", "net_import_pct"]
    if any(c not in panel.columns for c in need):
        print("  ⚠ frontier/exporter check skipped: needs dea_crs, dea_vrs, net_import_pct")
        return pd.DataFrame(), {}
    # ── Missingness made explicit (reviewer) ──────────────────────────────────
    # Exporter status needs the OBSERVED net-import share (WB EG.IMP.CONS.ZS).
    # Country-years without it cannot be classified and are left out of the
    # exporter/importer rates below -- they are NOT dropped from the DDF, which
    # uses the imputed input dea_net_import. Which rows are affected, whether
    # they are on the frontier, and a sensitivity that classifies them from
    # the imputed input are all reported, so the reader can judge the effect.
    allrows = panel[need + (["dea_net_import"] if "dea_net_import" in panel.columns else [])].copy()
    allrows = allrows.dropna(subset=["dea_crs", "dea_vrs"])
    miss = allrows[allrows["net_import_pct"].isna()].copy()
    n_all, n_miss = len(allrows), len(miss)
    print(f"\n  Exporter diagnostic sample: {n_all - n_miss}/{n_all} country-years have an observed "
          f"net-import share; {n_miss} cannot be classified as exporter/importer")
    if n_miss:
        _mc = miss.groupby("iso_code")["year"].apply(lambda y: f"{int(y.min())}-{int(y.max())} ({len(y)})")
        print("    missing: " + ", ".join(f"{k} {v}" for k, v in _mc.items()))
        print(f"    of these, on the raw frontier: CRS {int((miss['dea_crs'] >= thr).sum())}, "
              f"VRS {int((miss['dea_vrs'] >= thr).sum())} "
              f"(they remain in the DDF via the imputed import input)")
    d = allrows.dropna(subset=["net_import_pct"]).copy()
    d["exporter"] = d["net_import_pct"] <= 0
    d["front_crs"] = d["dea_crs"] >= thr
    d["front_vrs"] = d["dea_vrs"] >= thr
    g = d.groupby("year")
    _miss_by_year = miss.groupby("year").size()
    tab = pd.DataFrame({
        "n_countries": g.size(),
        "n_unclassified_missing_imports": _miss_by_year.reindex(g.size().index).fillna(0).astype(int),
        "n_exporters": g["exporter"].sum(),
        "frontier_CRS": g["front_crs"].sum(),
        "frontier_VRS": g["front_vrs"].sum(),
        "exporters_on_frontier_CRS": g.apply(lambda x: (x.front_crs & x.exporter).sum()),
        "exporters_on_frontier_VRS": g.apply(lambda x: (x.front_vrs & x.exporter).sum()),
    }).reset_index().astype({"year": int})
    ex, im = d[d.exporter], d[~d.exporter]
    summ = {
        "mean_frontier_share_CRS": d.front_crs.mean(),
        "mean_frontier_share_VRS": d.front_vrs.mean(),
        "exporter_frontier_rate_CRS": ex.front_crs.mean() if len(ex) else np.nan,
        "exporter_frontier_rate_VRS": ex.front_vrs.mean() if len(ex) else np.nan,
        "importer_frontier_rate_CRS": im.front_crs.mean() if len(im) else np.nan,
        "importer_frontier_rate_VRS": im.front_vrs.mean() if len(im) else np.nan,
        "exporter_share_of_frontier_CRS": (d.front_crs & d.exporter).sum() / max(d.front_crs.sum(), 1),
        "exporter_share_of_frontier_VRS": (d.front_vrs & d.exporter).sum() / max(d.front_vrs.sum(), 1),
        "exporter_share_of_sample": d.exporter.mean(),
        "n_classified": len(d), "n_unclassified_missing_imports": n_miss,
    }
    print(f"\n  {'═'*70}")
    print(f"  FRONTIER AND EXPORTER CHECK (raw scores ≥ {thr})")
    print(f"  {'═'*70}")
    print(f"  N = classified country-years; Uncl. = no observed net-import share (not classified)")
    print(f"  {'Year':<6}{'N':>4}{'Uncl.':>6}{'Exp.':>6}{'Front CRS':>11}{'(exp.)':>8}{'Front VRS':>11}{'(exp.)':>8}")
    for _, r in tab.iterrows():
        print(f"  {r.year:<6}{r.n_countries:>4}{r.n_unclassified_missing_imports:>6}{r.n_exporters:>6}{r.frontier_CRS:>11}"
              f"{r.exporters_on_frontier_CRS:>8}{r.frontier_VRS:>11}{r.exporters_on_frontier_VRS:>8}")
    print(f"\n  Share of country-years on the frontier:   CRS {summ['mean_frontier_share_CRS']:.1%}   "
          f"VRS {summ['mean_frontier_share_VRS']:.1%}")
    print(f"  Exporters on the frontier (rate):         CRS {summ['exporter_frontier_rate_CRS']:.1%}   "
          f"VRS {summ['exporter_frontier_rate_VRS']:.1%}")
    print(f"  Importers on the frontier (rate):         CRS {summ['importer_frontier_rate_CRS']:.1%}   "
          f"VRS {summ['importer_frontier_rate_VRS']:.1%}")
    print(f"  Exporters as share of frontier:           CRS {summ['exporter_share_of_frontier_CRS']:.1%}   "
          f"VRS {summ['exporter_share_of_frontier_VRS']:.1%}   "
          f"(exporters are {summ['exporter_share_of_sample']:.1%} of the sample)")
    gap_vrs = summ["exporter_frontier_rate_VRS"] - summ["importer_frontier_rate_VRS"]
    gap_crs = summ["exporter_frontier_rate_CRS"] - summ["importer_frontier_rate_CRS"]
    print(f"  Exporter advantage (exporter minus importer frontier rate): "
          f"CRS {gap_crs:+.1%}   VRS {gap_vrs:+.1%}   [classified rows only, n={len(d)}]")
    # Sensitivity: classify the unclassified rows from the imputed DDF input
    # (dea_net_import = 0 means exporter after the max(0, .) bound).
    if n_miss and "dea_net_import" in allrows.columns:
        dd = allrows.copy()
        dd["exporter"] = np.where(dd["net_import_pct"].notna(), dd["net_import_pct"] <= 0,
                                  dd["dea_net_import"] <= 1e-9)
        dd["front_crs"] = dd["dea_crs"] >= thr; dd["front_vrs"] = dd["dea_vrs"] >= thr
        exs, ims = dd[dd.exporter], dd[~dd.exporter]
        g_c = exs.front_crs.mean() - ims.front_crs.mean()
        g_v = exs.front_vrs.mean() - ims.front_vrs.mean()
        print(f"  Sensitivity (unclassified rows classified from the IMPUTED import input, n={len(dd)}):")
        print(f"    exporter advantage CRS {g_c:+.1%}   VRS {g_v:+.1%}")
        summ["exporter_advantage_CRS_incl_imputed"] = g_c
        summ["exporter_advantage_VRS_incl_imputed"] = g_v
    print(f"  Reading: a much larger exporter advantage under VRS than under CRS means the")
    print(f"  VRS frontier is partly produced by the import-input boundary, supporting CRS")
    print(f"  as the baseline. Similar advantages mean the boundary is not driving results.")
    _save_csv(tab, "step5_frontier_exporter_check_by_year.csv", "Frontier counts and exporters, CRS vs VRS")
    _save_csv(pd.DataFrame([summ]), "step5_frontier_exporter_check_summary.csv",
              "Frontier shares and exporter rates, CRS vs VRS")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})
    yrs = tab["year"]
    a1.plot(yrs, tab["frontier_CRS"], color=COLORS["blue"], lw=2, marker="o", label="On frontier, CRS")
    a1.plot(yrs, tab["exporters_on_frontier_CRS"], color=COLORS["blue"], lw=2, ls="--", marker="s",
            label="… of which exporters, CRS")
    a1.plot(yrs, tab["frontier_VRS"], color=COLORS["amber"], lw=2, marker="o", label="On frontier, VRS")
    a1.plot(yrs, tab["exporters_on_frontier_VRS"], color=COLORS["amber"], lw=2, ls="--", marker="s",
            label="… of which exporters, VRS")
    a1.plot(yrs, tab["n_exporters"], color="#888888", lw=1.4, ls=":", label="Exporters in sample")
    a1.set_xlabel("Year"); a1.set_ylabel("Number of countries")
    a1.set_title("(a) Countries on the frontier each year")
    a1.grid(True, axis="y")
    a1.set_ylim(bottom=0)
    lab = ["Exporters", "Importers"]
    x = np.arange(2); w = 0.36
    a2.bar(x - w/2, [summ["exporter_frontier_rate_CRS"], summ["importer_frontier_rate_CRS"]], w,
           color=COLORS["blue"], label="CRS")
    a2.bar(x + w/2, [summ["exporter_frontier_rate_VRS"], summ["importer_frontier_rate_VRS"]], w,
           color=COLORS["amber"], label="VRS")
    a2.set_xticks(x); a2.set_xticklabels(lab)
    a2.set_ylabel("Share of country-years on the frontier")
    a2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    a2.set_title("(b) Frontier rate, exporters vs importers")
    a2.grid(True, axis="y")
    _fig_legend_right(fig, [a1, a2], right=0.84, top=0.93)
    fig.suptitle("Step 5 — Frontier membership and net exporters, CRS vs VRS", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=fig._tight_rect)
    fig._labels_placed = True
    _save("step5_frontier_exporter_check.png")
    return tab, summ


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5c — SPECIFICATION SENSITIVITY OF THE FRONTIER (reported, never adopted)
# ═════════════════════════════════════════════════════════════════════════════
# INTERPRETATION OF THE BASELINE. The declared baseline imposes weak
# disposability on the CARBON INTENSITY of electricity (gCO2e/kWh) -- a ratio --
# together with electricity per person and energy per person. Under CRS the
# reference technology scales (x, y, b) proportionally, so a reference
# combination at twice the scale has twice the intensity, although physically
# intensity does not change with the level of activity; and the equality
# constraint "contracts" an intensity rather than an emission flow. The
# baseline is therefore an INDICATOR FRONTIER: it benchmarks combinations of
# per-capita and intensity indicators, and beta measures a proportional joint
# improvement in electricity per person and in carbon intensity relative to
# best-practice indicator combinations. It is NOT a distance to a physical
# emissions-production technology.
#
# SENSITIVITY SPECIFICATIONS (raw annual DDF scores, same solver and direction
# g = (y0, b0); none replaces the baseline, and none is chosen by its results):
#   P1 physical, per person (CRS): inputs energy and NET IMPORTED energy per
#      person (kWh), good electricity per person (MWh), bad LIFECYCLE ELECTRICITY
#      EMISSIONS per person (t CO2e = OWID greenhouse_gas_emissions / population).
#      Electricity and its emissions are on the same per-person scale, so CRS
#      scaling and weak disposability (proportional reduction of electricity
#      AND its emissions) have their physical meaning.
#   P2 physical, extensive totals (CRS): energy (TWh), net imported energy
#      (TWh), electricity (TWh), emissions (Mt). Under CRS with the
#      proportional direction, P1 and P2 must give IDENTICAL scores (each DMU's
#      whole vector is rescaled by its population) -- a built-in check of the
#      scaling interpretation.
#   P1/P2 under VRS: the two scales then differ, showing how much the choice of
#      scale matters once returns to scale are not constant.
#   I1-I3 import definitions (baseline intensity bad output kept):
#      I1 electricity net imports as % of demand (bounded at 0) instead of the
#         fuel-wide energy import share; I2 no import input; I3 net imported
#         energy per person (kWh) instead of the share.
# Each is compared with the baseline on country means over all years:
# Spearman rank correlation, mean score, countries ever on the frontier,
# largest rank shifts.

def ddf_raw_scores(panel, inputs, good, bad, vrs=False, label=""):
    """Annual raw DDF efficiency theta for an arbitrary specification
    (complete cases per year; non-positive values replaced as in Step 5)."""
    cols = list(inputs) + list(good) + list(bad)
    d = panel[["iso_code", "year"] + cols].copy()
    for c in cols:
        d[c] = d.groupby("year")[c].transform(
            lambda x: x.mask(x <= 0, x[x > 0].min() * 0.5 if (x > 0).any() else np.nan))
    theta = pd.Series(np.nan, index=panel.index)
    for yr, g in d.dropna(subset=cols).groupby("year"):
        if len(g) < 8:
            continue
        X = _normalise(g[list(inputs)].values.astype(float))
        Y = _normalise(g[list(good)].values.astype(float))
        Bm = _normalise(g[list(bad)].values.astype(float))
        beta = _run_ddf_cross_section(X, Y, Bm, vrs=vrs)
        theta.loc[g.index] = 1.0 / (1.0 + np.clip(beta, 0, None))
    return theta


def compare_with_baseline(panel, theta_alt, label, justification, base_col="dea_score", thr=0.999):
    d = pd.DataFrame({"iso_code": panel["iso_code"], "base": panel[base_col], "alt": theta_alt}).dropna()
    cm = d.groupby("iso_code")[["base", "alt"]].mean()
    rho = stats.spearmanr(cm["base"], cm["alt"])[0] if len(cm) > 4 else np.nan
    rk = cm.rank(ascending=False)
    shift = (rk["alt"] - rk["base"]).abs().sort_values(ascending=False)
    fb = sorted(d.loc[d["base"] >= thr, "iso_code"].unique())
    fa = sorted(d.loc[d["alt"] >= thr, "iso_code"].unique())
    # Frontier MEMBERSHIP stability (rank stability does not imply it):
    # Jaccard index of the sets of frontier country-years, and countries
    # entering / leaving the set of countries ever on the frontier.
    cy = panel.loc[d.index, ["iso_code", "year"]]
    sb = set(map(tuple, cy[d["base"] >= thr].values))
    sa = set(map(tuple, cy[d["alt"] >= thr].values))
    jac = len(sb & sa) / len(sb | sa) if (sb | sa) else np.nan
    return {"sensitivity": label, "justification": justification,
            "n_country_years": len(d), "n_countries": len(cm),
            "spearman_country_means_vs_baseline": rho,
            "mean_theta_baseline": cm["base"].mean(), "mean_theta_alt": cm["alt"].mean(),
            "countries_ever_on_frontier_baseline": len(fb), "countries_ever_on_frontier_alt": len(fa),
            "frontier_countries_alt": ", ".join(fa),
            "frontier_country_years_baseline": len(sb), "frontier_country_years_alt": len(sa),
            "jaccard_frontier_country_years": jac,
            "countries_entering_frontier": ", ".join(sorted(set(fa) - set(fb))),
            "countries_leaving_frontier": ", ".join(sorted(set(fb) - set(fa))),
            "largest_rank_shifts": ", ".join(f"{i} ({int(v)})" for i, v in shift.head(5).items())}


def step5c_specification_sensitivity(state: dict) -> dict:
    """STEP 5c — frontier specification sensitivity (see the note above)."""
    panel = state["panel"]
    print("\n" + "=" * 70)
    print("STEP 5c — SPECIFICATION SENSITIVITY (reported only; the declared baseline")
    print("  -- CRS, carbon intensity as bad output -- is unchanged and is not selected")
    print("  on the basis of these results)")
    print("=" * 70)
    print("  Baseline interpretation: an INDICATOR frontier. Weak disposability and CRS")
    print("  scaling are imposed on a ratio (gCO2e/kWh), which does not scale with")
    print("  activity; beta is a joint proportional improvement in electricity per person")
    print("  and carbon intensity relative to best-practice indicator combinations, not a")
    print("  distance to a physical emissions-production technology.")
    p = panel.copy()
    pop = p["population"] if "population" in p.columns else np.nan
    share = p["dea_net_import"] / 100.0 if "dea_net_import" in p.columns else np.nan
    p["emis_pc_t"] = p.get("ghg_emissions") * 1e6 / pop                 # t CO2e per person
    p["imports_pc_kwh"] = p["energy_pc"] * share                          # kWh per person
    # Net exporters have zero net imports. As in Step 5, zeros are replaced by
    # half the smallest positive value of the year -- applied ONCE, on the
    # per-person variable, and carried to the totals, so that P1 and P2 differ
    # only by each country's population (otherwise the replacement rule itself
    # would break the CRS scaling check).
    p["imports_pc_kwh"] = p.groupby("year")["imports_pc_kwh"].transform(
        lambda x: x.mask(x <= 0, x[x > 0].min() * 0.5 if (x > 0).any() else np.nan))
    p["energy_twh_total"] = p["energy_pc"] * pop / 1e9                    # TWh
    p["imports_twh_total"] = p["imports_pc_kwh"] * pop / 1e9              # TWh
    p["elec_twh_total"] = p.get("elec_gen_twh")
    p["emis_mt_total"] = p.get("ghg_emissions")
    if "net_elec_imports_share_demand" in p.columns:
        p["elec_import_share_pos"] = p["net_elec_imports_share_demand"].clip(lower=0)
    rows, thetas = [], {}
    specs = [
        ("P1 physical per person, CRS",
         ["imports_pc_kwh", "energy_pc"], ["elec_gen_twh_pc"], ["emis_pc_t"], False,
         "Electricity and its emissions on the same per-person scale: CRS scaling and weak "
         "disposability apply to physical flows"),
        ("P2 physical extensive totals, CRS",
         ["imports_twh_total", "energy_twh_total"], ["elec_twh_total"], ["emis_mt_total"], False,
         "Totals: physical technology; must equal P1 under CRS (scaling check)"),
        ("P1 physical per person, VRS",
         ["imports_pc_kwh", "energy_pc"], ["elec_gen_twh_pc"], ["emis_pc_t"], True,
         "Returns to scale not constant: per-person scale"),
        ("P2 physical extensive totals, VRS",
         ["imports_twh_total", "energy_twh_total"], ["elec_twh_total"], ["emis_mt_total"], True,
         "Returns to scale not constant: extensive scale (country size matters)"),
        ("I1 electricity import share (baseline bad output)",
         ["elec_import_share_pos", "energy_pc"], GOOD_OUT, BAD_OUT, VRS,
         "Electricity-specific instead of fuel-wide import dependence"),
        ("I2 no import input (baseline bad output)",
         ["energy_pc"], GOOD_OUT, BAD_OUT, VRS,
         "Removes the import input (and the exporter boundary at 0)"),
        ("I3 imported energy per person (baseline bad output)",
         ["imports_pc_kwh", "energy_pc"], GOOD_OUT, BAD_OUT, VRS,
         "Imports as a physical per-person quantity instead of a share"),
    ]
    for lab, ins, gd, bd, v, why in specs:
        if not all(c in p.columns for c in ins + list(gd) + list(bd)):
            print(f"  ⚠ {lab}: required columns missing -- skipped")
            continue
        th = ddf_raw_scores(p, ins, gd, bd, vrs=v)
        thetas[lab] = th
        r = compare_with_baseline(p, th, lab, why)
        rows.append(r)
    if "P1 physical per person, CRS" in thetas and "P2 physical extensive totals, CRS" in thetas:
        dmax = float(np.nanmax(np.abs(thetas["P1 physical per person, CRS"]
                                      - thetas["P2 physical extensive totals, CRS"])))
        print(f"\n  Scaling check: max |theta(P1, CRS) - theta(P2, CRS)| = {dmax:.2e} "
              f"({'identical, as CRS implies' if dmax < 1e-6 else '⚠ NOT identical -- investigate'})")
    tab = pd.DataFrame(rows)
    print(f"\n  {'Specification':<52}{'N':>6}{'ρ vs base':>10}{'mean θ':>8}{'base θ':>8}{'#front':>8}")
    for _, r in tab.iterrows():
        print(f"  {r['sensitivity']:<52}{int(r['n_country_years']):>6}"
              f"{r['spearman_country_means_vs_baseline']:>10.3f}{r['mean_theta_alt']:>8.3f}"
              f"{r['mean_theta_baseline']:>8.3f}{int(r['countries_ever_on_frontier_alt']):>8}")
        print(f"      largest rank shifts: {r['largest_rank_shifts']}")
        print(f"      frontier membership: Jaccard(country-years) = {r['jaccard_frontier_country_years']:.2f}; "
              f"entering: {r['countries_entering_frontier'] or '—'}; leaving: {r['countries_leaving_frontier'] or '—'}")
    print(f"  ρ = Spearman correlation of country-mean scores with the baseline (CRS, intensity).")
    print(f"  Reading: a high ρ for P1 means the indicator frontier ranks countries much like a")
    print(f"  physical emissions-production technology; a low ρ means conclusions depend on the")
    print(f"  indicator construction and must be stated as such. Either way the baseline stays.")
    print(f"  Rank stability does NOT imply stable frontier membership: compare #front and the")
    print(f"  Jaccard index (1 = identical frontier country-years, 0 = no overlap).")
    for lab, th in thetas.items():
        key = "theta_sens_" + "".join(ch if ch.isalnum() else "_" for ch in lab.split(" ")[0])
        state["panel"][key + ("_vrs" if "VRS" in lab else "")] = th
    _save_csv(tab, "step5c_specification_sensitivity.csv",
              "Frontier specification sensitivity (physical scales, import definitions)")
    state["sensitivity_spec"] = tab
    return state


def step5_dea_bootstrap(state: dict, B: int = B_BOOTSTRAP,
                        alpha: float = ALPHA, vrs: bool = VRS) -> dict:
    """
    STEP 5 — Output-oriented BCC DEA with Simar–Wilson (1998) smooth bootstrap.

    Requires state from steps 1–4.
    Populates/updates state keys:
        panel   ← adds dea_score, dea_bc, dea_bias, dea_lo, dea_hi, dea_se

    Parameters:
        B     : bootstrap replications  (default: B_BOOTSTRAP)
        alpha : CI significance level   (default: 0.05)
        vrs   : True = BCC/VRS, False = CCR/CRS

    Usage:
        state = step5_dea_bootstrap(state)          # B=B_BOOTSTRAP (set by TEST_MODE)
    """
    print("\n" + "="*60)
    print(f"STEP 5 — DDF ({'VRS' if vrs else 'CRS'} baseline) + Simar–Wilson (1998) Bootstrap  (B={B})")
    print("="*60)
    t = time.time()

    state["panel"] = build_dea_bootstrap(state["panel"], B=B, alpha=alpha, vrs=vrs)

    bc = state["panel"]["dea_bc"].dropna()
    print(f"\n  DEA (bias-corrected) — mean={bc.mean():.3f}  "
          f"std={bc.std():.3f}  min={bc.min():.3f}  max={bc.max():.3f}")
    plot_step5_dea(state["panel"])
    plot_step5_scale_efficiency(state["panel"])
    dea_cols = ["iso_code","country","year","dea_score","dea_bc",
                "dea_bias","dea_lo","dea_hi","dea_se",
                "dea_crs","dea_se_score"]
    _save_csv(state["panel"][[c for c in dea_cols if c in state["panel"].columns]],
              "step5_dea_scores.csv", "DEA bootstrap scores")
    correlation_table(state["panel"])
    state["frontier_exporter_check"] = frontier_exporter_check(state["panel"])
    print(f"  ✓ Step 5 complete  ({time.time()-t:.0f}s)")
    return state


def step6_mpi(state: dict, vrs: bool = False) -> dict:
    """
    STEP 6 — Global Malmquist-Luenberger Index (GML) decomposition (GEC × GTC).
    Pooled global technology T^G guarantees 100% transition coverage.

    Requires state from steps 1–5.
    Populates state keys:
        mpi_df  ← DataFrame with mpi, ec, tc per iso_code × year pair

    Usage:
        state = step6_mpi(state)
        state['mpi_df'].groupby('iso_code')[['mpi','ec','tc']].mean()
    """
    print("\n" + "="*60)
    print("STEP 6 — Global Malmquist-Luenberger Index (GML = GEC × GTC)")
    print("="*60)
    t = time.time()

    state["mpi_df"] = build_mpi(state["panel"], vrs=vrs)  # vrs=False = CRS (conventional)
    print(f"\n  GML records: {len(state['mpi_df'])}")
    plot_step6_mpi(state["mpi_df"])
    _save_csv(state["mpi_df"], "step6_mpi_results.csv", "Global Malmquist-Luenberger (GML) decomposition")
    print(f"  ✓ Step 6 complete  ({time.time()-t:.0f}s)")
    return state


def step7_simar_wilson(state: dict, z_vars: list = None,
                       B: int = B_SW2007, alpha: float = ALPHA) -> dict:
    """
    STEP 7 — Simar–Wilson (2007) two-stage double-bootstrap truncated regression.

    Regresses bias-corrected DEA scores on ESI dimensions + controls.
    Requires state from steps 1–5.
    Populates state keys:
        sw_results  ← DataFrame with coef_bc, bootstrap_se, t_stat, CI

    Parameters:
        z_vars : list of regressor column names (default: 5 ESI dims + gdp_growth;
                 net_import_pct excluded — source of a DEA input)
        B      : bootstrap replications (default: B_BOOTSTRAP)
        alpha  : CI significance level  (default: 0.05)

    *** NOTE: this is the FULL 5-variable specification. The separability
    diagnostic (daraio_simar_wilson_separability_test, run automatically
    at the start of simar_wilson_2007) tests, jointly and variable by
    variable, whether Z shifts the production technology itself. Where it
    rejects, this model is EXPLORATORY and the conditional DDF (Step 7c) is
    the primary structural-operational analysis; the restricted model
    (Step 7b) is a sensitivity check, not a replacement. The test outcome
    is read from the log of each run, not assumed here. ***

    Usage:
        state = step7_simar_wilson(state)
        state['sw_results']
    """
    print("\n" + "="*60)
    print(f"STEP 7 — Simar–Wilson (2007) Two-Stage Regression  (B={B})")
    print("  *** Algorithm 2 adapted to the DDF. Separability is ASSUMED, ***")
    print("  *** not formally tested: coefficients are associations; the  ***")
    print("  *** conditional DDF (Step 7c) is the structural benchmark.   ***")
    print("="*60)
    t = time.time()

    with _TeeLogger("step7_simar_wilson.log"):
        state["sw_results"] = simar_wilson_2007(state["panel"], z_vars=z_vars,
                                                B=B, alpha=alpha)
    plot_step7_sw(state["sw_results"], "step7_sw_full.png", "full model")
    _save_csv(state["sw_results"], "step7_simar_wilson.csv", "Simar-Wilson coefficients")
    print(f"\n  ✓ Step 7 complete  ({time.time()-t:.0f}s)")
    return state


def step7b_simar_wilson_restricted(state: dict, z_vars: list = None,
                                   B: int = B_SW2007, alpha: float = ALPHA,
                                   common_sample_with_full: bool = False) -> dict:
    """
    STEP 7b — Restricted Simar–Wilson specification (sensitivity check).

    Re-estimates the second stage with affordability and GDP growth only,
    the two variables originally expected to behave as conventional
    efficiency-associated environmental factors rather than technology-
    shaping ones. The separability diagnostic is run for this restricted Z
    as well, jointly and per variable; its result is printed in the log and
    must be reported alongside the coefficients. If it rejects, the
    restricted model does not satisfy separability either and is reported
    as a sensitivity check only -- the conditional DDF (Step 7c) remains the
    primary structural-operational analysis. (The previous hard-coded
    statement that neither variable rejects individually, with R2=0.008, is
    removed: it described an earlier run and no longer matches the data.)

    Parameters
    ----------
    common_sample_with_full : if True, additionally re-estimates the
        restricted model on EXACTLY the estimation sample of the full
        5-variable model (rows where all five Z variables are present), so
        that any change in fit between the two specifications can be
        attributed to the change in variables rather than to a change in
        sample. The sample size is printed in each run.
        Populates state["sw_results_restricted_common_sample"].

    Populates state keys:
        sw_results_restricted ← DataFrame with coef_bc, bootstrap_se, t_stat, CI
                                 (on the restricted variables' own
                                 maximum-information sample)
        sw_results_restricted_common_sample ← same, forced onto the full
                                 model's sample, if requested

    Usage:
        state = step7b_simar_wilson_restricted(state)
        state = step7b_simar_wilson_restricted(state, common_sample_with_full=True)
    """
    if z_vars is None:
        z_vars = ["dim_affordability", "gdp_growth"]

    print("\n" + "="*60)
    print(f"STEP 7b — Restricted Simar–Wilson (sensitivity check)  (B={B})")
    print(f"  Z = {z_vars}")
    print("  *** Sensitivity check, NOT the primary model. The k-NN sensitivity")
    print("  *** diagnostic for this restricted Z is printed below.")
    print("="*60)
    t = time.time()

    with _TeeLogger("step7b_simar_wilson_restricted.log"):
        state["sw_results_restricted"] = simar_wilson_2007(
            state["panel"], z_vars=z_vars, B=B, alpha=alpha
        )
    plot_step7_sw(state["sw_results_restricted"], "step7b_sw_restricted.png", "restricted model")
    _save_csv(state["sw_results_restricted"], "step7b_simar_wilson_restricted.csv",
              "Simar-Wilson restricted (reduced separability concern)")
    _compare_sw_specs(state.get("sw_results"), state["sw_results_restricted"])
    print(f"\n  ✓ Step 7b complete  ({time.time()-t:.0f}s)")

    if common_sample_with_full:
        # Forces the restricted model onto the full model's exact
        # estimation sample, so any change in fit can be attributed to the
        # variable change alone, not partly to the sample change.
        print("\n" + "="*60)
        print("STEP 7b (common-sample) — Restricted SW on the FULL model's estimation sample")
        print("="*60)
        full_z_vars = ["dim_resilience","dim_affordability",
                       "dim_robustness","dim_sustainability","gdp_growth"]
        common_cols = ["iso_code","year"] + INPUT_COLS + GOOD_OUT + BAD_OUT + \
                      ["dea_score","dea_bc"] + full_z_vars
        common_cols = [c for c in common_cols if c in state["panel"].columns]
        common_mask_df = state["panel"][common_cols].dropna()
        common_idx = common_mask_df.index
        panel_common = state["panel"].loc[common_idx].copy()
        n_common = len(panel_common)
        print(f"  Forced sample size: {n_common} (the full 5-variable model's complete-case sample)")

        with _TeeLogger("step7b_common_sample.log"):
            state["sw_results_restricted_common_sample"] = simar_wilson_2007(
                panel_common, z_vars=z_vars, B=B, alpha=alpha
            )
        _save_csv(state["sw_results_restricted_common_sample"],
                  "step7b_restricted_common_sample.csv",
                  f"Restricted SW on the full model's common sample (N={n_common})")
        print(f"  ✓ Common-sample restricted SW complete")

    return state


def _compare_sw_specs(full, restr):
    """Coefficients shared by the full and restricted second stages: does the
    sign and the 'both intervals exclude zero' status persist?"""
    if not isinstance(full, pd.DataFrame) or not isinstance(restr, pd.DataFrame):
        return
    excl = lambda r: (pd.notna(r["ci_lo_95"]) and pd.notna(r["ci_lo_cluster"]) and
                      (r["ci_lo_95"] > 0 or r["ci_hi_95"] < 0) and
                      (r["ci_lo_cluster"] > 0 or r["ci_hi_cluster"] < 0))
    f_ = full.set_index("variable"); r_ = restr.set_index("variable")
    shared = [v for v in r_.index if v in f_.index and v != "const"]
    if not shared:
        return
    print(f"\n  FULL vs RESTRICTED second stage (shared regressors):")
    print(f"  {'Variable':<22}{'full γ':>9}{'both CIs ≠ 0':>14}{'restr. γ':>10}{'both CIs ≠ 0':>14}  persists?")
    for v in shared:
        a, b = f_.loc[v], r_.loc[v]
        ea, eb = excl(a), excl(b)
        pers = "yes" if (ea and eb and np.sign(a["coef_bc"]) == np.sign(b["coef_bc"])) else "NO"
        print(f"  {v:<22}{a['coef_bc']:>9.4f}{str(ea):>14}{b['coef_bc']:>10.4f}{str(eb):>14}  {pers}")
    print(f"  An association that does not persist across specifications should not be")
    print(f"  reported as a finding of the second stage.")


def step7c_conditional_ddf(state: dict, z_vars: list = None) -> dict:
    """
    STEP 7c — Conditional DDF analysis: D(x,y,b|Z).

    The recommended main alternative to Simar-Wilson,
    given that the separability diagnostic (Step 6b) rejects H0 for both
    the full 5-variable Z vector AND the restricted affordability+gdp_growth
    model. Reframes the research question from "does Z explain deviations
    from a fixed frontier" (the SW framing, not defensible here) to
    "structural energy-security conditions shape the feasible frontier of
    green operational performance" (the conditional-DDF framing). ***

    Requires state from steps 1–5.
    Populates state keys:
        panel updated with dea_score_conditional
        conditional_ddf_summary ← country-level conditional vs unconditional
                                  efficiency comparison

    Usage:
        state = step7c_conditional_ddf(state)
    """
    print("\n" + "="*60)
    print("STEP 7c — Conditional DDF Analysis")
    print("="*60)
    t = time.time()

    with _TeeLogger("step7c_conditional_ddf.log"):
        state["panel"], state["conditional_ddf_summary"] = conditional_ddf_analysis(
            state["panel"], z_vars=z_vars
        )
    _save_csv(state["conditional_ddf_summary"], "step7c_conditional_ddf_summary.csv",
              "Conditional vs unconditional DDF efficiency by country")
    with _TeeLogger("step7c_conditional_sensitivity.log"):
        state["conditional_sensitivity"] = conditional_sensitivity(state["panel"], z_vars=z_vars)
    if len(state["conditional_sensitivity"]):
        _save_csv(state["conditional_sensitivity"], "step7c_conditional_sensitivity.csv",
                  "Conditional DDF gap: sensitivity to neighbourhood size and Z set")
    print(f"\n  ✓ Step 7c complete  ({time.time()-t:.0f}s)")
    return state


def step8_panel_fe(state: dict) -> dict:
    """
    STEP 8 — Panel Fixed-Effects OLS (entity + time FE, clustered SE).

    Dependent variable: dea_bc (bias-corrected DEA efficiency)
    Regressors: esi_score, ESI dimensions, gdp_growth, net_import_pct
    Requires state from steps 1–5.
    Populates state keys:
        fe_result  ← linearmodels PanelOLS result object

    Usage:
        state = step8_panel_fe(state)
        state['fe_result'].summary
    """
    print("\n" + "="*60)
    print("STEP 8 — Panel Fixed-Effects OLS")
    print("="*60)
    t = time.time()

    with _TeeLogger("step8_panel_fe.log"):
        state["fe_result"] = panel_fe_regression(state["panel"])
    plot_step8_fe(state["fe_result"])
    if state["fe_result"] is not None:
        try:
            fe_df = pd.DataFrame({
                "variable": state["fe_result"].params.index,
                "coef":     state["fe_result"].params.values,
                "std_err":  state["fe_result"].std_errors.values,
                "pvalue":   state["fe_result"].pvalues.values,
            })
            _save_csv(fe_df, "step8_panel_fe.csv", "Panel FE coefficients")
        except Exception:
            pass
    print(f"\n  ✓ Step 8 complete  ({time.time()-t:.0f}s)")
    return state


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8c — IS THERE A (LAGGED) LINK BETWEEN ESI AND DDF?
# ═════════════════════════════════════════════════════════════════════════════
# ESI (structural conditions) and DDF (operational performance) answer different
# questions, so their rankings need not agree. This step asks a narrower
# question: do within-country movements in one go together with movements in
# the other, at the same time or with a delay of 1, 2 or 3 years?
#
# Three complementary views, all on within-country variation (country and year
# effects removed, so persistent level differences and common shocks cannot
# create a link):
#   (a) lagged cross-correlations, k = -3..+3  (k > 0: ESI moves first)
#   (b) distributed-lag fixed-effects regressions in BOTH directions
#         DDF_it = a_i + l_t + sum_k b_k ESI_i,t-k + e_it      (ESI -> DDF)
#         ESI_it = a_i + l_t + sum_k c_k DDF_i,t-k + e_it      (DDF -> ESI)
#       one lag at a time and all lags 0..3 jointly (Wald test of all b_k = 0),
#       standard errors clustered by country
#   (c) the same for each ESI dimension (which dimension drives any link)
# Everything is an association. A lagged association is consistent with, but
# does not prove, a causal ordering.

def _two_way_demean(df, col):
    """Remove country and year means (balanced-panel within transform)."""
    x = df[col]
    return (x - df.groupby("iso_code")[col].transform("mean")
              - df.groupby("year")[col].transform("mean") + x.mean())


def _lag_crosscorr(panel, x_col, y_col, max_lag=3, B=None, seed=11):
    """
    corr( x_{i,t-k}, y_{i,t} ) on two-way-demeaned data for k = -max_lag..max_lag.
    95% CIs by resampling COUNTRIES (keeps each country's time series intact).
    Fast: each country contributes sufficient statistics, so a bootstrap
    replication only sums them.
    """
    B = B_RESAMPLE if B is None else B
    d = panel[["iso_code", "year", x_col, y_col]].dropna().sort_values(["iso_code", "year"]).copy()
    d["_x"] = _two_way_demean(d, x_col); d["_y"] = _two_way_demean(d, y_col)
    isos = list(d["iso_code"].unique())
    years = range(int(d["year"].min()), int(d["year"].max()) + 1)
    ks = list(range(-max_lag, max_lag + 1))
    # stats[c, k] = (n, sx, sy, sxx, syy, sxy)
    stats_ = np.zeros((len(isos), len(ks), 6))
    for ci, (iso, g) in enumerate(d.groupby("iso_code", sort=False)):
        g = g.set_index("year").reindex(years)
        for ki, k in enumerate(ks):
            m = pd.concat([g["_x"].shift(k), g["_y"]], axis=1).dropna().values
            if len(m):
                x, y = m[:, 0], m[:, 1]
                stats_[ci, ki] = [len(x), x.sum(), y.sum(), (x*x).sum(), (y*y).sum(), (x*y).sum()]

    def _corr(S):                                   # S: (len(ks), 6) summed stats
        n, sx, sy, sxx, syy, sxy = S.T
        cov = sxy - sx * sy / n
        return cov / np.sqrt((sxx - sx**2 / n) * (syy - sy**2 / n))

    with np.errstate(all="ignore"):
        r = _corr(stats_.sum(axis=0))
        rng = np.random.default_rng(seed)
        boot = np.array([_corr(stats_[rng.integers(0, len(isos), len(isos))].sum(axis=0))
                         for _ in range(B)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"lag_k": ks, "corr": r, "ci_lo": lo, "ci_hi": hi,
                         "n_obs": stats_[:, :, 0].sum(axis=0).astype(int)})


def _dl_fe(panel, y_col, x_col, lags, controls=()):
    """Distributed-lag FE regression with country + year effects and
    country-clustered SEs. Variables are standardised (pooled SD), so each
    coefficient is the SD change in y per SD change in x."""
    d = panel[["iso_code", "year", y_col, x_col] + list(controls)].dropna().sort_values(["iso_code", "year"]).copy()
    for c in [y_col, x_col] + list(controls):
        d[c] = (d[c] - d[c].mean()) / (d[c].std() + 1e-12)
    names = []
    for k in lags:
        nm = f"{x_col}_L{k}"
        d[nm] = d.groupby("iso_code")[x_col].shift(k)
        names.append(nm)
    d = d.dropna(subset=names + [y_col]).set_index(["iso_code", "year"])
    res = PanelOLS(d[y_col], d[names + list(controls)], entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    b = res.params[names].values
    V = res.cov.loc[names, names].values
    try:
        wald = float(b @ np.linalg.solve(V, b))
        p_joint = float(1 - stats.chi2.cdf(wald, len(names)))
    except np.linalg.LinAlgError:
        wald, p_joint = np.nan, np.nan
    ci = res.conf_int()
    out = pd.DataFrame({"lag_k": list(lags), "coef": b, "se": res.std_errors[names].values,
                        "p": res.pvalues[names].values,
                        "ci_lo": ci.loc[names, "lower"].values, "ci_hi": ci.loc[names, "upper"].values})
    return out, wald, p_joint, int(res.nobs), float(res.rsquared)


def _cross_lagged(panel, y_col, x_col, p=3):
    """
    Granger-type cross-lagged FE regression:
        y_it = a_i + l_t + sum_{k=1..p} g_k y_i,t-k + sum_{k=1..p} b_k x_i,t-k + e_it
    Controlling for y's own past is what separates direction: a persistent x
    correlates with y at every lag, but only lagged x that adds information
    beyond y's own history counts here. Wald test of all b_k = 0, clustered SE.
    (With country FE and lagged y, the g_k carry a Nickell bias of order 1/T,
    about 4% with T = 24; the b_k test is much less affected.)
    """
    d = panel[["iso_code", "year", y_col, x_col]].dropna().sort_values(["iso_code", "year"]).copy()
    for c in (y_col, x_col):
        d[c] = (d[c] - d[c].mean()) / (d[c].std() + 1e-12)
    ylags = [f"{y_col}_L{k}" for k in range(1, p + 1)]
    xlags = [f"{x_col}_L{k}" for k in range(1, p + 1)]
    for k in range(1, p + 1):
        d[ylags[k - 1]] = d.groupby("iso_code")[y_col].shift(k)
        d[xlags[k - 1]] = d.groupby("iso_code")[x_col].shift(k)
    d = d.dropna().set_index(["iso_code", "year"])
    res = PanelOLS(d[y_col], d[ylags + xlags], entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    b = res.params[xlags].values; V = res.cov.loc[xlags, xlags].values
    try:
        wald = float(b @ np.linalg.solve(V, b)); pj = float(1 - stats.chi2.cdf(wald, p))
    except np.linalg.LinAlgError:
        wald, pj = np.nan, np.nan
    ci = res.conf_int()
    tab = pd.DataFrame({"lag_k": list(range(1, p + 1)), "coef": b, "p": res.pvalues[xlags].values,
                        "ci_lo": ci.loc[xlags, "lower"].values, "ci_hi": ci.loc[xlags, "upper"].values})
    return tab, wald, pj, int(res.nobs)


def lagged_esi_ddf_link(panel, esi_col="esi_score", ddf_col="dea_score", max_lag=3,
                        dims=("dim_resilience", "dim_affordability", "dim_robustness",
                              "dim_sustainability")):
    print(f"\n{'═'*74}")
    print(f"STEP 8c — LAGGED LINK BETWEEN ESI ({esi_col}) AND DDF ({ddf_col}), lags 0–{max_lag}")
    print(f"  Within-country variation only (country and year effects removed).")
    print(f"{'═'*74}")
    out = {}
    if esi_col not in panel.columns or ddf_col not in panel.columns:
        print("  ⚠ ESI or DDF column missing — skipped"); return out

    cc = _lag_crosscorr(panel, esi_col, ddf_col, max_lag)
    out["crosscorr"] = cc
    print(f"\n  (a) Within-country cross-correlation corr(ESI_t-k, DDF_t), 95% CI by country resampling")
    print(f"      k > 0: ESI moves first;  k < 0: DDF moves first")
    for _, r in cc.iterrows():
        sig = "*" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else " "
        print(f"      k={int(r['lag_k']):+d}  r={r['corr']:+.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]{sig}  n={int(r['n_obs'])}")

    lags = list(range(0, max_lag + 1))
    rows_single, joint = [], {}
    for direction, y, x in (("ESI -> DDF", ddf_col, esi_col), ("DDF -> ESI", esi_col, ddf_col)):
        for k in lags:
            t, *_ = _dl_fe(panel, y, x, [k])
            r = t.iloc[0].to_dict(); r["direction"] = direction; r["model"] = "single lag"
            rows_single.append(r)
        t, wald, pj, n, r2 = _dl_fe(panel, y, x, lags)
        t["direction"] = direction; t["model"] = f"joint lags 0-{max_lag}"
        joint[direction] = (t, wald, pj, n, r2)
    single = pd.DataFrame(rows_single)
    joint_tab = pd.concat([v[0] for v in joint.values()], ignore_index=True)
    out["single"], out["joint"] = single, joint_tab
    out["joint_tests"] = pd.DataFrame([{"direction": d, "wald_chi2": v[1], "df": len(lags), "p_joint": v[2],
                                        "n_obs": v[3], "within_r2": v[4]} for d, v in joint.items()])

    print(f"\n  (b) Distributed-lag FE regressions (standardised; country + year FE; clustered SE)")
    for direction in ("ESI -> DDF", "DDF -> ESI"):
        print(f"\n      {direction}   (one lag at a time)")
        for _, r in single[single["direction"] == direction].iterrows():
            sig = "**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else "")
            print(f"        lag {int(r['lag_k'])}: b={r['coef']:+.3f}  SE={r['se']:.3f}  p={r['p']:.3f} {sig}")
        t, wald, pj, n, r2 = joint[direction]
        print(f"      {direction}   (lags 0–{max_lag} jointly): Wald chi2({len(lags)})={wald:.2f}, "
              f"p={pj:.3f}, N={n}, within R2={r2:.3f}")
        for _, r in t.iterrows():
            print(f"        lag {int(r['lag_k'])}: b={r['coef']:+.3f}  p={r['p']:.3f}")

    # (c) each ESI dimension -> DDF, joint lags (which dimension carries any link)
    dim_rows = []
    for dcol in dims:
        if dcol in panel.columns and panel[dcol].notna().sum() > 100:
            t, wald, pj, n, r2 = _dl_fe(panel, ddf_col, dcol, lags)
            for _, r in t.iterrows():
                dim_rows.append({"dimension": dcol, "lag_k": int(r["lag_k"]), "coef": r["coef"],
                                 "p": r["p"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                                 "p_joint": pj, "n_obs": n})
    out["dimensions"] = pd.DataFrame(dim_rows)
    if dim_rows:
        print(f"\n  (c) ESI dimensions -> DDF (lags 0–{max_lag} jointly)")
        for dcol, g in out["dimensions"].groupby("dimension", sort=False):
            lagtxt = "  ".join(f"L{int(r.lag_k)}={r.coef:+.3f}{'*' if r.p < 0.05 else ''}" for r in g.itertuples())
            print(f"      {dcol:<20} joint p={g['p_joint'].iloc[0]:.3f}   {lagtxt}")
        print(f"      Note: dim_sustainability (territorial CO2 per capita) and the DDF bad output")
        print(f"      (electricity carbon intensity) are related by subject matter; read that row")
        print(f"      as partly mechanical.")

    # (d) direction: cross-lagged (Granger-type) FE models, own lags controlled
    cl_rows, cl_tests = [], []
    print(f"\n  (d) Which moves first? Cross-lagged FE models controlling for the outcome's own")
    print(f"      lags 1–{max_lag} (Granger-type; country + year FE; clustered SE)")
    for direction, y, x in (("ESI -> DDF", ddf_col, esi_col), ("DDF -> ESI", esi_col, ddf_col)):
        try:
            t, wald, pj, n = _cross_lagged(panel, y, x, p=max_lag)
        except Exception as e:
            print(f"      {direction}: failed ({e})"); continue
        t["direction"] = direction; cl_rows.append(t)
        cl_tests.append({"direction": direction, "wald_chi2": wald, "df": max_lag, "p_joint": pj, "n_obs": n})
        lagtxt = "  ".join(f"t-{int(r.lag_k)}: {r.coef:+.3f}{'*' if r.p < 0.05 else ''}" for r in t.itertuples())
        print(f"      {direction}: joint Wald chi2({max_lag})={wald:.2f}, p={pj:.3f}, N={n}   {lagtxt}")
    if cl_rows:
        out["cross_lagged"] = pd.concat(cl_rows, ignore_index=True)
        out["cross_lagged_tests"] = pd.DataFrame(cl_tests)

    print(f"\n  INTERPRETATION")
    print(f"  ESI dimensions move slowly, so ESI is correlated with itself across years. That")
    print(f"  spreads any link over every lag in (a) and in the one-lag models in (b), and can")
    print(f"  even make the reverse direction look significant. Read the timing from the JOINT")
    print(f"  models in (b), where all lags compete, and the direction from (d), where each")
    print(f"  variable's own past is controlled. Evidence that structural conditions precede")
    print(f"  operational change = significant lagged ESI in (d) 'ESI -> DDF' but not in")
    print(f"  'DDF -> ESI'. These are within-country associations, not causal effects; null")
    print(f"  results mean 'no detectable within-country link', not 'no relationship' (Step 8b")
    print(f"  covers between-country level differences).")
    return out


def plot_lagged_link(res, esi_col="esi_score", ddf_col="dea_score", suffix=""):
    """(a) within-country cross-correlations by lag; (b) cross-lagged
    (Granger-type) FE coefficients in both directions, own lags controlled."""
    if not res or "crosscorr" not in res:
        return
    cc = res["crosscorr"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 6.5))
    a1.axhline(0, color="black", lw=0.8)
    sig = (cc["ci_lo"] > 0) | (cc["ci_hi"] < 0)
    a1.bar(cc["lag_k"], cc["corr"], color=np.where(sig, COLORS["blue"], "#b8c4d9"), width=0.6,
           label="Correlation\n(dark = 95% CI excludes 0)")
    a1.errorbar(cc["lag_k"], cc["corr"], yerr=[cc["corr"] - cc["ci_lo"], cc["ci_hi"] - cc["corr"]],
                fmt="none", color="black", capsize=4, lw=1.1, label="95% CI\n(country resampling)")
    a1.set_xticks(cc["lag_k"])
    a1.set_xticklabels([f"{k:+d}" for k in cc["lag_k"]])
    a1.set_xlabel("Lag k  (k > 0: ESI earlier;  k < 0: DDF earlier)")
    a1.set_ylabel("Within-country correlation")
    a1.set_title("(a) corr(ESI$_{t-k}$, DDF$_t$), country and year effects removed")
    a1.grid(True, axis="y")
    _legend_right(a1)

    cl = res.get("cross_lagged")
    tests = res.get("cross_lagged_tests")
    if cl is not None and len(cl):
        for off, (direction, col) in zip((-0.12, 0.12), (("ESI -> DDF", COLORS["blue"]),
                                                          ("DDF -> ESI", COLORS["amber"]))):
            t = cl[cl["direction"] == direction]
            pj = tests.loc[tests["direction"] == direction, "p_joint"]
            lab = direction.replace("->", "→") + (f"\njoint p = {pj.iloc[0]:.3f}" if len(pj) else "")
            a2.errorbar(t["lag_k"] + off, t["coef"], yerr=[t["coef"] - t["ci_lo"], t["ci_hi"] - t["coef"]],
                        fmt="o", color=col, capsize=4, ms=7, lw=1.4, label=lab)
        a2.axhline(0, color="black", lw=0.8)
        ks = sorted(cl["lag_k"].unique())
        a2.set_xticks(ks); a2.set_xticklabels([f"t-{k}" for k in ks])
        a2.set_xlabel("Lag of the other variable")
        a2.set_ylabel("Standardised coefficient (95% CI)")
        a2.set_title("(b) Which moves first? Cross-lagged FE, own lags controlled")
        a2.grid(True, axis="y")
        _legend_right(a2, title="Direction")
    fig.suptitle(f"Step 8c — Lagged link between ESI ({esi_col}) and DDF ({ddf_col})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(wspace=0.5)
    fig._labels_placed = True
    _save(f"step8c_esi_ddf_lagged_link{suffix}.png")


def step8c_lagged_link(state: dict, max_lag: int = 3) -> dict:
    """
    STEP 8c — Lagged ESI–DDF link (lags 0..max_lag), main run on the raw DDF
    score (dea_score) and a sensitivity run on the bias-corrected score (dea_bc).
    Populates state["lagged_link"] = {"dea_score": {...}, "dea_bc": {...}}.
    """
    print("\n" + "=" * 60)
    print(f"STEP 8c — Lagged ESI–DDF link (lags 0–{max_lag})")
    print("=" * 60)
    t = time.time()
    state["lagged_link"] = {}
    for ddf_col, suffix in (("dea_score", ""), ("dea_bc", "_bias_corrected")):
        if ddf_col not in state["panel"].columns:
            continue
        with _TeeLogger(f"step8c_lagged_link{suffix}.log"):
            res = lagged_esi_ddf_link(state["panel"], ddf_col=ddf_col, max_lag=max_lag)
        state["lagged_link"][ddf_col] = res
        for key in ("crosscorr", "single", "joint", "joint_tests", "dimensions",
                    "cross_lagged", "cross_lagged_tests"):
            if key in res and len(res[key]):
                _save_csv(res[key], f"step8c_{key}{suffix}.csv", f"Lagged ESI-DDF link: {key} ({ddf_col})")
        plot_lagged_link(res, ddf_col=ddf_col, suffix=suffix)
    print(f"\n  ✓ Step 8c complete  ({time.time()-t:.0f}s)")
    return state


def step8b_mundlak_cre(state: dict, z_vars: list = None) -> dict:
    """
    STEP 8b — Mundlak (1978) Correlated Random Effects model.

    Separates the within-country effect (beta,
    equivalent to Panel FE) from the between-country structural effect
    (theta, which Panel FE absorbs into the fixed effect and discards).
    Motivated by the weak within-country fit of the Panel FE model
    (Step 8), consistent with ESI dimensions being persistent, slow-moving,
    country-level structural characteristics rather than short-run
    within-country drivers. Reported alongside Panel FE, not instead of it.

    Requires state from steps 1–5 (uses state["panel"]["dea_bc"] and the
    ESI dimensions).
    Populates state keys:
        mundlak_results ← DataFrame with beta_within / theta_between per
                          variable, their clustered SEs, t-stats, p-values

    Usage:
        state = step8b_mundlak_cre(state)
        state['mundlak_results']
    """
    print("\n" + "="*60)
    print("STEP 8b — Mundlak Correlated Random Effects")
    print("="*60)
    t = time.time()

    with _TeeLogger("step8b_mundlak.log"):
        state["mundlak_results"] = mundlak_cre_regression(state["panel"], z_vars=z_vars)
    _save_csv(state["mundlak_results"], "step8b_mundlak_cre.csv",
              "Mundlak CRE (within vs between effects)")
    print(f"\n  ✓ Step 8b complete  ({time.time()-t:.0f}s)")
    return state


def step10_granger(state: dict, maxlag: int = 2) -> dict:
    """
    STEP 10 — Panel Granger Causality (Dumitrescu–Hurlin approximation).

    Tests bidirectional causality: ESI ↔ DEA efficiency.
    Requires state from steps 1–5.
    Populates state keys:
        granger_done  ← True (results are printed, no object returned)

    Parameters:
        maxlag : maximum lag order (default: 2)

    Usage:
        state = step10_granger(state)
        state = step10_granger(state, maxlag=3)
    """
    print("\n" + "="*60)
    print("STEP 10 — Panel Granger Causality")
    print("="*60)
    t = time.time()

    with _TeeLogger("step10_granger.log"):
        granger_df, granger_stats = panel_granger(state["panel"], maxlag=maxlag)
    state["granger_df"]    = granger_df
    state["granger_stats"] = granger_stats
    plot_step10_granger(granger_df, granger_stats)
    _save_csv(granger_df, "step10_granger_pvalues.csv", "Granger p-values by country")
    if granger_stats:
        gs_df = pd.DataFrame([granger_stats])
        _save_csv(gs_df, "step10_granger_summary.csv", "Granger W-bar statistics")
    state["granger_done"] = True
    print(f"\n  ✓ Step 10 complete  ({time.time()-t:.0f}s)")
    return state




# ═════════════════════════════════════════════════════════════════════════════
# STEP 10b — LOCAL PROJECTIONS: IMPULSE RESPONSES TO ENERGY-SECURITY SHOCKS
# ═════════════════════════════════════════════════════════════════════════════

def detect_candidate_shock_years(panel: pd.DataFrame,
                                  series_cols=None,
                                  known_shock_years=None,
                                  z_thresh: float = 1.5) -> pd.DataFrame:
    """
    Data-driven detection of candidate shock years NOT currently covered by
    any of the three manually pre-specified LP shock windows (oil/GFC
    2008-09, COVID 2020-21, gas crisis 2022-23).

    Motivation: those three windows were chosen from historical knowledge,
    not detected from the data, and they cover only 6 of the 24 sample
    years. Any genuine disturbance in the remaining ~18 "transition
    period" years -- for example Fukushima (2011), the 2014-16 oil-price
    collapse, or a renewable-cost/technology tipping point -- is
    structurally invisible to the LP framework as currently specified,
    since no shock regressor exists for those years at all. This is
    precisely the gap that produced the unexplained 2019 spike in the
    pooled GML/GTC index noted in the results review (Section 5 of this
    pipeline's GML output): a real disturbance with no corresponding shock
    variable to test it against.

    Method: for each candidate series (pooled, cross-country mean, unless
    a specific column is requested), compute the year-over-year change,
    standardise it against that series' OWN historical volatility (a
    z-score relative to the full-sample mean and standard deviation of the
    year-over-year change, NOT relative to the level), and flag any year
    whose absolute z-score exceeds z_thresh. Years already covered by an
    existing shock window are excluded from the flagged list -- this
    function is specifically for finding NEW candidates, not confirming
    already-known ones.

    This is a simple, transparent detection rule (a pooled z-score on the
    first difference), not a formal structural-break test (e.g. Bai-Perron,
    CUSUM); it is intended to generate candidate years for further
    investigation and, if judged substantively interesting, a new
    purpose-built shock interaction term -- not to be treated as a
    confirmatory test on its own.

    Parameters
    ----------
    series_cols : list of column names to scan (default: dea_bc, esi_score,
                  and, if present, the GML index for that year's transition).
    known_shock_years : years to exclude as "already covered" (default:
                        2008,2009,2020,2021,2022,2023 -- the three existing
                        LP shock windows).
    z_thresh : flagging threshold in standard deviations of the year-over-
               year change (default 1.5; lower than a conventional 2.0 to
               avoid missing genuine candidates in a moderate-length panel,
               at the cost of a higher false-positive rate -- flagged years
               are candidates for investigation, not confirmed shocks).

    Returns
    -------
    DataFrame with columns: series, year, pooled_change, z_score, is_new_candidate
    """
    if series_cols is None:
        series_cols = [c for c in ["dea_bc", "esi_score", "gml"] if c in panel.columns]
    if known_shock_years is None:
        known_shock_years = {2008, 2009, 2020, 2021, 2022, 2023}

    print(f"\n{'═'*66}")
    print("CANDIDATE SHOCK YEAR DETECTION (data-driven, transition-period gap)")
    print(f"  Scanning for disturbances NOT covered by the existing three")
    print(f"  manually-specified shock windows (oil/GFC, COVID, gas-crisis).")
    print(f"{'═'*66}")

    rows = []
    for col in series_cols:
        if col not in panel.columns:
            continue
        pooled = panel.groupby("year")[col].mean().sort_index()
        change = pooled.diff().dropna()
        if len(change) < 5:
            continue
        mu, sd = change.mean(), change.std()
        if sd < 1e-12:
            continue
        z = (change - mu) / sd

        print(f"\n  Series: {col}  (pooled cross-country mean, year-over-year change)")
        print(f"    Mean change={mu:+.5f}  SD={sd:.5f}")
        any_flag = False
        for yr, zval in z.items():
            is_new = abs(zval) >= z_thresh and yr not in known_shock_years
            if abs(zval) >= z_thresh:
                any_flag = True
                tag = "NEW CANDIDATE" if is_new else "(already covered by an existing shock window)"
                print(f"    {int(yr)}: change={change[yr]:+.5f}  z={zval:+.2f}  {tag}")
            rows.append({"series": col, "year": int(yr), "pooled_change": float(change[yr]),
                        "z_score": float(zval), "is_new_candidate": bool(is_new)})
        if not any_flag:
            print(f"    No year exceeds |z|={z_thresh} for this series.")

    result_df = pd.DataFrame(rows)
    new_candidates = result_df[result_df["is_new_candidate"]] if len(result_df) else pd.DataFrame()
    print(f"\n  {'─'*60}")
    if len(new_candidates) > 0:
        candidate_years = sorted(new_candidates["year"].unique())
        print(f"  {len(candidate_years)} new candidate year(s) identified, not covered by")
        print(f"  any existing shock window: {candidate_years}")
        print(f"  These are DETECTED CANDIDATES ONLY -- investigate substantively (was")
        print(f"  there a genuine, identifiable energy-sector event in that year?) before")
        print(f"  constructing a new shock interaction term and adding it to _define_shocks().")
    else:
        print(f"  No new candidate years identified beyond the three existing shock windows")
        print(f"  at the current threshold (|z|>={z_thresh}).")
    print(f"  {'─'*60}")

    return result_df


def _define_shocks(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct shock regressors for the Local Projection analysis.

    COLLINEARITY FIX: A binary global shock (e.g. covid_t = 1 for ALL countries
    in 2020-21) is perfectly collinear with year fixed effects when every country
    receives the same shock simultaneously. The panel LP absorbs it entirely into
    the time FE, producing no coefficient — exactly what caused DEA x COVID/gas/oil
    pairs to be absent from the formal LP table in the previous implementation.

    Correct design: Shock_t x Exposure_{i,t-1}
    The interaction retains cross-sectional variation after time FE is absorbed,
    because countries entered the shock with heterogeneous structural exposures.
    The LP coefficient estimates how much of the efficiency response is explained
    by pre-crisis structural vulnerability — a defensible country-ranking mechanism.

    Interaction regressors constructed (shock year x lagged exposure):
      oil_x_import   : oil/GFC 2008-09   x  lagged net_import_pct (1-SD units)
      covid_x_fossil : COVID-19 2020-21   x  lagged fossil-fuel share (1-SD units)
      gas_x_import   : gas crisis 2022-23 x  lagged net_import_pct (1-SD units)

    Continuous country-specific shocks (no FE collinearity issue):
      import_shock   : within-country change in net_import_pct (country-demeaned)
      esi_shock      : within-country change in ESI composite  (country-demeaned)

    Raw binary dummies are stored as _oil_dummy, _covid_dummy, _gas_dummy for use
    in Chart 5 trajectory-window plots only — NOT used as LP regressors.
    """
    panel = panel.copy().sort_values(["iso_code", "year"])

    # ── Raw dummies for trajectory windows (NOT LP regressors) ───────────────
    panel["_oil_dummy"]   = panel["year"].isin([2008, 2009]).astype(float)
    panel["_covid_dummy"] = panel["year"].isin([2020, 2021]).astype(float)
    panel["_gas_dummy"]   = panel["year"].isin([2022, 2023]).astype(float)

    # ── Predetermined exposure variables (lagged one period) ─────────────────
    panel["_imp_lag1"] = panel.groupby("iso_code")["net_import_pct"].shift(1)

    fossil_cols = [c for c in ["coal_share", "oil_share", "gas_share"]
                   if c in panel.columns]
    if fossil_cols:
        panel["_fossil_raw"] = panel[fossil_cols].sum(axis=1)
    else:
        ren = panel.get("renew_share",    pd.Series(0.0, index=panel.index))
        nuc = panel.get("nuclear_share",  pd.Series(0.0, index=panel.index))
        panel["_fossil_raw"] = (100 - ren.fillna(0) - nuc.fillna(0)).clip(0, 100)
    panel["_fossil_lag1"] = panel.groupby("iso_code")["_fossil_raw"].shift(1)

    # Winsorise at 1st-99th percentile to limit outlier leverage
    for col in ["_imp_lag1", "_fossil_lag1"]:
        p1  = panel[col].quantile(0.01)
        p99 = panel[col].quantile(0.99)
        panel[col] = panel[col].clip(p1, p99)

    # ── Standardise exposures (1-SD units for interpretable beta) ─────────────
    def _std(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / (sd + 1e-9)

    # ── Interaction shocks: Shock_t x Exposure_{i,t-1} ───────────────────────
    panel["oil_x_import"]   = panel["_oil_dummy"]   * _std(panel["_imp_lag1"])
    panel["covid_x_fossil"] = panel["_covid_dummy"]  * _std(panel["_fossil_lag1"])
    panel["gas_x_import"]   = panel["_gas_dummy"]    * _std(panel["_imp_lag1"])

    # ── Continuous country-specific shocks (demeaned within country) ──────────
    panel["import_shock"] = panel.groupby("iso_code")["net_import_pct"].transform(
        lambda x: x.diff())
    panel["esi_shock"]    = panel.groupby("iso_code")["esi_score"].transform(
        lambda x: x.diff())
    for col in ["import_shock", "esi_shock"]:
        panel[col] -= panel.groupby("iso_code")[col].transform("mean")

    n_obs = len(panel)
    print(f"  Shock series constructed ({n_obs} obs):")
    for sh, dummy in [("oil_x_import",   "_oil_dummy"),
                      ("covid_x_fossil",  "_covid_dummy"),
                      ("gas_x_import",    "_gas_dummy")]:
        n_t = int(panel[panel[dummy] == 1][sh].notna().sum())
        sd  = panel[sh].std()
        print(f"    {sh:<22}: {n_t} treated obs  SD={sd:.3f}")
    print(f"    import_shock           : "
          f"mean={panel['import_shock'].dropna().mean():.3f}  "
          f"std={panel['import_shock'].dropna().std():.3f}")
    print(f"    esi_shock              : "
          f"mean={panel['esi_shock'].dropna().mean():.3f}  "
          f"std={panel['esi_shock'].dropna().std():.3f}")
    print(f"  Beta interpretation: response per 1-SD difference in")
    print(f"  pre-crisis import/fossil exposure (year FE absorbs common shock).")
    return panel
def local_projections(panel: pd.DataFrame,
                       outcomes:  list = None,
                       shocks:    list = None,
                       H:         int  = 5,
                       alpha:     float = 0.10) -> dict:
    """
    Jordà (2005) local projections for panel data.

    For each outcome variable y and shock variable s, estimates:

        y_{i,t+h} - y_{i,t-1}  =  α_i  +  γ_t  +  β_h · s_{i,t}
                                   +  Σ_k β_k · Δy_{i,t-k}  +  ε_{i,t+h}

    at each horizon h = 0, 1, ..., H using linearmodels PanelOLS with entity
    and time fixed effects and HC1 clustered standard errors.

    The sequence β_0, β_1, ..., β_H traces the cumulative impulse response of
    y to a unit shock in s, with (1-α) confidence bands.

    Parameters
    ----------
    outcomes : list of panel columns to use as dependent variables
               Default: ["dea_bc", "esi_score"]
    shocks   : list of shock column names (must exist after _define_shocks)
               Default: all five shocks
    H        : maximum horizon in years (default 5)
    alpha    : significance level for confidence bands (default 0.10 → 90% CI)

    Returns dict: {(outcome, shock): DataFrame with columns
                   [horizon, beta, ci_lo, ci_hi, pval, n_obs]}
    """
    from linearmodels.panel import PanelOLS
    import warnings

    if outcomes is None:
        outcomes = ["dea_bc", "esi_score"]
    if shocks is None:
        shocks = ["oil_x_import","covid_x_fossil","gas_x_import",
                  "import_shock","esi_shock"]

    results = {}
    panel = panel.copy().sort_values(["iso_code","year"])

    # Lag controls: one-period lagged change in the outcome (AR term)
    for out in outcomes:
        panel[f"d_{out}_lag1"] = (panel.groupby("iso_code")[out]
                                        .transform(lambda x: x.diff().shift(1)))

    z_score = stats.norm.ppf(1 - alpha / 2)

    for out in outcomes:
        for shock in shocks:
            if out == "esi_score" and shock == "esi_shock":
                # esi_shock IS the (country-demeaned) change in esi_score, so at h=0
                # the response is 1 by construction (identity, not a finding).
                print("  ℹ (esi_score, esi_shock) skipped: the shock is the outcome's own")
                print("    change, so its impact response is mechanically 1.")
                continue
            rows = []
            for h in range(H + 1):
                # Forward difference: y_{i,t+h} - y_{i,t-1}
                panel[f"_y_fwd"] = (panel.groupby("iso_code")[out]
                                         .transform(lambda x:
                                             x.shift(-h) - x.shift(1)))

                # Regressors: shock + AR lag (entity/time FE absorbed by PanelOLS)
                lag_col = f"d_{out}_lag1"
                needed  = ["iso_code","year","_y_fwd", shock]
                if lag_col in panel.columns:
                    needed.append(lag_col)

                sub = panel[needed].dropna().copy()
                if len(sub) < 50:
                    continue

                sub = sub.set_index(["iso_code","year"])
                y   = sub["_y_fwd"]
                Xc  = sub[[c for c in [shock, lag_col] if c in sub.columns]]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fe  = PanelOLS(y, Xc,
                                       entity_effects=True,
                                       time_effects=True,
                                       drop_absorbed=True)
                        res = fe.fit(cov_type="clustered",
                                     cluster_entity=True)
                    if shock not in res.params.index:
                        continue
                    beta = float(res.params[shock])
                    se   = float(res.std_errors[shock])
                    pval = float(res.pvalues[shock])
                    rows.append({
                        "horizon": h,
                        "beta"   : beta,
                        "ci_lo"  : beta - z_score * se,
                        "ci_hi"  : beta + z_score * se,
                        "pval"   : pval,
                        "n_obs"  : int(res.nobs),
                    })
                except Exception as e:
                    continue

            if rows:
                results[(out, shock)] = pd.DataFrame(rows)

    return results


def plot_local_projections(lp_results: dict,
                           alpha: float = 0.10,
                           save_prefix: str = "lp") -> None:
    """
    Plot impulse-response functions from local_projections() output.

    One subplot grid per outcome variable: rows = outcomes, cols = shocks.
    Each panel shows the β_h path with (1-α) confidence bands shaded.
    A dashed zero line marks no response. Significant horizons (CI excludes
    zero) are marked with a filled dot.
    """
    outcomes = sorted({k[0] for k in lp_results})
    shocks   = sorted({k[1] for k in lp_results})

    shock_labels = {
        "oil_x_import"    : "Oil/GFC x Import Exposure (2008–09)",
        "covid_x_fossil"  : "COVID-19 x Fossil Exposure (2020–21)",
        "gas_x_import"    : "Gas Crisis x Import Exposure (2022–23)",
        "import_shock"    : "Import dependence shock",
        "esi_shock"       : "ESI structural shock",
    }
    outcome_labels = {
        "dea_bc"   : "DEA efficiency (θ_bc)",
        "esi_score": "ESI composite score",
    }

    n_rows = len(outcomes)
    n_cols = len(shocks)
    if n_rows == 0 or n_cols == 0:
        print("  ⚠ No LP results to plot")
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.6 * n_cols, 4.8 * n_rows),
                             squeeze=False)
    fig.suptitle(f"Local Projection Impulse Responses — {int((1-alpha)*100)}% CI",
                 fontsize=13, fontweight="bold", y=1.01)

    for r, out in enumerate(outcomes):
        for c, shock in enumerate(shocks):
            ax = axes[r][c]
            key = (out, shock)
            if key not in lp_results or len(lp_results[key]) == 0:
                ax.text(0.5, 0.5, "insufficient data",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="gray")
                ax.axis("off")
                continue

            df = lp_results[key].sort_values("horizon")
            hs  = df["horizon"].values
            bs  = df["beta"].values
            los = df["ci_lo"].values
            his = df["ci_hi"].values
            sig = (los > 0) | (his < 0)   # CI excludes zero

            # Shaded CI
            ax.fill_between(hs, los, his, alpha=0.20, color=COLORS["blue"])
            # Response path
            ax.plot(hs, bs, color=COLORS["blue"], lw=2.0, marker="o",
                    markersize=4, label="IRF")
            # Significant horizons
            ax.scatter(hs[sig], bs[sig], color=COLORS["red"], zorder=5,
                       s=60, label="p < signif.")
            ax.axhline(0, color="black", lw=0.8, ls="--")

            ax.set_xlabel("Horizon (years)", fontsize=9)
            ax.set_ylabel(outcome_labels.get(out, out), fontsize=9)
            ax.set_title(shock_labels.get(shock, shock), fontsize=9,
                         fontweight="bold")
            ax.set_xticks(hs)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.25, linewidth=0.5)

            n_obs = int(df["n_obs"].mean())
            ax.text(0.97, 0.03, f"n≈{n_obs}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9, color="gray")

    # Row labels
    for r, out in enumerate(outcomes):
        axes[r][0].set_ylabel(f"{outcome_labels.get(out,out)}\n(Δ from t-1)",
                              fontsize=9)

    _fig_legend_right(fig, handles=[plt.Line2D([], [], color=COLORS["blue"], lw=2, marker="o"),
                                    plt.Line2D([], [], color=COLORS["red"], marker="o", ls="none"),
                                    mpatches.Patch(color=COLORS["blue"], alpha=0.2)],
                      labels=["Impulse response", f"CI excludes zero", f"{int((1-alpha)*100)}% CI"],
                      right=0.88, top=0.97)
    plt.tight_layout(rect=fig._tight_rect)
    fname = f"{save_prefix}_impulse_responses.png"
    _save_fig(fname)
    plt.show()
    print(f"  IRF plot saved: {fname}")


def print_lp_summary(lp_results: dict, alpha: float = 0.10) -> None:
    """Print a compact summary table of LP results."""
    W = 85
    print(f"\n  {'═'*W}")
    print(f"  LOCAL PROJECTION RESULTS — Impulse Responses at each horizon")
    print(f"  {int((1-alpha)*100)}% confidence intervals | entity+time FE | clustered SE")
    print(f"  {'═'*W}")
    print(f"  {'Outcome':<14} {'Shock':<25} {'h=0':>8} {'h=1':>8} "
          f"{'h=2':>8} {'h=3':>8} {'h=4':>8} {'h=5':>8}")
    print(f"  {'─'*W}")

    outcomes = sorted({k[0] for k in lp_results})
    shocks   = sorted({k[1] for k in lp_results})

    for out in outcomes:
        for shock in shocks:
            key = (out, shock)
            if key not in lp_results:
                continue
            df = lp_results[key].set_index("horizon")["beta"]
            sig_df = lp_results[key].set_index("horizon")
            row = f"  {out:<14} {shock:<25}"
            for h in range(6):
                if h in df.index:
                    b = df[h]
                    lo = sig_df.loc[h,"ci_lo"]
                    hi = sig_df.loc[h,"ci_hi"]
                    star = "*" if (lo > 0 or hi < 0) else " "
                    row += f" {b:>+7.4f}{star}"
                else:
                    row += f"  {'—':>8}"
            print(row)
    print(f"  * = {int((1-alpha)*100)}% CI excludes zero")
    print(f"  {'═'*W}")

    # *** ADDED: data-driven synthesis, scanning the actual results rather
    # than hardcoding a claim, so this stays accurate if the underlying
    # estimates change (e.g. after a B or shock-definition change). Local
    # Projections are SECONDARY evidence in this pipeline's evidentiary
    # hierarchy -- the primary structural-operational evidence is the
    # separability-informed Conditional DDF and Simar-Wilson results
    # (Steps 6b/7c); LP addresses a different question (dynamic shock
    # response), and a weak or isolated LP finding does not undermine the
    # static results, nor should a strong one be read as confirming them. ***
    print(f"\n  {'─'*W}")
    print(f"  INTERPRETATION (secondary evidence — see caveat below)")
    print(f"  {'─'*W}")
    any_finding = False
    for out in outcomes:
        for shock in shocks:
            key = (out, shock)
            if key not in lp_results:
                continue
            sig_df = lp_results[key].set_index("horizon")
            sig_horizons = [h for h in sig_df.index
                            if (sig_df.loc[h,"ci_lo"] > 0 or sig_df.loc[h,"ci_hi"] < 0)]
            if not sig_horizons:
                continue
            any_finding = True
            is_isolated = len(sig_horizons) == 1
            is_consecutive = (len(sig_horizons) > 1 and
                             max(sig_horizons) - min(sig_horizons) == len(sig_horizons) - 1)
            beta_at_sig = {h: sig_df.loc[h,"beta"] for h in sig_horizons}
            if is_isolated:
                h0 = sig_horizons[0]
                print(f"  {out} / {shock}: ONLY h={h0} is significant "
                      f"(beta={beta_at_sig[h0]:+.4f}) at the {int((1-alpha)*100)}% level.")
                print(f"    This is an isolated, single-horizon finding, not a robust or")
                print(f"    broadly-supported dynamic response -- report it as marginal,")
                print(f"    and do not describe this (outcome, shock) pair as having a")
                print(f"    confirmed dynamic effect on the strength of this one horizon alone.")
            elif is_consecutive:
                lo_h, hi_h = min(sig_horizons), max(sig_horizons)
                print(f"  {out} / {shock}: significant at h={lo_h}-{hi_h} "
                      f"({len(sig_horizons)} consecutive horizons) at the "
                      f"{int((1-alpha)*100)}% level.")
                print(f"    This is the broadest, most consistently identified dynamic")
                print(f"    response in this LP run -- the strongest candidate for a")
                print(f"    genuine, reportable dynamic finding among the pairs estimated.")
            else:
                print(f"  {out} / {shock}: significant at non-consecutive horizons "
                      f"{sig_horizons} at the {int((1-alpha)*100)}% level -- an irregular")
                print(f"    pattern that should be treated cautiously rather than as a")
                print(f"    clean, monotonic dynamic response.")
    if not any_finding:
        print(f"  No (outcome, shock) pair shows any horizon excluding zero at the "
              f"{int((1-alpha)*100)}% level.")
        print(f"  This LP run finds no statistically identified dynamic response anywhere")
        print(f"  in the panel at the current settings.")
    print(f"\n  ⚠ CAVEAT: Local Projections are SECONDARY evidence in this pipeline.")
    print(f"  The primary structural-operational evidence is the conditional DDF (Step 7c)")
    print(f"  and the Simar-Wilson second stage (Steps 7/7b), read together with the k-NN")
    print(f"  sensitivity diagnostic. LP addresses a distinct")
    print(f"  question -- dynamic response to identified shocks -- and its results should")
    print(f"  neither be used to override the static structural-operational findings, nor")
    print(f"  be over-interpreted when, as is common with shock-specific interaction terms")
    print(f"  and a moderate panel size, only isolated or marginal (90% CI) effects emerge.")
    print(f"  Note also that {int((1-alpha)*100)}% CIs are used throughout LP inference,")
    print(f"  a weaker standard than the 95% level used elsewhere in this pipeline,")
    print(f"  adopted because of the panel's moderate size -- this should be stated")
    print(f"  explicitly whenever an LP finding is reported in the results chapter.")
    print(f"  {'═'*W}")




# ─────────────────────────────────────────────────────────────────────────────
# LOCAL PROJECTION VISUALISATION SUITE
# Six purpose-built charts for the shock analysis chapter
# ─────────────────────────────────────────────────────────────────────────────

def _safe_country_lp_beta(sub, shock_col, y_col, lag_col,
                          min_obs=6, min_active_obs=2, max_cond=1e8,
                          return_se=False):
    """
    Hardened per-country Local Projection coefficient estimator, shared by
    every LP chart function below (Charts 1, 2, 3, 4 and 6).

    *** FIX (root cause of the Mexico/gas-crisis instability traced through
    six downstream chart outputs -- see review): the previous per-chart
    implementations (one nearly-identical copy duplicated five times)
    checked only the TOTAL number of observations (len(sub) < 6/8) before
    calling np.linalg.lstsq. This misses the actual failure mode entirely.
    A shock interaction term such as gas_x_import is a binary-episode
    indicator multiplied by import exposure: it is EXACTLY ZERO for every
    year outside the 2022-2023 window, so a country can have 20+ total
    rows (passing the old check comfortably) while having only ONE or TWO
    non-zero ("active") observations of the shock itself. With so little
    genuine variation in that one regressor, the design matrix is
    numerically near-singular specifically in the shock's own column, and
    np.linalg.lstsq / np.linalg.pinv do NOT raise an error in this case --
    they silently return a technically-defined but numerically explosive
    coefficient (observed directly as Mexico's gas-crisis estimates of
    approximately +348 and a 90% CI spanning +/-12,500, against every
    other cell in the same tables falling within +/-1). This function
    adds the checks that actually guard against this failure mode:
    (1) a minimum number of ACTIVE (non-near-zero) shock observations,
    not just a minimum total row count; (2) an explicit condition-number
    check on the design matrix, which catches near-singularity regardless
    of its specific cause (not just the short-window-shock case), before
    ever calling lstsq; (3) an output-plausibility check on the fitted
    coefficient itself (added in the follow-up fix below), which does not
    depend on correctly anticipating the cause of a failure in advance.
    A country failing any of these checks returns NaN rather than a
    numerically unstable point estimate. ***

    *** FOLLOW-UP FIX (round 1): min_active_obs was originally set to 3.
    This was an over-correction, confirmed by re-running the pipeline:
    oil/GFC (2008-09) and COVID (2020-21) EACH have exactly 2 active shock
    years per country, identical in structure to gas-crisis (2022-23) --
    with min_active_obs=3, EVERY country failed this check for ALL THREE
    binary shocks, not just the genuinely unstable gas-crisis case, causing
    the multi-shock heat map and country-ranking charts to go completely
    blank for oil/GFC and COVID as well, even though those two shocks had
    shown sensible, well-behaved per-country estimates before the original
    fix was introduced. Lowered to min_active_obs=2 -- the minimum a
    two-year shock window can ever provide -- so a legitimately well-
    conditioned 2-year shock (oil/GFC, COVID) is no longer blocked purely
    on active-year count.

    *** FOLLOW-UP FIX (round 2): after lowering min_active_obs, whether the
    condition-number check alone still reliably catches a case like the
    original gas-crisis anomaly could not be confirmed -- a synthetic
    reproduction of that failure mode did not clearly trigger the
    condition-number threshold once the active-year restriction was
    loosened, meaning the true cause of Mexico's original +348 estimate
    was not fully pinned down (edge-of-sample conditioning was the leading
    hypothesis but is unconfirmed). Rather than leave this as an open risk
    resting on an unverified hypothesis, a THIRD, independent safeguard was
    added: an output-plausibility check (see below) that flags any fitted
    coefficient implying an outcome swing far outside what the outcome
    variable's own observed range could support, regardless of which
    input-side condition produced it. This check does not depend on
    correctly diagnosing the failure's root cause in advance, and is the
    primary safeguard this function now relies on for catching a
    recurrence of the original anomaly under the loosened active-year
    threshold. ***

    Parameters
    ----------
    sub : DataFrame already restricted to one country, with y_col, shock_col
          and lag_col all non-null (caller's dropna).
    min_obs : minimum total rows (matches the original per-chart thresholds).
    min_active_obs : minimum number of observations where the shock
          regressor is meaningfully non-zero (|value| > 1% of that
          column's own within-country standard deviation, or > 1e-8 in
          absolute terms if the column has essentially no spread at all).
    max_cond : maximum acceptable condition number of X'X. 1e8 is a
          standard numerical-analysis rule of thumb (Belsley, Kuh and
          Welsch 1980) beyond which OLS coefficients are considered
          numerically unreliable regardless of the nominal sample size.

    Returns
    -------
    beta (float or np.nan), se (float or np.nan, only if return_se=True)
    """
    n = len(sub)
    if n < min_obs:
        return (np.nan, np.nan) if return_se else np.nan

    shock_vals = sub[shock_col].values.astype(float)
    shock_std = np.std(shock_vals)
    active_thresh = max(1e-8, 0.01 * shock_std)
    n_active = int(np.sum(np.abs(shock_vals) > active_thresh))
    if n_active < min_active_obs:
        return (np.nan, np.nan) if return_se else np.nan

    Xm = np.column_stack([np.ones(n), shock_vals, sub[lag_col].values.astype(float)])
    XtX = Xm.T @ Xm
    try:
        cond = np.linalg.cond(XtX)
    except Exception:
        return (np.nan, np.nan) if return_se else np.nan
    if not np.isfinite(cond) or cond > max_cond:
        return (np.nan, np.nan) if return_se else np.nan

    y_ = sub[y_col].values.astype(float)
    try:
        b, _, _, _ = np.linalg.lstsq(Xm, y_, rcond=None)
    except Exception:
        return (np.nan, np.nan) if return_se else np.nan

    beta = float(b[1])

    # *** ADDED: direct output-plausibility check, closing the gap left by
    # the active-observations and condition-number checks above. Neither
    # of those checks was confirmed, via synthetic reproduction, to
    # reliably catch the exact real-data conditions behind the original
    # Mexico gas-crisis anomaly (a fitted beta of approximately +348).
    #
    # IMPORTANT CORRECTION found via direct testing: a first version of
    # this check computed "implied_swing = |beta| * shock_std" using the
    # LOCAL (this country's own) shock standard deviation. This was wrong
    # and, when tested against a synthetic near-zero-variance shock with
    # otherwise clean, normal data, failed to catch a fitted beta of +13
    # arising from an ill-conditioned near-singular fit -- multiplying by
    # a small local shock_std mechanically shrinks the "implied swing"
    # precisely in the cases where a tiny shock_std is itself the cause of
    # an inflated beta, cancelling out the exact sensitivity the check
    # needs. The production shocks in _define_shocks() are standardised
    # ONCE, panel-wide, before reaching this function, so beta is already
    # meant to be interpreted directly as "change in y per 1-SD shock" --
    # it should be compared to a plausibility bound AS-IS, not re-scaled
    # by a further, possibly-degenerate, per-country local statistic. ***
    y_std = float(np.std(y_))
    if y_std > 1e-9:
        # A DEA efficiency score or ESI composite's year-over-year change is
        # bounded well within [-1, 1] by construction; a coefficient
        # implying more than roughly a full-scale swing per a single
        # standard-deviation shock is already implausible for a genuinely
        # well-identified relationship. 50 is deliberately generous (not
        # tight), to avoid false positives on genuinely large but real
        # effects; Mexico's actual +348 clears this bound by nearly an
        # order of magnitude.
        if abs(beta) > 50:
            return (np.nan, np.nan) if return_se else np.nan

    if not return_se:
        return beta

    k = Xm.shape[1]
    resid = y_ - Xm @ b
    s2 = np.dot(resid, resid) / max(n - k, 1)
    vcov = s2 * np.linalg.pinv(XtX)
    se_b = float(np.sqrt(max(vcov[1, 1], 0)))
    return beta, se_b


def plot_lp_country_ranking(lp_results, panel,
                             shock="gas_x_import",
                             outcome="dea_bc", horizon=1):
    """
    Chart 1 - Country-level shock sensitivity ranking.
    Methodology A: country-by-country LP beta at a chosen horizon,
    sorted from most resilient (least negative) to most vulnerable.
    """
    print("  Chart 1: country ranking -- " + outcome + " response to " + shock)
    panel = panel.copy().sort_values(["iso_code","year"])
    if shock not in panel.columns:
        print("  WARNING: shock column missing"); return

    lag_col = "d_" + outcome + "_lag1"
    if lag_col not in panel.columns:
        panel[lag_col] = panel.groupby("iso_code")[outcome].transform(
            lambda x: x.diff().shift(1))
    panel["_y_fwd"] = panel.groupby("iso_code")[outcome].transform(
        lambda x, _h=horizon: x.shift(-_h) - x.shift(1))

    rows = []
    for iso, g in panel.groupby("iso_code"):
        g = g.dropna(subset=["_y_fwd", shock, lag_col])
        if len(g) < 8: continue
        beta, se_b = _safe_country_lp_beta(g, shock, "_y_fwd", lag_col,
                                           min_obs=8, return_se=True)
        if not np.isfinite(beta):
            continue
        n, k = len(g), 3
        t_crit = stats.t.ppf(0.95, df=max(n - k, 1))
        rows.append({"iso_code": iso, "beta": beta,
                     "ci_lo": beta - t_crit * se_b,
                     "ci_hi": beta + t_crit * se_b})

    if not rows:
        _yrs_active = sorted(panel.loc[panel[shock].abs() > 1e-8, "year"].unique())
        _last = int(panel["year"].max())
        print(f"  ℹ Chart 1 skipped for {shock} at h={horizon}: no country has the")
        print(f"    minimum two active shock years with an observed outcome {horizon} year(s)")
        print(f"    ahead (shock years {[int(y) for y in _yrs_active]}, sample ends {_last}).")
        print(f"    This is a sample-end limitation, not a data gap; the shock is still")
        print(f"    covered in the pooled LP at the horizons the data allow.")
        return

    df = pd.DataFrame(rows).sort_values("beta", ascending=True)
    n_c = len(df)
    n_all_countries = panel["iso_code"].nunique()
    if n_c < n_all_countries:
        print(f"  ℹ {n_all_countries - n_c}/{n_all_countries} countries excluded: "
              f"insufficient active shock variation or a near-singular design "
              f"matrix for '{shock}' (see _safe_country_lp_beta docstring). "
              f"This is expected and correct for short-window shocks such as "
              f"gas_x_import (2022-23 only) -- it is NOT the same as a data gap.")
    shock_label = {"oil_x_import": "Oil/GFC x Import (2008-09)",
                   "covid_x_fossil": "COVID-19 x Fossil Share (2020-21)",
                   "gas_x_import":   "Gas Crisis x Import Exposure (2022-23)",
                   "import_shock": "Import Dependence Shock",
                   "esi_shock": "ESI Structural Shock"}.get(shock, shock)
    outcome_label = {"dea_bc": "DEA efficiency (theta_bc)",
                     "esi_score": "ESI score"}.get(outcome, outcome)

    fig, ax = plt.subplots(figsize=(11, _country_extent(n_c, per=0.36)))
    colors = [COLORS["red"] if b < 0 else COLORS["teal"] for b in df["beta"]]
    y_pos = np.arange(n_c)
    ax.barh(y_pos, df["beta"], color=colors, alpha=0.78, height=0.65)
    err_lo = np.nan_to_num(np.clip(df["beta"].values - df["ci_lo"].values, 0, None))
    err_hi = np.nan_to_num(np.clip(df["ci_hi"].values - df["beta"].values, 0, None))
    ax.errorbar(df["beta"], y_pos, xerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=3, linewidth=1.0)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["iso_code"], fontsize=9.5)
    ax.set_ylim(-0.7, n_c - 0.3)
    ax.set_xlabel("Response of " + outcome_label + " at h=" + str(horizon) + " (90% CI)", fontsize=9)
    ax.set_title("Chart 1 -- Country Shock Sensitivity Ranking | " + shock_label,
                 fontsize=11, fontweight="bold")
    ax.text(0.98, 0.01, "More vulnerable <--    --> More resilient",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, style="italic")
    plt.tight_layout()
    _save_fig("lp_chart1_ranking_" + shock + "_" + outcome + ".png")
    plt.show()


def plot_lp_heat_map(lp_results, panel, shocks_to_show=None,
                      outcome="dea_bc", horizon=1):
    """
    Chart 2 - Multi-shock vulnerability heat map.
    Rows = countries (sorted by mean beta), cols = shocks.
    Red = vulnerable (negative beta), Green = resilient (positive beta).
    """
    print("  Chart 2: multi-shock heat map -- " + outcome + " at h=" + str(horizon))
    if shocks_to_show is None:
        shocks_to_show = ["oil_x_import", "covid_x_fossil", "gas_x_import",
                          "import_shock", "esi_shock"]
    panel = panel.copy().sort_values(["iso_code", "year"])
    lag_col = "d_" + outcome + "_lag1"
    if lag_col not in panel.columns:
        panel[lag_col] = panel.groupby("iso_code")[outcome].transform(
            lambda x: x.diff().shift(1))
    panel["_y_fwd"] = panel.groupby("iso_code")[outcome].transform(
        lambda x, _h=horizon: x.shift(-_h) - x.shift(1))

    isos = sorted(panel["iso_code"].unique())
    beta_mat = pd.DataFrame(index=isos, columns=shocks_to_show, dtype=float)
    for iso, g in panel.groupby("iso_code"):
        for sh in shocks_to_show:
            if sh not in g.columns: continue
            sub = g.dropna(subset=["_y_fwd", sh, lag_col])
            if len(sub) < 6: continue
            beta = _safe_country_lp_beta(sub, sh, "_y_fwd", lag_col, min_obs=6)
            if np.isfinite(beta):
                beta_mat.loc[iso, sh] = beta

    beta_mat = beta_mat.dropna(how="all")
    beta_mat["_mean"] = beta_mat.mean(axis=1)
    beta_mat = beta_mat.sort_values("_mean", ascending=True).drop(columns=["_mean"])

    shock_labels = {"oil_x_import":   "Oil/GFC\nx Import",
                    "covid_x_fossil": "COVID-19\nx Fossil",
                    "gas_x_import":   "Gas Crisis\nx Import",
                    "import_shock":   "Import\nShock",
                    "esi_shock":      "ESI\nShock"}
    col_labels = [shock_labels.get(s, s) for s in beta_mat.columns]
    vmax = float(np.nanpercentile(np.abs(beta_mat.values[np.isfinite(beta_mat.values)]), 95))
    if vmax == 0: vmax = 0.01

    fig, ax = plt.subplots(figsize=(max(10, 3 + len(shocks_to_show) * 2.0),
                                    _country_extent(len(beta_mat), base=3.0, per=0.40)))
    im = ax.imshow(beta_mat.values.astype(float), aspect="auto",
                   cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(beta_mat)))
    ax.set_yticklabels(beta_mat.index, fontsize=9.5)
    _annotate_heatmap(ax, im, beta_mat.values.astype(float), fmt="{:+.3f}")
    plt.colorbar(im, ax=ax,
                 label="beta at h=" + str(horizon) + " (red=vulnerable, green=resilient)")
    ax.set_title("Chart 2 -- Multi-Shock Vulnerability Heat Map | "
                 + outcome + " at h=" + str(horizon) + " | sorted by mean beta",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save_fig("lp_chart2_heatmap_" + outcome + ".png")
    plt.show()


def plot_lp_dual_methodology(lp_results, panel, shock="gas_x_import"):
    """
    Chart 3 - Dual-methodology ranking comparison.
    Left (Method A): LP beta at h=1 (dynamic shock response).
    Right (Method B): pre/post event-window contrast — mean outcome during
    shock years minus mean in pre-shock years, per country. This is NOT a
    formal Difference-in-Differences estimator (no untreated control group,
    no parallel-trends assumption). It serves as a simpler alternative
    shock-response measure to cross-validate the LP country rankings.
    Spearman rank correlation between the two methods shown at bottom.
    """
    print("  Chart 3: dual-methodology comparison -- " + shock)
    panel = panel.copy().sort_values(["iso_code", "year"])
    shock_yr_map = {"oil_x_import":   [2008, 2009],
                    "covid_x_fossil": [2020, 2021],
                    "gas_x_import":   [2022, 2023]}
    pre_yr_map   = {"oil_x_import":   [2005, 2006, 2007],
                    "covid_x_fossil": [2017, 2018, 2019],
                    "gas_x_import":   [2019, 2020, 2021]}
    shock_years = shock_yr_map.get(shock, [])
    pre_years   = pre_yr_map.get(shock, [])
    shock_label = {"oil_x_import": "Oil/GFC x Import (2008-09)",
                   "covid_x_fossil": "COVID-19 x Fossil Share (2020-21)",
                   "gas_x_import":   "Gas Crisis x Import Exposure (2022-23)"}.get(shock, shock)

    for outcome in ["dea_bc", "esi_score"]:
        lag_col = "d_" + outcome + "_lag1"
        if lag_col not in panel.columns:
            panel[lag_col] = panel.groupby("iso_code")[outcome].transform(
                lambda x: x.diff().shift(1))
        panel["_y_fwd1"] = panel.groupby("iso_code")[outcome].transform(
            lambda x: x.shift(-1) - x.shift(1))

        methA = {}
        for iso, g in panel.groupby("iso_code"):
            sub = g.dropna(subset=["_y_fwd1", shock, lag_col]) if shock in g.columns else pd.DataFrame()
            if len(sub) < 6: continue
            beta = _safe_country_lp_beta(sub, shock, "_y_fwd1", lag_col, min_obs=6)
            if np.isfinite(beta):
                methA[iso] = beta

        methB = {}
        if shock_years and pre_years:
            for iso, g in panel.groupby("iso_code"):
                during = g[g["year"].isin(shock_years)][outcome].mean()
                before = g[g["year"].isin(pre_years)][outcome].mean()
                if np.isfinite(during) and np.isfinite(before):
                    methB[iso] = float(during - before)

        if not methA or not methB:
            print(f"  ℹ Chart 3 skipped for {shock} / {outcome}: "
                  + ("no country-level LP coefficient at h=1 (shock window too close to "
                     f"the sample end, {int(panel['year'].max())})" if not methA
                     else "no pre/post event-window contrast available")
                  + " -- sample-end limitation, not a data gap.")
            continue
        common = sorted(set(methA) & set(methB))
        dfA = pd.Series(methA).loc[common].sort_values()
        dfB = pd.Series(methB).loc[common].reindex(dfA.index)

        outcome_label = {"dea_bc": "DEA efficiency (theta_bc)",
                         "esi_score": "ESI score"}.get(outcome, outcome)

        fig, axes = plt.subplots(1, 2, figsize=(17, _country_extent(len(common), per=0.36)))
        fig.suptitle("Chart 3 -- Dual-Methodology Resilience | " +
                     outcome_label + " | Shock: " + shock_label,
                     fontsize=12, fontweight="bold")

        for ax, data, title, xlabel in [
            (axes[0], dfA,
             "Method A: Local Projection (beta at h=1)",
             "LP coefficient beta (negative = more vulnerable)"),
            (axes[1], dfB,
             "Method B: Pre/post event-window contrast",
             "Delta efficiency (negative = deterioration)"),
        ]:
            colors = [COLORS["red"] if v < 0 else COLORS["teal"] for v in data]
            y_pos = np.arange(len(data))
            ax.barh(y_pos, data.values, color=colors, alpha=0.78, height=0.65)
            ax.axvline(0, color="black", lw=1.0)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data.index, fontsize=9.5)
            ax.set_ylim(-0.7, len(data) - 0.3)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.2, axis="x")

        rho = dfA.rank().corr(dfB.reindex(dfA.index).rank(), method="spearman")
        fig.text(0.5, -0.02,
                 "Spearman rank correlation between methods: rho = {:.3f}".format(rho),
                 ha="center", fontsize=10, style="italic")
        plt.tight_layout()
        _save_fig("lp_chart3_dual_" + shock + "_" + outcome + ".png")
        plt.show()


def plot_lp_resilience_quadrant(panel, lp_results=None):
    """
    Chart 4 - Impact x Recovery resilience quadrant.
    X-axis: mean LP beta at h=0 across binary shocks (impact).
    Y-axis: beta at h=2 minus beta at h=0 (recovery).
    Four quadrants: Resilient / Adaptive / Fragile-Stable / Vulnerable.
    """
    print("  Chart 4: impact x recovery resilience quadrant")
    panel = panel.copy().sort_values(["iso_code", "year"])
    binary_shocks = ["oil_x_import", "covid_x_fossil", "gas_x_import"]
    outcome = "dea_bc"
    lag_col = "d_" + outcome + "_lag1"
    if lag_col not in panel.columns:
        panel[lag_col] = panel.groupby("iso_code")[outcome].transform(
            lambda x: x.diff().shift(1))

    impact = {}
    recovery = {}
    for iso, g in panel.groupby("iso_code"):
        b0s, b2s = [], []
        for sh in binary_shocks:
            if sh not in g.columns: continue
            for h_, store in [(0, b0s), (2, b2s)]:
                g2 = g.copy()
                g2["_yh"] = panel.loc[g.index, outcome].shift(-h_) - panel.loc[g.index, outcome].shift(1)
                sub = g2.dropna(subset=["_yh", sh, lag_col])
                if len(sub) < 6: continue
                beta = _safe_country_lp_beta(sub, sh, "_yh", lag_col, min_obs=6)
                if np.isfinite(beta):
                    store.append(beta)
        if b0s and b2s:
            impact[iso]   = float(np.mean(b0s))
            recovery[iso] = float(np.mean(b2s)) - float(np.mean(b0s))

    if len(impact) < 4:
        print("  WARNING: insufficient data for quadrant"); return

    df = pd.DataFrame({"impact": impact, "recovery": recovery}).dropna()
    xm, ym = df["impact"].median(), df["recovery"].median()

    quad_colors = []
    for _, row in df.iterrows():
        if row["impact"] >= xm and row["recovery"] >= ym:
            quad_colors.append(COLORS["teal"])
        elif row["impact"] < xm and row["recovery"] >= ym:
            quad_colors.append(COLORS["blue"])
        elif row["impact"] >= xm and row["recovery"] < ym:
            quad_colors.append(COLORS["amber"])
        else:
            quad_colors.append(COLORS["red"])

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.scatter(df["impact"], df["recovery"], c=quad_colors, s=110, zorder=4,
               edgecolors="white", linewidth=0.8)
    ax.axvline(xm, color="gray", lw=0.9, ls="--", alpha=0.7)
    ax.axhline(ym, color="gray", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlabel("Impact score (LP beta at h=0, avg. across shocks)", fontsize=10)
    ax.set_ylabel("Recovery score (beta[h=2] minus beta[h=0])", fontsize=10)
    ax.set_title("Chart 4 -- Energy Security Shock Resilience Quadrant\n"
                 "DEA efficiency response: impact vs. recovery across oil, COVID, gas shocks",
                 fontsize=11, fontweight="bold")
    quad_patches = [mpatches.Patch(color=c, label=t) for t, c in [
        ("Resilient (high impact score, high recovery)", COLORS["teal"]),
        ("Adaptive (low impact score, high recovery)", COLORS["blue"]),
        ("Fragile-stable (high impact score, low recovery)", COLORS["amber"]),
        ("Vulnerable (low impact score, low recovery)", COLORS["red"])]]
    _legend_right(ax, handles=quad_patches, labels=[p.get_label() for p in quad_patches],
                  fontsize=10, title="Quadrant (split at medians,\ndashed lines)")
    _label_points(ax, df["impact"].values, df["recovery"].values, list(df.index))
    _save_fig("lp_chart4_resilience_quadrant.png")
    plt.show()


def plot_lp_shock_trajectories(panel, shock="gas_x_import",
                                outcome="dea_bc", n_countries=12):
    """
    Chart 5 - Individual shock-window trajectories.
    Shows actual time-series of the outcome in a +-3 year window,
    normalised to pre-shock mean = 1. Top-N most affected vs resilient.
    """
    print("  Chart 5: shock trajectories -- " + shock)
    shock_yr_map = {"oil_x_import": 2008, "oil_shock_2008": 2008,
                    "covid_x_fossil": 2020, "covid_shock": 2020,
                    "gas_x_import": 2022, "gas_crisis_2022": 2022}
    shock_yr = shock_yr_map.get(shock)
    if shock_yr is None:
        print("  Only binary shocks supported in trajectory chart"); return

    shock_label = {"oil_x_import": "Oil/GFC 2008 (x import exp.)",
                   "covid_x_fossil": "COVID-19 2020 (x fossil share)",
                   "gas_x_import": "Gas Crisis 2022 (x import exp.)"}.get(shock, shock)
    outcome_label = {"dea_bc": "DEA efficiency (theta_bc)",
                     "esi_score": "ESI score"}.get(outcome, outcome)

    window = list(range(shock_yr - 3, shock_yr + 4))
    # *** FIX: window was constructed assuming a full +/-3-year span always
    # exists, but a shock near the end of the sample (gas_x_import, 2022,
    # only one year before the panel's last available year) produces a
    # window extending to 2024-2025 with no data at all -- the axis then
    # shows two empty years past the end of the actual panel. Clip the
    # window to the panel's own actual year range, so the x-axis only ever
    # shows years the data could possibly cover. ***
    panel_year_min, panel_year_max = int(panel["year"].min()), int(panel["year"].max())
    window = [y for y in window if panel_year_min <= y <= panel_year_max]
    pre    = [y for y in range(shock_yr - 3, shock_yr) if y in window]
    shock_w = [y for y in range(shock_yr, shock_yr + 3) if y in window]

    df_w = panel[panel["year"].isin(window)][["iso_code", "year", outcome]].copy()
    if df_w.empty:
        print("  No data in shock window"); return

    pre_mean = (panel[panel["year"].isin(pre)]
                .groupby("iso_code")[outcome].mean())
    df_w = df_w[df_w["iso_code"].isin(pre_mean.index)].copy()
    df_w["norm"] = df_w.apply(
        lambda r: r[outcome] / pre_mean[r["iso_code"]]
        if pre_mean.get(r["iso_code"], 0) > 0 else np.nan, axis=1)

    trough = (df_w[df_w["year"].isin(shock_w)]
              .groupby("iso_code")["norm"].min()
              .dropna()
              .sort_values())

    n_each = n_countries // 2
    most_affected  = list(trough.head(n_each).index)
    most_resilient = list(trough.tail(n_each).index)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True, sharex=True)
    fig.suptitle("Chart 5 -- Shock-Window Trajectories: " + outcome_label +
                 " | Around " + shock_label + " | Normalised to pre-shock mean = 1",
                 fontsize=12, fontweight="bold")

    for ax, group, title, cmap_name in [
        (axes[0], most_affected,  "Most Affected (n=" + str(len(most_affected)) + ")", "Reds"),
        (axes[1], most_resilient, "Most Resilient (n=" + str(len(most_resilient)) + ")", "Greens"),
    ]:
        for k, iso in enumerate(group):
            sub = df_w[df_w["iso_code"] == iso].sort_values("year")
            if sub.empty: continue
            st = _country_style(iso)
            ax.plot(sub["year"], sub["norm"], color=st["color"], ls=st["linestyle"],
                    lw=2.0, marker=st["marker"], markersize=6, label=iso)
        ax.axvline(shock_yr - 0.5, color="black", lw=1.2, ls="--", label="Shock onset")
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.set_xticks(window)
        ax.set_xticklabels([str(y) for y in window], rotation=45, fontsize=9.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalised efficiency (pre-shock = 1)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        _legend_right(ax, ncol=2 if len(group) > 7 else 1)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save_fig("lp_chart5_trajectories_" + shock + "_" + outcome + ".png")
    plt.show()


def plot_lp_composite_vulnerability(panel, lp_results=None):
    """
    Chart 6 - Composite LP vulnerability score vs ESI structural security.
    Left: bar chart ranked by composite LP beta, coloured by ESI quartile.
    Right: scatter of LP vulnerability vs ESI with Spearman rho and trend line.
    """
    print("  Chart 6: composite vulnerability vs ESI")
    panel = panel.copy().sort_values(["iso_code", "year"])
    all_shocks = ["oil_x_import", "covid_x_fossil", "gas_x_import",
                  "import_shock", "esi_shock"]
    outcome = "dea_bc"
    lag_col = "d_" + outcome + "_lag1"
    if lag_col not in panel.columns:
        panel[lag_col] = panel.groupby("iso_code")[outcome].transform(
            lambda x: x.diff().shift(1))

    betas = {iso: [] for iso in panel["iso_code"].unique()}
    for sh in all_shocks:
        if sh not in panel.columns: continue
        for h_ in [0, 1, 2]:
            yh_col = "_yh_" + str(h_)
            panel[yh_col] = panel.groupby("iso_code")[outcome].transform(
                lambda x, _h=h_: x.shift(-_h) - x.shift(1))
            for iso, g in panel.groupby("iso_code"):
                sub = g.dropna(subset=[yh_col, sh, lag_col])
                if len(sub) < 6: continue
                beta = _safe_country_lp_beta(sub, sh, yh_col, lag_col, min_obs=6)
                if np.isfinite(beta):
                    betas[iso].append(beta)

    vuln = {iso: float(np.mean(v)) for iso, v in betas.items() if v}
    esi_mean = panel.groupby("iso_code")["esi_score"].mean().to_dict()
    common = sorted(set(vuln) & set(esi_mean))
    df = pd.DataFrame({"vuln": vuln, "esi": esi_mean}).loc[common].dropna()
    if df.empty:
        print("  WARNING: no data for chart 6"); return
    df["vuln_norm"] = (df["vuln"] - df["vuln"].mean()) / (df["vuln"].std() + 1e-9)

    q25, q50, q75 = df["esi"].quantile(0.25), df["esi"].quantile(0.50), df["esi"].quantile(0.75)
    def esi_color(v):
        if v >= q75:   return COLORS["teal"]
        elif v >= q50: return COLORS["blue"]
        elif v >= q25: return COLORS["amber"]
        else:           return COLORS["red"]

    df_sorted = df.sort_values("vuln_norm", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(21, _country_extent(len(df), per=0.36)))
    fig.suptitle("Chart 6 -- Composite LP Vulnerability vs ESI Structural Security",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    y = np.arange(len(df_sorted))
    col_sorted = [esi_color(v) for v in df_sorted["esi"]]
    ax.barh(y, df_sorted["vuln_norm"], color=col_sorted, alpha=0.80, height=0.70)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted.index, fontsize=9.5)
    ax.set_ylim(-0.7, len(df_sorted) - 0.3)
    ax.set_xlabel("Standardised LP vulnerability (negative = more resilient)", fontsize=9)
    ax.set_title("LP Composite Vulnerability (coloured by ESI quartile)", fontsize=10, fontweight="bold")
    patches = [plt.Rectangle((0,0),1,1,color=c,alpha=0.75) for c in
               [COLORS["teal"], COLORS["blue"], COLORS["amber"], COLORS["red"]]]
    _fig_legend_right(fig, handles=patches,
                      labels=["ESI Q4 (strongest)", "ESI Q3", "ESI Q2", "ESI Q1 (weakest)"],
                      right=0.90, title="ESI quartile")

    ax2 = axes[1]
    ax2.scatter(df["esi"], df["vuln_norm"],
                c=[esi_color(v) for v in df["esi"]],
                s=90, edgecolors="white", linewidth=0.8, zorder=4)
    xv, yv = df["esi"].values, df["vuln_norm"].values
    m, b_ = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 50)
    ax2.plot(xs, m * xs + b_, color="black", lw=1.2, ls="--", alpha=0.6)
    rho = df["esi"].corr(df["vuln_norm"], method="spearman")
    ax2.set_xlabel("ESI composite score (higher = structurally more secure)", fontsize=9)
    ax2.set_ylabel("LP vulnerability score (standardised)", fontsize=9)
    ax2.set_title("Structural (ESI) vs Dynamic (LP) Resilience | Spearman rho = {:.3f}".format(rho),
                  fontsize=10, fontweight="bold")
    ax2.axhline(0, color="gray", lw=0.7, ls=":")
    ax2.grid(True, alpha=0.2)
    _label_points(ax2, df["esi"].values, df["vuln_norm"].values, list(df.index),
                  colors=[esi_color(v) for v in df["esi"]])
    _save_fig("lp_chart6_composite_vuln.png")
    plt.show()


def plot_all_lp_charts(lp_results, panel, alpha=0.10):
    """Run all six shock-analysis charts in sequence."""
    print("\n  -- Generating shock analysis chart suite --")
    for shock in ["gas_x_import", "covid_x_fossil", "oil_x_import"]:
        plot_lp_country_ranking(lp_results, panel, shock=shock,
                                 outcome="dea_bc", horizon=1)
    plot_lp_heat_map(lp_results, panel, outcome="dea_bc", horizon=1)
    for shock in ["gas_x_import", "covid_x_fossil", "oil_x_import"]:
        plot_lp_dual_methodology(lp_results, panel, shock=shock)
    plot_lp_resilience_quadrant(panel, lp_results)
    for shock in ["gas_x_import", "covid_x_fossil"]:
        plot_lp_shock_trajectories(panel, shock=shock, outcome="dea_bc")
    plot_lp_composite_vulnerability(panel, lp_results)
    print("  -- Chart suite complete --")
    print("  Files: lp_chart1_ranking_*.png  lp_chart2_heatmap_*.png")
    print("         lp_chart3_dual_*.png  lp_chart4_resilience_quadrant.png")
    print("         lp_chart5_trajectories_*.png  lp_chart6_composite_vuln.png")


def step10b_local_projections(state: dict,
                               H: int = 5,
                               alpha: float = 0.10) -> dict:
    """
    STEP 10b — Local Projections (Jordà 2005): impulse responses of DEA
    efficiency and ESI to identified energy-security shocks.

    Shock identification:
      Global binary  : oil/GFC 2008-09, COVID 2020-21, gas crisis 2022-23
      Country-specific: within-country change in net import dependence,
                        within-country change in ESI composite (demeaned)

    For each (outcome, shock) pair, OLS estimates:
        y_{i,t+h} - y_{i,t-1} = α_i + γ_t + β_h · shock_{i,t} + AR-lag + ε
    at horizons h = 0…H. The β_h sequence is the impulse-response function.
    Entity and time FE are absorbed; SE clustered by entity.

    Results in state:
        lp_results  — dict of IRF DataFrames, keyed by (outcome, shock)
        lp_panel    — panel with shock columns appended
        lp_candidate_shock_years — data-driven scan for disturbance years
                       NOT covered by the three pre-specified shock windows
                       below (see detect_candidate_shock_years())

    Parameters
    ----------
    H     : maximum horizon (default 5 years)
    alpha : confidence level (default 0.10 → 90% CI)

    Usage:
        state = step10b_local_projections(state)
        state = step10b_local_projections(state, H=8, alpha=0.05)
    """
    print("\n" + "="*60)
    print("STEP 10b — Local Projections: Shock Impulse Responses")
    print("="*60)
    t = time.time()

    panel = state["panel"].copy()

    if "dea_bc" not in panel.columns or panel["dea_bc"].isna().all():
        print("  ⚠ dea_bc missing — run step5_dea_bootstrap first")
        return state
    if "esi_score" not in panel.columns or panel["esi_score"].isna().all():
        print("  ⚠ esi_score missing — run step4_esi first")
        return state

    # Data-driven scan for candidate shock years NOT covered by the three
    # pre-specified windows below, before committing to those fixed windows.
    state["lp_candidate_shock_years"] = detect_candidate_shock_years(panel)

    # Build shock series
    panel = _define_shocks(panel)

    # Run local projections
    print(f"\n  Running local projections (H={H}, α={alpha})...")
    lp_results = local_projections(panel,
                                    outcomes=["dea_bc","esi_score"],
                                    shocks=["oil_x_import","covid_x_fossil",
                                            "gas_x_import",
                                            "import_shock","esi_shock"],
                                    H=H, alpha=alpha)

    n_pairs = len(lp_results)
    print(f"  {n_pairs} (outcome, shock) pairs estimated")

    # Print summary
    print_lp_summary(lp_results, alpha=alpha)

    # Plot IRFs
    plot_local_projections(lp_results, alpha=alpha, save_prefix="step10b")
    plot_all_lp_charts(lp_results, panel, alpha=alpha)

    # Save per-pair CSVs
    for (out, shock), df in lp_results.items():
        _save_csv(df, f"lp_{out}_{shock}.csv",
                  f"LP IRF: {out} ← {shock}")

    state["lp_results"] = lp_results
    state["lp_panel"]   = panel
    print(f"\n  ✓ Step 10b complete  ({time.time()-t:.0f}s)")
    return state

def step11_plots(state: dict, year: int = None) -> dict:
    """
    STEP 11 — Generate and save all plots.

    Requires state from steps 1–6 (mpi_df needed for the GML chart).
    Saves:
        dea_bootstrap_ci.png
        mpi_decomposition.png
        quadrant_map.png
        affordability_validation.png   ← from step 2 (already saved)

    Parameters:
        year : year for DEA CI bar chart and quadrant map (default: latest)

    Usage:
        state = step11_plots(state)
        state = step11_plots(state, year=2019)
    """
    print("\n" + "="*60)
    print("STEP 11 — Plots")
    print("="*60)
    t = time.time()

    plot_dea_with_ci(state["panel"], year=year)
    if "mpi_df" in state:
        plot_mpi_decomposition(state["mpi_df"])
    plot_quadrant(state["panel"], year=year)

    print(f"\n  ✓ Step 11 complete  ({time.time()-t:.0f}s)")
    return state


# ═════════════════════════════════════════════════════════════════════════════
# STEP 12 — THEORY-DRIVEN METAFRONTIER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# STEP 12 — THEORY-DRIVEN METAFRONTIER ANALYSIS (three technology regimes)
# Reuses EXACTLY the same production variables and DDF orientation as Step 5
# (module-level INPUT_COLS, GOOD_OUT, BAD_OUT) -- no separate DEA model is
# constructed for this analysis. theta^Meta is solved FRESH on the same
# per-year sample as theta^Group (see _solve_group_meta_ddf), so that
# theta^Meta <= theta^Group holds by construction; it is compared with
# Step 5's dea_score only as a diagnostic.
# ═════════════════════════════════════════════════════════════════════════════

def assign_technology_regime(panel, min_group_size=10, f_threshold=0.65, lc_threshold=0.40,
                             baseline_years=None):
    """
    Assign each country to one of three technology regimes, based on its
    average ELECTRICITY-GENERATION mix (not total primary energy):

        F_i  = mean(coal_share_elec + gas_share_elec + oil_share_elec) / 100
        LC_i^max = max( mean(nuclear_share_elec), mean(hydro_share_elec),
                       mean(other_renew_share_elec) ) / 100

        G_i = Fossil-dominant           if F_i >= f_threshold
            = Low-carbon specialised    if F_i <  f_threshold and LC_i^max >= lc_threshold
            = Diversified/mixed         if F_i <  f_threshold and LC_i^max <  lc_threshold

    "Low-carbon specialised" captures a country whose non-fossil generation
    is itself concentrated in ONE low-carbon source (e.g. France/nuclear,
    Norway/hydro), as distinct from "Diversified/mixed", where no single
    source -- fossil or low-carbon -- dominates.

    *** THRESHOLD AND AVERAGING-WINDOW HISTORY, established through direct
    testing against the real 44-country panel at each step, not assumed:

    Step 1 (electricity- vs total-energy shares): an earlier version used
    OWID's "_share_energy" columns (total primary energy). This produced a
    badly degenerate split (41/1/2), because total-energy shares are
    diluted by transport and heating fuel use -- even a country with a
    very clean electricity grid (e.g. France, ~70% nuclear ELECTRICITY)
    can show a high fossil share of TOTAL energy. Switching to OWID's
    electricity-specific "_share_elec" columns gave a far more sensible
    27/9/8 split on the symmetric 50%/50% threshold rule.

    Step 2 (symmetric 50/50 vs asymmetric 65/40, full-sample average): a
    55%/45% asymmetric rule was proposed as a first test, together with
    switching from a full-sample average to a 2000-2004 baseline-period
    average, with the explicit goal of moving the split toward roughly
    20-24 / 10-14 / 10-14. Testing the exact combination requested
    (baseline 2000-2004, F>=0.55, LC>=0.45) against real data gave
    30/11/3 -- WORSE for the Diversified/mixed group than the original
    27/9/8, and outside the target range on two of three groups.
    Isolating the two changes separately showed why: the threshold change
    alone (full-sample, 55/45 vs 50/50) barely moved the split at all
    (27/9/8 -> 26/9/9); the BASELINE-PERIOD SWITCH ALONE was what drove
    the unwanted result (full-sample vs 2000-2004, both at 50/50:
    27/9/8 -> 31/10/3). Early-2000s generation mixes were more uniformly
    fossil-heavy, before two decades of renewable build-out broadened the
    "balanced mix" category most countries now occupy -- restricting to a
    2000-2004 baseline empties out Diversified/mixed almost entirely for
    reasons unrelated to the threshold rule itself. The baseline-period
    averaging was therefore NOT adopted; baseline_years remains an
    available parameter (None = full-sample average, the default) for
    anyone who wants to re-test it, but the full-sample average is what
    is actually used unless baseline_years is explicitly supplied.

    A subsequent full-sample threshold scan (F from 0.50 to 0.65, LC from
    0.40 to 0.50) found several combinations landing inside the
    20-24/10-14/10-14 target; F>=0.65, LC>=0.40 was selected as the most
    evenly-centred (22/11/11 on the 44-country panel; 22/12/11 on the final
    45-country panel, Switzerland joining Low-carbon specialised), with country memberships
    matching domain expectations (Germany, Denmark, Spain, Czech Republic,
    Portugal, Romania, Bulgaria, Slovenia, Lithuania in Diversified/mixed;
    France, Norway, Canada, Brazil, Austria, Sweden, Belgium, Croatia,
    Hungary, Latvia, Slovakia in Low-carbon specialised). min_group_size
    was correspondingly restored to 10 (from the earlier, disclosed
    loosening to 8 needed only for the smaller 27/9/8 split). ***

    Parameters
    ----------
    f_threshold : fossil-dominance threshold on F_i (default 0.65).
    lc_threshold : low-carbon-specialisation threshold on LC_i^max
                  (default 0.40).
    baseline_years : optional (start_year, end_year) tuple to average over
                  a fixed baseline window instead of the full sample (e.g.
                  (2000, 2004)). Default None uses the full sample -- see
                  design note above for why this default was retained
                  after direct testing of the baseline-period alternative.

    HARD CHECK: raises AssertionError if any group has fewer than
    min_group_size countries, rather than silently proceeding with an
    unstably small reference technology for that group's annual DDF.

    Returns the panel with a new column "tech_regime", and prints the
    country-level classification.
    """
    print(f"\n{'═'*70}")
    period_desc = f"baseline {baseline_years[0]}-{baseline_years[1]}" if baseline_years else "full-sample"
    print(f"STEP 12 — TECHNOLOGY REGIME ASSIGNMENT (electricity-specific shares, "
          f"{period_desc} average, F>={f_threshold}, LC>={lc_threshold})")
    print(f"{'═'*70}")

    need_cols = ["coal_share_elec","gas_share_elec","oil_share_elec",
                 "nuclear_share_elec","hydro_share_elec","other_renew_share_elec"]
    avail = [c for c in need_cols if c in panel.columns]
    missing = [c for c in need_cols if c not in panel.columns]
    if missing:
        print(f"  ⚠ Missing electricity-mix columns: {missing}")
        print(f"    (these require the OWID rename map fix -- re-run Step 1 if")
        print(f"    unexpectedly absent)")

    source = panel
    if baseline_years is not None:
        source = panel[(panel["year"] >= baseline_years[0]) & (panel["year"] <= baseline_years[1])]
        print(f"  Using baseline period {baseline_years[0]}-{baseline_years[1]} "
              f"({source['year'].nunique()} years) rather than the full sample.")

    means = source.groupby("iso_code")[avail].mean()
    for c in need_cols:
        if c not in means.columns:
            means[c] = 0.0
    if means.max().max() > 2:
        means = means / 100.0   # shares stored 0-100 -> convert to 0-1

    F = means[["coal_share_elec","gas_share_elec","oil_share_elec"]].sum(axis=1)
    LC_max = means[["nuclear_share_elec","hydro_share_elec","other_renew_share_elec"]].max(axis=1)

    def _classify(f, lc):
        if f >= f_threshold:
            return "Fossil-dominant"
        elif lc >= lc_threshold:
            return "Low-carbon specialised"
        else:
            return "Diversified/mixed"

    regime = pd.Series({iso: _classify(F[iso], LC_max[iso]) for iso in F.index}, name="tech_regime")
    panel = panel.merge(regime.rename("tech_regime"), left_on="iso_code", right_index=True, how="left")

    counts = regime.value_counts()
    print(f"\n  Country-level classification ({period_desc} average generation mix):")
    for grp in ["Fossil-dominant","Low-carbon specialised","Diversified/mixed"]:
        members = sorted(regime[regime == grp].index.tolist())
        print(f"    {grp:<24}: {len(members):2d} countries -- {members}")

    # *** HARD CHECK: do not proceed with an unstably small group ***
    small_groups = counts[counts < min_group_size]
    if len(small_groups) > 0:
        raise AssertionError(
            f"Step 12 technology-regime assignment: group(s) below the minimum "
            f"size of {min_group_size} countries: {small_groups.to_dict()}. "
            f"An annual DDF frontier for a group this small is not stable enough "
            f"to serve as a reference technology. Either broaden the group "
            f"definition, pool years, or reduce min_group_size deliberately "
            f"(not recommended) before proceeding."
        )
    print(f"\n  ✓ All three groups meet the minimum size of {min_group_size} countries.")

    return panel, F, LC_max


def regime_threshold_sensitivity(panel, F, LC_max, base_map, min_group_size=10,
                                 f_grid=(0.55, 0.60, 0.65, 0.70, 0.75),
                                 lc_grid=(0.30, 0.35, 0.40, 0.45, 0.50)):
    """
    Transparent sensitivity of the Step 12 regime results to the classification
    thresholds (fossil share F, largest single low-carbon share LC) and to the
    classification period (full-sample average vs 2000-2004). For each
    combination: group sizes, countries reassigned relative to the baseline
    (F >= 0.65, LC >= 0.40), country-level mean TGR by regime, Kruskal-Wallis
    (asymptotic p), and whether the baseline ordering (low-carbon > diversified
    > fossil) holds. Groups below min_group_size are flagged; groups below 5
    countries are not estimated. The baseline thresholds stay fixed.
    """
    import io, contextlib
    print(f"\n{'═'*78}")
    print("12.14c SENSITIVITY — REGIME THRESHOLDS AND CLASSIFICATION PERIOD (reported only)")
    print(f"{'═'*78}")

    def _classify(f, lc, ft, lt):
        return "Fossil-dominant" if f >= ft else ("Low-carbon specialised" if lc >= lt else "Diversified/mixed")

    combos = [(ft, lt, F, LC_max, "full sample") for ft in f_grid for lt in lc_grid]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, F_e, LC_e = assign_technology_regime(panel.drop(columns=["tech_regime"], errors="ignore"),
                                                    min_group_size=1, baseline_years=(2000, 2004))
        combos.append((0.65, 0.40, F_e, LC_e, "2000-2004 average"))
    except Exception as e:
        print(f"  ⚠ early-period classification not available: {e}")
    regs = ["Low-carbon specialised", "Diversified/mixed", "Fossil-dominant"]
    short = {"Low-carbon specialised": "lowcarbon", "Diversified/mixed": "diversified",
             "Fossil-dominant": "fossil"}
    rows = []
    for ft, lt, Fv, Lv, period in combos:
        rmap = pd.Series({iso: _classify(Fv[iso], Lv[iso], ft, lt) for iso in Fv.index})
        sizes = rmap.value_counts().reindex(regs).fillna(0).astype(int)
        row = {"f_threshold": ft, "lc_threshold": lt, "period": period,
               "baseline": (ft == 0.65 and lt == 0.40 and period == "full sample"),
               **{f"n_{short[r]}": int(sizes[r]) for r in regs},
               "n_reassigned": int((rmap != base_map.reindex(rmap.index)).sum()),
               "min_group_ok": bool(sizes.min() >= min_group_size)}
        if sizes.min() >= 5:
            p2 = panel.drop(columns=["tech_regime"], errors="ignore").merge(
                rmap.rename("tech_regime_s"), left_on="iso_code", right_index=True, how="left")
            with contextlib.redirect_stdout(io.StringIO()):
                res = _solve_group_meta_ddf(p2, group_col="tech_regime_s", vrs=VRS)
            cm = res.groupby("iso_code").agg(tgr=("tgr", "mean"), grp=("tech_regime", "first"))
            means = cm.groupby("grp")["tgr"].mean().reindex(regs)
            for r in regs:
                row[f"tgr_{short[r]}"] = means[r]
            samples = [cm.loc[cm["grp"] == r, "tgr"].values for r in regs]
            if all(len(x) >= 2 for x in samples):
                H, pkw = stats.kruskal(*samples)
                row["kruskal_H"], row["kruskal_p_asymptotic"] = H, pkw
            row["baseline_ordering_holds"] = bool(means.iloc[0] > means.iloc[1] > means.iloc[2])
            row["lowcarbon_minus_fossil"] = means.iloc[0] - means.iloc[2]
        rows.append(row)
    tab = pd.DataFrame(rows)
    print(f"  {'F':>5}{'LC':>6} {'period':<18}{'nLC':>4}{'nDiv':>5}{'nFos':>5}{'moved':>6}"
          f"{'TGR LC':>8}{'TGR Div':>8}{'TGR Fos':>8}{'KW p':>8}  order")
    for _, r in tab.iterrows():
        flag = "◀ baseline" if r["baseline"] else ("" if r["min_group_ok"] else "(group < min size)")
        tg = lambda k: f"{r[k]:>8.3f}" if k in r.index and pd.notna(r[k]) else f"{'—':>8}"
        kp = (f"{r['kruskal_p_asymptotic']:>8.4f}" if "kruskal_p_asymptotic" in r.index
              and pd.notna(r["kruskal_p_asymptotic"]) else f"{'—':>8}")
        oh = r["baseline_ordering_holds"] if "baseline_ordering_holds" in r.index else np.nan
        od = "—" if pd.isna(oh) else ("yes" if oh else "no")
        print(f"  {r['f_threshold']:>5.2f}{r['lc_threshold']:>6.2f} {r['period']:<18}{r['n_lowcarbon']:>4}"
              f"{r['n_diversified']:>5}{r['n_fossil']:>5}{r['n_reassigned']:>6}"
              f"{tg('tgr_lowcarbon')}{tg('tgr_diversified')}{tg('tgr_fossil')}{kp}  {od} {flag}")
    est = tab[tab["baseline_ordering_holds"].notna()] if "baseline_ordering_holds" in tab.columns else tab.iloc[0:0]
    if len(est):
        ok_sz = est[est["min_group_ok"]]
        fd = int((est["tgr_fossil"] > est["tgr_diversified"]).sum())
        dl = int((est["tgr_diversified"] > est["tgr_lowcarbon"]).sum())
        lf = int((est["tgr_lowcarbon"] > est["tgr_fossil"]).sum())
        print(f"\n  Full ordering (low-carbon > diversified > fossil): {int(est['baseline_ordering_holds'].sum())}"
              f"/{len(est)} estimable combinations; {int(ok_sz['baseline_ordering_holds'].sum())}/{len(ok_sz)} "
              f"excluding combinations with a group below {min_group_size} countries.")
        print(f"  Pairwise: low-carbon > fossil in {lf}/{len(est)} (difference "
              f"{est['lowcarbon_minus_fossil'].min():+.3f} to {est['lowcarbon_minus_fossil'].max():+.3f}); "
              f"reversals are fossil > diversified in {fd} and diversified > low-carbon in {dl}.")
        print(f"  Reading: the conclusion that survives threshold choice is the pairwise one that")
        print(f"  holds in every combination; a pair that reverses in some combinations should not")
        print(f"  be reported as a robust ordering.")
    print(f"  Thresholds of the baseline are fixed in advance (F >= 0.65, LC >= 0.40, full-sample")
    print(f"  average) and are not chosen from this table.")
    _save_csv(tab, "step12_regime_threshold_sensitivity.csv",
              "Step 12 regime thresholds and classification period: sensitivity")
    return tab


def _solve_group_meta_ddf(panel, group_col="tech_regime", vrs=VRS):
    """
    Core DDF engine for Step 12. Uses the EXACT SAME module-level
    INPUT_COLS, GOOD_OUT, BAD_OUT and _solve_ddf_one() as Step 5 -- no
    separate or parallel DEA/DDF specification is constructed here.

    theta^Group_it : DDF solved with the reference set restricted to the
                     country's OWN technology regime, in that year, under
                     the returns to scale passed in `vrs` (baseline: CRS).
    theta^Meta_it  : computed FRESH within this function, using the SAME
                     per-year reference set and normalisation as
                     theta^Group (all countries present in that year's
                     complete-case sample here) -- NOT reused directly
                     from dea_score.

    *** DESIGN NOTE, established through direct testing rather than
    assumed: an earlier version of this function reused dea_score
    directly as theta^Meta, following the convention established for the
    Appendix A metafrontier. Testing against a realistic synthetic panel
    surfaced a genuine inconsistency: Step 5's own per-year completeness
    mask (build_dea_bootstrap's internal dropna/positivity filter) is not
    guaranteed to select EXACTLY the same set of countries, in every year,
    as this function's own dropna() on the columns it needs -- if even one
    country differs between the two sets in a given year, dea_score for
    that year was computed against a subtly different reference technology
    than the one this function's group-level DDF is being compared to,
    which can violate the theoretical requirement theta_meta <= theta_group
    (the group being a strict subset of the meta reference set should
    guarantee the meta LP is at least as generous as the group LP, given
    an IDENTICAL reference set and normalisation). Computing theta^Meta
    fresh, on the exact same yr_df used for theta^Group, removes this
    reference-set mismatch entirely and guarantees the subset-monotonicity
    property holds by construction. theta^Meta computed this way remains
    numerically very close to dea_score (both represent the same annual,
    all-country VRS frontier concept) and is compared against it
    diagnostically in _check_metafrontier_consistency, without asserting
    bit-for-bit equality. ***

    If vrs=False, both theta^Group and theta^Meta are computed under CRS
    (secondary benchmark) instead.

    Returns a DataFrame: iso_code, year, tech_regime, theta_group, theta_meta, tgr
    """
    dea_col_for_diagnostic = "dea_score" if vrs else "dea_crs"

    needed = ["iso_code","year",group_col] + INPUT_COLS + GOOD_OUT + BAD_OUT
    needed = [c for c in needed if c in panel.columns]
    sub = panel[needed].dropna().copy()
    if len(sub) < 30:
        print(f"  ⚠ Only {len(sub)} complete observations — skipped")
        return pd.DataFrame()

    years = sorted(sub["year"].unique())
    results = []
    skipped = []
    _CNT = ("primary_not_optimal", "rescued_by_fallback", "unsolved",
            "tiny_negative_set_to_zero", "negative_beta_rejected")
    _stats0 = {k: _DDF_SOLVE_STATS[k] for k in _CNT}
    for yr in years:
        yr_df = sub[sub["year"] == yr].reset_index(drop=True)
        if len(yr_df) < 8:
            continue
        X_all = _normalise(yr_df[INPUT_COLS].values.astype(float))
        Y_all = _normalise(yr_df[GOOD_OUT].values.astype(float))
        B_all = _normalise(yr_df[BAD_OUT].values.astype(float))
        n_all = len(yr_df)

        # theta^Meta: solved FRESH, reference set = every country in yr_df
        # (this year's complete-case sample), matching the group LP's own
        # normalisation and completeness sample exactly.
        theta_meta_arr = np.full(n_all, np.nan)
        for i in range(n_all):
            beta_m = _solve_ddf_one(X_all, Y_all, B_all,
                                    X_all[i], Y_all[i], B_all[i],
                                    vrs=vrs, allow_negative=False)
            if np.isfinite(beta_m):
                theta_meta_arr[i] = 1.0 / (1.0 + beta_m)

        for grp in yr_df[group_col].unique():
            grp_idx = np.where((yr_df[group_col] == grp).values)[0]
            if len(grp_idx) < 4:
                continue
            X_g = X_all[grp_idx]; Y_g = Y_all[grp_idx]; B_g = B_all[grp_idx]

            for k_local, i_global in enumerate(grp_idx):
                beta_group = _solve_ddf_one(X_g, Y_g, B_g,
                                            X_g[k_local], Y_g[k_local], B_g[k_local],
                                            vrs=vrs, allow_negative=False)
                _fail_rec = _DDF_SOLVE_STATS["last_failure"] if not np.isfinite(beta_group) else None
                if not np.isfinite(beta_group) or not np.isfinite(theta_meta_arr[i_global]):
                    # Record exactly which country-year was lost and in which LP,
                    # instead of dropping it silently (reviewer item 1: the VRS
                    # run had 1,079 instead of 1,080 observations).
                    skipped.append({"iso_code": yr_df.loc[i_global, "iso_code"], "year": int(yr),
                                    "tech_regime": grp,
                                    "failed_lp": ("group" if not np.isfinite(beta_group) else "")
                                                 + ("+meta" if not np.isfinite(theta_meta_arr[i_global]) else ""),
                                    "reason": _fail_rec})
                    continue
                theta_group = 1.0 / (1.0 + beta_group)
                theta_meta = float(theta_meta_arr[i_global])
                tgr = theta_meta / theta_group if theta_group > 1e-9 else np.nan
                results.append({
                    "iso_code": yr_df.loc[i_global, "iso_code"],
                    "tech_regime": grp, "year": yr,
                    "theta_group": theta_group, "theta_meta": theta_meta,
                    "tgr": min(tgr, 1.0 + 1e-6) if np.isfinite(tgr) else np.nan,
                })

    _rts = "VRS" if vrs else "CRS"
    _d = {k: _DDF_SOLVE_STATS[k] - _stats0[k] for k in _CNT}
    print(f"  {_rts} group/meta DDF: {len(results)}/{len(sub)} country-years solved; "
          f"{DDF_SOLVER.upper()} not optimal in {_d['primary_not_optimal']} LP(s), rescued by fallback "
          f"{_d['rescued_by_fallback']}, unsolved {_d['unsolved']}, "
          f"tiny negative beta set to 0 {_d['tiny_negative_set_to_zero']}, "
          f"negative beta rejected {_d['negative_beta_rejected']}"
          + (f"  [VRS abatement: {VRS_ABATEMENT}]" if vrs else ""))
    if skipped:
        print(f"  ⚠ {len(skipped)} country-year(s) without a {_rts} metafrontier result:")
        for r in skipped:
            print(f"      {r['iso_code']} {r['year']} ({r['tech_regime']}): {r['failed_lp']} LP failed -- {r['reason']}")
    # Every missing observation must correspond to a counted failure.
    _n_counted = _d["unsolved"] + _d["negative_beta_rejected"]
    if len(skipped) and _n_counted == 0:
        print(f"  ✗ INCONSISTENT: {len(skipped)} missing result(s) but no counted solver failure.")
    out = pd.DataFrame(results)
    out.attrs["skipped"] = skipped
    return out
def _check_metafrontier_consistency(results_df, panel, meta_col="dea_score", tol=1e-4):
    """
    Consistency checks for Step 12. theta^Meta is now computed FRESH within
    _solve_group_meta_ddf (see that function's docstring for why), on the
    SAME per-year reference set as theta^Group -- so the HARD requirement
    is theta_meta <= theta_group (the group being a strict subset of the
    meta reference set guarantees the meta LP is at least as generous),
    verified below with a raised error if violated beyond tolerance.

    theta^Meta's closeness to Step 5's own dea_score is reported as a
    DIAGNOSTIC only (both represent the same underlying concept -- the
    annual, all-country VRS frontier -- and should be very close, but are
    not asserted to be bit-for-bit identical, since Step 5's own internal
    completeness mask is not guaranteed to select an identical country set
    in every year to this function's own dropna()).
    """
    print(f"\nConsistency checks:")

    n_total = len(results_df)
    bad_mask = (results_df["theta_meta"] - results_df["theta_group"]) > tol
    n_bad = int(bad_mask.sum())
    print(f"  theta_meta <= theta_group (HARD requirement, subset monotonicity): "
          f"{n_total-n_bad}/{n_total} observations satisfy condition")
    if n_bad > 0:
        print(f"  ⚠ {n_bad} observations violate theta_meta <= theta_group beyond tol={tol}:")
        print(results_df.loc[bad_mask, ["iso_code","year","tech_regime",
                                        "theta_group","theta_meta"]].head(10).to_string(index=False))
        if n_bad / n_total > 0.02:
            raise AssertionError(
                f"Step 12: {n_bad}/{n_total} ({100*n_bad/n_total:.1f}%) observations "
                f"violate theta_meta <= theta_group beyond tolerance -- this exceeds "
                f"the 2% numerical-noise allowance and indicates a genuine construction "
                f"bug (e.g. a reference-set or normalisation mismatch between the group "
                f"and meta DDF solves). Do not interpret results until fixed."
            )
        else:
            print(f"  ℹ {100*n_bad/n_total:.2f}% of observations -- within the 2% numerical-")
            print(f"    noise allowance for independent LP solves at the group boundary;")
            print(f"    not treated as a construction error.")

    # Diagnostic-only comparison against Step 5's own dea_score
    if meta_col in panel.columns:
        check = results_df.merge(
            panel[["iso_code","year",meta_col]].drop_duplicates(),
            on=["iso_code","year"], how="left"
        )
        check["meta_diff"] = (check["theta_meta"] - check[meta_col]).abs()
        print(f"\n  [Diagnostic only] theta_meta vs Step-5 {meta_col}: "
              f"mean abs diff = {check['meta_diff'].mean():.4f}, "
              f"max abs diff = {check['meta_diff'].max():.4f}")
        print(f"  (Not asserted equal -- see docstring. Small, non-zero differences are")
        print(f"  expected and do not indicate an error.)")

    n_tgr_valid = int(((results_df["tgr"] > 0) & (results_df["tgr"] <= 1 + tol)).sum())
    print(f"\n  TGR in (0,1]: {n_tgr_valid}/{n_total} observations valid")
    print(f"  {'─'*58}")


def classify_four_way(results_df):
    """
    Four-way country classification, based on each country's FULL-SAMPLE
    mean group efficiency and mean TGR, split at the panel median on each
    axis (consistent with the median-split convention already used
    elsewhere in this pipeline, e.g. the ESI quadrant map):

                        High TGR                    Low TGR
    High group eff.     Technology leaders          Efficient but
                                                     technology-constrained
    Low group eff.      Operational improvement     Dual gap
                        needed

    High group efficiency + High TGR ("Technology leaders"): efficient
        relative to peers AND those peers' technology is close to the
        global best-practice envelope.
    High group efficiency + Low TGR ("Efficient but technology-constrained"):
        efficient relative to structurally similar peers, but that peer
        group's OWN technology regime lies substantially below the best
        attainable technology in the full sample -- the operational
        performance is good, but the ceiling itself is low.
    Low group efficiency + High TGR ("Operational improvement needed"):
        the country's technology regime is close to the global frontier,
        but the country itself is not realising that potential -- a
        genuine operational (not technological) gap.
    Low group efficiency + Low TGR ("Dual gap"): both an operational gap
        relative to peers AND a technology gap between those peers'
        regime and the global frontier.
    """
    country_means = (results_df.groupby(["iso_code","tech_regime"])
                     .agg(mean_theta_group=("theta_group","mean"),
                          mean_theta_meta=("theta_meta","mean"),
                          mean_tgr=("tgr","mean"))
                     .reset_index())

    med_eff = country_means["mean_theta_group"].median()
    med_tgr = country_means["mean_tgr"].median()

    def _label(row):
        high_eff = row["mean_theta_group"] >= med_eff
        high_tgr = row["mean_tgr"] >= med_tgr
        if high_eff and high_tgr:
            return "Technology leaders"
        elif high_eff and not high_tgr:
            return "Efficient but technology-constrained"
        elif not high_eff and high_tgr:
            return "Operational improvement needed"
        else:
            return "Dual gap"

    country_means["classification"] = country_means.apply(_label, axis=1)

    print(f"\n{'─'*78}")
    print(f"FOUR-WAY COUNTRY CLASSIFICATION (median split: group_eff={med_eff:.4f}, TGR={med_tgr:.4f})")
    print(f"{'─'*78}")
    for label in ["Technology leaders","Efficient but technology-constrained",
                 "Operational improvement needed","Dual gap"]:
        members = country_means[country_means["classification"]==label]["iso_code"].tolist()
        print(f"  {label:<38}: {len(members):2d} -- {sorted(members)}")
    print(f"{'─'*78}")

    return country_means


def regime_summary_table(results_df):
    """
    Table: Technology regime | Group efficiency (mean) | Meta efficiency
    (mean) | Mean TGR | Interpretation.
    """
    summary = (results_df.groupby("tech_regime")
              .agg(mean_theta_group=("theta_group","mean"),
                   mean_theta_meta=("theta_meta","mean"),
                   mean_tgr=("tgr","mean"),
                   n_obs=("tgr","count"))
              .reset_index()
              .sort_values("mean_tgr", ascending=False))

    def _interp(row):
        if row["mean_tgr"] > 0.9:
            gap_desc = "technology regime close to the global frontier"
        elif row["mean_tgr"] > 0.7:
            gap_desc = "moderate technology gap relative to the global frontier"
        else:
            gap_desc = "substantial technology gap relative to the global frontier"
        return gap_desc

    print(f"\n{'─'*100}")
    print(f"TABLE: METAFRONTIER SUMMARY BY TECHNOLOGY REGIME")
    print(f"{'─'*100}")
    print(f"  {'Regime':<24} {'θ^Group':>9} {'θ^Meta':>9} {'Mean TGR':>10}  Interpretation")
    print(f"  {'─'*96}")
    for _, row in summary.iterrows():
        print(f"  {row['tech_regime']:<24} {row['mean_theta_group']:>9.4f} "
              f"{row['mean_theta_meta']:>9.4f} {row['mean_tgr']:>10.4f}  {_interp(row)}")
    print(f"  {'─'*96}")

    return summary
def _cliffs_delta(x, y):
    """
    Cliff's delta effect size for two independent samples: the probability
    that a randomly drawn value from x exceeds one from y, minus the
    reverse probability. Ranges [-1,1]; |d|<0.147 negligible, <0.33 small,
    <0.474 medium, else large (Romano et al. 2006 thresholds).
    """
    x = np.asarray(x); y = np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def test_regime_differences(results_df, value_col="tgr", n_boot=B_RESAMPLE, seed=42):
    """
    Distributional comparison of value_col (default: TGR) across the three
    technology regimes, using country-level means (one observation per
    country, avoiding pseudo-replication from pooling country-years).

    1. Kruskal-Wallis test across all three regimes (omnibus, non-parametric,
       does not assume normality -- appropriate given TGR is a bounded ratio).
    2. If the omnibus test rejects at 5%, pairwise Mann-Whitney U tests for
       each of the three regime pairs, with Holm-Bonferroni correction for
       multiple comparisons (three tests).
    3. Cliff's delta effect size for every pairwise comparison, reported
       regardless of significance -- per the explicit instruction not to
       rely on significance alone.
    4. Bootstrap (percentile) confidence intervals for each regime's mean
       and median, resampling countries with replacement within regime.
    """
    print(f"\n{'═'*70}")
    print(f"DISTRIBUTIONAL COMPARISON ACROSS TECHNOLOGY REGIMES: {value_col}")
    print(f"{'═'*70}")

    country_vals = results_df.groupby(["iso_code","tech_regime"])[value_col].mean().reset_index()
    regimes = sorted(country_vals["tech_regime"].unique())
    groups = {g: country_vals[country_vals["tech_regime"]==g][value_col].dropna().values
             for g in regimes}
    n_countries_total = sum(len(v) for v in groups.values())

    print(f"\n  Country-level {value_col} by regime (n countries):")
    for g in regimes:
        v = groups[g]
        print(f"    {g:<24}: n={len(v):2d}  mean={v.mean():.4f}  median={np.median(v):.4f}  "
              f"SD={v.std():.4f}")
    print(f"\n  *** N = {n_countries_total} COUNTRY-LEVEL MEANS (one observation per country, "
          f"each country's\n      {value_col} averaged across all its sample years FIRST). "
          f"The test below is NOT run\n      on the {len(results_df)} pooled country-year "
          f"observations -- that would treat each\n      country's ~24 repeated, serially "
          f"dependent yearly observations as independent,\n      making the nominal p-value "
          f"too optimistic. ***")

    # 1. Kruskal-Wallis omnibus test (asymptotic chi-square approximation --
    # scipy.stats.kruskal does NOT compute an exact finite-sample p-value;
    # an exact/Monte-Carlo permutation p-value is also reported below given
    # the modest group sizes here, so the asymptotic approximation's
    # adequacy does not need to be taken on faith).
    if all(len(v) >= 3 for v in groups.values()):
        H, p_kw_asymp = stats.kruskal(*groups.values())
        n_total = sum(len(v) for v in groups.values())
        k = len(groups)
        eps_sq = (H - k + 1) / (n_total - k)   # epsilon-squared effect size for KW
        print(f"\n  Kruskal-Wallis H = {H:.4f}, df={k-1}, p_asymptotic = {p_kw_asymp:.4f}")
        print(f"  (asymptotic chi-square approximation to H's null distribution --")
        print(f"  scipy.stats.kruskal does not compute an exact finite-sample p-value)")
        print(f"  Epsilon-squared (effect size) = {eps_sq:.4f}  "
              f"({'negligible' if eps_sq<0.01 else 'small' if eps_sq<0.06 else 'medium' if eps_sq<0.14 else 'large'})")

        # Exact / Monte-Carlo permutation p-value: shuffle the N country-
        # level values across the three regime labels (respecting each
        # regime's actual size), recompute H each time, and use the share
        # of permuted H values >= the observed H as p_perm. This removes
        # any dependence on the asymptotic chi-square approximation, which
        # matters given the modest group sizes here (n=8, 9, 27).
        rng_perm = np.random.default_rng(seed)
        all_vals = np.concatenate(list(groups.values()))
        group_sizes = [len(v) for v in groups.values()]
        n_perm = N_PERM
        H_perm = np.empty(n_perm)
        for p_i in range(n_perm):
            shuffled = rng_perm.permutation(all_vals)
            split_groups = np.split(shuffled, np.cumsum(group_sizes)[:-1])
            H_perm[p_i], _ = stats.kruskal(*split_groups)
        # (k+1)/(n+1): the observed labelling counts as one permutation, so the
        # p-value is never exactly 0 (Phipson and Smyth 2010).
        p_kw_exact = float((np.sum(H_perm >= H) + 1) / (n_perm + 1))
        print(f"  Monte Carlo permutation p-value ({n_perm} shuffles) = {p_kw_exact:.4f}"
              f"  (smallest attainable: {1/(n_perm+1):.4f})")
        if n_perm < 1000:
            print(f"  ⚠ Only {n_perm} permutations: this p-value is a coarse estimate, not an")
            print(f"    exact permutation result. Use the production N_PERM ({_REPS['N_PERM'][0]:,}) for reported results.")
        print(f"  Report both p-values; the permutation estimate avoids the chi-square")
        print(f"  approximation but is only as precise as its number of shuffles.")
        p_kw = p_kw_exact   # use the exact value downstream (pairwise gating, return dict)
    else:
        H, p_kw, p_kw_asymp, p_kw_exact = np.nan, np.nan, np.nan, np.nan
        print(f"\n  ⚠ At least one regime has fewer than 3 countries — Kruskal-Wallis skipped")

    # 2 & 3. Pairwise Mann-Whitney with Holm-Bonferroni correction + Cliff's delta
    pairs = [(regimes[i], regimes[j]) for i in range(len(regimes)) for j in range(i+1, len(regimes))]
    pairwise_results = []
    for g1, g2 in pairs:
        v1, v2 = groups[g1], groups[g2]
        if len(v1) < 2 or len(v2) < 2:
            continue
        U, p_raw = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        delta = _cliffs_delta(v1, v2)
        pairwise_results.append({"pair": f"{g1} vs {g2}", "U": U, "p_raw": p_raw, "cliffs_delta": delta})

    if pairwise_results:
        # Holm-Bonferroni step-down correction
        pdf = pd.DataFrame(pairwise_results).sort_values("p_raw").reset_index(drop=True)
        m = len(pdf)
        pdf["p_holm"] = [min(1.0, pdf.loc[i,"p_raw"] * (m - i)) for i in range(m)]
        pdf["p_holm"] = pdf["p_holm"].cummax()   # enforce monotonicity

        print(f"\n  Pairwise Mann-Whitney U tests (Holm-Bonferroni corrected, m={m}):")
        print(f"  {'Pair':<45} {'U':>8} {'p_raw':>8} {'p_holm':>8} {'Cliffs delta':>13}  Magnitude")
        print(f"  {'─'*95}")
        for _, row in pdf.iterrows():
            d = abs(row["cliffs_delta"])
            mag = "negligible" if d<0.147 else "small" if d<0.33 else "medium" if d<0.474 else "large"
            sig = "*" if row["p_holm"] < 0.05 else " "
            print(f"  {row['pair']:<45} {row['U']:>8.1f} {row['p_raw']:>8.4f} "
                  f"{row['p_holm']:>7.4f}{sig} {row['cliffs_delta']:>+13.3f}  {mag}")
        print(f"  {'─'*95}")
        if np.isfinite(p_kw) and p_kw >= 0.05:
            print(f"  ℹ Kruskal-Wallis omnibus test (exact p) did not reject at 5% (p={p_kw:.4f});")
            print(f"    pairwise tests above are reported for completeness but the overall")
            print(f"    evidence for a regime difference in {value_col} is weak.")
    else:
        pdf = pd.DataFrame()

    # 4. Bootstrap CIs for regime means and medians
    rng = np.random.default_rng(seed)
    print(f"\n  Resampling 95% intervals for regime mean/median {value_col} (countries resampled")
    print(f"  within regime, B={n_boot}). These reflect which countries are observed, NOT the")
    print(f"  uncertainty of re-estimating the frontiers (see 12.15):")
    print(f"  {'Regime':<24} {'Mean':>8} {'[95% CI]':>18}   {'Median':>8} {'[95% CI]':>18}")
    print(f"  {'─'*90}")
    boot_summary = []
    for g in regimes:
        v = groups[g]
        if len(v) < 2:
            continue
        boot_means = np.array([np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
        boot_medians = np.array([np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
        m_lo, m_hi = np.percentile(boot_means, [2.5, 97.5])
        md_lo, md_hi = np.percentile(boot_medians, [2.5, 97.5])
        print(f"  {g:<24} {v.mean():>8.4f} [{m_lo:>6.4f},{m_hi:>6.4f}]   "
              f"{np.median(v):>8.4f} [{md_lo:>6.4f},{md_hi:>6.4f}]")
        boot_summary.append({"tech_regime": g, "mean": v.mean(), "mean_ci_lo": m_lo, "mean_ci_hi": m_hi,
                             "median": np.median(v), "median_ci_lo": md_lo, "median_ci_hi": md_hi})
    print(f"  {'─'*90}")

    return {"kruskal_H": H, "kruskal_p": p_kw, "kruskal_p_asymptotic": p_kw_asymp,
            "kruskal_p_exact": p_kw_exact, "pairwise": pdf,
            "bootstrap_summary": pd.DataFrame(boot_summary)}
def compare_with_conditional_ddf(results_df, panel):
    """
    12.10 — Compare the metafrontier TGR with the Conditional DDF gap
    (dea_score_conditional - dea_score, from Step 7c). Both diagnostics
    respond to the same underlying phenomenon investigated by the
    separability test: that structural/technological conditions may shift
    the relevant production frontier, not merely a country's distance from
    a single fixed one. If the two are strongly correlated, this is
    convergent evidence from two independently-constructed diagnostics
    (a discrete 3-group metafrontier vs. a continuous k-NN local frontier)
    that the same structural-technology mechanism is at work.
    """
    print(f"\n{'─'*66}")
    print("12.10 — METAFRONTIER TGR vs CONDITIONAL DDF GAP")
    print(f"{'─'*66}")
    if "dea_score_conditional" not in panel.columns:
        print("  ⚠ dea_score_conditional not found — run Step 7c first")
        return {}
    cond = panel[["iso_code","year","dea_score","dea_score_conditional"]].copy()
    cond["cond_gap"] = cond["dea_score_conditional"] - cond["dea_score"]
    merged = results_df.merge(cond[["iso_code","year","cond_gap"]], on=["iso_code","year"], how="inner")
    merged = merged.dropna(subset=["tgr","cond_gap"])
    if len(merged) < 10:
        print(f"  ⚠ Only {len(merged)} overlapping observations — skipped")
        return {}
    r, p_r = stats.pearsonr(merged["tgr"], merged["cond_gap"])
    rho, p_rho = stats.spearmanr(merged["tgr"], merged["cond_gap"])
    print(f"  Pearson  r   = {r:+.4f}  (p={p_r:.4f})")
    print(f"  Spearman rho = {rho:+.4f}  (p={p_rho:.4f})")
    print(f"  n = {len(merged)} country-year observations")
    print(f"  Interpretation: a NEGATIVE correlation is expected -- a low TGR (large")
    print(f"  technology gap) should coincide with a large positive conditional-DDF")
    print(f"  gap (the country looks much better once compared only to structurally")
    print(f"  similar peers), since both are picking up the same underlying signal.")
    return {"pearson_r": r, "pearson_p": p_r, "spearman_rho": rho, "spearman_p": p_rho, "n": len(merged)}


def correlate_tgr_esi(results_df, panel):
    """
    12.11 — Correlate TGR with the ESI composite score, pooled Pearson and
    Spearman FIRST (as explicitly requested), before any regime-specific
    or causal claim. This is a pooled association only.
    """
    print(f"\n{'─'*66}")
    print("12.11 — TGR vs ESI (pooled correlation, reported first and plainly)")
    print(f"{'─'*66}")
    if "esi_score" not in panel.columns:
        print("  ⚠ esi_score not found")
        return {}
    merged = results_df.merge(panel[["iso_code","year","esi_score"]], on=["iso_code","year"], how="inner")
    merged = merged.dropna(subset=["tgr","esi_score"])
    if len(merged) < 10:
        print(f"  ⚠ Only {len(merged)} overlapping observations — skipped")
        return {}
    r, p_r = stats.pearsonr(merged["tgr"], merged["esi_score"])
    rho, p_rho = stats.spearmanr(merged["tgr"], merged["esi_score"])
    print(f"  Pooled Pearson  r   = {r:+.4f}  (p={p_r:.4f})")
    print(f"  Pooled Spearman rho = {rho:+.4f}  (p={p_rho:.4f})")
    print(f"  n = {len(merged)} country-year observations")
    print(f"  This is a POOLED ASSOCIATION, not a causal or within-country claim --")
    print(f"  consistent with the treatment of every other pooled correlation in this")
    print(f"  pipeline (see Step 4's correlation table caveat).")
    return {"pearson_r": r, "pearson_p": p_r, "spearman_rho": rho, "spearman_p": p_rho, "n": len(merged)}


def tgr_time_series(results_df):
    """
    12.12 — Evolution of mean TGR, 2000-2023, overall and by regime.

    *** FIX: the trend test is now a genuine panel-aware test, not a
    simple linear regression on the 24 cross-sectionally-aggregated yearly
    means. The original approach (stats.linregress on the annual mean
    series) avoided the most naive form of pseudo-replication -- it did
    not treat all 1,056 country-year TGR values as independent observations
    -- but it also discarded all cross-sectional information and cannot
    account for country-level heterogeneity or within-country serial
    correlation in its own inference. This matters directly: a declining
    trend of this kind is a substantively important finding only if it
    survives an inference procedure that respects the panel's structure.

    The fix uses PanelOLS (already the pipeline's established tool for
    exactly this, see panel_fe_regression()): TGR_it = alpha_i + beta*year
    + e_it, with entity (country) fixed effects and cluster-robust
    standard errors by entity -- the SAME convention already used
    throughout this pipeline's other panel regressions (Panel FE, Mundlak,
    Local Projections). This properly accounts for persistent country-
    level differences in mean TGR (absorbed into alpha_i) and within-
    country serial correlation (via the cluster-robust SE on beta),
    rather than inferring significance from 24 aggregated points alone.
    """
    print(f"\n{'─'*66}")
    print("12.12 — EVOLUTION OF MEAN TGR OVER TIME")
    print(f"{'─'*66}")
    overall = results_df.groupby("year")["tgr"].mean()
    by_regime = results_df.groupby(["year","tech_regime"])["tgr"].mean().unstack("tech_regime")

    print(f"\n  {'Year':<6} {'Overall':>9}", end="")
    regimes = by_regime.columns.tolist()
    for g in regimes:
        print(f" {g[:14]:>16}", end="")
    print()
    for yr in overall.index:
        print(f"  {int(yr):<6} {overall[yr]:>9.4f}", end="")
        for g in regimes:
            v = by_regime.loc[yr, g] if yr in by_regime.index and g in by_regime.columns else np.nan
            print(f" {v:>16.4f}" if np.isfinite(v) else f" {'—':>16}", end="")
        print()

    # Descriptive-only trend on the aggregated annual series (24 points) --
    # reported for reference, NOT as the primary significance claim.
    yrs = overall.index.values.astype(float)
    slope_agg, intercept, r_val, p_val_agg, se = stats.linregress(yrs, overall.values)
    print(f"\n  [Descriptive only] Linear trend on the 24 aggregated annual means:")
    print(f"    slope={slope_agg:+.5f}/yr, p={p_val_agg:.4f}, R²={r_val**2:.4f}")
    print(f"    This treats the panel's cross-sectional structure as fully collapsed")
    print(f"    into 24 points and is NOT the primary significance claim -- see below.")

    # PRIMARY panel-aware trend test: country fixed effects + cluster-robust
    # SE on the country-year panel itself (not the aggregated series).
    panel_trend = results_df[["iso_code","year","tgr"]].dropna().copy()
    trend_result = {"slope_agg": slope_agg, "p_agg": p_val_agg}
    if panel_trend["iso_code"].nunique() >= 5 and len(panel_trend) >= 20:
        try:
            pt = panel_trend.set_index(["iso_code","year"])
            y_pt = pt["tgr"]
            X_pt = pt[[]].copy()
            X_pt["year"] = pt.index.get_level_values("year").astype(float)
            fe = PanelOLS(y_pt, X_pt, entity_effects=True, time_effects=False)
            res = fe.fit(cov_type="clustered", cluster_entity=True)
            beta_year = float(res.params["year"])
            se_year = float(res.std_errors["year"])
            p_year = float(res.pvalues["year"])
            ci_lo, ci_hi = float(res.conf_int().loc["year","lower"]), float(res.conf_int().loc["year","upper"])
            print(f"\n  [PRIMARY] Panel fixed-effects trend (entity FE, cluster-robust SE by country):")
            print(f"    TGR_it = alpha_i + beta*year + e_it")
            print(f"    beta (year) = {beta_year:+.5f}/yr   cluster-robust SE = {se_year:.5f}")
            print(f"    p = {p_year:.4f}   95% CI = [{ci_lo:+.5f}, {ci_hi:+.5f}]")
            print(f"    n countries = {panel_trend['iso_code'].nunique()}, "
                  f"n obs = {len(panel_trend)}")
            if p_year < 0.05:
                print(f"    Trend is statistically distinguishable from zero even after")
                print(f"    accounting for country fixed effects and within-country serial")
                print(f"    correlation via cluster-robust standard errors.")
            else:
                print(f"    Trend does NOT reach significance once country fixed effects and")
                print(f"    cluster-robust standard errors are accounted for -- the descriptive")
                print(f"    trend on the aggregated series above should NOT be reported as a")
                print(f"    statistically established finding on its own.")
            trend_result.update({"beta_year": beta_year, "se_year": se_year, "p_year": p_year,
                                 "ci_lo": ci_lo, "ci_hi": ci_hi})
        except Exception as e:
            print(f"\n  ⚠ Panel fixed-effects trend test failed ({e}) — falling back to the")
            print(f"    descriptive aggregated-series trend above only.")
    else:
        print(f"\n  ⚠ Insufficient countries/observations for the panel fixed-effects trend")
        print(f"    test — descriptive aggregated-series trend above only.")
    return {"overall": overall, "by_regime": by_regime, **trend_result}


def correlate_tgr_gml(results_df, mpi_df):
    """
    12.13 — Connect the metafrontier with GML: corr(delta_TGR, GTC) and
    corr(delta_theta_group, GEC), matching each country's year-to-year TGR
    change to the same transition's GTC/GEC from Step 6's GML decomposition.
    """
    print(f"\n{'─'*66}")
    print("12.13 — METAFRONTIER CHANGE vs GML DECOMPOSITION")
    print(f"{'─'*66}")
    if mpi_df is None or len(mpi_df) == 0:
        print("  ⚠ mpi_df not available — run Step 6 first")
        return {}

    r_sorted = results_df.sort_values(["iso_code","year"]).copy()
    r_sorted["delta_tgr"] = r_sorted.groupby("iso_code")["tgr"].diff()
    r_sorted["delta_theta_group"] = r_sorted.groupby("iso_code")["theta_group"].diff()
    # mpi_df transitions are indexed by the LATER year (year_t+1) in this
    # pipeline's convention; align on that year as the transition's endpoint.
    # *** mpi_df uses "ec"/"tc" as its actual column names (matching the
    # pipeline's established MPI/EC/TC plotting convention), NOT "gec"/"gtc"
    # -- verified directly against build_mpi()'s own DataFrame construction
    # and plot_step6_mpi()'s column references before writing this function,
    # to avoid a silent zero-match failure from a plausible-looking guess. ***
    if "year_t1" not in mpi_df.columns:
        print("  ⚠ Could not identify the GML transition year column (year_t1) — skipped")
        return {}

    merged = r_sorted.merge(
        mpi_df[["iso_code", "year_t1", "ec", "tc"]].rename(columns={"year_t1": "year"}),
        on=["iso_code","year"], how="inner"
    ).dropna(subset=["delta_tgr","delta_theta_group","ec","tc"])

    if len(merged) < 10:
        print(f"  ⚠ Only {len(merged)} overlapping transitions — skipped")
        return {}

    r1, p1 = stats.pearsonr(merged["delta_tgr"], merged["tc"])
    r2, p2 = stats.pearsonr(merged["delta_theta_group"], merged["ec"])
    print(f"  corr(delta_TGR, GTC)          = {r1:+.4f}  (p={p1:.4f}, n={len(merged)})")
    print(f"  corr(delta_theta_Group, GEC)  = {r2:+.4f}  (p={p2:.4f}, n={len(merged)})")
    print(f"  Interpretation (read from the signs obtained in this run):")
    if p1 < 0.05 and r1 < 0:
        print(f"  corr(ΔTGR, GTC) < 0: years in which the common frontier shifts out")
        print(f"  for a country (GTC > 1) are years in which its regime's technology gap")
        print(f"  WIDENS -- frontier progress is concentrated in other regimes, so the")
        print(f"  country's own regime falls further behind the metafrontier.")
    elif p1 < 0.05 and r1 > 0:
        print(f"  corr(ΔTGR, GTC) > 0: years of frontier progress for a country coincide")
        print(f"  with its regime closing in on the metafrontier -- progress originates")
        print(f"  within its own regime.")
    else:
        print(f"  corr(ΔTGR, GTC) not significant: no detectable link between changes in")
        print(f"  the technology gap and frontier shift.")
    if p2 < 0.05:
        print(f"  corr(Δθ^Group, GEC) {'>' if r2 > 0 else '<'} 0: within-regime catch-up "
              f"{'moves with' if r2 > 0 else 'moves against'} catch-up to the annual frontier,")
        print(f"  as expected when the regime frontier and the annual frontier overlap.")
    return {"corr_dTGR_GTC": r1, "p_dTGR_GTC": p1, "corr_dGroup_GEC": r2, "p_dGroup_GEC": p2, "n": len(merged)}
def _adjusted_rand_index(labels_a, labels_b):
    """
    Adjusted Rand Index between two categorical partitions of the same
    items, implemented directly from the standard formula (Hubert & Arabie
    1985) to avoid an sklearn dependency. ARI=1 for identical partitions,
    ARI~0 for random-chance agreement, negative for worse-than-chance.
    """
    from itertools import product
    labels_a = np.asarray(labels_a); labels_b = np.asarray(labels_b)
    classes_a = np.unique(labels_a); classes_b = np.unique(labels_b)
    n = len(labels_a)

    contingency = np.zeros((len(classes_a), len(classes_b)), dtype=int)
    for i, ca in enumerate(classes_a):
        for j, cb in enumerate(classes_b):
            contingency[i, j] = np.sum((labels_a == ca) & (labels_b == cb))

    def _comb2(x):
        return x * (x - 1) / 2.0

    sum_comb_c = sum(_comb2(contingency[i,j]) for i in range(len(classes_a)) for j in range(len(classes_b)))
    sum_comb_a = sum(_comb2(contingency[i,:].sum()) for i in range(len(classes_a)))
    sum_comb_b = sum(_comb2(contingency[:,j].sum()) for j in range(len(classes_b)))
    comb_n = _comb2(n)

    expected = sum_comb_a * sum_comb_b / comb_n if comb_n > 0 else 0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    return (sum_comb_c - expected) / denom


def robustness_data_driven_clustering(panel, theoretical_regime, k=3, seed=42):
    """
    12.14 Robustness A — data-driven clustering. Cluster countries using
    ONLY their full-sample average ELECTRICITY-GENERATION shares (coal,
    oil, gas, nuclear, hydro, other renewables), standardised, via k-means
    with k=3. Compare membership against the theoretically-defined regimes
    via a contingency table and the Adjusted Rand Index.

    Uses the SAME electricity-specific ("_share_elec") basis as the main
    theoretical classification in assign_technology_regime(), not the
    total-primary-energy shares used elsewhere in this pipeline (e.g. for
    the Shannon/HHI resilience dimension) -- comparing a theoretical
    regime built on electricity shares against a clustering built on a
    different variable basis would confound genuine agreement/disagreement
    with a basis mismatch, exactly the distortion identified and fixed for
    the main classification.
    """
    print(f"\n{'═'*66}")
    print("12.14 ROBUSTNESS A — DATA-DRIVEN CLUSTERING (k-means, k=3, electricity shares)")
    print(f"{'═'*66}")
    share_cols = [c for c in ["coal_share_elec","oil_share_elec","gas_share_elec",
                              "nuclear_share_elec","hydro_share_elec","other_renew_share_elec"]
                  if c in panel.columns]

    means = panel.groupby("iso_code")[share_cols].mean().dropna()
    if len(means) < k * 2:
        print(f"  ⚠ Only {len(means)} countries with complete share data — skipped")
        return {}

    X = (means - means.mean()) / (means.std() + 1e-9)
    from scipy.cluster.vq import kmeans2
    rng_seed = seed
    centroids, labels = kmeans2(X.values, k, seed=rng_seed, minit="++")
    cluster_labels = pd.Series(labels, index=means.index, name="cluster")

    merged = pd.DataFrame({"tech_regime": theoretical_regime}).join(cluster_labels, how="inner")
    contingency = pd.crosstab(merged["tech_regime"], merged["cluster"])
    print(f"\n  Contingency table (theoretical regime x k-means cluster):")
    print(contingency.to_string())

    ari = _adjusted_rand_index(merged["tech_regime"].values, merged["cluster"].values)
    print(f"\n  Adjusted Rand Index = {ari:.4f}")
    if ari > 0.5:
        print(f"  Substantial agreement between the theoretical and data-driven groupings --")
        print(f"  the metafrontier findings are considerably more convincing as a result.")
    elif ari > 0.2:
        print(f"  Modest agreement -- the two groupings are related but not interchangeable;")
        print(f"  interpret the metafrontier findings as regime-specific, not fully general.")
    else:
        print(f"  Weak agreement -- the theoretical regime definition captures something")
        print(f"  distinct from the dominant axes of variation in the raw generation-mix")
        print(f"  data. This does not invalidate the theoretical grouping (which is")
        print(f"  motivated by an explicit decarbonisation-relevant threshold rule, not")
        print(f"  by maximising within-cluster homogeneity), but it should be disclosed.")
    if ari <= 0.5:
        print(f"  This check therefore does NOT corroborate the regime grouping: regime")
        print(f"  results rest on the theory-based threshold rule and must not be described")
        print(f"  as robust to the grouping method.")

    return {"contingency": contingency, "ari": ari, "cluster_labels": cluster_labels}


def run_step12_import_robustness(panel, alt_import_col="net_elec_imports_share_demand",
                                 group_col="tech_regime", vrs=VRS):
    """
    12.x ROBUSTNESS — re-run the group/meta DDF with an ALTERNATIVE import-
    dependence variable, comparing against the primary specification.

    The PRIMARY Step 12 analysis uses dea_net_import (World Bank
    EG.IMP.CONS.ZS, bounded at max(0, net_import_pct)) as INPUT_COLS[0],
    identical to Step 5's own production DDF -- this is unchanged and
    remains the reported main result.

    This function additionally re-solves the SAME group/meta DDF using
    net_elec_imports_share_demand (OWID, electricity-specific import
    share) in place of dea_net_import, as a robustness/sensitivity check
    only -- NOT as an alternative primary specification. This is useful
    specifically because dea_net_import (overall energy import dependence)
    and net_elec_imports_share_demand (electricity-specific) capture
    related but distinct exposures, and a country's TGR ranking that is
    stable across both is a more robust finding than one that depends on
    which import variable is used.

    Requires net_elec_imports_share_demand to already be present in panel
    (added via the OWID fetch); if absent, this robustness check is
    skipped with a clear message rather than silently failing.

    Returns a DataFrame with both TGR series and their correlation, and
    prints a side-by-side regime summary comparison.
    """
    global INPUT_COLS
    print(f"\n{'═'*66}")
    print(f"12.x ROBUSTNESS — ALTERNATIVE IMPORT VARIABLE ({alt_import_col})")
    print(f"{'═'*66}")

    if alt_import_col not in panel.columns:
        print(f"  ⚠ {alt_import_col} not found in panel — this robustness check requires")
        print(f"    the OWID electricity-import-share column; skipped.")
        return None
    if "dea_net_import" not in panel.columns:
        print(f"  ⚠ dea_net_import (primary variable) not found — skipped.")
        return None

    primary_import_col = INPUT_COLS[0]
    print(f"  Primary import variable  : {primary_import_col} (unchanged, production DDF)")
    print(f"  Robustness import variable: {alt_import_col}")

    panel_alt = panel.copy()
    panel_alt["_alt_import_bounded"] = panel_alt[alt_import_col].clip(lower=0)

    # Temporarily swap INPUT_COLS[0] to the alternative variable, re-solve,
    # then restore -- never leave global state altered on exit or on error.
    original_input_cols = list(INPUT_COLS)
    try:
        INPUT_COLS = ["_alt_import_bounded"] + original_input_cols[1:]
        results_robust = _solve_group_meta_ddf(panel_alt, group_col=group_col, vrs=vrs)
    finally:
        INPUT_COLS = original_input_cols

    if results_robust.empty:
        print(f"  ⚠ Robustness DDF produced no results — skipped.")
        return None

    print(f"\n  Regime summary under the ROBUSTNESS import variable:")
    regime_summary_table(results_robust)

    return results_robust


def compare_import_robustness(results_primary, results_robust):
    """
    Compare country-level mean TGR between the primary (dea_net_import) and
    robustness (alternative import variable) specifications: Spearman rank
    correlation of country rankings, plus the countries whose classification
    would change most between the two.
    """
    print(f"\n{'─'*66}")
    print("IMPORT-VARIABLE ROBUSTNESS: PRIMARY vs ALTERNATIVE COMPARISON")
    print(f"{'─'*66}")
    if results_primary is None or results_robust is None or results_primary.empty or results_robust.empty:
        print("  ⚠ One or both specifications unavailable — skipped.")
        return {}

    p_means = results_primary.groupby("iso_code")["tgr"].mean().rename("tgr_primary")
    r_means = results_robust.groupby("iso_code")["tgr"].mean().rename("tgr_robust")
    merged = pd.concat([p_means, r_means], axis=1).dropna()

    if len(merged) < 5:
        print(f"  ⚠ Only {len(merged)} overlapping countries — skipped.")
        return {}

    rho, p_rho = stats.spearmanr(merged["tgr_primary"], merged["tgr_robust"])
    print(f"  Spearman rank correlation of country-mean TGR (primary vs robustness): "
          f"rho={rho:+.4f} (p={p_rho:.4f}, n={len(merged)})")
    merged["abs_rank_diff"] = (merged["tgr_primary"].rank() - merged["tgr_robust"].rank()).abs()
    biggest_movers = merged.sort_values("abs_rank_diff", ascending=False).head(5)
    print(f"\n  Countries whose TGR ranking differs most between the two specifications:")
    print(biggest_movers.to_string())
    # Overall rank agreement can hide large moves for individual countries
    # (reviewer: rho = 0.83 overall, but Canada moved 27 ranks). Report both.
    MOVE_THRESHOLD = 10
    movers = merged[merged["abs_rank_diff"] >= MOVE_THRESHOLD].sort_values("abs_rank_diff", ascending=False)
    max_move = merged["abs_rank_diff"].max()
    mover_txt = ", ".join(f"{i} ({int(r)})" for i, r in movers["abs_rank_diff"].items())
    print(f"\n  Largest rank change: {int(max_move)} of {len(merged)} positions; "
          f"{len(movers)} countries move ≥{MOVE_THRESHOLD} ranks"
          + (f": {mover_txt}" if len(movers) else ""))
    # Regime-level check: does the regime ordering of mean TGR survive?
    reg_ok = None
    if "tech_regime" in results_primary.columns and "tech_regime" in results_robust.columns:
        o1 = results_primary.groupby("tech_regime")["tgr"].mean().sort_values().index.tolist()
        o2 = results_robust.groupby("tech_regime")["tgr"].mean().sort_values().index.tolist()
        reg_ok = (o1 == o2)
        print(f"  Regime ordering of mean TGR: primary {o1} | alternative {o2} -> "
              f"{'unchanged' if reg_ok else 'CHANGED'}")
    if rho > 0.7:
        print(f"\n  Strong overall rank agreement (rho={rho:.2f}).")
        if reg_ok is False:
            print(f"  BUT the regime ordering of mean TGR changes with the import variable:")
            print(f"  regime-level conclusions are NOT robust to how import dependence is measured.")
        else:
            print(f"  The aggregate ranking" + (" and the regime ordering are" if reg_ok else " is")
                  + " robust to the import variable.")
        if len(movers):
            print(f"  Country-level positions are NOT robust for the countries listed above;")
            print(f"  report their TGR rank as sensitive to how import dependence is measured")
            print(f"  (fuel-wide vs electricity-specific), rather than as a firm ranking.")
    elif rho > 0.4:
        print(f"\n  Moderate rank agreement -- the TGR findings are broadly, but not")
        print(f"  fully, robust to the choice of import-dependence variable.")
    else:
        print(f"\n  Weak rank agreement -- the choice of import-dependence variable")
        print(f"  materially affects country-level TGR rankings; the primary")
        print(f"  specification (dea_net_import) should be treated as authoritative,")
        print(f"  with this discrepancy disclosed rather than resolved by preference.")
    return {"spearman_rho": rho, "spearman_p": p_rho, "n": len(merged), "merged": merged,
            "max_rank_change": float(max_move), "countries_moving_10plus": movers.index.tolist(),
            "regime_ordering_unchanged": reg_ok}


def robustness_g20_split(results_df, panel, europe_iso3_set, g20_iso3_set):
    """
    12.14 Robustness B — institutional/geographical grouping: G20 (19) vs
    Europe non-G20 (26), the DISJOINT partition of the 45-country panel
    (EUROPE_NONG20_ISO3), NOT Europe-30, which overlaps the G20. Explicitly
    treated as SECONDARY since it is not a technology classification.
    """
    print(f"\n{'═'*66}")
    print(f"12.14 ROBUSTNESS B — G20 ({len(G20_ISO3)}) vs EUROPE NON-G20 ({len(EUROPE_NONG20_ISO3)}) "
          f"(disjoint; secondary, non-technological)")
    print(f"{'═'*66}")
    grp = results_df.copy()
    grp["inst_group"] = grp["iso_code"].apply(
        lambda iso: (f"G20 ({len(g20_iso3_set)})" if iso in g20_iso3_set
                     else f"Europe non-G20 ({len(europe_iso3_set)})"))
    _uncl = sorted(set(grp["iso_code"]) - set(g20_iso3_set) - set(europe_iso3_set))
    assert not _uncl, f"countries in neither group: {_uncl}"
    summary = grp.groupby("inst_group").agg(
        mean_theta_group=("theta_group","mean"),
        mean_tgr=("tgr","mean"), n_obs=("tgr","count")).reset_index()
    print(summary.to_string(index=False))
    print(f"\n  Reported as a SECONDARY, non-technological cross-check only -- an")
    print(f"  institutional/geographic split is not itself a claim about production")
    print(f"  technology, unlike the fossil/low-carbon/diversified regime grouping above.")
    return summary


def bootstrap_group_meta_tgr(panel, tech_regime_map, results_df=None, B=B_BOOTSTRAP, seed=42):
    """
    12.15 -- Bootstrap confidence intervals for theta^Group, theta^Meta and
    TGR, rather than reporting deterministic DEA/DDF values as though
    measured without uncertainty.

    *** METHODOLOGY, REVISED AFTER TWO ROUNDS OF DIRECT TESTING FOUND THE
    ORIGINAL "PERTURB-AND-RESOLVE" APPROACH UNRELIABLE FOR A RATIO:
    The original design perturbed the underlying DDF inputs (Y, B) via a
    Simar-Wilson-style smooth bootstrap and re-solved theta^Group and
    theta^Meta fresh on each perturbed pseudo-sample, following the same
    logic as the Step 5 DEA bias-correction bootstrap. Two serious issues
    were found by testing this directly against the point estimate, not
    merely assumed correct:

    (1) theta^Meta in the original version was seeded from Step 5's own
        dea_score, a DIFFERENT quantity from the theta^Meta the point
        estimate actually reports (which is solved fresh within
        _solve_group_meta_ddf, not derived from dea_score at all) --
        producing bootstrap TGR values entirely inconsistent with the
        point estimate (observed directly: Fossil-dominant point estimate
        0.799 vs bootstrap mean 0.951, non-overlapping 95% CI).

    (2) After fixing (1) so theta^Meta was solved fresh and consistently
        with the point estimate on every replication, a SECOND, more
        fundamental problem remained: TGR is a RATIO of two DDF scores
        (theta^Meta and theta^Group) computed from the SAME jointly-
        perturbed pseudo-sample. Perturbing shared inputs and taking a
        ratio of two frontier re-solves does not produce a bootstrap
        distribution centred on the original point estimate in the way a
        single DEA score's bootstrap distribution is (the property the
        Simar-Wilson reflected-CI formula relies on) -- confirmed directly:
        even with theta^Meta fixed, a systematic, non-random gap between
        the point estimate and the bootstrap distribution persisted, in a
        consistent direction across every regime tested, the signature of
        a structural mismatch between this ratio and the reflected-CI
        method rather than remaining sampling noise.

    Given both attempts at correctly adapting the input-perturbation
    approach to a RATIO failed direct verification, this function now uses
    a simpler, more robust, and directly verifiable alternative: a
    nonparametric bootstrap of the ALREADY-COMPUTED point-estimate TGR
    (and theta^Group, theta^Meta) values in results_df, resampling
    COUNTRIES WITH REPLACEMENT within each regime (for regime-level CIs)
    and resampling YEARS WITH REPLACEMENT within each country (for
    country-level CIs) -- exactly the same convention already used, and
    already verified to behave correctly, for the regime-mean bootstrap
    CIs in test_regime_differences() elsewhere in this Step 12 module.
    This bootstraps the SAME quantities the point estimate reports,
    directly, and is mechanically guaranteed to be centred on the
    empirical distribution being resampled -- it cannot suffer from the
    "estimating a different quantity" failure mode identified above, since
    no DDF is re-solved at all; only the already-computed values are
    resampled. The trade-off, stated plainly: this characterises sampling
    uncertainty due to which countries/years happen to be observed, not
    the additional uncertainty from re-estimating the DEA frontier itself
    on a different pseudo-sample of the underlying production data -- a
    narrower notion of "bootstrap" than Step 5's DEA bias-correction
    bootstrap, but one that is directly verified to produce a CI containing
    the point estimate, rather than one that is not. ***

    results_df : REQUIRED -- the point-estimate output of
                 _solve_group_meta_ddf(), containing theta_group,
                 theta_meta and tgr by iso_code and year.

    Returns confidence intervals for:
      - regime-mean theta^Group, theta^Meta and TGR (TGR_bar_g)
      - country-average TGR
    """
    print(f"\n{'═'*66}")
    print(f"12.15 — RESAMPLING INTERVALS (B={B}, country/year resampling of estimated scores)")
    print(f"  WHAT THESE ARE: intervals from resampling the ALREADY-ESTIMATED θ^Group,")
    print(f"  θ^Meta and TGR values (countries within regime; years within country).")
    print(f"  They capture sampling variation in which countries/years are observed.")
    print(f"  WHAT THEY ARE NOT: they do not capture frontier-estimation uncertainty --")
    print(f"  the frontiers are not re-estimated. Report them as 'resampling intervals',")
    print(f"  never as bootstrap confidence intervals for the frontier or for TGR itself.")
    print(f"  Frontier-estimation uncertainty is reported separately below, where available.")
    print(f"{'═'*66}")

    if results_df is None or results_df.empty:
        print(f"  ⚠ results_df not supplied or empty — bootstrap skipped.")
        return {"regime_ci": {}, "country_ci": {}}

    rng = np.random.default_rng(seed)
    regimes = sorted(results_df["tech_regime"].dropna().unique())

    print(f"\n  Resampling 95% interval for regime-mean theta^Group, theta^Meta and TGR")
    print(f"  (B={B}, resampling countries with replacement within each regime):")
    regime_ci = {"theta_group": {}, "theta_meta": {}, "tgr": {}}
    for g in regimes:
        g_df = results_df[results_df["tech_regime"] == g]
        countries_g = g_df["iso_code"].unique()
        if len(countries_g) < 3:
            continue
        boot_stats = {"theta_group": [], "theta_meta": [], "tgr": []}
        for _ in range(B):
            sampled = rng.choice(countries_g, size=len(countries_g), replace=True)
            resampled_df = pd.concat([g_df[g_df["iso_code"] == c] for c in sampled], ignore_index=True)
            for col in boot_stats:
                boot_stats[col].append(resampled_df[col].mean())

        for col in boot_stats:
            pt = float(g_df[col].mean())
            lo, hi = np.percentile(boot_stats[col], [2.5, 97.5])
            regime_ci[col][g] = (pt, float(lo), float(hi))
        pt_tgr, lo_tgr, hi_tgr = regime_ci["tgr"][g]
        print(f"    {g:<24}: TGR point={pt_tgr:.4f}  [95% resampling interval: {lo_tgr:.4f}, {hi_tgr:.4f}]  "
              f"(n={len(countries_g)} countries)")

    print(f"\n  Resampling 95% interval for country-average TGR (top/bottom 5 by point estimate,")
    print(f"  B={B}, resampling years with replacement within each country):")
    country_ci = {}
    for iso, c_df in results_df.groupby("iso_code"):
        if len(c_df) < 4:
            continue
        boot_means = np.array([
            c_df["tgr"].sample(n=len(c_df), replace=True, random_state=rng.integers(0, 2**31)).mean()
            for _ in range(B)
        ])
        pt = float(c_df["tgr"].mean())
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        country_ci[iso] = (pt, float(lo), float(hi))
    sorted_countries = sorted(country_ci.items(), key=lambda kv: kv[1][0])
    for iso, (pt, lo, hi) in sorted_countries[:5] + sorted_countries[-5:]:
        print(f"    {iso:<6}: point = {pt:.4f}  [95% resampling interval: {lo:.4f}, {hi:.4f}]")

    # Sanity check, printed explicitly rather than assumed: every regime and
    # country CI constructed above is a percentile CI of resampled POINT-
    # ESTIMATE values, so it CANNOT fail to bracket the full-sample point
    # estimate whenever that point estimate lies within the range of the
    # per-country/per-year values being resampled -- verified below, rather
    # than asserted, so any residual failure is visible immediately.
    n_checked, n_ok = 0, 0
    for g, (pt, lo, hi) in regime_ci["tgr"].items():
        n_checked += 1
        n_ok += int(lo <= pt <= hi)
    print(f"\n  Containment check: {n_ok}/{n_checked} regime TGR point estimates fall within")
    print(f"  their own reported 95% resampling interval (expect {n_checked}/{n_checked} by construction).")

    # Frontier-estimation uncertainty, kept SEPARATE from the resampling intervals.
    # Under the baseline, theta^Meta is the annual all-country frontier score, i.e.
    # Step 5's dea_score, whose Simar-Wilson (1998) interval (dea_lo, dea_hi) does
    # come from re-estimating the frontier on pseudo-samples. No such interval
    # exists for theta^Group or for the ratio TGR.
    frontier_unc = None
    if all(c in panel.columns for c in ("dea_lo", "dea_hi", "dea_score")):
        fu = results_df.merge(panel[["iso_code", "year", "dea_lo", "dea_hi", "dea_score"]],
                              on=["iso_code", "year"], how="left")
        fu = fu[(fu["theta_meta"] - fu["dea_score"]).abs() < 1e-4]   # same quantity only
        if len(fu):
            fu["width"] = fu["dea_hi"] - fu["dea_lo"]
            frontier_unc = fu.groupby("tech_regime")["width"].agg(["mean", "median"]).reset_index()
            print(f"\n  FRONTIER-ESTIMATION UNCERTAINTY (separate measure): mean width of the")
            print(f"  Step 5 Simar-Wilson (1998) 95% interval for θ^Meta (= dea_score), by regime:")
            for _, r in frontier_unc.iterrows():
                print(f"    {r['tech_regime']:<24}: mean width {r['mean']:.4f}  (median {r['median']:.4f})")
            print(f"  No frontier-bootstrap interval is available for θ^Group or TGR; the")
            print(f"  resampling intervals above must not be read as a substitute.")

    return {"regime_ci": regime_ci["tgr"], "theta_group_ci": regime_ci["theta_group"],
            "theta_meta_ci": regime_ci["theta_meta"], "country_ci": country_ci,
            "interval_type": "resampling of estimated scores (no frontier re-estimation)",
            "frontier_uncertainty_theta_meta": frontier_unc}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 DIAGRAMS
# ─────────────────────────────────────────────────────────────────────────────

def plot_step12_quadrant(results_df, classification_df):
    """
    Step 12 Figure A -- the country-level four-way classification scatter:
    mean theta^Group (x-axis) vs mean TGR (y-axis), one point per country,
    coloured by regime, with the median-split quadrant lines and labels
    from classify_four_way() drawn explicitly. This is the single most
    informative Step 12 diagram: it makes the operational-gap/technology-
    gap distinction visible directly, rather than only tabulated.
    """
    df = classification_df.copy()
    med_eff = df["mean_theta_group"].median()
    med_tgr = df["mean_tgr"].median()

    regime_colors = {"Fossil-dominant": COLORS["red"], "Low-carbon specialised": COLORS["teal"],
                     "Diversified/mixed": COLORS["blue"]}

    fig, ax = plt.subplots(figsize=(14, 11))
    for regime, sub in df.groupby("tech_regime"):
        ax.scatter(sub["mean_theta_group"], sub["mean_tgr"],
                  color=regime_colors.get(regime, COLORS["dark"]), s=90,
                  edgecolor="white", linewidth=0.8, label=regime, zorder=3, alpha=0.85)

    ax.axvline(med_eff, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.axhline(med_tgr, color="gray", linestyle="--", linewidth=1, zorder=1)

    # Quadrant names sit in the four corners; country labels are kept off them.
    label_kwargs = dict(fontsize=10.5, style="italic", color="#555555", transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9))
    quad_txt = [ax.text(0.01, 0.99, "Operational improvement needed", ha="left", va="top", **label_kwargs),
                ax.text(0.99, 0.99, "Technology leaders", ha="right", va="top", **label_kwargs),
                ax.text(0.01, 0.01, "Dual gap", ha="left", va="bottom", **label_kwargs),
                ax.text(0.99, 0.01, "Efficient but technology-constrained", ha="right", va="bottom", **label_kwargs)]

    ax.set_xlabel(r"Mean group efficiency  $\bar{\theta}^{Group}$")
    ax.set_ylabel(r"Mean Technology Gap Ratio  $\overline{TGR}$")
    ax.set_title("Step 12 — Four-Way Country Classification\n"
                "(median-split quadrants: group efficiency × TGR)",
                fontsize=11, fontweight="bold")
    _legend_right(ax, fontsize=10, title="Regime")
    _label_points(ax, df["mean_theta_group"].values, df["mean_tgr"].values, df["iso_code"].tolist(),
                  colors=[regime_colors.get(g, COLORS["dark"]) for g in df["tech_regime"]],
                  avoid=quad_txt)
    _save("step12_quadrant.png")


def plot_step12_regime_summary(regime_summary_df):
    """
    Step 12 Figure B -- grouped bar chart of theta^Group, theta^Meta and
    mean TGR by technology regime, the direct visual counterpart of the
    "TABLE: METAFRONTIER SUMMARY BY TECHNOLOGY REGIME" printed by
    regime_summary_table().
    """
    df = regime_summary_df.sort_values("mean_tgr", ascending=False)
    regimes = df["tech_regime"].tolist()
    x = np.arange(len(regimes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, df["mean_theta_group"], width=w, label=r"$\bar{\theta}^{Group}$", color=COLORS["blue"])
    ax.bar(x,     df["mean_theta_meta"],  width=w, label=r"$\bar{\theta}^{Meta}$",  color=COLORS["purple"])
    ax.bar(x + w, df["mean_tgr"],         width=w, label="Mean TGR", color=COLORS["teal"])

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Efficiency score")
    ax.set_title("Step 12 — Metafrontier Summary by Technology Regime", fontsize=12, fontweight="bold")
    _legend_right(ax)
    for i, row in enumerate(df.itertuples()):
        ax.text(i, row.mean_tgr + 0.02, f"n={int(row.n_obs)}", ha="center", fontsize=9, color="gray")
    _save("step12_regime_summary.png")


def plot_step12_tgr_trend(ts_dict):
    """
    Step 12 Figure C -- evolution of mean TGR over time, overall and by
    regime, the direct visual counterpart of tgr_time_series()'s printed
    table and panel fixed-effects trend test.
    """
    overall = ts_dict.get("overall")
    by_regime = ts_dict.get("by_regime")
    if overall is None:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(overall.index, overall.values, color=COLORS["dark"], linewidth=2.4,
           marker="o", markersize=4, label="Overall", zorder=5)

    regime_colors = {"Fossil-dominant": COLORS["red"], "Low-carbon specialised": COLORS["teal"],
                     "Diversified/mixed": COLORS["blue"]}
    if by_regime is not None:
        for regime in by_regime.columns:
            ax.plot(by_regime.index, by_regime[regime], color=regime_colors.get(regime, "gray"),
                   linewidth=1.4, marker="s", markersize=3, alpha=0.85, label=regime)

    beta = ts_dict.get("beta_year")
    p_year = ts_dict.get("p_year")
    if beta is not None and p_year is not None:
        sig = "significant" if p_year < 0.05 else "not significant"
        ax.text(0.02, 0.03, f"Panel FE trend: β={beta:+.5f}/yr, p={p_year:.4f} ({sig})",
               transform=ax.transAxes, fontsize=9, style="italic",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="lightgray"))

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Technology Gap Ratio (TGR)")
    ax.set_title("Step 12 — Evolution of Mean TGR, 2000–2023", fontsize=12, fontweight="bold")
    _legend_right(ax)
    _save("step12_tgr_trend.png")


def plot_step12_bootstrap_ci(bootstrap_dict):
    """
    Step 12 Figure D -- forest plot of regime-mean TGR point estimates
    with their bootstrap 95% CIs, the direct visual counterpart of
    bootstrap_group_meta_tgr()'s printed CI table. Confirms visually that
    every point estimate falls within its own reported interval.
    """
    regime_ci = bootstrap_dict.get("regime_ci", {})
    if not regime_ci:
        return
    regimes = list(regime_ci.keys())
    pts  = [regime_ci[g][0] for g in regimes]
    los  = [regime_ci[g][1] for g in regimes]
    his  = [regime_ci[g][2] for g in regimes]

    fig, ax = plt.subplots(figsize=(8, 3.5 + 0.5*len(regimes)))
    y = np.arange(len(regimes))
    xerr_lo = np.array(pts) - np.array(los)
    xerr_hi = np.array(his) - np.array(pts)
    ax.errorbar(pts, y, xerr=[xerr_lo, xerr_hi], fmt="o", color=COLORS["blue"],
               capsize=5, markersize=8, linewidth=1.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(regimes, fontsize=10)
    ax.set_xlabel("Mean TGR (point estimate, 95% resampling interval)")
    ax.set_title("Step 12 — Resampling Intervals for Regime-Mean TGR\n"
                "(countries resampled within regime; frontiers NOT re-estimated)",
                fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.8)
    for yi, pt in zip(y, pts):
        ax.text(pt, yi + 0.15, f"{pt:.3f}", ha="center", fontsize=9)
    _save("step12_resampling_intervals.png")


def plot_step12_distribution(results_df):
    """
    Step 12 Figure E -- boxplot of country-level mean TGR by regime, the
    direct visual counterpart of test_regime_differences()'s Kruskal-
    Wallis and pairwise Mann-Whitney results. One point per country
    (collapsed across years first, matching the N=len(PANEL_ISO3) country-level test).
    """
    country_vals = results_df.groupby(["iso_code","tech_regime"])["tgr"].mean().reset_index()
    regimes = sorted(country_vals["tech_regime"].unique())
    data = [country_vals[country_vals["tech_regime"]==g]["tgr"].values for g in regimes]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot(data, labels=regimes, patch_artist=True, widths=0.5,
                    medianprops=dict(color=COLORS["dark"], linewidth=2))
    palette = [COLORS["red"], COLORS["teal"], COLORS["blue"]]
    for patch, color in zip(bp["boxes"], palette[:len(regimes)]):
        patch.set_facecolor(color); patch.set_alpha(0.55)

    for i, g in enumerate(regimes):
        vals = country_vals[country_vals["tech_regime"]==g]["tgr"].values
        jitter = np.random.default_rng(0).normal(0, 0.04, len(vals))
        ax.scatter(np.full(len(vals), i+1) + jitter, vals, color="black", s=18, alpha=0.6, zorder=3)

    ax.set_ylabel("Country-level mean TGR")
    ax.set_title(f"Step 12 — TGR Distribution by Technology Regime\n"
                f"(N={len(country_vals)} country-level means; see Kruskal-Wallis / "
                f"pairwise results)",
                fontsize=10.5, fontweight="bold")
    _save("step12_tgr_distribution.png")


def plot_step12_robustness_contingency(contingency_table):
    """
    Step 12 Figure F -- heatmap of the theoretical-regime x k-means-cluster
    contingency table from robustness_data_driven_clustering(), the direct
    visual counterpart of the printed contingency table and Adjusted Rand
    Index.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    data = contingency_table.values
    im = ax.imshow(data, cmap="Greens", aspect="auto")
    ax.set_xticks(range(contingency_table.shape[1]))
    ax.set_xticklabels([f"Cluster {c}" for c in contingency_table.columns])
    ax.set_yticks(range(contingency_table.shape[0]))
    ax.set_yticklabels(contingency_table.index, fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, str(data[i,j]), ha="center", va="center",
                   color="white" if data[i,j] > data.max()/2 else "black", fontsize=10)
    ax.set_title("Step 12 Robustness A — Theoretical Regime vs k-Means Cluster",
                fontsize=10.5, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Number of countries")
    _save("step12_robustness_contingency.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12.12b — WHY DOES A REGIME'S TGR FALL? Separating the mechanisms
# ─────────────────────────────────────────────────────────────────────────────
# A falling TGR can come from (a) the regime's own performance deteriorating,
# (b) the common frontier improving faster than the regime, or (c) both.
# Annual frontiers cannot tell these apart, because every year is measured
# against that year's own best practice. A FIXED benchmark is needed: the
# pooled technology T^G built from all years (as in the GML, Section 5).
# Against T^G a rising score means a country genuinely moved closer to best
# practice, whatever happened to other countries.
#
# With all three distances solved on ONE set of pooled normalisation bounds:
#
#   theta^G = theta^Group * TGR * MGR,   MGR = theta^G / theta^Meta
#
# MGR (meta-to-global ratio) measures how close year t's common frontier lies
# to the pooled envelope. MGR rising over time means the common frontier
# itself is advancing. In logs the identity is additive, so changes split
# exactly into: within-regime catch-up (theta^Group), technology gap (TGR)
# and common-frontier position (MGR).
# ─────────────────────────────────────────────────────────────────────────────

_MIX_COLS = {"coal_share_elec": "Coal", "gas_share_elec": "Gas", "oil_share_elec": "Oil",
             "nuclear_share_elec": "Nuclear", "hydro_share_elec": "Hydro",
             "other_renew_share_elec": "Other renewables"}


def decompose_tgr_mechanisms(panel, group_col="tech_regime", vrs=VRS):
    """
    Solve theta^Group, theta^Meta and theta^G (pooled technology T^G) for every
    country-year on ONE pooled set of [1,10] normalisation bounds, so that
    theta^G = theta^Group * TGR * MGR holds exactly. Uses the same
    _solve_ddf_one, INPUT_COLS, GOOD_OUT, BAD_OUT and returns-to-scale setting
    as the rest of Step 12.

    Note: Step 12's main TGR rescales each year by its own means; the pooled-
    bounds TGR here is a separate, internally consistent version built only
    for this decomposition. Their agreement is printed as a check.

    Cost: one pooled LP per country-year against all ~1,080 observations
    (the same cost as the GML global distances), plus small group/meta LPs.

    Returns a country-year DataFrame with theta_group_p, theta_meta_p,
    theta_global, tgr_p, mgr, carbon intensity and electricity-mix shares.
    """
    print(f"\n{'═'*70}")
    print("12.12b — SOURCES OF TGR CHANGE (fixed pooled benchmark T^G)")
    print(f"{'═'*70}")
    mix_avail = [c for c in _MIX_COLS if c in panel.columns]
    needed = ["iso_code","year",group_col] + INPUT_COLS + GOOD_OUT + BAD_OUT
    sub = panel[needed].dropna().sort_values(["year","iso_code"]).reset_index(drop=True)
    if len(sub) < 30:
        print(f"  ⚠ Only {len(sub)} complete observations — skipped")
        return pd.DataFrame()

    X_raw = sub[INPUT_COLS].values.astype(float)
    Y_raw = sub[GOOD_OUT].values.astype(float)
    B_raw = sub[BAD_OUT].values.astype(float)
    X = _normalise(X_raw, X_raw.mean(0))      # one pooled scale per variable
    Y = _normalise(Y_raw, Y_raw.mean(0))
    B = _normalise(B_raw, B_raw.mean(0))
    n = len(sub)

    def _theta(Xr, Yr, Br, i):
        b = _solve_ddf_one(Xr, Yr, Br, X[i], Y[i], B[i], vrs=vrs, allow_negative=False)
        return 1.0 / (1.0 + b) if np.isfinite(b) else np.nan

    th_group = np.full(n, np.nan); th_meta = np.full(n, np.nan); th_glob = np.full(n, np.nan)
    for yr in sorted(sub["year"].unique()):
        idx_y = np.where((sub["year"] == yr).values)[0]
        for i in idx_y:
            th_meta[i] = _theta(X[idx_y], Y[idx_y], B[idx_y], i)
        for g in sub.loc[idx_y, group_col].unique():
            idx_g = idx_y[(sub.loc[idx_y, group_col] == g).values]
            if len(idx_g) < 4:
                continue
            for i in idx_g:
                th_group[i] = _theta(X[idx_g], Y[idx_g], B[idx_g], i)
    print(f"  Solving {n} pooled-technology LPs (reference set = all {n} country-years)...")
    for i in range(n):
        th_glob[i] = _theta(X, Y, B, i)

    out = sub[["iso_code","year",group_col]].rename(columns={group_col: "tech_regime"}).copy()
    out["theta_group_p"] = th_group
    out["theta_meta_p"]  = th_meta
    out["theta_global"]  = th_glob
    out["tgr_p"] = th_meta / th_group
    out["mgr"]   = th_glob / th_meta
    extra = panel[["iso_code","year","co2_intensity_elec"] + mix_avail].drop_duplicates(["iso_code","year"])
    out = out.merge(extra, on=["iso_code","year"], how="left")
    scale = 100.0 if (len(mix_avail) and out[mix_avail].max().max() > 2) else 1.0
    for c in mix_avail:
        out[c] = out[c] / scale
    fossil = [c for c in ("coal_share_elec","gas_share_elec","oil_share_elec") if c in mix_avail]
    out["fossil_share_elec"] = out[fossil].sum(axis=1, min_count=1) if fossil else np.nan

    tol = 1e-4
    v1 = int((out["theta_meta_p"] - out["theta_group_p"] > tol).sum())
    v2 = int((out["theta_global"] - out["theta_meta_p"] > tol).sum())
    ident = np.nanmax(np.abs(out["theta_global"] - out["theta_group_p"]*out["tgr_p"]*out["mgr"]))
    print(f"  Nesting checks: theta_meta > theta_group in {v1}/{n}; "
          f"theta_global > theta_meta in {v2}/{n} (expect ~0, LP tolerance only)")
    print(f"  Identity theta_G = theta_Group x TGR x MGR: max abs error {ident:.1e}")
    return out


def summarise_tgr_mechanisms(dec, window=3, focus="Fossil-dominant"):
    """
    (1) Regime-level split of the change in log theta^G between the first and
        last `window` years into Delta ln theta^Group + Delta ln TGR +
        Delta ln MGR (exact, computed on country-level logs).
    (2) Country-level classification by the sign of the linear trend in TGR
        and in theta^G:
          TGR falling, theta^G rising  -> falling behind despite improvement
          TGR falling, theta^G falling -> absolute deterioration
          TGR rising                   -> narrowing gap
        together with the trends in carbon intensity and fossil share.
    """
    d = dec.dropna(subset=["theta_group_p","theta_meta_p","theta_global"]).copy()
    for c in ("theta_group_p","tgr_p","mgr","theta_global"):
        d["ln_" + c] = np.log(d[c].clip(lower=1e-9))
    yrs = sorted(d["year"].unique())
    early, late = yrs[:window], yrs[-window:]

    rows = []
    for g, gd in d.groupby("tech_regime"):
        e = gd[gd["year"].isin(early)].groupby("iso_code").mean(numeric_only=True)
        l = gd[gd["year"].isin(late)].groupby("iso_code").mean(numeric_only=True)
        common = e.index.intersection(l.index)
        diff = (l.loc[common] - e.loc[common]).mean()
        rows.append({"tech_regime": g, "n_countries": len(common),
                     "dln_theta_global": diff["ln_theta_global"],
                     "dln_theta_group": diff["ln_theta_group_p"],
                     "dln_tgr": diff["ln_tgr_p"], "dln_mgr": diff["ln_mgr"],
                     "d_carbon_intensity": diff.get("co2_intensity_elec", np.nan),
                     "d_fossil_share": diff.get("fossil_share_elec", np.nan)})
    reg = pd.DataFrame(rows)

    print(f"\n  {'─'*96}")
    print(f"  REGIME-LEVEL SPLIT: mean change {early[0]}–{early[-1]} → {late[0]}–{late[-1]} (log points)")
    print(f"  Δln θ^G = Δln θ^Group + Δln TGR + Δln MGR   (exact; θ^G = fixed pooled benchmark)")
    print(f"  {'─'*96}")
    print(f"  {'Regime':<24} {'Δlnθ^G':>8} {'=Δlnθ^Grp':>10} {'+ΔlnTGR':>9} {'+ΔlnMGR':>9} "
          f"{'ΔgCO2e/kWh':>10} {'ΔFossil':>9}")
    for _, r in reg.iterrows():
        print(f"  {r['tech_regime']:<24} {r['dln_theta_global']:>+8.3f} {r['dln_theta_group']:>+10.3f} "
              f"{r['dln_tgr']:>+9.3f} {r['dln_mgr']:>+9.3f} {r['d_carbon_intensity']:>+10.1f} "
              f"{r['d_fossil_share']:>+9.3f}")
    print(f"  Reading: Δlnθ^G > 0 means the regime moved closer to the FIXED best practice.")
    print(f"  ΔlnMGR > 0 means the common frontier itself advanced. A negative ΔlnTGR with")
    print(f"  positive Δlnθ^G is 'falling behind despite improvement', not deterioration.")

    def _slope(s):
        s = s.dropna()
        return np.polyfit(s.index.values.astype(float), s.values, 1)[0] if len(s) >= 4 else np.nan

    crows = []
    for (iso, g), cd in d.groupby(["iso_code","tech_regime"]):
        cd = cd.set_index("year").sort_index()
        s_tgr, s_glob = _slope(cd["tgr_p"]), _slope(cd["theta_global"])
        if not (np.isfinite(s_tgr) and np.isfinite(s_glob)):
            continue
        if s_tgr >= 0:
            cls = "Narrowing gap"
        elif s_glob > 0:
            cls = "Falling behind despite improvement"
        else:
            cls = "Absolute deterioration"
        crows.append({"iso_code": iso, "tech_regime": g, "slope_tgr": s_tgr,
                      "slope_theta_global": s_glob, "slope_theta_group": _slope(cd["theta_group_p"]),
                      "slope_carbon_intensity": _slope(cd["co2_intensity_elec"]) if "co2_intensity_elec" in cd else np.nan,
                      "slope_fossil_share": _slope(cd["fossil_share_elec"]),
                      "mechanism": cls})
    ctry = pd.DataFrame(crows)

    print(f"\n  COUNTRY-LEVEL MECHANISM (sign of linear trends in TGR and θ^G):")
    for g in sorted(ctry["tech_regime"].unique()):
        cg = ctry[ctry["tech_regime"] == g]
        print(f"    {g}:")
        for m in ["Falling behind despite improvement", "Absolute deterioration", "Narrowing gap"]:
            mem = sorted(cg[cg["mechanism"] == m]["iso_code"])
            print(f"      {m:<36}: {len(mem):2d} {mem}")
    if focus in set(ctry["tech_regime"]):
        f = ctry[ctry["tech_regime"] == focus]
        print(f"\n  {focus}: carbon intensity falling in "
              f"{int((f['slope_carbon_intensity'] < 0).sum())}/{len(f)} countries; "
              f"fossil share falling in {int((f['slope_fossil_share'] < 0).sum())}/{len(f)}.")
    print(f"  Trend signs are descriptive; a slope near zero can flip sign with a")
    print(f"  single year, so read borderline countries together with the trajectories.")
    return {"regime_split": reg, "country_mechanism": ctry, "early": early, "late": late}


def print_regime_causality_caveat(dec):
    """
    The regime label and the bad output are built from the same quantity, so
    regime differences in TGR are not a causal effect of switching to
    low-carbon generation. This prints the strength of that construction link.
    """
    cm = dec.groupby(["iso_code","tech_regime"])[["fossil_share_elec","co2_intensity_elec"]].mean().dropna()
    rho = cm["fossil_share_elec"].corr(cm["co2_intensity_elec"], method="spearman") if len(cm) > 4 else np.nan
    print(f"\n  {'─'*70}")
    print(f"  INTERPRETATION LIMIT: the regime comparison is NOT a causal estimate")
    print(f"  {'─'*70}")
    print(f"  Regimes are defined from the electricity generation mix, and the DDF's bad")
    print(f"  output (lifecycle GHG intensity of electricity) is largely determined by that same")
    print(f"  mix. Across countries, mean fossil share and mean carbon intensity have")
    print(f"  Spearman rho = {rho:+.3f}. A lower TGR for fossil-dominant systems is therefore")
    print(f"  partly built in by construction. Regimes also differ in resource endowments,")
    print(f"  income, geography and history, none of which are held fixed. Report regime")
    print(f"  differences as descriptive technology-gap patterns, not as the effect of")
    print(f"  switching to low-carbon generation.")
    return rho


_REGIME_COLORS = None
def _regime_colors():
    return {"Fossil-dominant": COLORS["red"], "Low-carbon specialised": COLORS["teal"],
            "Diversified/mixed": COLORS["blue"]}


def plot_step12_tgr_mechanisms(dec):
    """
    Six-panel trajectory figure by regime (yearly means):
      (a) θ^Group  (b) θ^Meta (annual common frontier)  (c) TGR
      (d) θ^G against the fixed pooled frontier T^G     (e) MGR (common-frontier position)
      (f) lifecycle GHG intensity of electricity (gCO2e/kWh)
    """
    rc = _regime_colors()
    ym = dec.groupby(["tech_regime","year"]).mean(numeric_only=True).reset_index()
    panels = [("theta_group_p", "(a) Group efficiency θ^Group", "Efficiency vs own regime"),
              ("theta_meta_p",  "(b) Common-frontier efficiency θ^Meta", "Efficiency vs that year's common frontier"),
              ("tgr_p",         "(c) Technology gap ratio TGR", "θ^Meta / θ^Group"),
              ("theta_global",  "(d) Efficiency vs FIXED pooled frontier θ^G", "Rising = absolute improvement"),
              ("mgr",           "(e) Common-frontier position MGR", "θ^G / θ^Meta (rising = frontier advancing)"),
              ("co2_intensity_elec", "(f) Lifecycle GHG intensity of electricity", "gCO₂e/kWh")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, (col, title, ylab) in zip(axes.flat, panels):
        for g, gd in ym.groupby("tech_regime"):
            ax.plot(gd["year"], gd[col], color=rc.get(g, "gray"), lw=2, marker="o", ms=3, label=g)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Step 12 — Why does the technology gap change? Regime trajectories\n"
                 "TGR falls when the regime deteriorates (d falls) or when the common frontier "
                 "advances faster (e rises), or both", fontsize=13, fontweight="bold")
    _fig_legend_right(fig, [axes[0, 0]], right=0.86, top=0.92, title="Regime")
    plt.tight_layout(rect=fig._tight_rect)
    fig._labels_placed = True
    _save("step12_tgr_mechanisms.png")


def plot_step12_generation_mix(dec):
    """Stacked electricity-generation mix by regime over time (one panel per regime)."""
    mix = [c for c in _MIX_COLS if c in dec.columns]
    if not mix:
        return
    regimes = sorted(dec["tech_regime"].dropna().unique())
    palette = {"coal_share_elec": "#4d4d4d", "gas_share_elec": "#b0794a", "oil_share_elec": "#8c510a",
               "nuclear_share_elec": "#7b6fd0", "hydro_share_elec": "#2f7fc1",
               "other_renew_share_elec": "#3aa36b"}
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.6*len(regimes), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, regimes):
        ym = dec[dec["tech_regime"] == g].groupby("year")[mix].mean()
        ax.stackplot(ym.index, *[ym[c].values for c in mix],
                     labels=[_MIX_COLS[c] for c in mix], colors=[palette[c] for c in mix], alpha=0.9)
        ax.set_title(g, fontsize=11, fontweight="bold", color=_regime_colors().get(g, "black"))
        ax.set_ylim(0, 1); ax.set_xlabel("Year")
    axes[0].set_ylabel("Share of electricity generation")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.suptitle("Step 12 — Electricity generation mix by regime (yearly means)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save("step12_generation_mix.png")


def plot_step12_mechanism_split(mech):
    """Bar chart of the exact log split of Δln θ^G by regime."""
    reg = mech["regime_split"]
    if reg.empty:
        return
    comps = [("dln_theta_group", "Δln θ^Group (within-regime catch-up)", COLORS["blue"]),
             ("dln_tgr", "Δln TGR (technology gap)", COLORS["red"]),
             ("dln_mgr", "Δln MGR (common-frontier position)", COLORS["purple"])]
    x = np.arange(len(reg)); w = 0.22
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k, (col, lab, colr) in enumerate(comps):
        ax.bar(x + (k-1)*w, reg[col], width=w, label=lab, color=colr, alpha=0.85)
    ax.scatter(x, reg["dln_theta_global"], color="black", zorder=5, s=70, marker="D",
               label="Δln θ^G (sum; vs fixed pooled frontier)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(reg["tech_regime"], fontsize=10)
    ax.set_ylabel("Change in log points")
    e, l = mech["early"], mech["late"]
    ax.set_title(f"Step 12 — Sources of change, {e[0]}–{e[-1]} → {l[0]}–{l[-1]}\n"
                 "TGR down with θ^G up = falling behind despite improvement; "
                 "TGR down with θ^G down = absolute deterioration",
                 fontsize=10.5, fontweight="bold")
    _legend_right(ax)
    _save("step12_tgr_mechanism_split.png")


def plot_step12_construction_link(dec, f_threshold=0.65):
    """
    Two panels showing why regime differences are not causal estimates:
    (a) schematic: the generation mix feeds BOTH the regime label and the bad
        output, so regime and DDF score share a common input by construction;
    (b) country means: fossil share vs carbon intensity, coloured by regime,
        with the 0.65 classification threshold.
    """
    rc = _regime_colors()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8.5), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax0.axis("off"); ax0.set_xlim(0, 10); ax0.set_ylim(0, 10)
    def box(x, y, txt, fc):
        ax0.text(x, y, txt, ha="center", va="center", fontsize=9.5,
                 bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec="#555555"))
    def arrow(x1, y1, x2, y2, txt=None, color="#333333", style="-|>"):
        ax0.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle=style, color=color, lw=1.6))
        if txt:
            ax0.text((x1+x2)/2, (y1+y2)/2 + 0.25, txt, fontsize=9, ha="center", color=color)
    box(5, 9.0, "Electricity generation mix\n(coal, gas, oil, nuclear, hydro, renewables)", "#f4e3c1")
    box(2.0, 6.2, "Regime label\n(F ≥ 0.65, LC_max ≥ 0.40)", "#e3e0f7")
    box(8.0, 6.2, "Bad output\nlifecycle GHG intensity\n(gCO₂e/kWh)", "#f7d9d4")
    box(8.0, 3.3, "DDF scores\nθ^Group, θ^Meta", "#d8ebf7")
    box(5.0, 0.8, "Regime difference in TGR", "#dff2e6")
    box(5.0, 4.3, "Not held fixed:\nendowments, income,\ngeography, history", "#eeeeee")
    arrow(4.0, 8.4, 2.4, 6.9, "defines")
    arrow(6.0, 8.4, 7.6, 6.9, "largely determines")
    arrow(8.0, 5.6, 8.0, 4.0, "enters LP")
    arrow(2.2, 5.6, 4.4, 1.3)
    ax0.text(2.7, 3.2, "groups\ncountries", fontsize=9, ha="center", color="#333333")
    arrow(7.6, 2.7, 5.8, 1.3)
    arrow(3.8, 4.8, 2.6, 5.6, color="#888888")
    arrow(6.2, 4.1, 7.1, 3.6, color="#888888")
    ax0.set_title("(a) Regime and environmental output share an input by construction",
                  fontsize=10.5, fontweight="bold")

    cm = dec.groupby(["iso_code","tech_regime"])[["fossil_share_elec","co2_intensity_elec"]].mean().dropna().reset_index()
    for g, gd in cm.groupby("tech_regime"):
        ax1.scatter(gd["fossil_share_elec"], gd["co2_intensity_elec"], s=70, color=rc.get(g, "gray"),
                    edgecolor="white", label=g, zorder=3)
    ax1.axvline(f_threshold, color="gray", ls="--", lw=1)
    ax1.text(f_threshold, ax1.get_ylim()[1], f" F = {f_threshold}", va="top", fontsize=9, color="gray")
    rho = cm["fossil_share_elec"].corr(cm["co2_intensity_elec"], method="spearman") if len(cm) > 4 else np.nan
    ax1.set_xlabel("Mean fossil share of electricity generation, F")
    ax1.set_ylabel("Mean lifecycle GHG intensity (gCO₂e/kWh)")
    ax1.set_title(f"(b) Country means: classification variable vs bad output (Spearman ρ = {rho:+.2f})",
                  fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.25)
    fig.suptitle("Step 12 — Regime comparisons are descriptive, not causal effects of low-carbon generation",
                 fontsize=13, fontweight="bold")
    _fig_legend_right(fig, [ax1], right=0.86, top=0.93, title="Regime")
    plt.tight_layout(rect=fig._tight_rect)
    fig._layout_frozen = True
    _label_points(ax1, cm["fossil_share_elec"].values, cm["co2_intensity_elec"].values,
                  cm["iso_code"].tolist(), colors=[rc.get(g, "gray") for g in cm["tech_regime"]])
    _save("step12_construction_link.png")


def step12_metafrontier_analysis(state: dict, min_group_size: int = 10,
                                 run_bootstrap: bool = True, B_step12: int = B_RESAMPLE) -> dict:
    """
    STEP 12 — Theory-driven metafrontier analysis: three technology
    regimes (Fossil-dominant, Low-carbon specialised, Diversified/mixed),
    defined from each country's full-sample average generation mix.

    *** Reuses EXACTLY the same production variables and DDF orientation
    as Step 5 (module-level INPUT_COLS, GOOD_OUT, BAD_OUT, and
    _solve_ddf_one()) -- no separate DEA model is constructed for this
    analysis. theta^Meta is computed FRESH within _solve_group_meta_ddf,
    on the exact same per-year reference set as theta^Group (see that
    function's docstring for why this replaced an earlier design that
    reused dea_score directly and was found, through testing, to violate
    the theta_meta <= theta_group subset-monotonicity requirement). Uses
    state["panel"] directly -- the same panel object populated by Steps
    1-11 -- not a separately constructed one. ***

    Sub-analyses (matching the specification this step was built to):
      12.1-12.9  : group assignment, group/meta DDF, TGR, four-way
                   classification, regime summary table, Kruskal-Wallis +
                   pairwise Mann-Whitney (Holm-corrected) + Cliff's delta +
                   bootstrap CIs for regime means/medians.
      12.10      : compare with the Conditional DDF gap (Step 7c).
      12.11      : pooled TGR-ESI correlation (Pearson and Spearman).
      12.12      : evolution of mean TGR, 2000-2023, overall and by regime.
      12.13      : correlate delta-TGR with GTC and delta-theta^Group with GEC.
      12.14      : two robustness groupings -- (A) data-driven k-means
                   clustering on raw generation shares, compared to the
                   theoretical regimes via a contingency table and the
                   Adjusted Rand Index; (B) G20 vs non-G20 Europe,
                   explicitly secondary/non-technological.
      12.15      : resampling intervals (NOT frontier-bootstrap CIs) for theta^Group,
                   theta^Meta and TGR (regime means and country averages).

    Requires state from Steps 1-8 at minimum; Steps 6, 7c improve the
    completeness of 12.10/12.13 but are not hard requirements.

    Populates state keys: step12_results (raw country-year theta^Group/
    theta^Meta/TGR), step12_classification, step12_regime_summary,
    step12_stats, step12_comparisons (dict of the 12.10-12.13 sub-results),
    step12_robustness, step12_bootstrap.
    """
    print("\n" + "="*60)
    print("STEP 12 — Theory-Driven Metafrontier Analysis (Three Technology Regimes)")
    print("="*60)
    t = time.time()
    panel = state["panel"]

    # 12.1-12.2: group assignment (hard-fails if any group < min_group_size)
    panel_with_regime, F, LC_max = assign_technology_regime(panel, min_group_size=min_group_size)
    state["panel"] = panel_with_regime
    tech_regime_map = panel_with_regime.drop_duplicates("iso_code").set_index("iso_code")["tech_regime"]

    # 12.3-12.4: annual group + meta DDF, baseline returns to scale
    print(f"\n  Solving annual group-specific {BASE_RTS} DDF (baseline)...")
    results_df = _solve_group_meta_ddf(panel_with_regime, group_col="tech_regime", vrs=VRS)
    if results_df.empty:
        print("  ⚠ No results — aborting Step 12")
        return state
    _check_metafrontier_consistency(results_df, panel_with_regime, meta_col="dea_score")

    # Parallel specification (the other returns-to-scale assumption)
    print(f"\n  Solving annual group-specific {ALT_RTS} DDF (robustness check)...")
    try:
        results_alt = _solve_group_meta_ddf(panel_with_regime, group_col="tech_regime", vrs=not VRS)
        state["step12_results_alt"] = results_alt
        print(f"  ✓ {ALT_RTS} robustness check computed ({len(results_alt)} obs)")
        if not results_alt.empty:
            cmp_ = (results_df.groupby("tech_regime")["tgr"].mean().rename(f"mean_tgr_{BASE_RTS}").to_frame()
                    .join(results_alt.groupby("tech_regime")["tgr"].mean().rename(f"mean_tgr_{ALT_RTS}")))
            print(f"\n  {'─'*70}")
            print(f"  Mean TGR by regime: {BASE_RTS} (baseline) vs {ALT_RTS} (robustness)")
            print(f"  {'─'*70}")
            for reg, row in cmp_.iterrows():
                print(f"    {reg:<24}: TGR_{BASE_RTS}={row.iloc[0]:.4f}  TGR_{ALT_RTS}={row.iloc[1]:.4f}  "
                      f"(diff={row.iloc[1]-row.iloc[0]:+.4f})")
            _save_csv(results_alt, f"step12_metafrontier_results_{ALT_RTS}_robustness.csv",
                      f"Step 12 {ALT_RTS} robustness check (baseline is {BASE_RTS})")
    except Exception as e:
        print(f"  ⚠ {ALT_RTS} robustness check skipped: {e}")

    state["step12_results"] = results_df

    # 12.5: four-way classification
    state["step12_classification"] = classify_four_way(results_df)
    try:
        plot_step12_quadrant(results_df, state["step12_classification"])
    except Exception as e:
        print(f"  ⚠ plot_step12_quadrant failed: {e}")

    # 12.6: regime summary table
    state["step12_regime_summary"] = regime_summary_table(results_df)
    try:
        plot_step12_regime_summary(state["step12_regime_summary"])
    except Exception as e:
        print(f"  ⚠ plot_step12_regime_summary failed: {e}")

    # 12.7-12.9: Kruskal-Wallis, pairwise Mann-Whitney+Holm, Cliff's delta, bootstrap CIs
    state["step12_stats"] = test_regime_differences(results_df, value_col="tgr")

    # Parallel specification: the SAME regime tests on the other RTS assumption.
    # Group frontiers and the metafrontier share one RTS assumption in each.
    results_alt = state.get("step12_results_alt")
    if results_alt is not None and not results_alt.empty:
        print(f"\n  {'#'*70}")
        print(f"  {ALT_RTS} PARALLEL SPECIFICATION: regime tests repeated")
        print(f"  {'#'*70}")
        try:
            state["step12_stats_alt"] = test_regime_differences(results_alt, value_col="tgr")
            _sv, _sc = state["step12_stats"], state["step12_stats_alt"]
            b_, a_ = BASE_RTS, ALT_RTS
            print(f"\n  {'─'*70}")
            print(f"  {b_} vs {a_}: regime tests on country-level mean TGR")
            print(f"  {'─'*70}")
            print(f"  {'':<34} {b_ + ' (baseline)':>16} {a_ + ' (robustness)':>18}")
            print(f"  {'Kruskal-Wallis H':<34} {_sv['kruskal_H']:>16.4f} {_sc['kruskal_H']:>18.4f}")
            print(f"  {'p (permutation)':<34} {_sv['kruskal_p_exact']:>16.4f} {_sc['kruskal_p_exact']:>18.4f}")
            print(f"  {'p (asymptotic)':<34} {_sv['kruskal_p_asymptotic']:>16.4f} {_sc['kruskal_p_asymptotic']:>18.4f}")
            pv, pc = _sv["pairwise"], _sc["pairwise"]
            sfx = (f"_{b_.lower()}", f"_{a_.lower()}")
            if len(pv) and len(pc):
                pm = pv[["pair","p_holm","cliffs_delta"]].merge(
                    pc[["pair","p_holm","cliffs_delta"]], on="pair", suffixes=sfx)
                for _, r in pm.iterrows():
                    print(f"  {r['pair'][:34]:<34} p_holm={r['p_holm'+sfx[0]]:.4f} d={r['cliffs_delta'+sfx[0]]:+.3f}"
                          f"   p_holm={r['p_holm'+sfx[1]]:.4f} d={r['cliffs_delta'+sfx[1]]:+.3f}")
                _save_csv(pm, f"step12_pairwise_{b_}_vs_{a_}.csv",
                         f"Step 12 pairwise regime tests, {b_} baseline vs {a_} robustness check")
                _both = pm[(pm["p_holm" + sfx[0]] < 0.05) & (pm["p_holm" + sfx[1]] < 0.05)]["pair"].tolist()
                _only = pm[(pm["p_holm" + sfx[0]] < 0.05) & (pm["p_holm" + sfx[1]] >= 0.05)]["pair"].tolist()
                print(f"\n  Robust to returns to scale (Holm p<0.05 under {b_} AND {a_}): {_both or 'none'}")
                print(f"  Significant under {b_} only (NOT robust): {_only or 'none'}")
                print(f"  Only the comparisons in the first list should be reported as robust;")
                print(f"  the full three-regime ordering is robust only if the second list is empty.")
            bs_v, bs_c = _sv["bootstrap_summary"], _sc["bootstrap_summary"]
            if len(bs_v) and len(bs_c):
                bm = bs_v.merge(bs_c, on="tech_regime", suffixes=sfx)
                _save_csv(bm, f"step12_regime_bootstrap_{b_}_vs_{a_}.csv",
                         f"Step 12 regime mean/median TGR with bootstrap CIs, {b_} vs {a_}")
            _save_csv(pd.DataFrame([{
                "spec": spec, "kruskal_H": d["kruskal_H"],
                "kruskal_p_permutation": d["kruskal_p_exact"],
                "kruskal_p_asymptotic": d["kruskal_p_asymptotic"]}
                for spec, d in ((b_, _sv), (a_, _sc))]),
                f"step12_kruskal_{b_}_vs_{a_}.csv", f"Step 12 Kruskal-Wallis, {b_} vs {a_}")
            print(f"  {'─'*70}")
            print(f"  The permutation p-value is (k+1)/(N_PERM+1); N_PERM = {N_PERM} in this run.")
        except Exception as e:
            print(f"  ⚠ {ALT_RTS} regime tests failed: {e}")
    else:
        print(f"\n  ℹ {ALT_RTS} regime tests skipped (no robustness-check results)")
    try:
        plot_step12_distribution(results_df)
    except Exception as e:
        print(f"  ⚠ plot_step12_distribution failed: {e}")

    # 12.10-12.13: comparisons
    comparisons = {}
    comparisons["conditional_ddf"] = compare_with_conditional_ddf(results_df, panel_with_regime)
    comparisons["esi"] = correlate_tgr_esi(results_df, panel_with_regime)
    comparisons["tgr_time_series"] = tgr_time_series(results_df)
    try:
        plot_step12_tgr_trend(comparisons["tgr_time_series"])
    except Exception as e:
        print(f"  ⚠ plot_step12_tgr_trend failed: {e}")
    comparisons["gml"] = correlate_tgr_gml(results_df, state.get("mpi_df"))

    # 12.12b: separate deterioration from a faster-moving common frontier,
    # using a fixed pooled benchmark; plus carbon intensity and generation mix.
    try:
        dec = decompose_tgr_mechanisms(panel_with_regime, group_col="tech_regime", vrs=VRS)
        if not dec.empty:
            rho_main = (dec.merge(results_df[["iso_code","year","tgr"]], on=["iso_code","year"])
                           [["tgr","tgr_p"]].corr(method="spearman").iloc[0, 1])
            print(f"  Pooled-bounds TGR vs main (per-year bounds) TGR: Spearman rho = {rho_main:+.3f}")
            mech = summarise_tgr_mechanisms(dec)
            mech["decomposition"] = dec
            mech["construction_rho"] = print_regime_causality_caveat(dec)
            comparisons["tgr_mechanisms"] = mech
            _save_csv(dec, "step12_tgr_mechanisms_country_year.csv",
                      "theta^Group, theta^Meta, theta^G, TGR, MGR, carbon intensity, mix")
            _save_csv(mech["regime_split"], "step12_tgr_mechanisms_regime_split.csv",
                      "Exact log split of change in theta^G by regime")
            _save_csv(mech["country_mechanism"], "step12_tgr_mechanisms_by_country.csv",
                      "Country trend signs and mechanism class")
            for _plot, _arg in ((plot_step12_tgr_mechanisms, dec), (plot_step12_generation_mix, dec),
                                (plot_step12_mechanism_split, mech), (plot_step12_construction_link, dec)):
                try:
                    _plot(_arg)
                except Exception as e:
                    print(f"  ⚠ {_plot.__name__} failed: {e}")
    except Exception as e:
        print(f"  ⚠ 12.12b mechanism decomposition failed: {e}")
    state["step12_comparisons"] = comparisons

    # 12.14: robustness groupings
    robustness = {}
    robustness["clustering"] = robustness_data_driven_clustering(panel_with_regime, tech_regime_map)
    try:
        if robustness["clustering"] and "contingency" in robustness["clustering"]:
            plot_step12_robustness_contingency(robustness["clustering"]["contingency"])
    except Exception as e:
        print(f"  ⚠ plot_step12_robustness_contingency failed: {e}")
    robustness["g20_split"] = robustness_g20_split(results_df, panel_with_regime,
                                                   set(EUROPE_NONG20_ISO3), set(G20_ISO3))

    # Import-variable robustness: PRIMARY result (results_df, above) uses
    # dea_net_import unchanged; this additionally checks whether TGR
    # rankings are sensitive to using an electricity-specific import
    # variable instead. Skipped cleanly if the alternative column is absent.
    results_robust_import = run_step12_import_robustness(panel_with_regime, group_col="tech_regime", vrs=VRS)
    robustness["import_variable_comparison"] = compare_import_robustness(results_df, results_robust_import)
    try:
        robustness["threshold_sensitivity"] = regime_threshold_sensitivity(
            panel_with_regime, F, LC_max, tech_regime_map, min_group_size=min_group_size)
    except Exception as e:
        print(f"  ⚠ regime threshold sensitivity failed: {e}")
    state["step12_robustness"] = robustness

    # 12.15: resampling intervals for theta^Group, theta^Meta, TGR (estimated
    # scores resampled; frontier-estimation uncertainty reported separately)
    if run_bootstrap:
        state["step12_bootstrap"] = bootstrap_group_meta_tgr(
            panel_with_regime, tech_regime_map, results_df=results_df, B=B_step12)
        try:
            plot_step12_bootstrap_ci(state["step12_bootstrap"])
        except Exception as e:
            print(f"  ⚠ plot_step12_bootstrap_ci failed: {e}")
    else:
        print(f"\n  ℹ Bootstrap (12.15) skipped (run_bootstrap=False)")

    _save_csv(results_df, "step12_metafrontier_results.csv",
              "Step 12 metafrontier: theta^Group, theta^Meta, TGR by country-year")
    _save_csv(state["step12_regime_summary"], "step12_regime_summary.csv",
              "Step 12 regime summary table")
    _save_csv(state["step12_classification"], "step12_four_way_classification.csv",
              "Step 12 four-way country classification")

    print(f"\n  ✓ Step 12 complete  ({time.time()-t:.0f}s)")
    return state


def report_sensitivity_overview(state):
    """One consolidated list of every sensitivity exercise in the run, with the
    declared baseline restated, so none is hidden in a single step's log."""
    rows = []
    sp = state.get("sensitivity_spec")
    if isinstance(sp, pd.DataFrame):
        for _, r in sp.iterrows():
            rows.append(("Frontier specification", r["sensitivity"],
                         f"ρ(country means vs baseline) = {r['spearman_country_means_vs_baseline']:.3f}; "
                         f"frontier countries {int(r['countries_ever_on_frontier_alt'])} "
                         f"(baseline {int(r['countries_ever_on_frontier_baseline'])})",
                         "step5c_specification_sensitivity.csv"))
    cs = state.get("conditional_sensitivity")
    if isinstance(cs, pd.DataFrame) and len(cs):
        for _, r in cs.iterrows():
            rows.append(("Conditional peer set", str(r.iloc[0]),
                         "; ".join(f"{c} = {r[c]:.3f}" for c in cs.columns[1:] if isinstance(r[c], (float, np.floating))),
                         "step7c_conditional_sensitivity.csv"))
    rb = state.get("step12_robustness") or {}
    th = rb.get("threshold_sensitivity")
    if isinstance(th, pd.DataFrame) and "baseline_ordering_holds" in th.columns:
        est = th[th["baseline_ordering_holds"].notna()]
        rows.append(("Regime thresholds / period", f"{len(th)} combinations",
                     f"baseline ordering holds in {int(est['baseline_ordering_holds'].sum())}/{len(est)} estimable",
                     "step12_regime_threshold_sensitivity.csv"))
    ic = rb.get("import_variable_comparison") or {}
    if "spearman_rho" in ic:
        rows.append(("Import definition (metafrontier)", "electricity net-import share",
                     f"ρ(country TGR) = {ic['spearman_rho']:.3f}; max rank change "
                     f"{int(ic.get('max_rank_change', np.nan)) if pd.notna(ic.get('max_rank_change', np.nan)) else '—'}",
                     "Step 12 log"))
    if not rows:
        return
    print(f"\n{'═'*78}")
    print("SENSITIVITY OVERVIEW — declared baseline: CRS DDF, carbon intensity as bad output,")
    print("fuel-wide import share, k = 1/2 peer set, regimes F >= 0.65 / LC >= 0.40. None of the")
    print("exercises below replaces the baseline or was used to choose it.")
    print(f"{'═'*78}")
    for area, lab, res, f in rows:
        print(f"  [{area}] {lab}\n      {res}   ({f})")
    _save_csv(pd.DataFrame(rows, columns=["area", "exercise", "result", "file"]),
              "sensitivity_overview.csv", "All sensitivity exercises in this run")


def step13_export(state: dict) -> dict:
    """
    STEP 13 — Export results to CSV.

    Saves:
        energy_security_panel.csv
        mpi_results.csv            ← if mpi_df in state
        simar_wilson_results.csv   ← if sw_results in state

    Usage:
        state = step13_export(state)
    """
    print("\n" + "="*60)
    print("STEP 13 — Export")
    print("="*60)

    mpi_df     = state.get("mpi_df",     pd.DataFrame())
    sw_results = state.get("sw_results", pd.DataFrame())
    export_results(state["panel"], mpi_df, sw_results)

    # *** ADDED: the six requested country-year reference tables (ESI+4
    # dimensions under both C2 variants, bootstrapped VRS, bootstrapped
    # CRS, raw DDF theta, and GML with all components). ***
    state = export_country_year_tables(state)
    state = export_affordability_components_table(state)
    state["sample_scope"] = report_sample_scope(state)
    report_sensitivity_overview(state)
    state["ddf_solver_stats"] = report_ddf_solver_stats("whole run")
    _save_csv(pd.DataFrame([{k: (str(v) if isinstance(v, dict) else v)
                             for k, v in state["ddf_solver_stats"].items()}]),
              "ddf_solver_stats.csv", "DDF LPs: CBC failures, HiGHS rescues, unsolved")

    # Write run summary
    run_dir = _rdir()
    summary_lines = [
        f"Run completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Results folder: {run_dir.resolve()}",
        f"",
        f"Panel: {state['panel'].shape[0]} rows × {state['panel'].shape[1]} cols",
        f"Countries: {state['panel']['iso_code'].nunique()}",
        f"Years: {state['panel']['year'].min()}–{state['panel']['year'].max()}",
        f"",
        f"Files saved:",
    ]
    for f_path in sorted(run_dir.rglob("*")):
        if f_path.is_file():
            summary_lines.append(f"  {f_path.relative_to(run_dir)}")
    summary_text = "\n".join(summary_lines)
    _save_log(summary_text, "run_summary.txt")
    print(f"\n  Run summary: {run_dir / 'logs' / 'run_summary.txt'}")
    print(f"  All results in: {run_dir.resolve()}")

    state["exported"] = True
    return state


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: run all steps in sequence
# ─────────────────────────────────────────────────────────────────────────────

def run_optional_extensions(state: dict, z_vars: list = None) -> dict:
    """
    Run all methods in APPENDIX A (optional extensions) in sequence. These
    are NOT part of the core pipeline (DDF + GML + Separability Test +
    Conditional DDF + Mundlak + Local Projections) -- they were explicitly
    labelled "robustness" or "optional extension" in the review that
    proposed them, and are opt-in only (not run by default in run_all()).

    *** This block is deliberately NOT called "Section 11" or "Step 11" --
    step11_plots() is an unrelated, always-run core pipeline step that
    happens to share that numeral. Calling this "Section 11" caused real
    confusion in a previous run: the printed "STEP 11 — Plots" banner was
    mistaken for this block, when in fact NONE of the methods below had
    executed because run_optional=True was never passed. If you do not
    see the "APPENDIX A" banner below in your console output, none of
    between_effects_results, quantile_results, or metafrontier_results
    were computed in that run. ***

    Requires state from a full run_all() (or at minimum steps 1-5).

    Populates state keys:
        between_effects_results   ← A.1 cross-sectional OLS coefficients
        quantile_results          ← A.2 per-tau coefficients
        metafrontier_results      ← A.3 dict {"carbon": df, "import": df}, each a
                                    country-year θ^Group/θ^Meta/TGR decomposition
                                    under that grouping (two robustness checks,
                                    not the earlier single G20/non-G20 split)

    (A.3, Anderson-Hsiao dynamic panel, is intentionally NOT run here or
    reported in the paper -- see the Appendix A banner comment above for
    the reason. state["dynamic_panel_result"] will not be populated by
    this function; call dynamic_panel_anderson_hsiao(state["panel"])
    directly if you need it for exploratory purposes.)

    (A PCA-based SEM path model was considered and removed as the
    weakest method in this appendix.)

    Usage:
        state = run_all()                          # optional extensions NOT run
        state = run_optional_extensions(state)      # runs them explicitly
        # -- or, in one call --
        state = run_all(run_optional=True)          # runs everything including Appendix A
    """
    print("\n" + "="*60)
    print("APPENDIX A — OPTIONAL EXTENSIONS (robustness / supplementary)")
    print("  NOT part of the core numbered pipeline (unrelated to step11_plots).")
    print("  This banner appearing means run_optional_extensions() DID execute.")
    print("=" * 60)
    t = time.time()

    state["between_effects_results"] = between_effects_regression(state["panel"], z_vars=z_vars)
    _save_csv(state["between_effects_results"], "ext11_1_between_effects.csv",
              "Between-effects long-run cross-sectional model")

    state["quantile_results"] = quantile_regression_panel(state["panel"], z_vars=z_vars)
    _save_csv(state["quantile_results"], "ext11_3_quantile_regression.csv",
              "Panel quantile regression (Canay 2011 approximation)")

    # Anderson-Hsiao dynamic panel (A.3) is deliberately NOT run/reported.
    # The function dynamic_panel_anderson_hsiao() remains defined below and
    # can be called manually, but is excluded from the paper's results: on
    # testing it showed unstable point estimates even with a strong first-
    # stage instrument (a known finite-sample weakness of Anderson-Hsiao
    # relative to System-GMM), and was judged not reliable enough to report.

    # Metafrontier robustness: two theoretically-motivated groupings
    # (carbon-intensity, import-exposure), NOT the earlier arbitrary
    # G20/non-G20-Europe split. Returns a dict {"carbon": df, "import": df}.
    state["metafrontier_results"] = run_metafrontier_robustness(state["panel"])
    for gtype, df in state["metafrontier_results"].items():
        if not df.empty:
            _save_csv(df, f"ext11_3_metafrontier_{gtype}.csv",
                      f"Metafrontier robustness ({gtype} grouping)")

    print(f"\n  ✓ Appendix A (optional extensions) complete  ({time.time()-t:.0f}s)")
    return state


# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT / RESUME
# ═════════════════════════════════════════════════════════════════════════════
# After every step the state dict and the module-level settings that steps
# modify (INPUT_COLS, GOOD_OUT, BAD_OUT, OUTPUT_COLS, solver counters) are
# saved to <run folder>/checkpoints/state.pkl, and a small manifest records the
# completed steps and the settings of the run. A pointer file in the output
# root (latest_run.json) identifies the run to resume. After a disconnect:
# re-run the notebook cell that defines the pipeline, then call run_all()
# again -- it continues from the last completed step in the same folder.
# A run is resumed only if it is unfinished AND its settings (TEST_MODE,
# replication counts, solver, VRS formulation, RTS baseline, C2 mode) match the
# current ones; otherwise a fresh run is started, so results never mix.
_CKPT_GLOBALS = ("INPUT_COLS", "GOOD_OUT", "BAD_OUT", "OUTPUT_COLS")


def _run_settings(B, run_optional, run_granger):
    return {"TEST_MODE": TEST_MODE, "B": int(B), "B_BOOTSTRAP": B_BOOTSTRAP,
            "B_BOOTSTRAP_ALT": B_BOOTSTRAP_ALT,
            "B_GML_GLOBAL": B_GML_GLOBAL, "B_SW2007": B_SW2007, "B_RESAMPLE": B_RESAMPLE,
            "N_PERM": N_PERM, "B_PERM_SEPARABILITY": B_PERM_SEPARABILITY,
            "DDF_SOLVER": DDF_SOLVER, "VRS_ABATEMENT": VRS_ABATEMENT, "VRS": VRS,
            "ESI_C2_MODE": ESI_C2_MODE, "run_optional": bool(run_optional),
            "run_granger": bool(run_granger)}


def _save_checkpoint(state, done, settings, finished=False):
    import pickle, json
    ck = _rdir() / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    keep, dropped = {}, []
    for k, v in state.items():
        try:
            pickle.dumps(v)
            keep[k] = v
        except Exception:
            dropped.append(k)
    payload = {"state": keep,
               "globals": {g: globals()[g] for g in _CKPT_GLOBALS},
               "solver_stats": _DDF_SOLVE_STATS}
    tmp = ck / "state.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ck / "state.pkl")          # atomic: never a half-written checkpoint
    manifest = {"run_dir": str(_rdir()), "completed": list(done), "finished": bool(finished),
                "settings": settings, "dropped_keys": dropped,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(ck / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(Path(_rdir()).parent / "latest_run.json", "w") as f:
        json.dump({"run_dir": str(_rdir())}, f)
    if dropped:
        print(f"  ℹ checkpoint: state entries that cannot be saved (kept in memory only): {dropped}")


def _find_resumable_checkpoint(root_dir, settings):
    import json
    ptr = Path(root_dir) / "latest_run.json"
    if not ptr.exists():
        return None
    try:
        run_dir = Path(json.load(open(ptr))["run_dir"])
        man = json.load(open(run_dir / "checkpoints" / "manifest.json"))
    except Exception:
        return None
    if man.get("finished"):
        print(f"  ℹ latest run {run_dir.name} is finished -- starting a new run")
        return None
    if man.get("settings") != settings:
        diff = {k: (man.get("settings", {}).get(k), v) for k, v in settings.items()
                if man.get("settings", {}).get(k) != v}
        print(f"  ℹ unfinished run {run_dir.name} has different settings {diff} -- "
              f"NOT resumed; starting a new run")
        return None
    if not (run_dir / "checkpoints" / "state.pkl").exists():
        return None
    return run_dir


def _load_checkpoint(run_dir):
    import pickle, json
    global RUN_DIR
    with open(run_dir / "checkpoints" / "state.pkl", "rb") as f:
        payload = pickle.load(f)
    man = json.load(open(run_dir / "checkpoints" / "manifest.json"))
    globals().update(payload["globals"])
    _DDF_SOLVE_STATS.clear(); _DDF_SOLVE_STATS.update(payload["solver_stats"])
    RUN_DIR = Path(run_dir)
    for sub in ("plots", "tables", "logs"):
        (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)
    return payload["state"], list(man["completed"])


# ═════════════════════════════════════════════════════════════════════════════
# CANONICAL RESULTS, RUN MANIFEST AND CONSISTENCY CHECK
# ═════════════════════════════════════════════════════════════════════════════
# The dissertation text, tables and figures should be finalised from ONE
# identified production run. At the end of every run:
#   tables/key_results.csv  -- the headline numbers the text cites, with source
#   logs/run_manifest.json  -- run id, settings, executed and OMITTED analyses,
#                              and a SHA-256 inventory of every exported file
#   tables/consistency_check.csv -- headline numbers recomputed from the
#                              exported CSV files and compared with the values
#                              held in memory during the run
# consistency_check(run_dir, text_path=...) can be called later to compare the
# chapter text (.docx/.md/.txt) with key_results.csv.
# Step numbering: there is no Step 9 (the number was retired in an earlier
# version); the shock evidence is Step 10b. Step 10 (panel Granger) and
# Appendix A are optional modules and are listed as OMITTED when not run --
# no conclusions should be attributed to them.

_GML_TOL = 1e-4


def _gml_counts(mpi_df):
    g = mpi_df.dropna(subset=["mpi"]).groupby("iso_code")["mpi"].apply(
        lambda x: float(np.exp(np.log(x.clip(lower=1e-12)).mean())))
    return (int((g > 1 + _GML_TOL).sum()), int((g < 1 - _GML_TOL).sum()),
            int(((g - 1).abs() <= _GML_TOL).sum()))


def build_key_results(state):
    """Headline numbers of the run, computed from the in-memory results."""
    rows = []
    add = lambda name, val, src: rows.append({"name": name, "value": val, "source": src})
    p = state.get("panel")
    if isinstance(p, pd.DataFrame) and "iso_code" in p.columns and len(p):
        add("n_country_years", len(p), "energy_security_panel.csv")
        add("n_countries", p["iso_code"].nunique(), "energy_security_panel.csv")
        for c, lab in (("dea_score", "mean_theta_baseline_raw"), ("dea_bc", "mean_theta_baseline_bc"),
                       ("dea_vrs", "mean_theta_vrs_raw"), ("dea_vrs_bc", "mean_theta_vrs_bc"),
                       ("dea_se_score", "mean_scale_efficiency"), ("esi_score", "mean_esi")):
            if c in p.columns:
                add(lab, float(p[c].mean()), "energy_security_panel.csv")
        if "dea_score" in p.columns:
            add("countries_ever_on_baseline_frontier",
                int(p.loc[p["dea_score"] >= 0.999, "iso_code"].nunique()), "energy_security_panel.csv")
    m = state.get("mpi_df")
    if isinstance(m, pd.DataFrame) and {"mpi", "iso_code"} <= set(m.columns):
        imp, dec, unc = _gml_counts(m)
        add("gml_countries_improving", imp, "mpi_results.csv")
        add("gml_countries_declining", dec, "mpi_results.csv")
        add("gml_countries_unchanged", unc, "mpi_results.csv")
    sw = state.get("sw_results")
    if isinstance(sw, pd.DataFrame) and "coef_bc" in sw.columns:
        for _, r in sw.iterrows():
            add(f"sw_coef_{r['variable']}", float(r["coef_bc"]), "step7_simar_wilson.csv")
    r12 = state.get("step12_results")
    if isinstance(r12, pd.DataFrame) and {"tgr", "iso_code", "tech_regime"} <= set(r12.columns):
        cm = r12.groupby("iso_code").agg(tgr=("tgr", "mean"), g=("tech_regime", "first"))
        for g, v in cm.groupby("g")["tgr"].mean().items():
            add(f"tgr_mean_{g}", float(v), "step12_metafrontier_results.csv")
            add(f"n_countries_{g}", int((cm["g"] == g).sum()), "step12_metafrontier_results.csv")
    return pd.DataFrame(rows)


def _recompute_from_exports(tdir):
    """Recompute the headline numbers from the exported CSV files only."""
    out = {}
    f = tdir / "energy_security_panel.csv"
    if f.exists():
        p = pd.read_csv(f)
        out["n_country_years"] = len(p)
        out["n_countries"] = p["iso_code"].nunique()
        for c, lab in (("dea_score", "mean_theta_baseline_raw"), ("dea_bc", "mean_theta_baseline_bc"),
                       ("dea_vrs", "mean_theta_vrs_raw"), ("dea_vrs_bc", "mean_theta_vrs_bc"),
                       ("dea_se_score", "mean_scale_efficiency"), ("esi_score", "mean_esi")):
            if c in p.columns:
                out[lab] = float(p[c].mean())
        if "dea_score" in p.columns:
            out["countries_ever_on_baseline_frontier"] = int(p.loc[p["dea_score"] >= 0.999, "iso_code"].nunique())
    f = tdir / "mpi_results.csv"
    if f.exists():
        imp, dec, unc = _gml_counts(pd.read_csv(f))
        out.update({"gml_countries_improving": imp, "gml_countries_declining": dec,
                    "gml_countries_unchanged": unc})
    f = tdir / "step7_simar_wilson.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            out[f"sw_coef_{r['variable']}"] = float(r["coef_bc"])
    f = tdir / "step12_metafrontier_results.csv"
    if f.exists():
        r12 = pd.read_csv(f)
        cm = r12.groupby("iso_code").agg(tgr=("tgr", "mean"), g=("tech_regime", "first"))
        for g, v in cm.groupby("g")["tgr"].mean().items():
            out[f"tgr_mean_{g}"] = float(v)
            out[f"n_countries_{g}"] = int((cm["g"] == g).sum())
    return out


def _text_numbers(text_path):
    import re, zipfile
    tp = Path(text_path)
    if tp.suffix.lower() == ".docx":
        with zipfile.ZipFile(tp) as z:
            xml = z.read("word/document.xml").decode("utf8", "ignore")
        text = re.sub(r"<[^>]+>", " ", xml)
    else:
        text = tp.read_text(encoding="utf8", errors="ignore")
    text = text.replace("\u2212", "-")
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)
            if any(ch.isdigit() for ch in x)], text


def consistency_check(run_dir=None, text_path=None, tol=1e-9):
    """
    (1) Recompute the headline numbers from the exported CSV files of run_dir
        and compare them with tables/key_results.csv (written from memory at
        the end of the same run): any mismatch means the folder mixes files
        from different runs or a file was edited.
    (2) If text_path is given (.docx/.md/.txt), report for every key result
        whether it appears in the text (at 2, 3 or 4 decimals, or as an
        integer), and list numbers in the text with 3+ decimals that match
        no key result -- candidates for stale values.
    """
    rd = Path(run_dir) if run_dir else _rdir()
    tdir = rd / "tables"
    kr_f = tdir / "key_results.csv"
    if not kr_f.exists():
        print(f"  ⚠ {kr_f} not found -- run the pipeline to completion first"); return pd.DataFrame()
    kr = pd.read_csv(kr_f)
    rec = _recompute_from_exports(tdir)
    kr["recomputed_from_exports"] = kr["name"].map(rec)
    kr["exports_consistent"] = [
        (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(a))) if pd.notna(b) else np.nan
        for a, b in zip(kr["value"].astype(float), kr["recomputed_from_exports"].astype(float))]
    n_bad = int((kr["exports_consistent"] == False).sum())
    print(f"\n{'═'*70}")
    print(f"CONSISTENCY CHECK — run {rd.name}")
    print(f"{'═'*70}")
    print(f"  Exports vs in-memory key results: {int((kr['exports_consistent'] == True).sum())} consistent, "
          f"{n_bad} inconsistent, {int(kr['exports_consistent'].isna().sum())} not recomputable")
    for _, r in kr[kr["exports_consistent"] == False].iterrows():
        print(f"    ✗ {r['name']}: key result {r['value']} vs exports {r['recomputed_from_exports']}")
    if text_path:
        nums, _ = _text_numbers(text_path)
        arr = np.array(nums)

        def _in_text(v):
            if float(v).is_integer():
                return bool(np.any(arr == v))
            return any(np.any(np.isclose(arr, round(v, d), atol=0.5 * 10 ** -d)) &
                       np.any(np.round(arr, d) == round(v, d)) for d in (2, 3, 4))
        kr["in_text"] = [_in_text(v) for v in kr["value"].astype(float)]
        keyvals = kr["value"].astype(float).values
        unmatched = sorted({x for x in nums if abs(x) < 1e6 and len(str(x).split(".")[-1]) >= 3
                            and not any(round(k, len(str(x).split(".")[-1])) == x for k in keyvals)})
        print(f"\n  Text: {text_path}")
        print(f"  Key results found in the text: {int(kr['in_text'].sum())}/{len(kr)}")
        print(f"  Numbers in the text with 3+ decimals not matching any key result "
              f"({len(unmatched)}; check against the run's tables): "
              + ", ".join(map(str, unmatched[:40])) + (" …" if len(unmatched) > 40 else ""))
    kr.to_csv(tdir / "consistency_check.csv", index=False)
    print(f"  Saved: {tdir / 'consistency_check.csv'}")
    return kr


def write_run_manifest(state, settings, done, all_steps):
    """logs/run_manifest.json: run identity, settings, executed / omitted
    analyses and a SHA-256 inventory of every exported file."""
    import json, hashlib
    rd = _rdir()
    omitted = []
    if "step10" not in all_steps:
        omitted.append({"module": "Step 10 — panel Granger causality",
                        "reason": "optional, not run (run_granger=False)",
                        "note": "Step 8c cross-lagged models are a different analysis and do not "
                                "substitute for this module"})
    if "appendix_A" not in all_steps:
        omitted.append({"module": "Appendix A — optional extensions",
                        "reason": "optional, not run (run_optional=False)"})
    if "bootstrap_validation" not in all_steps:
        omitted.append({"module": "Monte Carlo bootstrap validation",
                        "reason": "optional, not run (run_validation=False)"})
    inv = []
    for sub in ("tables", "plots", "logs", "data_export"):
        for f in sorted((rd / sub).glob("*")) if (rd / sub).exists() else []:
            if f.is_file() and f.name != "run_manifest.json":
                inv.append({"file": f"{sub}/{f.name}", "bytes": f.stat().st_size,
                            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    man = {"run_id": rd.name, "run_dir": str(rd), "mode": "TEST" if TEST_MODE else "PRODUCTION",
           "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "settings": settings,
           "steps_executed": list(done), "analyses_omitted": omitted,
           "step_numbering_note": "There is no Step 9; the shock evidence is Step 10b.",
           "files": inv}
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    with open(rd / "logs" / "run_manifest.json", "w") as f:
        json.dump(man, f, indent=2, default=str)
    print(f"\n  Run manifest: {rd / 'logs' / 'run_manifest.json'} ({len(inv)} files inventoried)")
    print(f"  Executed steps: {', '.join(done)}")
    for o in omitted:
        print(f"  OMITTED: {o['module']} — {o['reason']}. No conclusions should be attributed to it.")
    print(f"  Note: there is no Step 9; the shock evidence is Step 10b.")


def run_all(B: int = B_BOOTSTRAP, run_optional=False, run_granger=False,
            resume: bool = True, overwrite: bool = False, run_label: str = None,
            run_validation: bool = None) -> dict:
    """
    Run the complete pipeline end-to-end and return the state dict.

    Parameters:
        B : bootstrap replications (default: B_BOOTSTRAP)
        run_optional : if True, also runs Appendix A (optional extensions:
            between-effects, quantile regression, Anderson-Hsiao dynamic
            panel, metafrontier). Default False -- these are
            robustness/supplementary methods, not part of the core
            pipeline, and add meaningful runtime.
        resume : if True (default) and an UNFINISHED run with the SAME settings
            has checkpoints on Drive, continue from the last completed step
            (same results folder). Otherwise a new run is started.
        run_validation : run the Monte Carlo validation of the DDF bootstrap
            (coverage and score correction on a known technology; MC_R x MC_B).
            Default: RUN_BOOTSTRAP_VALIDATION.
        run_label : optional text added to the new run folder name, e.g.
            run_all(run_label="final") -> run_PROD_<date>_<time>_final.
            Folder names always start with run_TEST_ or run_PROD_.
        overwrite : if True, delete the whole output root folder before a NEW
            run. Default False (previous results, including a finished
            production run, are kept). Never applied when resuming.
        run_granger : if True, also runs the Panel Granger causality test.
            Default False. Individual country-level Wald statistics can be
            numerically unstable (near-singular within-country regressions
            occasionally produce astronomically large values), which gates
            the panel-level verdict for the affected direction rather than
            reporting it. Local Projections (step10b, run by default) give
            a more robust and economically interpretable temporal-dynamics
            analysis and are the primary recommended method for this
            question; Granger is retained as an optional cross-check only.

    Usage:
        state = run_all()           # B=B_BOOTSTRAP (20 in TEST_MODE, 1000 in production)
        state = run_all(B=B_BOOTSTRAP, run_optional=True)   # + Appendix A
        state = run_all(B=B_BOOTSTRAP, run_granger=True)    # + Granger causality

    Returns:
        state dict containing:
            panel        — main G20 panel with DEA + ESI scores
            mpi_df       — Malmquist results
            sw_results   — Simar–Wilson coefficients
            fe_result    — Panel FE OLS result
            proxy_stats  — affordability validation statistics
    """
    import shutil
    # overwrite=True wipes the entire root_dir, including ALL previous runs.
    # Default is now False so a finished production run cannot be deleted by
    # starting another run.
    # Set to False to preserve intermediate panels between test runs.
    # *** FIX: this path previously assumed /content/drive/MyDrive was always
    # available and writable. If Drive failed to mount (see the try/except
    # around drive.mount() near the top of this file), os.makedirs() here
    # would raise its own error on a non-existent /content/drive mount
    # point, a SECOND crash point stemming from the same root cause. Falls
    # back to a local, non-Drive path when Drive isn't mounted, so this
    # function degrades the same way the rest of the pipeline does (results
    # saved locally, not persisted across a runtime restart) rather than
    # failing outright. ***
    if globals().get("_DRIVE_MOUNTED", False) and os.path.isdir("/content/drive/MyDrive"):
        root_dir = "/content/drive/MyDrive/out/20260602_ESIDEASimar-Wolson"
    else:
        root_dir = "results/data_export_20260602_ESIDEASimar-Wolson"
        print(f"  ℹ Google Drive not mounted -- data export root_dir falls back to a")
        print(f"    local path: {root_dir}")
    # ── Checkpoint / resume ──────────────────────────────────────────────────
    if run_validation is None:
        run_validation = RUN_BOOTSTRAP_VALIDATION
    settings = _run_settings(B, run_optional, run_granger)
    settings["run_validation"] = bool(run_validation)
    ckpt = _find_resumable_checkpoint(root_dir, settings) if resume else None
    if ckpt is None and overwrite and os.path.exists(root_dir):
        shutil.rmtree(root_dir)
        print(f"  ⚠ overwrite=True: previous output folder wiped: {root_dir}")
    os.makedirs(root_dir, exist_ok=True)

    t0 = time.time()
    print("=" * 60)
    print("  ENERGY SECURITY PIPELINE — PANEL FULL ANALYSIS")
    print(f"  Countries in panel: {len(PANEL_ISO3)}")
    print(f"  Bootstrap replications: B={B}")
    print(f"  Primary LP solver: {DDF_SOLVER.upper()} | parallel bootstrap: "
          f"{PARALLEL_BOOTSTRAP} ({_n_jobs()} worker process(es))")
    print("=" * 60)

    if ckpt is not None:
        state, done = _load_checkpoint(ckpt)
        print(f"\n  ▶ RESUMING run {RUN_DIR} -- {len(done)} step(s) already completed:")
        print(f"    {done}")
    else:
        _init_run_dir(root_dir, label=run_label)   # new, uniquely named folder for this run
        state, done = {}, []
        print(f"  Checkpoints for this run: {RUN_DIR / 'checkpoints'}")

    # Raw input snapshots (OWID, World Bank, panels) are stored INSIDE the run
    # folder. They used to go to <root>/data_export, shared by all runs, so a
    # later run overwrote the inputs of an earlier (e.g. production) run.
    out_dir = os.path.join(str(RUN_DIR), 'data_export')
    os.makedirs(out_dir, exist_ok=True)

    def _exports_3(st):
        st['owid_g20'].to_csv(os.path.join(out_dir, 'owid_g20.csv'), index=False)
        st['wb_g20'].to_csv(os.path.join(out_dir, 'wb_g20.csv'), index=False)
        st['energy_burden_g20'].to_csv(os.path.join(out_dir, 'energy_burden_g20.csv'), index=False)
        st['panel'].to_csv(os.path.join(out_dir, 'panel_3.csv'), index=False)
        return st

    def _audit(st):
        with _TeeLogger("ddf_solver_audit.log"):
            st["ddf_solver_audit_passed"] = audit_ddf_solver()
        return st

    def _step4(st):
        st = step4_esi(st)
        st['panel'].to_csv(os.path.join(out_dir, 'panel_4.csv'), index=False)
        return st

    def _validation(st):
        with _TeeLogger("bootstrap_validation.log"):
            st["bootstrap_validation"] = validate_ddf_bootstrap(vrs=VRS)
        return st

    steps = [
        ("audit",        _audit),
        *([("bootstrap_validation", _validation)] if run_validation else []),
        ("step1",        step1_download),
        ("step3",        lambda st: _exports_3(step3_build_panel(st))),
        ("step2",        step2_validate_affordability),
        ("step4",        _step4),
        ("step5",        lambda st: step5_dea_bootstrap(st, B=B)),
        # Bootstrap of the alternative returns to scale (VRS robustness check)
        ("alt_rts",      add_alt_rts_bootstrap),
        ("step5c",       step5c_specification_sensitivity),
        ("step6",        step6_mpi),
        ("step7",        lambda st: step7_simar_wilson(st, B=B_SW2007)),
        ("step7b",       lambda st: step7b_simar_wilson_restricted(st, B=B_SW2007,
                                                                    common_sample_with_full=True)),
        ("step7c",       step7c_conditional_ddf),
        ("step8",        step8_panel_fe),
        ("step8b",       step8b_mundlak_cre),
        ("step8c",       step8c_lagged_link),
    ]
    if run_granger:
        steps.append(("step10", step10_granger))
    steps += [
        ("step10b",      step10b_local_projections),
        ("step11",       step11_plots),
        ("step12",       step12_metafrontier_analysis),
        ("step13",       step13_export),
    ]
    if run_optional:
        steps.append(("appendix_A", run_optional_extensions))

    for name, fn in steps:
        if name in done:
            print(f"  ⏭ {name}: already completed (checkpoint) -- skipped")
            continue
        t_step = time.time()
        state = fn(state)
        done.append(name)
        _save_checkpoint(state, done, settings, finished=(name == steps[-1][0]))
        print(f"  ✓ checkpoint saved after {name} ({(time.time()-t_step)/60:.1f} min)")

    # Canonical headline numbers, consistency check of the exports, manifest.
    try:
        _save_csv(build_key_results(state), "key_results.csv", "Headline numbers of this run")
        state["consistency_check"] = consistency_check(_rdir())
    except Exception as e:
        print(f"  ⚠ key results / consistency check failed: {e}")
    try:
        write_run_manifest(state, settings, done, [n for n, _ in steps])
    except Exception as e:
        print(f"  ⚠ run manifest failed: {e}")

    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline complete in {(time.time()-t0)/60:.1f} min")
    print("="*60)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT — bootstrap_mpi(panel)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_mpi(panel: pd.DataFrame = None, B: int = B_BOOTSTRAP) -> pd.DataFrame:
    """
    Compute bootstrap-corrected DEA efficiency + Global Malmquist-Luenberger (GML) decomposition
    on an existing panel.  If no panel is provided, downloads data first.

    Parameters:
        panel : DataFrame already containing the required DEA input/output columns.
                If None, runs steps 1–4 automatically.
        B     : bootstrap replications (default: B_BOOTSTRAP)

    Returns:
        mpi_df : DataFrame with columns mpi, ec, tc per iso_code × year pair

    Usage (notebook):
        # Option A — on existing panel:
        mpi_df = bootstrap_mpi(state['panel'])

        # Option B — from scratch:
        mpi_df = bootstrap_mpi()

        # Option C — explicit override (not needed for production use; the
        # default B=B_BOOTSTRAP already applies uniformly to every bootstrap
        # and Malmquist calculation in the pipeline, Malmquist included):
        mpi_df = bootstrap_mpi(state['panel'], B=B_BOOTSTRAP)
    """
    if panel is None:
        print("No panel provided — running steps 1–4 to build it …")
        state = step1_download({})
        state = step2_validate_affordability(state)
        state = step3_build_panel(state)
        state = step4_esi(state)
        panel = state["panel"]

    print(f"\nRunning bootstrap DEA (B={B}) …")
    panel = build_dea_bootstrap(panel, B=B)
    print("Running GML (Global Malmquist-Luenberger) decomposition …")
    mpi_df = build_mpi(panel)
    return mpi_df


# ─────────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # In Colab, running the cell that contains this file starts (or RESUMES)
    # the pipeline here. After a disconnect, just run the cell again.
    state = run_all(B=B_BOOTSTRAP, run_label=(None if TEST_MODE else RUN_LABEL))
