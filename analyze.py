"""
analyze.py – Demography dashboard sheet analysis
Reads all country Excel files from data/ and produces visualisation plots in output/.

Column layout in each sheet (0-indexed):
  0=ID code, 1=Label, 2=All (aa), 3=Native Female (nf), 4=Native Male (nm),
  5=Migrant Female (mf), 6=Migrant Male (mm),
  7=Utility Native M+F (na), 8=Utility Migrant M+F (ma)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# Colors for native/migrant split (used in per-country plots)
COLORS = {
    "native_f":  "#4C72B0",
    "native_m":  "#1A4E8C",
    "migrant_f": "#DD8452",
    "migrant_m": "#C44E52",
    "native":    "#4C72B0",
    "migrant":   "#DD8452",
}

# Per-country palette for comparison plots: each country gets a native + migrant color
# and a line style so they stay distinguishable on the same axes.
COUNTRY_PALETTE = [
    {"native": "#1A4E8C", "migrant": "#C44E52", "ls": "-",  "lw": 2.0},
    {"native": "#2ca02c", "migrant": "#9467bd", "ls": "--", "lw": 2.0},
    {"native": "#8c564b", "migrant": "#e377c2", "ls": ":",  "lw": 2.2},
]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi":     120,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_country(path: Path) -> dict:
    return pd.read_excel(path, sheet_name=None, header=None)


def get_meta(sheets: dict) -> dict:
    ov = sheets.get("overview", pd.DataFrame())
    meta = {}
    for _, row in ov.iterrows():
        code = str(row.get(0, ""))
        if code == "na":
            meta["acronym"] = str(row.get(2, "")).strip()
        elif code == "nn":
            meta["name"] = str(row.get(2, "")).strip()
    return meta


def get_overview_stats(sheets: dict) -> dict:
    ov = sheets.get("overview", pd.DataFrame())
    stats = {}
    targets = {
        "Current population":                                   ("pop_all", "pop_migrant_share", "pop_native", "pop_migrant"),
        "Share of migrant population":                          ("migrant_share",),
        "Birth / year used for estimation":                     ("births_native", "births_migrant"),
        "Deaths / year":                                        ("deaths_native", "deaths_migrant"),
        "Migration balance / year":                             ("mig_native", "mig_migrant"),
        "Population delta / year":                              ("delta_all", "delta_native", "delta_migrant"),
        "Years until  33% share of migrants":              ("years_33pct",),
        "Estimated tipping point for  33% share of migrants":  ("tipping_33",),
        "Years until  50% share of migrants":              ("years_50pct",),
        "Estimated tipping point for  50% share of migrants":  ("tipping_50",),
    }
    for _, row in ov.iterrows():
        label = str(row.get(1, "")).strip()
        if label in targets:
            keys = targets[label]
            nums = [v for v in row[2:] if pd.notna(v) and isinstance(v, (int, float))]
            for i, key in enumerate(keys):
                if i < len(nums):
                    stats[key] = nums[i]
    return stats


def get_fertility(sheets: dict) -> dict:
    births = sheets.get("births", pd.DataFrame())
    rates = {}
    for _, row in births.iterrows():
        if str(row.get(0, "")) == "f":
            if pd.notna(row.get(3)):
                rates["native"] = float(row[3])
            if pd.notna(row.get(5)):
                rates["migrant"] = float(row[5])
            break
    return rates


def _age_frame(sheet: pd.DataFrame, id_pattern: str) -> pd.DataFrame | None:
    cols_base = ["code", "label", "all", "native_f", "native_m", "migrant_f", "migrant_m",
                 "util_na", "util_ma", "util_af", "util_am", "util_other"]
    rows = sheet[sheet[0].astype(str).str.match(id_pattern, na=False)].copy()
    if rows.empty:
        return None
    rows.columns = cols_base[: rows.shape[1]] + [f"x{i}" for i in range(max(0, rows.shape[1] - len(cols_base)))]
    rows["age"] = rows["code"].str[1:].astype(int)
    for col in ["all", "native_f", "native_m", "migrant_f", "migrant_m"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    if rows[["native_f", "native_m", "migrant_f", "migrant_m"]].isna().all().all():
        return None
    return rows.set_index("age").sort_index()


def get_population_pyramid(sheets: dict) -> pd.DataFrame | None:
    return _age_frame(sheets.get("population", pd.DataFrame()), r"^p\d{2}$")


def get_births_by_age(sheets: dict) -> pd.DataFrame | None:
    df = _age_frame(sheets.get("births", pd.DataFrame()), r"^b\d{2}$")
    if df is None or df[["native_f", "migrant_f"]].isna().all().all():
        return None
    return df


def get_deaths_by_age(sheets: dict) -> pd.DataFrame | None:
    return _age_frame(sheets.get("deaths", pd.DataFrame()), r"^d\d{2}$")


def get_migration_by_age(sheets: dict) -> pd.DataFrame | None:
    return _age_frame(sheets.get("migration", pd.DataFrame()), r"^m\d{2}$")


def get_origin(sheets: dict) -> pd.DataFrame | None:
    origin = sheets.get("origin", pd.DataFrame())
    rows = origin[origin[0].astype(str).str.match(r"^c\d", na=False)].copy()
    if rows.empty:
        return None
    rows.columns = ["code", "country", "iso2", "native_f", "native_m", "migrant_f", "migrant_m",
                    "util_na", "util_ma", "util_af", "util_am", "util_other"][: rows.shape[1]]
    for col in ["migrant_f", "migrant_m"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows[rows[["migrant_f", "migrant_m"]].notna().any(axis=1)].copy()
    rows["total"] = rows["migrant_f"].fillna(0) + rows["migrant_m"].fillna(0)
    return rows[rows["total"] > 0].sort_values("total", ascending=False).reset_index(drop=True)


def get_naturalization(sheets: dict) -> pd.DataFrame | None:
    """
    Read the naturalisation block from the patria sheet.

    The patria sheet has two year-keyed blocks:
      1. Naturalisations  — col 8 (Utility Migrant M+F) is populated
      2. Expatriations    — col 7 (Utility Native M+F)  is populated, col 8 absent

    Filtering on col 8 being non-NaN isolates block 1 and avoids the duplicate-year
    zigzag that arises when both blocks are merged. Years with total == 0 are skipped.
    """
    patria = sheets.get("patria", pd.DataFrame())
    years, totals = [], []
    for _, row in patria.iterrows():
        yr  = row.get(1)
        tot = row.get(2)
        util_migrant = row.get(8)
        if (
            pd.notna(yr) and isinstance(yr, (int, float)) and 1900 < yr < 2100
            and pd.notna(tot) and isinstance(tot, (int, float)) and tot > 0
            and pd.notna(util_migrant)
        ):
            years.append(int(yr))
            totals.append(int(tot))
    if not years:
        return None
    return pd.DataFrame({"year": years, "total": totals}).set_index("year").sort_index()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_k(x, _=None):
    ax = abs(x)
    if ax >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if ax >= 1_000:
        return f"{int(ax / 1_000)}k"
    return str(int(ax))


def _save(fig, path: Path, name: str):
    fig.savefig(path / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def _label(meta: dict) -> str:
    return f"{meta.get('name', '')} ({meta.get('acronym', '')})"


# ---------------------------------------------------------------------------
# Per-country plots
# ---------------------------------------------------------------------------

def plot_population_pyramid(pop: pd.DataFrame, meta: dict, stats: dict, out_dir: Path):
    acronym = meta.get("acronym", "XX")
    migrant_share = stats.get("migrant_share")
    share_str = f" — migrant share: {migrant_share:.1%}" if migrant_share else ""

    tp_lines = []
    for pct, key in [("33%", "tipping_33"), ("50%", "tipping_50")]:
        if stats.get(key):
            tp_lines.append(f"{pct} migrant share by ~{int(stats[key])}")
    footnote = " | ".join(tp_lines)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    fig.subplots_adjust(wspace=0.04)
    ages = pop.index.values

    for ax_idx, (ax, title, ncol, mcol, nc, mc) in enumerate(zip(
        axes,
        ["Males", "Females"],
        ["native_m", "native_f"],
        ["migrant_m", "migrant_f"],
        [COLORS["native_m"], COLORS["native_f"]],
        [COLORS["migrant_m"], COLORS["migrant_f"]],
    )):
        sign = -1 if ax_idx == 0 else 1
        nv = pop[ncol].fillna(0).values
        mv = pop[mcol].fillna(0).values
        ax.barh(ages, sign * nv, height=0.8, color=nc, label="Native",  alpha=0.9)
        ax.barh(ages, sign * mv, height=0.8, color=mc, label="Migrant", alpha=0.75,
                left=sign * nv)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Population", labelpad=6)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: fmt_k(abs(x))))
        ax.axvline(0, color="black", linewidth=0.6)
        ax.legend(loc="lower right" if ax_idx == 0 else "lower left", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right" if ax_idx == 0 else "left"].set_visible(False)

    axes[0].set_yticks(range(0, 100, 5))
    axes[0].set_ylabel("Age")
    fig.suptitle(f"Population Pyramid – {_label(meta)}{share_str}",
                 fontsize=14, fontweight="bold", y=1.01)
    if footnote:
        fig.text(0.5, -0.01, footnote, ha="center", fontsize=9, style="italic", color="#555")
    _save(fig, out_dir, f"{acronym}_population_pyramid.png")


def plot_births_by_age(births: pd.DataFrame, meta: dict, out_dir: Path):
    acronym = meta.get("acronym", "XX")
    ages  = births.index.values
    nv    = births["native_f"].fillna(0).values
    mv    = births["migrant_f"].fillna(0).values
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ages - width / 2, nv, width=width, color=COLORS["native_f"],  label="Native mothers",  alpha=0.9)
    ax.bar(ages + width / 2, mv, width=width, color=COLORS["migrant_f"], label="Migrant mothers", alpha=0.9)
    ax.set_xlabel("Age of mother")
    ax.set_ylabel("Births per year")
    ax.set_title(f"Births by Mother's Age – {_label(meta)}", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, f"{acronym}_births_by_mother_age.png")


def plot_deaths_by_age(deaths: pd.DataFrame, meta: dict, out_dir: Path):
    acronym = meta.get("acronym", "XX")
    ages = deaths.index.values
    groups = [
        ("native_f",  "Native Female",  COLORS["native_f"],  "-"),
        ("native_m",  "Native Male",    COLORS["native_m"],  "--"),
        ("migrant_f", "Migrant Female", COLORS["migrant_f"], "-"),
        ("migrant_m", "Migrant Male",   COLORS["migrant_m"], "--"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, lbl, color, ls in groups:
        ax.plot(ages, deaths[col].fillna(0), color=color, label=lbl, lw=1.8, linestyle=ls)
    ax.set_xlabel("Age")
    ax.set_ylabel("Deaths per year")
    ax.set_title(f"Deaths by Age – {_label(meta)}", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, f"{acronym}_deaths_by_age.png")


def plot_migration_by_age(migration: pd.DataFrame, meta: dict, out_dir: Path):
    acronym = meta.get("acronym", "XX")
    ages    = migration.index.values
    native  = (migration["native_f"].fillna(0)  + migration["native_m"].fillna(0)).values
    migrant = (migration["migrant_f"].fillna(0) + migration["migrant_m"].fillna(0)).values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(ages, native,  0, where=(native  < 0), color=COLORS["native"],  alpha=0.6, label="Native (net emigration)")
    ax.fill_between(ages, native,  0, where=(native  > 0), color=COLORS["native"],  alpha=0.6)
    ax.fill_between(ages, migrant, 0, where=(migrant > 0), color=COLORS["migrant"], alpha=0.6, label="Migrant (net immigration)")
    ax.fill_between(ages, migrant, 0, where=(migrant < 0), color=COLORS["migrant"], alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Age")
    ax.set_ylabel("Net migration balance per year")
    ax.set_title(f"Net Migration by Age – {_label(meta)}", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, f"{acronym}_migration_by_age.png")


def plot_origin(origin: pd.DataFrame, meta: dict, out_dir: Path, top_n: int = 25):
    acronym = meta.get("acronym", "XX")
    df  = origin.head(top_n).copy().sort_values("total", ascending=True)
    y   = np.arange(len(df))
    mf  = df["migrant_f"].fillna(0).values
    mm  = df["migrant_m"].fillna(0).values

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.32)))
    ax.barh(y, mf, color=COLORS["migrant_f"], label="Female", alpha=0.9)
    ax.barh(y, mm, left=mf, color=COLORS["migrant_m"], label="Male", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["country"].values, fontsize=8)
    ax.set_xlabel("Number of migrants")
    ax.set_title(f"Top {top_n} Migrant Origin Countries – {_label(meta)}", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, f"{acronym}_migrant_origins.png")


def plot_naturalization(patria: pd.DataFrame, meta: dict, out_dir: Path):
    acronym = meta.get("acronym", "XX")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(patria.index, patria["total"], color=COLORS["migrant"], lw=2)
    ax.fill_between(patria.index, patria["total"], alpha=0.2, color=COLORS["migrant"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Naturalisations per year")
    ax.set_title(f"Naturalisations per Year – {_label(meta)}", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, f"{acronym}_naturalisations.png")


# ---------------------------------------------------------------------------
# Comparison plots  (all countries on one chart)
# ---------------------------------------------------------------------------

def _palette(idx: int) -> dict:
    return COUNTRY_PALETTE[idx % len(COUNTRY_PALETTE)]


def cmp_population(countries: list[dict], out_dir: Path):
    """Stacked bar: native vs migrant population, one bar per country."""
    rows = [c for c in countries if c["stats"].get("pop_native")]
    if not rows:
        return
    labels   = [_label(c["meta"]) for c in rows]
    natives  = [c["stats"]["pop_native"]  for c in rows]
    migrants = [c["stats"].get("pop_migrant", 0) for c in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 2.5), 5))
    ax.bar(x, natives,  color=COLORS["native"],  label="Native",  alpha=0.9)
    ax.bar(x, migrants, bottom=natives, color=COLORS["migrant"], label="Migrant", alpha=0.85)
    for xi, (n, m) in zip(x, zip(natives, migrants)):
        total = n + m
        ax.text(xi, total * 1.01, f"{m/total:.1%}\nmigrant",
                ha="center", va="bottom", fontsize=9, color=COLORS["migrant_m"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Population")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.set_title("Total Population: Native vs Migrant", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "comparison_population.png")


def cmp_fertility(countries: list[dict], out_dir: Path):
    """Grouped bar: native and migrant TFR per country."""
    rows = [c for c in countries if c.get("fertility")]
    if not rows:
        return
    labels        = [_label(c["meta"]) for c in rows]
    native_rates  = [c["fertility"].get("native",  0) for c in rows]
    migrant_rates = [c["fertility"].get("migrant", 0) for c in rows]
    x, w = np.arange(len(rows)), 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 2.5), 5))
    ax.bar(x - w / 2, native_rates,  width=w, color=COLORS["native"],  label="Native",  alpha=0.9)
    ax.bar(x + w / 2, migrant_rates, width=w, color=COLORS["migrant"], label="Migrant", alpha=0.9)
    ax.axhline(2.1, color="gray", lw=1, linestyle="--", label="Replacement rate (2.1)")
    for xi, (n, m) in zip(x, zip(native_rates, migrant_rates)):
        ax.text(xi - w / 2, n + 0.02, f"{n:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Fertility Rate (TFR)")
    ax.set_title("Fertility Rate Comparison: Native vs Migrant", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, max(migrant_rates + [2.5]) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "comparison_fertility_rates.png")


def cmp_pyramid_migrant_share(countries: list[dict], out_dir: Path):
    """
    Migrant share (%) at each age, one line per country.
    Reveals whether migrants are concentrated in working-age cohorts.
    """
    rows = [c for c in countries if c.get("pop") is not None]
    if len(rows) < 2:
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, c in enumerate(rows):
        pal = _palette(i)
        pop = c["pop"]
        total = (pop["native_f"].fillna(0) + pop["native_m"].fillna(0)
                 + pop["migrant_f"].fillna(0) + pop["migrant_m"].fillna(0))
        migrant = pop["migrant_f"].fillna(0) + pop["migrant_m"].fillna(0)
        share = (migrant / total.replace(0, np.nan) * 100)
        ax.plot(share.index, share.values, color=pal["native"], lw=pal["lw"],
                linestyle=pal["ls"], label=_label(c["meta"]))

    ax.set_xlabel("Age")
    ax.set_ylabel("Migrant share at that age (%)")
    ax.set_title("Migrant Share by Age – Country Comparison", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}%"))
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "comparison_pyramid_migrant_share.png")


def cmp_births(countries: list[dict], out_dir: Path):
    """
    Births per 1,000 women of that age (age-specific rate), native vs migrant,
    one line style per country.
    """
    rows = [c for c in countries if c.get("births") is not None and c.get("pop") is not None]
    if len(rows) < 2:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, c in enumerate(rows):
        pal    = _palette(i)
        births = c["births"]
        pop    = c["pop"]
        ages   = births.index.values
        nf_pop = pop["native_f"].reindex(ages).fillna(0).replace(0, np.nan)
        mf_pop = pop["migrant_f"].reindex(ages).fillna(0).replace(0, np.nan)
        native_rate  = births["native_f"].fillna(0)  / nf_pop * 1000
        migrant_rate = births["migrant_f"].fillna(0) / mf_pop * 1000
        lbl = c["meta"].get("acronym", "?")
        ax.plot(ages, native_rate,  color=pal["native"],  lw=pal["lw"], linestyle=pal["ls"],
                label=f"{lbl} – Native")
        ax.plot(ages, migrant_rate, color=pal["migrant"], lw=pal["lw"], linestyle=pal["ls"],
                label=f"{lbl} – Migrant")

    ax.set_xlabel("Age of mother")
    ax.set_ylabel("Births per 1,000 women of that age")
    ax.set_title("Age-Specific Fertility Rate – Country Comparison", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "comparison_births_by_mother_age.png")


def cmp_deaths(countries: list[dict], out_dir: Path):
    """Deaths per 100,000 of each group, native vs migrant (sexes combined), by country."""
    rows = [c for c in countries if c.get("deaths") is not None and c.get("pop") is not None]
    if len(rows) < 2:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, c in enumerate(rows):
        pal    = _palette(i)
        deaths = c["deaths"]
        pop    = c["pop"]
        ages   = deaths.index.values
        np_n = (pop["native_f"].reindex(ages).fillna(0) + pop["native_m"].reindex(ages).fillna(0)).replace(0, np.nan)
        np_m = (pop["migrant_f"].reindex(ages).fillna(0) + pop["migrant_m"].reindex(ages).fillna(0)).replace(0, np.nan)
        d_n  = deaths["native_f"].fillna(0)  + deaths["native_m"].fillna(0)
        d_m  = deaths["migrant_f"].fillna(0) + deaths["migrant_m"].fillna(0)
        lbl  = c["meta"].get("acronym", "?")
        ax.plot(ages, d_n / np_n * 100_000, color=pal["native"],  lw=pal["lw"],
                linestyle=pal["ls"], label=f"{lbl} – Native")
        ax.plot(ages, d_m / np_m * 100_000, color=pal["migrant"], lw=pal["lw"],
                linestyle=pal["ls"], label=f"{lbl} – Migrant")

    ax.set_xlabel("Age")
    ax.set_ylabel("Deaths per 100,000 of group")
    ax.set_title("Death Rate by Age – Country Comparison", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "comparison_deaths_by_age.png")


def cmp_migration(countries: list[dict], out_dir: Path):
    """Net migration per 1,000 of group (native / migrant), by country."""
    rows = [c for c in countries if c.get("migration") is not None and c.get("pop") is not None]
    if len(rows) < 2:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, c in enumerate(rows):
        pal = _palette(i)
        mig = c["migration"]
        pop = c["pop"]
        ages = mig.index.values
        np_n = (pop["native_f"].reindex(ages).fillna(0) + pop["native_m"].reindex(ages).fillna(0)).replace(0, np.nan)
        np_m = (pop["migrant_f"].reindex(ages).fillna(0) + pop["migrant_m"].reindex(ages).fillna(0)).replace(0, np.nan)
        m_n  = mig["native_f"].fillna(0)  + mig["native_m"].fillna(0)
        m_m  = mig["migrant_f"].fillna(0) + mig["migrant_m"].fillna(0)
        lbl  = c["meta"].get("acronym", "?")
        ax.plot(ages, m_n / np_n * 1000, color=pal["native"],  lw=pal["lw"],
                linestyle=pal["ls"], label=f"{lbl} – Native")
        ax.plot(ages, m_m / np_m * 1000, color=pal["migrant"], lw=pal["lw"],
                linestyle=pal["ls"], label=f"{lbl} – Migrant")

    ax.axhline(0, color="black", lw=0.7, linestyle=":")
    ax.set_xlabel("Age")
    ax.set_ylabel("Net migration per 1,000 of group")
    ax.set_title("Net Migration Rate by Age – Country Comparison", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "comparison_migration_by_age.png")


def cmp_naturalisations(countries: list[dict], out_dir: Path):
    """
    Naturalisation time series for all countries on one chart.
    Uses dual Y-axes when the scales differ by more than 5×.
    """
    rows = [c for c in countries if c.get("patria") is not None]
    if len(rows) < 2:
        return

    maxima = [c["patria"]["total"].max() for c in rows]
    use_dual = (max(maxima) / min(maxima)) > 5

    if use_dual:
        fig, ax1 = plt.subplots(figsize=(13, 5))
        axes_list = [ax1, ax1.twinx()]
    else:
        fig, ax1 = plt.subplots(figsize=(13, 5))
        axes_list = [ax1] * len(rows)

    for i, (c, ax) in enumerate(zip(rows, axes_list)):
        pal = _palette(i)
        pat = c["patria"]
        lbl = _label(c["meta"])
        ax.plot(pat.index, pat["total"], color=pal["native"], lw=pal["lw"],
                linestyle=pal["ls"], label=lbl)
        ax.fill_between(pat.index, pat["total"], alpha=0.12, color=pal["native"])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
        if use_dual:
            ax.set_ylabel(f"Naturalisations – {lbl}", color=pal["native"])
            ax.tick_params(axis="y", colors=pal["native"])

    if not use_dual:
        ax1.set_ylabel("Naturalisations per year")
        ax1.legend(fontsize=8)
    else:
        # Collect legend from both axes
        lines1, labels1 = axes_list[0].get_legend_handles_labels()
        lines2, labels2 = axes_list[1].get_legend_handles_labels()
        axes_list[0].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax1.set_xlabel("Year")
    ax1.set_title("Naturalisations per Year – Country Comparison", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    _save(fig, out_dir, "comparison_naturalisations.png")


def cmp_origins(countries: list[dict], out_dir: Path, top_n: int = 20):
    """
    Common origin countries: % of each country's migrant stock, side by side.
    Only countries present in every dataset are shown.
    """
    origin_dfs = [(c["meta"], c["origin"]) for c in countries if c.get("origin") is not None]
    if len(origin_dfs) < 2:
        return

    # Normalise to % of migrant stock (aggregate duplicates first)
    pct_dfs = {}
    for meta, df in origin_dfs:
        agg = df.groupby("iso2")["total"].sum()
        pct_dfs[meta["acronym"]] = (agg / agg.sum() * 100).rename(meta["acronym"])

    # Common countries only
    combined = pd.concat(pct_dfs.values(), axis=1).dropna()
    if combined.empty:
        return

    # Pick top N by average share across countries
    combined["avg"] = combined.mean(axis=1)
    combined = combined.nlargest(top_n, "avg").drop(columns="avg")

    # Attach country names from the first dataset that has them
    iso_to_name = {}
    for _, df in origin_dfs:
        iso_to_name.update(dict(zip(df["iso2"], df["country"])))
    combined.index = [iso_to_name.get(iso, iso) for iso in combined.index]
    combined = combined.sort_values(combined.columns[0], ascending=True)

    n_countries = len(combined.columns)
    y    = np.arange(len(combined))
    h    = 0.8 / n_countries
    fig, ax = plt.subplots(figsize=(11, max(6, len(combined) * 0.38)))

    for i, col in enumerate(combined.columns):
        pal    = _palette(i)
        offset = (i - (n_countries - 1) / 2) * h
        ax.barh(y + offset, combined[col].values, height=h * 0.9,
                color=pal["native"], label=col, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(combined.index, fontsize=8)
    ax.set_xlabel("% of country's total migrant stock")
    ax.set_title(f"Top {top_n} Common Migrant Origin Countries – Country Comparison",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1f}%"))
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "comparison_migrant_origins.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    xlsx_files = sorted(f for f in DATA_DIR.glob("*.xlsx") if not f.name.startswith("~$"))
    if not xlsx_files:
        print(f"No .xlsx files found in {DATA_DIR}")
        return

    countries = []

    for path in xlsx_files:
        print(f"\nProcessing: {path.name}")
        sheets = load_country(path)
        meta   = get_meta(sheets)
        print(f"  Country: {meta.get('name', '?')} ({meta.get('acronym', '?')})")

        stats     = get_overview_stats(sheets)
        fertility = get_fertility(sheets)
        pop       = get_population_pyramid(sheets)
        births    = get_births_by_age(sheets)
        deaths    = get_deaths_by_age(sheets)
        migration = get_migration_by_age(sheets)
        origin    = get_origin(sheets)
        patria    = get_naturalization(sheets)

        countries.append({
            "meta": meta, "stats": stats, "fertility": fertility,
            "pop": pop, "births": births, "deaths": deaths,
            "migration": migration, "origin": origin, "patria": patria,
        })

        if pop is not None:
            plot_population_pyramid(pop, meta, stats, OUT_DIR)
        else:
            print("  Skipping population pyramid (no age data)")

        if births is not None:
            plot_births_by_age(births, meta, OUT_DIR)
        else:
            print("  Skipping births chart (no data)")

        if deaths is not None:
            plot_deaths_by_age(deaths, meta, OUT_DIR)
        else:
            print("  Skipping deaths chart (no data)")

        if migration is not None:
            plot_migration_by_age(migration, meta, OUT_DIR)
        else:
            print("  Skipping migration chart (no data)")

        if origin is not None:
            plot_origin(origin, meta, OUT_DIR)
        else:
            print("  Skipping origin chart (no data)")

        if patria is not None:
            plot_naturalization(patria, meta, OUT_DIR)
        else:
            print("  Skipping naturalisations chart (no data)")

    print("\nGenerating comparison plots...")
    cmp_population(countries, OUT_DIR)
    cmp_fertility(countries, OUT_DIR)
    cmp_pyramid_migrant_share(countries, OUT_DIR)
    cmp_births(countries, OUT_DIR)
    cmp_deaths(countries, OUT_DIR)
    cmp_migration(countries, OUT_DIR)
    cmp_naturalisations(countries, OUT_DIR)
    cmp_origins(countries, OUT_DIR)

    print(f"\nDone. All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
