"""
data_loader.py – shared data parsing for the Streamlit dashboard.
Exports load_all_countries(data_dir) -> list[dict]
"""

from pathlib import Path
import pandas as pd
import numpy as np


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
        "Years until  33% share of migrants":                   ("years_33pct",),
        "Estimated tipping point for  33% share of migrants":   ("tipping_33",),
        "Years until  50% share of migrants":                   ("years_50pct",),
        "Estimated tipping point for  50% share of migrants":   ("tipping_50",),
    }
    # Normalize target keys to collapse any whitespace (including line separators)
    norm_targets = {" ".join(k.split()): v for k, v in targets.items()}
    for _, row in ov.iterrows():
        raw_label = str(row.get(1, ""))
        label = " ".join(raw_label.split())  # collapse all whitespace to single spaces
        if label in norm_targets:
            keys = norm_targets[label]
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


def load_all_countries(data_dir: Path) -> list[dict]:
    xlsx_files = sorted(f for f in data_dir.glob("*.xlsx") if not f.name.startswith("~$"))
    countries = []
    for path in xlsx_files:
        sheets    = load_country(path)
        meta      = get_meta(sheets)
        if not meta.get("acronym"):
            continue
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
    return countries
