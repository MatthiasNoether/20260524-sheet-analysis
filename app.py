"""
app.py – Demography Dashboard (Streamlit + Plotly)
Run: streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_all_countries

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "native_f":  "#4C72B0",
    "native_m":  "#1A4E8C",
    "migrant_f": "#DD8452",
    "migrant_m": "#C44E52",
}

COUNTRY_COLORS = [
    {"native": "#1A4E8C", "migrant": "#C44E52"},
    {"native": "#1A4E8C", "migrant": "#C44E52"},
    {"native": "#1A4E8C", "migrant": "#C44E52"},
    {"native": "#1A4E8C", "migrant": "#C44E52"},
]

COUNTRY_COMPARISON_COLORS = {
    "DE": "#FFD700", "GERMANY": "#FFD700",
    "FR": "#0055A4", "FRANCE": "#0055A4",
    "IT": "#009246", "ITALY": "#009246",
    "ES": "#AA151B", "SPAIN": "#AA151B",
    "US": "#3C3B6E", "USA": "#3C3B6E", "UNITED STATES": "#3C3B6E",
    "GB": "#012169", "UK": "#012169", "UNITED KINGDOM": "#012169",
    "CA": "#FF8C00", "CANADA": "#FF8C00",
    "AT": "#ED2939", "AUSTRIA": "#ED2939",
    "AU": "#00008B", "AUSTRALIA": "#00008B",
    "JP": "#BC002D", "JAPAN": "#BC002D",
    "IN": "#FF9933", "INDIA": "#FF9933",
    "BR": "#009C3B", "BRAZIL": "#009C3B",
    "RU": "#0033A0", "RUSSIA": "#0033A0",
    "SE": "#005B99", "SWEDEN": "#005B99",
    "MX": "#006847", "MEXICO": "#006847",
    "CN": "#DE2910", "CHINA": "#DE2910",
    "NL": "#FF6600", "NETHERLANDS": "#FF6600",
    "NO": "#BA0C2F", "NORWAY": "#BA0C2F",
    "PT": "#006600", "PORTUGAL": "#006600",
    "CH": "#D52B1E", "SWITZERLAND": "#D52B1E",
    "TR": "#E30A17", "TURKEY": "#E30A17",
    "GR": "#0D5EAF", "GREECE": "#0D5EAF",
}

COUNTRY_COMPARISON_FALLBACK = [
    "#1A4E8C", "#C44E52", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22"
]

# ---------------------------------------------------------------------------
# ISO2 → Plotly country name mapping for choropleth
# ---------------------------------------------------------------------------
ISO2_NAME_FIXES = {
    "TR": "Turkey", "CZ": "Czechia", "KR": "South Korea",
    "VN": "Vietnam", "TW": "Taiwan", "IR": "Iran",
    "SY": "Syria", "BO": "Bolivia", "RU": "Russia",
    "MD": "Moldova", "TZ": "Tanzania", "CD": "Democratic Republic of the Congo",
    "CI": "Cote d'Ivoire", "MK": "North Macedonia",
    "PS": "Palestine", "XK": "Kosovo",
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading country data…")
def load_data():
    return load_all_countries(DATA_DIR)


def fmt_k(x):
    ax = abs(x)
    if ax >= 1_000_000:
        return f"{x / 1_000_000:.1f} M"
    if ax >= 1_000:
        return f"{x / 1_000:.0f} k"
    return str(int(x))


def country_label(meta):
    return f"{meta.get('name', '')} ({meta.get('acronym', '')})"


def get_country_color(meta: dict) -> str:
    acronym = str(meta.get("acronym", "")).strip().upper()
    name = str(meta.get("name", "")).strip().upper()
    if acronym in COUNTRY_COMPARISON_COLORS:
        return COUNTRY_COMPARISON_COLORS[acronym]
    if name in COUNTRY_COMPARISON_COLORS:
        return COUNTRY_COMPARISON_COLORS[name]
    return COUNTRY_COMPARISON_FALLBACK[(len(acronym) + len(name)) % len(COUNTRY_COMPARISON_FALLBACK)]


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def page_overview(countries):
    st.title("Demography Dashboard")
    st.markdown("Key indicators for all loaded countries. Data sourced from national statistics offices.")

    for c in countries:
        meta  = c["meta"]
        stats = c["stats"]
        fert  = c.get("fertility", {})

        with st.expander(f"**{country_label(meta)}**", expanded=True):
            cols = st.columns(6)
            t50 = stats.get("tipping_50")
            if t50:
                cols[0].metric("50 % tipping point", f"~{int(t50)}")
            ms = stats.get("migrant_share")
            if ms:
                cols[1].metric("Migrant share", f"{ms:.1%}")
            if fert.get("migrant"):
                cols[2].metric("Migrant TFR", f"{fert['migrant']:.2f}")
            if fert.get("native"):
                cols[3].metric("Native TFR", f"{fert['native']:.2f}")
            pop_all = stats.get("pop_all")
            if pop_all:
                cols[4].metric("Total population", fmt_k(pop_all))
            delta = stats.get("delta_all")
            if delta:
                cols[5].metric("Pop. Δ / year", fmt_k(delta), delta_color="normal")


# ---------------------------------------------------------------------------
# Page: Population Pyramid
# ---------------------------------------------------------------------------

def page_pyramid(countries):
    st.header("Population Pyramid")

    options = [c for c in countries if c.get("pop") is not None]
    if not options:
        st.info("No population age data available.")
        return

    choice_idx = st.selectbox("Country", list(range(len(options))), format_func=lambda i: country_label(options[i]["meta"]))
    choice = options[choice_idx]
    pop    = choice["pop"]
    meta   = choice["meta"]
    stats  = choice["stats"]
    ages   = pop.index.values

    # Optional 5-year grouping
    group5 = st.checkbox("Group into 5-year bands", value=False)
    if group5:
        pop = pop.copy()
        pop["band"] = (pop.index // 5) * 5
        pop = pop.groupby("band")[["native_f", "native_m", "migrant_f", "migrant_m"]].sum()
        ages = pop.index.values

    nf = pop["native_f"].fillna(0).values
    nm = pop["native_m"].fillna(0).values
    mf = pop["migrant_f"].fillna(0).values
    mm = pop["migrant_m"].fillna(0).values

    fig = go.Figure()
    # Males (negative x)
    fig.add_trace(go.Bar(
        y=ages, x=-nm, orientation="h", name="Native Male",
        marker_color=COLORS["native_m"], offsetgroup="male",
        customdata=nm, hovertemplate="Age %{y}<br>Native Male: %{customdata:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=ages, x=-mm, orientation="h", name="Migrant Male",
        marker_color=COLORS["migrant_m"], offsetgroup="male", base=-nm,
        customdata=mm, hovertemplate="Age %{y}<br>Migrant Male: %{customdata:,}<extra></extra>",
    ))
    # Females (positive x)
    fig.add_trace(go.Bar(
        y=ages, x=nf, orientation="h", name="Native Female",
        marker_color=COLORS["native_f"], offsetgroup="female",
        customdata=nf, hovertemplate="Age %{y}<br>Native Female: %{customdata:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=ages, x=mf, orientation="h", name="Migrant Female",
        marker_color=COLORS["migrant_f"], offsetgroup="female", base=nf,
        customdata=mf, hovertemplate="Age %{y}<br>Migrant Female: %{customdata:,}<extra></extra>",
    ))

    ms = stats.get("migrant_share", 0)
    fig.update_layout(
        title=f"Population Pyramid – {country_label(meta)}  |  migrant share {ms:.1%}",
        barmode="relative",
        bargap=0.05,
        xaxis=dict(title="Population", tickformat=",d",
                   tickvals=[-500_000, -250_000, 0, 250_000, 500_000],
                   ticktext=["500k", "250k", "0", "250k", "500k"]),
        yaxis=dict(title="Age", dtick=5),
        legend=dict(orientation="h", y=-0.12),
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tipping-point callouts
    t33 = stats.get("tipping_33")
    t50 = stats.get("tipping_50")
    if t33 or t50:
        c1, c2 = st.columns(2)
        if t33:
            c1.info(f"33 % migrant share ≈ **{int(t33)}**")
        if t50:
            c2.warning(f"50 % migrant share ≈ **{int(t50)}**")


# ---------------------------------------------------------------------------
# Page: Fertility
# ---------------------------------------------------------------------------

def page_fertility(countries):
    st.header("Fertility")

    fert_rows = [c for c in countries if c.get("fertility")]
    if not fert_rows:
        st.info("No fertility data available.")
        return

    labels = [country_label(c["meta"]) for c in fert_rows]
    n_tfr  = [c["fertility"].get("native",  0) for c in fert_rows]
    m_tfr  = [c["fertility"].get("migrant", 0) for c in fert_rows]
    fig_tfr = go.Figure()
    fig_tfr.add_trace(go.Bar(name="Native",  x=labels, y=n_tfr, marker_color=COLORS["native_f"]))
    fig_tfr.add_trace(go.Bar(name="Migrant", x=labels, y=m_tfr, marker_color=COLORS["migrant_f"]))
    fig_tfr.add_hline(y=2.1, line_dash="dash", line_color="gray",
                      annotation_text="Replacement (2.1)", annotation_position="top right")
    fig_tfr.update_layout(
        title="Total Fertility Rate (TFR): Native vs Migrant",
        barmode="group", yaxis_title="TFR", height=350,
    )
    st.plotly_chart(fig_tfr, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Births
# ---------------------------------------------------------------------------

def page_births(countries):
    st.header("Births")

    birth_opts = [c for c in countries if c.get("births") is not None and c.get("pop") is not None]
    if not birth_opts:
        st.info("No age-specific birth data available.")
        return

    st.subheader("Age-Specific Birth Rate")
    selected_indices = st.multiselect(
        "Countries", list(range(len(birth_opts))),
        default=list(range(min(2, len(birth_opts)))),
        format_func=lambda i: country_label(birth_opts[i]["meta"]),
    )
    selected = [birth_opts[i] for i in selected_indices]
    fig = go.Figure()
    for c in selected:
        color = get_country_color(c["meta"])
        births = c["births"]
        pop    = c["pop"]
        ages   = births.index.values
        nf_pop = pop["native_f"].reindex(ages).fillna(0).replace(0, np.nan)
        mf_pop = pop["migrant_f"].reindex(ages).fillna(0).replace(0, np.nan)
        nr = births["native_f"].fillna(0) / nf_pop * 1000
        mr = births["migrant_f"].fillna(0) / mf_pop * 1000
        lbl = c["meta"].get("acronym", "?")
        fig.add_trace(go.Scatter(x=ages, y=nr, mode="lines", name=f"{lbl} Native",
                                 line=dict(color=color, dash="solid", width=2)))
        fig.add_trace(go.Scatter(x=ages, y=mr, mode="lines", name=f"{lbl} Migrant",
                                 line=dict(color=color, dash="dash", width=2)))
    fig.update_layout(
        title="Births per 1,000 Women of That Age",
        xaxis_title="Age of mother", yaxis_title="Births per 1,000 women",
        legend=dict(orientation="h", y=-0.2), height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Deaths
# ---------------------------------------------------------------------------

def page_deaths(countries):
    st.header("Deaths by Age")

    opts = [c for c in countries if c.get("deaths") is not None and c.get("pop") is not None]
    if not opts:
        st.info("No death data available.")
        return

    selected_indices = st.multiselect(
        "Countries", list(range(len(opts))), default=list(range(min(2, len(opts)))),
        format_func=lambda i: country_label(opts[i]["meta"]),
    )
    selected = [opts[i] for i in selected_indices]
    groups = st.multiselect(
        "Groups", ["Native Female", "Native Male", "Migrant Female", "Migrant Male"],
        default=["Native Male", "Migrant Male"],
    )
    group_map = {
        "Native Female":  ("native_f",  "native_f"),
        "Native Male":    ("native_m",  "native_m"),
        "Migrant Female": ("migrant_f", "migrant_f"),
        "Migrant Male":   ("migrant_m", "migrant_m"),
    }

    fig = go.Figure()
    all_ages = np.concatenate([c["deaths"].index.values for c in selected]) if selected else np.array([70])
    for c in selected:
        color = get_country_color(c["meta"])
        deaths = c["deaths"]
        pop    = c["pop"]
        ages   = deaths.index.values
        lbl    = c["meta"].get("acronym", "?")
        for grp in groups:
            d_col, p_col = group_map[grp]
            d_vals = deaths[d_col].fillna(0) if d_col in deaths.columns else pd.Series(0, index=ages)
            p_vals = pop[p_col].reindex(ages).fillna(0).replace(0, np.nan) if p_col in pop.columns else None
            if p_vals is not None:
                rate = d_vals / p_vals * 100_000
            else:
                rate = d_vals
            dash = "solid" if "Native" in grp else "dash"
            fig.add_trace(go.Scatter(
                x=ages, y=rate, mode="lines", name=f"{lbl} – {grp}",
                line=dict(color=color, dash=dash, width=2),
            ))

    fig.update_layout(
        title="Deaths per 100,000 of Group",
        xaxis_title="Age", yaxis_title="Deaths per 100,000",
        legend=dict(orientation="h", y=-0.25), height=450,
    )
    if selected:
        fig.update_xaxes(range=[70, int(all_ages.max())])
        fig.update_yaxes(range=[0, 60000])
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Migration
# ---------------------------------------------------------------------------

def page_migration(countries):
    st.header("Net Migration by Age")

    opts = [c for c in countries if c.get("migration") is not None and c.get("pop") is not None]
    if not opts:
        st.info("No migration data available.")
        return

    selected_indices = st.multiselect(
        "Countries", list(range(len(opts))), default=list(range(min(1, len(opts)))),
        format_func=lambda i: country_label(opts[i]["meta"]),
    )
    selected = [opts[i] for i in selected_indices]
    normalize = st.checkbox("Normalize to per 1,000 of group", value=True)

    fig = go.Figure()
    for c in selected:
        color = get_country_color(c["meta"])
        mig  = c["migration"]
        pop  = c["pop"]
        ages = mig.index.values
        lbl  = c["meta"].get("acronym", "?")

        np_n = (pop["native_f"].reindex(ages).fillna(0) + pop["native_m"].reindex(ages).fillna(0)).replace(0, np.nan)
        np_m = (pop["migrant_f"].reindex(ages).fillna(0) + pop["migrant_m"].reindex(ages).fillna(0)).replace(0, np.nan)
        m_n  = mig["native_f"].fillna(0)  + mig["native_m"].fillna(0)
        m_m  = mig["migrant_f"].fillna(0) + mig["migrant_m"].fillna(0)

        yn = (m_n / np_n * 1000) if normalize else m_n
        ym = (m_m / np_m * 1000) if normalize else m_m

        fig.add_trace(go.Scatter(
            x=ages, y=yn, mode="lines", name=f"{lbl} Native",
            line=dict(color=color, dash="solid", width=2),
            fill="tozeroy", fillcolor=f"rgba(26,78,140,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=ages, y=ym, mode="lines", name=f"{lbl} Migrant",
            line=dict(color=color, dash="dash", width=2),
            fill="tozeroy", fillcolor=f"rgba(196,78,82,0.15)",
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
    ylabel = "Net migration per 1,000 of group" if normalize else "Net migration per year"
    fig.update_layout(
        title="Net Migration by Age",
        xaxis_title="Age", yaxis_title=ylabel,
        legend=dict(orientation="h", y=-0.2), height=430,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Origins
# ---------------------------------------------------------------------------

def _clean_country_name(name: str, iso2: str) -> str:
    import re
    name = re.sub(r"\[.*?\]", "", name).strip()
    name = name.replace(" (the)", "").strip()
    return ISO2_NAME_FIXES.get(iso2.upper(), name) if isinstance(iso2, str) else name


def page_origins(countries):
    st.header("Migrant Origins")

    opts = [c for c in countries if c.get("origin") is not None]
    if not opts:
        st.info("No origin data available.")
        return

    choice_idx = st.selectbox("Country", list(range(len(opts))), format_func=lambda i: country_label(opts[i]["meta"]))
    choice = opts[choice_idx]
    top_n  = 10
    view   = st.radio("View", ["Bar chart", "World map"], horizontal=True)

    origin = choice["origin"].copy()
    origin["country_clean"] = [
        _clean_country_name(r["country"], r["iso2"]) for _, r in origin.iterrows()
    ]
    # Filter out aggregated regions/continents (keep only 2-letter ISO2 codes),
    # but retain any 'Middle East' aggregate entry if present.
    mask = (
        origin["iso2"].astype(str).str.match(r'^[A-Za-z]{2}$', na=False)
        | origin["country_clean"].str.lower().str.contains(r"middle\s*east")
    )
    origin = origin[mask].head(top_n).copy()

    if view == "Bar chart":
        origin_sorted = origin.sort_values("total", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=origin_sorted["country_clean"], x=origin_sorted["total"],
            orientation="h", name="Migrants", marker_color=COLORS["migrant_m"],
        ))
        fig.update_layout(
            title=f"Top {top_n} Migrant Origins – {country_label(choice['meta'])}",
            barmode="stack", xaxis_title="Number of migrants",
            height=max(400, top_n * 22),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # World map
        map_df = origin.copy()
        map_df["pct"] = map_df["total"] / map_df["total"].sum() * 100
        fig = px.choropleth(
            map_df,
            locations="country_clean",
            locationmode="country names",
            color="pct",
            hover_name="country_clean",
            hover_data={"total": ":,", "pct": ":.2f"},
            color_continuous_scale="Oranges",
            title=f"Migrant Origins – {country_label(choice['meta'])} (% of migrant stock)",
        )
        fig.update_layout(height=520, coloraxis_colorbar_title="% share")
        st.plotly_chart(fig, use_container_width=True)

    # Comparison: origins across all countries
    if len(opts) >= 2:
        st.divider()
        st.subheader("Common Origins: Cross-Country Comparison")
        top_cmp = 10
        pct_dfs = {}
        iso_to_name = {}
        for c in opts:
            df = c["origin"].copy()
            # Filter out aggregated regions/continents but keep Middle East
            mask = (
                df["iso2"].astype(str).str.match(r'^[A-Za-z]{2}$', na=False)
                | df["country"].astype(str).str.lower().str.contains(r"middle\s*east")
            )
            df = df[mask]
            iso_to_name.update(dict(zip(df["iso2"], df["country"])))
            if df.empty:
                continue
            agg = df.groupby("iso2")["total"].sum()
            if agg.sum() == 0:
                continue
            pct_dfs[c["meta"]["acronym"]] = (agg / agg.sum() * 100)
        combined = pd.concat(pct_dfs.values(), axis=1, keys=pct_dfs.keys()).dropna()
        if not combined.empty:
            combined["avg"] = combined.mean(axis=1)
            combined = combined.nlargest(top_cmp, "avg").drop(columns="avg")
            combined.index = [
                _clean_country_name(iso_to_name.get(iso, iso), iso) for iso in combined.index
            ]
            combined = combined.sort_values(combined.columns[0], ascending=True)
            fig2 = go.Figure()
            for i, col in enumerate(combined.columns):
                pal = COUNTRY_COLORS[i % len(COUNTRY_COLORS)]
                fig2.add_trace(go.Bar(
                    y=combined.index, x=combined[col],
                    orientation="h", name=col, marker_color=pal["native"],
                ))
            fig2.update_layout(
                title=f"Top {top_cmp} Common Origins (% of each country's migrant stock)",
                barmode="group",
                xaxis_title="% of migrant stock",
                height=max(400, top_cmp * 28),
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Naturalizations
# ---------------------------------------------------------------------------

def page_naturalizations(countries):
    st.header("Naturalizations")

    opts = [c for c in countries if c.get("patria") is not None]
    if not opts:
        st.info("No naturalization data available.")
        return

    selected_indices = st.multiselect(
        "Countries", list(range(len(opts))), default=list(range(len(opts))),
        format_func=lambda i: country_label(opts[i]["meta"]),
    )
    selected = [opts[i] for i in selected_indices]
    if not selected:
        return

    # Determine common year range across selected
    all_years = []
    for c in selected:
        all_years.extend(c["patria"].index.tolist())
    yr_min, yr_max = int(min(all_years)), int(max(all_years))
    yr_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    normalize = st.checkbox("Normalize to per 1,000 of native population", value=False)

    maxima = [c["patria"]["total"].max() for c in selected]
    use_dual = len(selected) == 2 and (max(maxima) / min(maxima)) > 5

    if use_dual:
        fig = go.Figure()
        for i, c in enumerate(selected):
            color = get_country_color(c["meta"])
            pat = c["patria"]
            pat = pat[(pat.index >= yr_range[0]) & (pat.index <= yr_range[1])]
            if normalize and c["stats"].get("pop_native"):
                y = pat["total"] / c["stats"]["pop_native"] * 1000
                ytitle = "Naturalizations per 1,000 natives"
            else:
                y = pat["total"]
                ytitle = "Naturalizations per year"
            yaxis = "y" if i == 0 else "y2"
            fig.add_trace(go.Scatter(
                x=pat.index, y=y, mode="lines", name=country_label(c["meta"]),
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba(26,78,140,{0.12 if i == 0 else 0.08})",
                yaxis=yaxis,
            ))
        c0_color = get_country_color(selected[0]["meta"])
        c1_color = get_country_color(selected[1]["meta"])
        fig.update_layout(
            title="Naturalizations – Dual Axis (large scale difference)",
            xaxis_title="Year",
            yaxis=dict(title=country_label(selected[0]["meta"]),
                       titlefont_color=c0_color,
                       tickfont_color=c0_color),
            yaxis2=dict(title=country_label(selected[1]["meta"]),
                        titlefont_color=c1_color,
                        tickfont_color=c1_color,
                        overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.15),
            height=430,
        )
    else:
        fig = go.Figure()
        for i, c in enumerate(selected):
            color = get_country_color(c["meta"])
            pat = c["patria"]
            pat = pat[(pat.index >= yr_range[0]) & (pat.index <= yr_range[1])]
            if normalize and c["stats"].get("pop_native"):
                y = pat["total"] / c["stats"]["pop_native"] * 1000
            else:
                y = pat["total"]
            fig.add_trace(go.Scatter(
                x=pat.index, y=y, mode="lines", name=country_label(c["meta"]),
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba(26,78,140,{0.12 if i == 0 else 0.08})",
            ))
        ytitle = "Naturalizations per 1,000 natives" if normalize else "Naturalizations per year"
        fig.update_layout(
            title="Naturalizations per Year",
            xaxis_title="Year", yaxis_title=ytitle,
            legend=dict(orientation="h", y=-0.15), height=430,
        )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Comparisons
# ---------------------------------------------------------------------------

def page_comparisons(countries):
    st.header("Country Comparisons")

    metric = st.selectbox("Metric", [
        "Total population (native vs migrant)",
        "Migrant share by age",
        "Age-specific fertility rate",
        "Death rate by age",
        "Net migration rate by age",
    ])

    if metric == "Total population (native vs migrant)":
        rows = [c for c in countries if c["stats"].get("pop_native")]
        if not rows:
            st.info("No data.")
            return
        labels   = [country_label(c["meta"]) for c in rows]
        natives  = [c["stats"]["pop_native"] for c in rows]
        migrants = [c["stats"].get("pop_migrant", 0) for c in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Native",  x=labels, y=natives,  marker_color=COLORS["native_m"]))
        fig.add_trace(go.Bar(name="Migrant", x=labels, y=migrants, marker_color=COLORS["migrant_m"]))
        for i, (n, m) in enumerate(zip(natives, migrants)):
            fig.add_annotation(x=labels[i], y=n + m, text=f"{m/(n+m):.1%}",
                               showarrow=False, yshift=10, font=dict(color=COLORS["migrant_m"]))
        fig.update_layout(
            barmode="stack", title="Population: Native vs Migrant",
            yaxis_title="Population", height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif metric == "Migrant share by age":
        rows = [c for c in countries if c.get("pop") is not None]
        if len(rows) < 2:
            st.info("Need at least 2 countries with population data.")
            return
        fig = go.Figure()
        for i, c in enumerate(rows):
            color = get_country_color(c["meta"])
            pop  = c["pop"]
            total   = (pop["native_f"].fillna(0) + pop["native_m"].fillna(0)
                       + pop["migrant_f"].fillna(0) + pop["migrant_m"].fillna(0))
            migrant = pop["migrant_f"].fillna(0) + pop["migrant_m"].fillna(0)
            share   = migrant / total.replace(0, np.nan) * 100
            dash = "dash" if i > 0 else "solid"
            fig.add_trace(go.Scatter(
                x=share.index, y=share.values, mode="lines",
                name=country_label(c["meta"]),
                line=dict(color=color, dash=dash, width=2),
            ))
        fig.update_layout(
            title="Migrant Share by Age (%)",
            xaxis_title="Age", yaxis_title="Migrant share (%)", height=430,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif metric == "Age-specific fertility rate":
        rows = [c for c in countries if c.get("births") is not None and c.get("pop") is not None]
        if not rows:
            st.info("No fertility data.")
            return
        fig = go.Figure()
        for i, c in enumerate(rows):
            color = get_country_color(c["meta"])
            births = c["births"]
            pop    = c["pop"]
            ages   = births.index.values
            nf_pop = pop["native_f"].reindex(ages).fillna(0).replace(0, np.nan)
            mf_pop = pop["migrant_f"].reindex(ages).fillna(0).replace(0, np.nan)
            nr = births["native_f"].fillna(0) / nf_pop * 1000
            mr = births["migrant_f"].fillna(0) / mf_pop * 1000
            lbl  = c["meta"].get("acronym", "?")
            fig.add_trace(go.Scatter(x=ages, y=nr, mode="lines", name=f"{lbl} Native",
                                     line=dict(color=color, dash="solid", width=2)))
            fig.add_trace(go.Scatter(x=ages, y=mr, mode="lines", name=f"{lbl} Migrant",
                                     line=dict(color=color, dash="dash", width=2)))
        fig.update_layout(
            title="Births per 1,000 Women of That Age",
            xaxis_title="Age of mother", yaxis_title="Births / 1,000 women",
            height=430,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif metric == "Death rate by age":
        rows = [c for c in countries if c.get("deaths") is not None and c.get("pop") is not None]
        if not rows:
            st.info("No death data.")
            return
        fig = go.Figure()
        for i, c in enumerate(rows):
            color = get_country_color(c["meta"])
            deaths = c["deaths"]
            pop    = c["pop"]
            ages   = deaths.index.values
            np_n = (pop["native_f"].reindex(ages).fillna(0) + pop["native_m"].reindex(ages).fillna(0)).replace(0, np.nan)
            np_m = (pop["migrant_f"].reindex(ages).fillna(0) + pop["migrant_m"].reindex(ages).fillna(0)).replace(0, np.nan)
            d_n  = deaths["native_f"].fillna(0) + deaths["native_m"].fillna(0)
            d_m  = deaths["migrant_f"].fillna(0) + deaths["migrant_m"].fillna(0)
            lbl  = c["meta"].get("acronym", "?")
            fig.add_trace(go.Scatter(x=ages, y=d_n / np_n * 100_000, mode="lines",
                                     name=f"{lbl} Native",
                                     line=dict(color=color, dash="solid", width=2)))
            fig.add_trace(go.Scatter(x=ages, y=d_m / np_m * 100_000, mode="lines",
                                     name=f"{lbl} Migrant",
                                     line=dict(color=color, dash="dash", width=2)))
        fig.update_layout(
            title="Deaths per 100,000 of Group",
            xaxis_title="Age", yaxis_title="Deaths / 100,000",
            height=430,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif metric == "Net migration rate by age":
        rows = [c for c in countries if c.get("migration") is not None and c.get("pop") is not None]
        if not rows:
            st.info("No migration data.")
            return
        fig = go.Figure()
        for i, c in enumerate(rows):
            color = get_country_color(c["meta"])
            mig  = c["migration"]
            pop  = c["pop"]
            ages = mig.index.values
            np_n = (pop["native_f"].reindex(ages).fillna(0) + pop["native_m"].reindex(ages).fillna(0)).replace(0, np.nan)
            np_m = (pop["migrant_f"].reindex(ages).fillna(0) + pop["migrant_m"].reindex(ages).fillna(0)).replace(0, np.nan)
            m_n  = mig["native_f"].fillna(0) + mig["native_m"].fillna(0)
            m_m  = mig["migrant_f"].fillna(0) + mig["migrant_m"].fillna(0)
            lbl  = c["meta"].get("acronym", "?")
            fig.add_trace(go.Scatter(x=ages, y=m_n / np_n * 1000, mode="lines",
                                     name=f"{lbl} Native",
                                     line=dict(color=color, dash="solid", width=2)))
            fig.add_trace(go.Scatter(x=ages, y=m_m / np_m * 1000, mode="lines",
                                     name=f"{lbl} Migrant",
                                     line=dict(color=color, dash="dash", width=2)))
        fig.add_hline(y=0, line_color="black", line_width=0.8, line_dash="dot")
        fig.update_layout(
            title="Net Migration per 1,000 of Group",
            xaxis_title="Age", yaxis_title="Net migration / 1,000",
            height=430,
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAGES = {
    "Overview":         page_overview,
    "Population Pyramid": page_pyramid,
    "Births":           page_births,
    "Fertility":        page_fertility,
    "Deaths":           page_deaths,
    "Migration":        page_migration,
    "Origins":          page_origins,
    "Naturalizations":  page_naturalizations,
    "Comparisons":      page_comparisons,
}


def main():
    st.set_page_config(
        page_title="Demography Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("📊 Demography")
        st.caption("Demographic indicators for migrants vs natives")
        st.divider()
        page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()

    countries = load_data()

    if not countries:
        st.error(f"No country Excel files found in `{DATA_DIR}`. "
                 "Place `*.xlsx` files there and reload.")
        return

    fn = PAGES[page]
    fn(countries)


if __name__ == "__main__":
    main()
