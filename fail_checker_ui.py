import math
import re
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Flood-Control AI Labelled Checker (FAIL Checker) Model", layout="wide")

# ─────────────────────────────────────────────────────────────
# Utilities (no-index table helper)
# ─────────────────────────────────────────────────────────────
def show_table(df: pd.DataFrame):
    st.dataframe(df.reset_index(drop=True))

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()

PHASE_RE = re.compile(r"\b(phase|package|lot|stage|section)\b", flags=re.I)
STOP = {
    "the","and","of","in","to","for","with","at","on","by","as","is","a","an","from","that","this","or",
    "flood","structure","construction","river","dike","project","works","improvement","rehabilitation",
    "barangay","city","province","municipality","drainage","control","bank","protection","road","bridge"
}

def has_phase_kw(text: str) -> bool:
    if not isinstance(text, str): return False
    return bool(PHASE_RE.search(text))

def desc_similar(a: str, b: str, min_common=3) -> bool:
    A = set(normalize_text(a).split()) - STOP
    B = set(normalize_text(b).split()) - STOP
    return len(A & B) >= int(min_common)

def parse_date_series(s: pd.Series):
    if s is None:
        return pd.Series([pd.NaT] * 0)
    s = (
        s.astype(str)
        .replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "None": np.nan, "null": np.nan})
    )
    parsed = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    remain_mask = parsed.isna() & s.notna()
    if remain_mask.any():
        parsed.loc[remain_mask] = pd.to_datetime(s[remain_mask], errors="coerce", dayfirst=False)
    return parsed

def haversine_m(lat1, lon1, lat2, lon2):
    try:
        if any(pd.isna([lat1, lon1, lat2, lon2])):
            return np.nan
        R = 6371000.0
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlmb = math.radians(float(lon2) - float(lon1))
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception:
        return np.nan

def peso(n):
    try:
        return f"₱{n/1e9:,.2f}B" if abs(n) >= 1e9 else f"₱{n/1e6:,.2f}M"
    except Exception:
        return "₱0.00"

def parse_cost(s):
    if pd.isna(s): return np.nan
    t = str(s)
    t = re.sub(r"[₱,]", "", t).strip()
    try:
        return float(t)
    except Exception:
        return pd.to_numeric(t, errors="coerce")

def _rounder(v, n):
    try: return round(float(v), n)
    except Exception: return np.nan

# ─────────────────────────────────────────────────────────────
# v23.3.1 RULES (portable versions)
# ─────────────────────────────────────────────────────────────
def _norm_province(s):
    s = normalize_text(s)
    s = s.replace("ncr", "national capital region")
    s = s.replace("metro manila", "national capital region")
    return s

def _norm_city(s):
    s = normalize_text(s)
    s = s.replace(" city", "")
    s = s.replace("quezon city", "quezon")
    return s

def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates
    for col in ["CompletionDateActual", "StartDate", "CompletionDateOriginal"]:
        if col not in df.columns:
            df[col] = pd.NaT
        df[col] = parse_date_series(df[col])

    # Coords
    for c in ["Latitude", "Longitude"]:
        if c not in df.columns:
            df[c] = np.nan
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Text fields for safety
    if "ProjectDescription" not in df.columns: df["ProjectDescription"] = ""
    if "ProjectTitle" not in df.columns: df["ProjectTitle"] = ""

    # Cost and Contractor
    if "ContractCost" not in df.columns: df["ContractCost"] = np.nan
    df["ContractCost"] = df["ContractCost"].map(parse_cost)

    if "Contractor" not in df.columns: df["Contractor"] = "Unknown"
    df["Contractor"] = df["Contractor"].fillna("Unknown").astype(str).str.strip()

    # Location
    for loc in ["Region", "Province", "Municipality"]:
        if loc not in df.columns: df[loc] = ""

    df["Province_norm"] = df["Province"].map(_norm_province)
    df["Municipality_norm"] = df["Municipality"].map(_norm_city)

    df["location_key_norm"] = (
        df["Region"].fillna("").map(normalize_text) + "_" +
        df["Province_norm"].fillna("") + "_" +
        df["Municipality_norm"].fillna("")
    ).str.strip("_")

    # Years
    df["CompletionYear"] = df["CompletionDateActual"].dt.year
    df["StartDateYear"] = df["StartDate"].dt.year
    if "InfraYear" in df.columns and not pd.api.types.is_numeric_dtype(df["InfraYear"]):
        df["InfraYear"] = pd.to_numeric(df["InfraYear"], errors="coerce")
    df["InfraYear"] = df.get("InfraYear", pd.Series(index=df.index, dtype="float")) \
                      .fillna(df["CompletionYear"]).fillna(df["StartDateYear"])

    # Durations
    df["duration_days"] = (df["CompletionDateActual"] - df["StartDate"]).dt.days
    df["ext_days"] = (df["CompletionDateActual"] - df["CompletionDateOriginal"]).dt.days
    df["crosses_calendar_year"] = (
        df["StartDate"].dt.year.notna()
        & df["CompletionDateActual"].dt.year.notna()
        & (df["CompletionDateActual"].dt.year > df["StartDate"].dt.year)
    )

    # Rounded coords
    df["_lat_rd5"] = df["Latitude"].apply(lambda x: _rounder(x, 5))
    df["_lon_rd5"] = df["Longitude"].apply(lambda x: _rounder(x, 5))
    return df

def compute_chop_chop(df, proximity_m=600, soft_proximity_m=1000, min_common_words=3):
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    reasons = np.array([""] * n, dtype=object)
    for _, g in df.groupby(["location_key_norm", "InfraYear"]):
        idxs = g.index.to_list()
        if len(idxs) < 2: continue
        for i, j in combinations(idxs, 2):
            lat1, lon1 = df.at[i, "Latitude"], df.at[i, "Longitude"]
            lat2, lon2 = df.at[j, "Latitude"], df.at[j, "Longitude"]
            dist_m = haversine_m(lat1, lon1, lat2, lon2)

            trig_round = (df.at[i, "_lat_rd5"] == df.at[j, "_lat_rd5"]) and (df.at[i, "_lon_rd5"] == df.at[j, "_lon_rd5"])
            trig_prox  = pd.notna(dist_m) and dist_m <= proximity_m
            trig_soft  = pd.notna(dist_m) and dist_m <= soft_proximity_m

            d1 = f"{df.at[i, 'ProjectTitle']} {df.at[i, 'ProjectDescription']}"
            d2 = f"{df.at[j, 'ProjectTitle']} {df.at[j, 'ProjectDescription']}"
            sim_desc   = desc_similar(d1, d2, min_common=min_common_words)
            phase_hint = has_phase_kw(d1) or has_phase_kw(d2)

            pair_split = trig_round or trig_prox or (trig_soft and (phase_hint or sim_desc)) or (sim_desc and phase_hint)
            if pair_split:
                mask[i] = True; mask[j] = True
    reasons[mask] = "Proximity/rounded match within location & InfraYear, or phase/similar description"
    return mask, reasons

def compute_doppelganger(
    df,
    tolerance_pct=0.02,
    min_common_words=1,
    allow_exact_cost_across_years=True,
    year_lag=1,
    deo_column_candidates=("DEO","DistrictEngineeringOffice","ImplementingOffice","ProcuringEntity")
):
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    reasons = np.array([""] * n, dtype=object)

    def near_equal(a, b, tol=0.02):
        if pd.isna(a) or pd.isna(b): return False
        denom = max(abs(a), abs(b), 1.0)
        return abs(a - b) / denom <= tol

    for _, g in df.groupby(["location_key_norm", "InfraYear"]):
        idxs = g.index.to_list()
        if len(idxs) < 2: continue
        for i, j in combinations(idxs, 2):
            c1, c2 = df.at[i, "ContractCost"], df.at[j, "ContractCost"]
            if not near_equal(c1, c2, tolerance_pct): 
                continue
            d1 = f"{df.at[i, 'ProjectTitle']} {df.at[i, 'ProjectDescription']}"
            d2 = f"{df.at[j, 'ProjectTitle']} {df.at[j, 'ProjectDescription']}"
            if desc_similar(d1, d2, min_common=min_common_words):
                mask[i] = True; mask[j] = True
                if not reasons[i]: reasons[i] = "Near-identical cost (±2%) + similar description within same area/year"
                if not reasons[j]: reasons[j] = "Near-identical cost (±2%) + similar description within same area/year"

    if allow_exact_cost_across_years:
        base_year = df["InfraYear"].fillna(df["CompletionYear"])
        for _, g in df.groupby(["Province_norm", "Municipality_norm"]):
            idxs = g.index.to_list()
            if len(idxs) < 2: continue
            for _, samecost in g.groupby("ContractCost"):
                sc_idxs = samecost.index.to_list()
                if len(sc_idxs) < 2: continue
                years = base_year.loc[sc_idxs]
                for i, j in combinations(sc_idxs, 2):
                    yi, yj = years.loc[i], years.loc[j]
                    ok_year = (pd.notna(yi) and pd.notna(yj) and abs(int(yi) - int(yj)) <= int(year_lag)) or (pd.isna(yi) or pd.isna(yj))
                    if not ok_year:
                        continue
                    same_rd = (df.at[i, "_lat_rd5"] == df.at[j, "_lat_rd5"]) and (df.at[i, "_lon_rd5"] == df.at[j, "_lon_rd5"])
                    close_xy = False
                    if pd.notna(df.at[i, "Latitude"]) and pd.notna(df.at[i, "Longitude"]) and \
                       pd.notna(df.at[j, "Latitude"]) and pd.notna(df.at[j, "Longitude"]):
                        d_m = haversine_m(df.at[i, "Latitude"], df.at[i, "Longitude"], df.at[j, "Latitude"], df.at[j, "Longitude"])
                        close_xy = (pd.notna(d_m) and d_m <= 1500)
                    if same_rd or close_xy:
                        mask[i] = True; mask[j] = True
                        if not reasons[i]: reasons[i] = "Exact-cost twin in same city+province + same/near coords (±year)"
                        if not reasons[j]: reasons[j] = "Exact-cost twin in same city+province + same/near coords (±year)"

    deo_col = None
    for c in deo_column_candidates:
        if c in df.columns:
            deo_col = c
            break
    if deo_col is not None:
        deo_norm = df[deo_col].fillna("").astype(str).str.strip().str.lower()
        key_df = pd.DataFrame({
            "Contractor": df["Contractor"].fillna("").astype(str).str.strip().str.lower(),
            "DEO": deo_norm,
            "StartDate": df["StartDate"],
            "CompletionDateActual": df["CompletionDateActual"]
        })
        for _, g in key_df.groupby(["Contractor","DEO","StartDate","CompletionDateActual"]):
            idxs = g.index.to_list()
            if len(idxs) >= 2:
                for i, j in combinations(idxs, 2):
                    mask[i] = True; mask[j] = True
                    if not reasons[i]: reasons[i] = "Same contractor+DEO and identical Start/Completion dates"
                    if not reasons[j]: reasons[j] = "Same contractor+DEO and identical Start/Completion dates"

    return mask, reasons

def _group_quantile_thresholds(df, q=0.20, floor_days=45, cap_days=180, min_group=20, group_keys=("Province_norm","InfraYear")):
    pos = df.loc[df["duration_days"].notna() & (df["duration_days"] > 0), ["duration_days", *group_keys]].copy()
    national_q = float(pos["duration_days"].quantile(q)) if len(pos) else 60.0
    prov_map = {}
    if "Province_norm" in group_keys and len(pos):
        prov_map = pos.groupby("Province_norm")["duration_days"].quantile(q).to_dict()
    thresh_map = {}
    if len(group_keys) > 0 and len(pos):
        grp = pos.groupby(list(group_keys))
        for key, sub in grp:
            if len(sub) >= min_group:
                thresh = float(sub["duration_days"].quantile(q))
            else:
                province = key[0] if isinstance(key, tuple) else key
                thresh = prov_map.get(province, national_q)
            thresh = max(floor_days, min(cap_days, thresh))
            thresh_map[key] = thresh
    return thresh_map, national_q

def compute_potentially_ghost(
    df,
    q=0.20, floor_days=45, cap_days=180, min_group=20,
    contractor_volume_threshold=5,
    group_keys=("Province_norm","InfraYear")
):
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    reasons = np.array([""] * n, dtype=object)

    contractor_counts = df["Contractor"].fillna("Unknown").astype(str).value_counts()
    high_volume = df["Contractor"].fillna("Unknown").astype(str).map(
        lambda x: contractor_counts.get(x, 0) > contractor_volume_threshold
    )

    thresh_map, national_q = _group_quantile_thresholds(
        df, q=q, floor_days=floor_days, cap_days=cap_days, min_group=min_group, group_keys=group_keys
    )
    if isinstance(group_keys, (list, tuple)) and len(group_keys) > 0:
        keys_df = df[list(group_keys)].copy()
        keys_tuples = [tuple(row) if len(group_keys) > 1 else row[0] for row in keys_df.values]
    else:
        keys_tuples = ["all"] * len(df)

    dur = df["duration_days"]

    cond_A = dur.notna() & (dur < 0)
    designed_from_start = (df["CompletionDateOriginal"] - df["StartDate"]).dt.days
    cond_B = designed_from_start.notna() & (designed_from_start <= 30)
    cond_C = dur.notna() & (dur <= 30)
    cond_D = dur.notna() & (dur <= 30) & high_volume.values
    designed_days = (df["CompletionDateOriginal"] - df["StartDate"]).dt.days
    cond_E = dur.notna() & (dur < 0.75 * designed_days)

    short_vs_peers = np.zeros(n, dtype=bool)
    for i in range(n):
        d = dur.iloc[i]
        if pd.isna(d) or d <= 0:
            short_vs_peers[i] = False
            continue
        key = keys_tuples[i]
        used_thresh = thresh_map.get(key, max(floor_days, min(cap_days, national_q)))
        short_vs_peers[i] = (d < used_thresh)
    cond_F = short_vs_peers & high_volume.values

    mask = cond_A | cond_B | cond_C | cond_D | cond_E | cond_F

    for i in range(n):
        if cond_A[i]:
            reasons[i] = "Negative duration (CompletionDateActual earlier than StartDate)"
        elif cond_B[i]:
            reasons[i] = "Very short design schedule: (CompletionDateOriginal − StartDate) ≤ 30 days"
        elif cond_C[i]:
            reasons[i] = "Very short actual duration: (CompletionDateActual − StartDate) ≤ 30 days"
        elif cond_D[i]:
            reasons[i] = "Very short duration (≤30 days) with high-volume contractor"
        elif cond_E[i]:
            reasons[i] = "Too fast vs designed: duration < 75% of (CompletionDateOriginal − StartDate)"
        elif cond_F[i]:
            key = keys_tuples[i]
            used_thresh = thresh_map.get(key, max(floor_days, min(cap_days, national_q)))
            reasons[i] = f"Abnormally short vs peers (< {int(used_thresh)} d for group) with high contractor load"

    return mask, reasons

def compute_siyam_siyam(df):
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    reasons = np.array([""] * n, dtype=object)
    ext_ok = df["ext_days"].notna() & (df["ext_days"] > 365)
    dur_ok = df["duration_days"].notna() & (df["duration_days"] > 365)
    cal_cross = df["crosses_calendar_year"].fillna(False)
    mask = ext_ok | dur_ok | cal_cross
    for i in df.index:
        rs = []
        if bool(ext_ok.iloc[i]): rs.append("Actual > Original by > 1 year")
        if bool(dur_ok.iloc[i]): rs.append("Project duration > 1 year")
        if bool(cal_cross.iloc[i]): rs.append("Crosses calendar year boundary from StartDate")
        reasons[i] = "; ".join(rs)
    return mask, reasons

def attach_category_labels(df, chop_mask, dop_mask, ghost_mask, siyam_mask):
    both = ghost_mask & siyam_mask
    if both.any():
        ghost_mask[both] = False
    green_mask = ~(chop_mask | dop_mask | ghost_mask | siyam_mask)

    def compose_labels(i):
        labs = []
        if chop_mask[i]: labs.append("Chop_chop_Project")
        if dop_mask[i]:  labs.append("Doppelganger_Project")
        if ghost_mask[i]: labs.append("Ghost_Project")
        if siyam_mask[i]: labs.append("Siyam-siyam_Project")
        if not labs: labs = ["Green_Flag"]
        return ";".join(labs)

    df["Project_Category"] = [compose_labels(i) for i in range(len(df))]
    return df, green_mask

def _category_color(cat: str) -> str:
    priority = [
        ("Ghost_Project", "#e74c3c"),
        ("Siyam-siyam_Project", "#f1c40f"),
        ("Doppelganger_Project", "#9b59b6"),
        ("Chop_chop_Project", "#e67e22"),
        ("Green_Flag", "#2ecc71"),
    ]
    cat = str(cat) if pd.notna(cat) else ""
    for label, color in priority:
        if label in cat:
            return color
    return "#95a5a6"

# ─────────────────────────────────────────────────────────────
# Sidebar: reference dataset (optional) & params
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Reference Dataset (optional)")
ref_file = st.sidebar.file_uploader("Upload CSV/XLSX for cross-record checks", type=["csv", "xlsx"])
st.sidebar.caption("Uploading a dataset enables Chop-chop & Doppelganger checks and stronger Ghost peer thresholds.")


# ─────────────────────────────────────────────────────────────
# Main form – Single project input
# ─────────────────────────────────────────────────────────────
st.title("Flood-Control AI Labelled Checker (FAIL Checker)")

with st.form("single_project_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        project_id = st.text_input("ProjectID / ContractID", "")
        contractor = st.text_input("Contractor", "")
        contract_cost = st.text_input("Contract Cost (₱)", "")
        infra_year = st.number_input("InfraYear", min_value=1990, max_value=2100, value=2024)

    with c2:
        region = st.text_input("Region", "")
        province = st.text_input("Province", "")
        municipality = st.text_input("Municipality/City", "")
        deo = st.text_input("District Engineering Office (DEO)", "")

    with c3:
        latitude = st.number_input("Latitude", value=0.0, format="%.6f")
        longitude = st.number_input("Longitude", value=0.0, format="%.6f")
        start_date = st.date_input("StartDate")
        comp_orig = st.date_input("CompletionDateOriginal")
        comp_actual = st.date_input("CompletionDateActual")

    st.text_area("Project Title", "", key="title")
    st.text_area("Project Description", "", key="desc")

    # Extra manual inputs (to support no-dataset cases)
    st.markdown("**Additional (if no dataset uploaded):**")
    c4, c5 = st.columns(2)
    with c4:
        contractor_total_projects = st.number_input("Contractor total DPWH projects (known/estimate)", min_value=0, value=0)
    with c5:
        peer_p20_days = st.number_input("Peer P20 threshold for Province/Year (days, if known)", min_value=1, value=60)

    submitted = st.form_submit_button("Tag this project")

# ─────────────────────────────────────────────────────────────
# Build input DataFrame and (optionally) reference dataset
# ─────────────────────────────────────────────────────────────
def load_ref_df(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        else:
            return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to load reference dataset: {e}")
        return None

ref_df_raw = load_ref_df(ref_file)
if ref_df_raw is not None:
    st.success("Reference dataset loaded.")
    # show small preview
    with st.expander("Preview reference data (first 15 rows)", expanded=False):
        show_table(ref_df_raw.head(15))

# Input row
if submitted:
    one = pd.DataFrame([{
        "ProjectID": project_id,
        "ContractID": project_id if not project_id else project_id,
        "Contractor": contractor,
        "ContractCost": contract_cost,
        "InfraYear": infra_year,
        "Region": region,
        "Province": province,
        "Municipality": municipality,
        "DistrictEngineeringOffice": deo,
        "Latitude": latitude,
        "Longitude": longitude,
        "StartDate": pd.to_datetime(str(start_date)),
        "CompletionDateOriginal": pd.to_datetime(str(comp_orig)),
        "CompletionDateActual": pd.to_datetime(str(comp_actual)),
        "ProjectTitle": st.session_state.get("title", ""),
        "ProjectDescription": st.session_state.get("desc", "")
    }])

    # If a reference dataset exists, append and run full checks.
    if ref_df_raw is not None and len(ref_df_raw) > 0:
        base = ref_df_raw.copy()
        # Make sure critical columns exist in base (so derived ops don't fail)
        for need in ["Contractor","ContractCost","InfraYear","Region","Province","Municipality",
                     "DistrictEngineeringOffice","Latitude","Longitude",
                     "StartDate","CompletionDateOriginal","CompletionDateActual",
                     "ProjectTitle","ProjectDescription"]:
            if need not in base.columns:
                base[need] = np.nan

        df = pd.concat([base, one], ignore_index=True)
        df = add_derived_cols(df)

        # Compute indicators
        chop_mask, chop_reason   = compute_chop_chop(df)
        dop_mask,  dop_reason    = compute_doppelganger(df)
        ghost_mask, ghost_reason = compute_potentially_ghost(
            df,
            q=ghost_q, floor_days=ghost_floor, cap_days=ghost_cap, min_group=ghost_min_group,
            contractor_volume_threshold=contractor_volume_threshold,
            group_keys=("Province_norm","InfraYear")
        )
        siyam_mask, siyam_reason = compute_siyam_siyam(df)

        # Attach categories
        df, green_mask = attach_category_labels(df, chop_mask, dop_mask, ghost_mask, siyam_mask)

        # Take the last row (the user’s project)
        row = df.tail(1).copy()
        category = row.iloc[0]["Project_Category"]
        color = _category_color(category)

        # Reasons summary for user's row
        i = row.index[0]
        reasons = []
        if chop_mask[i]: reasons.append(f"Chop-chop: {chop_reason[i]}")
        if dop_mask[i]: reasons.append(f"Doppelganger: {dop_reason[i]}")
        if ghost_mask[i]: reasons.append(f"Ghost: {ghost_reason[i]}")
        if siyam_mask[i]: reasons.append(f"Siyam-siyam: {siyam_reason[i]}")
        if not reasons: reasons.append("Residual Green Flag (no anomaly conditions triggered)")

    else:
        # No reference dataset: compute what we can from the single row
        df = add_derived_cols(one)

        # Approximate Ghost logic using manual inputs:
        # - Use contractor_total_projects vs threshold
        # - Use provided peer_p20_days as threshold for (F)
        dur = df["duration_days"].iloc[0]
        designed_days = (df["CompletionDateOriginal"].iloc[0] - df["StartDate"].iloc[0]).days if \
                        pd.notna(df["CompletionDateOriginal"].iloc[0]) and pd.notna(df["StartDate"].iloc[0]) else np.nan

        # Conditions
        cond_A = pd.notna(dur) and (dur < 0)
        cond_B = pd.notna(designed_days) and (designed_days <= 30)
        cond_C = pd.notna(dur) and (dur <= 30)
        high_vol = contractor_total_projects > contractor_volume_threshold
        cond_D = pd.notna(dur) and (dur <= 30) and high_vol
        cond_E = (pd.notna(dur) and pd.notna(designed_days) and (dur < 0.75 * designed_days))
        cond_F = (pd.notna(dur) and (dur > 0) and high_vol and (dur < max(ghost_floor, min(ghost_cap, peer_p20_days))))

        is_ghost = cond_A or cond_B or cond_C or cond_D or cond_E or cond_F

        # Siyam-siyam
        ext_days = df["ext_days"].iloc[0]
        crosses_cal = bool(df["crosses_calendar_year"].iloc[0])
        cond_sy1 = pd.notna(ext_days) and (ext_days > 365)
        cond_sy2 = pd.notna(dur) and (dur > 365)
        is_siyam = (cond_sy1 or cond_sy2 or crosses_cal)

        # Exclusivity: Siyam wins over Ghost
        if is_siyam:
            is_ghost = False

        # Category
        cats = []
        # Chop & Doppelganger cannot be computed without peers → omitted here
        if is_ghost: cats.append("Ghost_Project")
        if is_siyam: cats.append("Siyam-siyam_Project")
        if not cats: cats = ["Green_Flag"]
        category = ";".join(cats)
        color = _category_color(category)

        # Reasons
        reasons = []
        if is_siyam:
            rr = []
            if cond_sy1: rr.append("Actual > Original by > 1 year")
            if cond_sy2: rr.append("Project duration > 1 year")
            if crosses_cal: rr.append("Crosses calendar year boundary from StartDate")
            reasons.append("Siyam-siyam: " + "; ".join(rr))
        elif is_ghost:
            if cond_A: reasons.append("Ghost: Negative duration (CompletionDateActual earlier than StartDate)")
            elif cond_B: reasons.append("Ghost: Very short design schedule (Original−Start ≤ 30 days)")
            elif cond_C: reasons.append("Ghost: Very short actual duration (Actual−Start ≤ 30 days)")
            elif cond_D: reasons.append("Ghost: Very short (≤30 days) with high-volume contractor")
            elif cond_E: reasons.append("Ghost: Too fast vs designed (< 75% of designed days)")
            elif cond_F: reasons.append(f"Ghost: Abnormally short vs peers (< {int(max(ghost_floor, min(ghost_cap, peer_p20_days)))} d) with high contractor load)")
        else:
            reasons.append("Residual Green Flag (no anomaly conditions triggered)")

    # ─────────────────────────────────────────────────────────
    # Display results
    # ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Result")

    st.markdown(
        f"""
        <div style="padding:12px;border-radius:10px;border:1px solid #ddd;background:#fafafa">
        <div style="font-weight:600">Project_Category:</div>
        <div style="font-size:1.2rem;font-weight:700;color:{color}">{category}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**Reasons**")
    for r in reasons:
        st.write(f"- {r}")

    # Small table
    show_cols = ["ProjectID","Contractor","ContractCost","InfraYear","Region","Province","Municipality",
                 "DistrictEngineeringOffice","StartDate","CompletionDateOriginal","CompletionDateActual",
                 "Latitude","Longitude","ProjectTitle","ProjectDescription"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.markdown("**Input Snapshot**")
    show_table(df.tail(1)[show_cols])

    # Export (on demand)
    exp = df.tail(1).copy()
    exp["Project_Category"] = category
    exp["Category_Color"] = color
    st.download_button(
        "Download Tagged Project (CSV)",
        data=exp.to_csv(index=False).encode("utf-8"),
        file_name="tagged_single_project.csv",
        mime="text/csv"
    )
