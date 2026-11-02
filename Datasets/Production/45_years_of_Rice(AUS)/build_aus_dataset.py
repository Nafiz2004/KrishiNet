import json, os
import numpy as np, pandas as pd
from parse_aus import main as parse_pdf

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)
ACRE_TO_HA = 0.40468564224

df, aud = parse_pdf()
for c in ["pajam_area", "pajam_prod"]:
    if c not in df: df[c] = np.nan
VAL = ["local_area", "local_prod", "hyv_area", "hyv_prod", "pajam_area", "pajam_prod", "total_area", "total_prod"]
df["pajam_col_in_source"] = df.source_table.isin([f"1.1.{i}" for i in range(4, 15)])

# ---------------- name standardisation / geography ----------------
STD = {"Bakergonj": "Barisal", "Chittagong H.T.": "Chittagong Hill Tracts", "Khustia": "Kushtia",
       "Mymenshing": "Mymensingh", "Kishoregonj": "Kishoreganj", "Sunamgonj": "Sunamganj",
       "Hobigonj": "Habiganj", "Maulavibazar": "Moulvibazar", "Manikgonj": "Manikganj",
       "Munshigonj": "Munshiganj", "Munsigonj": "Munshiganj", "Narayangonj": "Narayanganj",
       "Gopalgonj": "Gopalganj", "Perojpur": "Pirojpur", "Noagaon": "Naogaon", "Nawabgonj": "Nawabganj",
       "Sirajgonj": "Sirajganj", "Jhalakathi": "Jhalokati", "BANGLADESH": "Bangladesh"}
df["district_std"] = df.district_raw.replace(STD)
df.loc[df.row_type == "division_subtotal", "district_std"] = df.district_raw.str.replace(r"\s+", " ", regex=True)

# 64 -> 23 legacy district groups (administrative knowledge, NOT printed in the PDF; verified for continuity below)
G23 = {
 "Bandarban": ["Bandarban"], "Chittagong": ["Chittagong", "Cox's Bazar"],
 "Comilla": ["Comilla", "Chandpur", "Brahmanbaria"], "Khagrachari": ["Khagrachari"],
 "Noakhali": ["Noakhali", "Lakshmipur", "Feni"], "Rangamati": ["Rangamati"],
 "Sylhet": ["Sylhet", "Moulvibazar", "Sunamganj", "Habiganj"],
 "Dhaka": ["Dhaka", "Gazipur", "Manikganj", "Munshiganj", "Narayanganj", "Narsingdi"],
 "Faridpur": ["Faridpur", "Rajbari", "Madaripur", "Gopalganj", "Shariatpur"],
 "Jamalpur": ["Jamalpur", "Sherpur"], "Kishoreganj": ["Kishoreganj", "Netrokona"],
 "Mymensingh": ["Mymensingh"], "Tangail": ["Tangail"],
 "Barisal": ["Barisal", "Jhalokati", "Pirojpur", "Bhola"],
 "Jessore": ["Jessore", "Jhenaidah", "Magura", "Narail"], "Khulna": ["Khulna", "Bagerhat", "Satkhira"],
 "Kushtia": ["Kushtia", "Chuadanga", "Meherpur"], "Patuakhali": ["Patuakhali", "Barguna"],
 "Bogra": ["Bogra", "Joypurhat"], "Dinajpur": ["Dinajpur", "Thakurgaon", "Panchagar"],
 "Pabna": ["Pabna", "Sirajganj"], "Rajshahi": ["Rajshahi", "Naogaon", "Natore", "Nawabganj"],
 "Rangpur": ["Rangpur", "Gaibandha", "Kurigram", "Nilphamari", "Lalmonirhat"],
}
D2G = {d: g for g, ds in G23.items() for d in ds}
assert len(D2G) == 64

# division scheme taken from the PDF's own Table 1.1.23 (division subtotal rows follow their districts)
t23 = df[(df.source_table == "1.1.23")].reset_index(drop=True)
div, pending = {}, []
for _, r in t23.iterrows():
    if r.row_type == "district": pending.append(r.district_std)
    elif r.row_type == "division_subtotal":
        for d in pending: div[d] = r.district_std.replace(" Division", "")
        pending = []
assert len(div) == 64
G2DIV = {}
for g, ds in G23.items():
    s = {div[d] for d in ds}; assert len(s) == 1, (g, s); G2DIV[g] = s.pop()
OLD20_DIV = {"Kishoreganj": "Dhaka", "Chittagong Hill Tracts": "Chittagong"}
def division_of(r):
    if r.row_type != "district": return np.nan
    d = r.district_std
    if r.district_system == "64-district": return div[d]
    return G2DIV.get(d, OLD20_DIV.get(d, np.nan))
def group_of(r):
    if r.row_type != "district": return np.nan
    return D2G[r.district_std] if r.district_system == "64-district" else r.district_std
df["division"] = df.apply(division_of, axis=1)
df["legacy_group_23"] = df.apply(group_of, axis=1)

# ---------------- year helpers ----------------
df["year_start"] = df.fiscal_year.str[:4].astype(int)
df["year_label_corrected"] = df.fy_label_raw != df.fiscal_year.where(df.fiscal_year.str.len() == 7, df.fy_label_raw)
df.loc[df.fy_label_raw == "2014-2015", "year_label_corrected"] = False
df["year_label_corrected"] = df.fy_label_raw.map(lambda s: s) != df.fiscal_year
df.loc[df.fy_label_raw == "2014-2015", "year_label_corrected"] = False
df["unit_production_label"] = np.where(df.year_start < 2006, "tons", "M.tons")

# ---------------- source anomalies (raw values never edited) ----------------
df["dq_flags"] = df.dq_flags.apply(lambda x: [f for f in x if f != "grid_blank_dropped"])   # parser artefact, not a data issue

m = (df.source_table == "1.1.13") & (df.fiscal_year == "1995-96")
df.loc[m, ["pajam_area", "pajam_prod", "total_area", "total_prod"]] = np.nan
for i in df.index[m]: df.at[i, "dq_flags"] = df.at[i, "dq_flags"] + ["src_1995_96_pajam_total_missing_cols_shifted"]

m = df.source_table == "1.1.11"
for i in df.index[m & (df.fiscal_year == "1992-93")]: df.at[i, "dq_flags"] = df.at[i, "dq_flags"] + ["src_1992_93_identical_to_1991_92"]
for i in df.index[m & (df.fiscal_year == "1991-92")]: df.at[i, "dq_flags"] = df.at[i, "dq_flags"] + ["src_1991_92_identical_to_1992_93"]

# ---------------- derived / cleaned values ----------------
z = lambda c: df[c].fillna(0)
df["total_area_calc"] = z("local_area") + z("hyv_area") + z("pajam_area")
df["total_prod_calc"] = z("local_prod") + z("hyv_prod") + z("pajam_prod")
df["chk_area_reported_minus_calc"] = df.total_area - df.total_area_calc
df["chk_prod_reported_minus_calc"] = df.total_prod - df.total_prod_calc

# 1987-88: local production lost its last digit in the source (yield ~0.04 t/acre vs ~0.38 for neighbouring years)
df["local_prod_clean"] = df.local_prod.astype("float")
yl = df.local_prod / df.local_area
fix = (df.fiscal_year == "1987-88") & (yl < 0.1) & (df.local_area > 0)
df.loc[fix, "local_prod_clean"] = df.loc[fix, "local_prod"] * 10
chk = (df.loc[fix, "local_prod_clean"] / df.loc[fix, "local_area"])
assert chk.between(0.2, 0.75).all(), chk.describe()
for i in df.index[fix]: df.at[i, "dq_flags"] = df.at[i, "dq_flags"] + ["adj_1987_88_local_prod_x10"]
# rows in 1987-88 whose yield is still implausibly low (<0.2) after the rule cannot be repaired -> NaN in clean column
bad = (df.fiscal_year == "1987-88") & ~fix & (yl < 0.2) & (df.local_area > 0)
df.loc[bad, "local_prod_clean"] = np.nan
for i in df.index[bad]: df.at[i, "dq_flags"] = df.at[i, "dq_flags"] + ["suspect_unrecoverable_1987_88_local_prod"]
for i in df.index[(df.fiscal_year == "1987-88") & ~fix & ~bad]: df.at[i, "dq_flags"] = df.at[i, "dq_flags"] + ["note_1987_88_row_plausible_not_adjusted"]
df["hyv_prod_clean"] = df.hyv_prod.astype("float")
blank = df.dq_flags.apply(lambda f: "blank_cell_in_source" in f)
df.loc[blank & df.hyv_prod.isna() & (df.total_prod == 0), "hyv_prod_clean"] = 0        # Lalmonirhat 2014-15
df["total_prod_clean"] = df.local_prod_clean.fillna(0) + df.hyv_prod_clean.fillna(0) + z("pajam_prod")
df.loc[df.local_prod_clean.isna() & (df.local_area > 0), "total_prod_clean"] = np.nan
df["total_area_clean"] = df.total_area_calc

# yields
def add_yields(d):
    pa = lambda p, a: np.where(d[a] > 0, d[p] / d[a], np.nan)
    d["yield_total_t_per_acre"] = pa("total_prod_clean", "total_area_clean")
    d["yield_local_t_per_acre"] = pa("local_prod_clean", "local_area")
    d["yield_hyv_incl_pajam_t_per_acre"] = np.where((d.hyv_area + d.pajam_area.fillna(0)) > 0,
                                                    (d.hyv_prod_clean.fillna(0) + d.pajam_prod.fillna(0)) / (d.hyv_area + d.pajam_area.fillna(0)), np.nan)
    d["yield_total_kg_per_ha"] = d.yield_total_t_per_acre * 1000 / ACRE_TO_HA
    return d
df = add_yields(df)

# remaining statistical outliers (flag only)
dd = df[(df.row_type == "district") & (df.total_area_clean >= 500)]
lo, hi = 0.12, 1.30
for i in dd.index[(dd.yield_total_t_per_acre < lo) | (dd.yield_total_t_per_acre > hi)]:
    df.at[i, "dq_flags"] = df.at[i, "dq_flags"] + ["yield_outlier_review"]
for i in df.index[(df.row_type != "national_total") & (df.chk_prod_reported_minus_calc.abs() == 1) & (df.source_table == "1.1.23")]:
    df.at[i, "dq_flags"] = df.at[i, "dq_flags"] + ["src_rounding_diff_1"]

df["is_cultivated"] = df.total_area_clean > 0
df["dq_flags_str"] = df.dq_flags.apply(lambda f: ";".join(sorted(set(f))))
df["dq_has_flag"] = df.dq_flags_str.str.contains(r"src_|adj_|blank|yield_outlier|suspect")
df["record_id"] = df.source_table + "|" + df.fiscal_year + "|" + df.district_raw.str.replace(r"\s+", "_", regex=True)

MASTER_COLS = ["record_id", "source_table", "source_pdf_page", "sl_no", "fy_label_raw", "fiscal_year", "year_start",
               "year_label_corrected", "district_raw", "district_std", "row_type", "district_system", "division",
               "legacy_group_23", "pajam_col_in_source", "unit_production_label"] + VAL + \
              ["total_area_calc", "total_prod_calc", "chk_area_reported_minus_calc", "chk_prod_reported_minus_calc",
               "local_prod_clean", "hyv_prod_clean", "total_prod_clean", "yield_total_t_per_acre", "yield_local_t_per_acre",
               "yield_hyv_incl_pajam_t_per_acre", "yield_total_kg_per_ha", "is_cultivated", "dq_flags_str"]
master = df[MASTER_COLS].copy()
for c in VAL + ["chk_area_reported_minus_calc", "chk_prod_reported_minus_calc"]:
    master[c] = master[c].astype("Int64")
master.to_csv(f"{OUT}/aus_rice_master_long.csv", index=False)

# lossless raw audit
aud["raw_cells"] = aud.raw_cells
aud.rename(columns={"table": "source_table", "pdf_page": "source_pdf_page", "name": "row_name_raw"}).to_csv(f"{OUT}/aus_rice_raw_cells_audit.csv", index=False)

# ---------------- ML feature engineering ----------------
def features(d, ent, nat):
    d = d.sort_values(ent + ["year_start"]).reset_index(drop=True)
    d["hyv_incl_pajam_area"] = d.hyv_area + d.pajam_area.fillna(0)
    d["hyv_incl_pajam_prod"] = d.hyv_prod_clean.fillna(0) + d.pajam_prod.fillna(0)
    d["local_prod"] = d.local_prod_clean
    d["total_area"] = d.total_area_clean; d["total_prod"] = d.total_prod_clean
    d["hyv_area_share"] = np.where(d.total_area > 0, d.hyv_incl_pajam_area / d.total_area, np.nan)
    d["local_area_share"] = np.where(d.total_area > 0, d.local_area / d.total_area, np.nan)
    d["yield_total_t_per_acre"] = np.where(d.total_area > 0, d.total_prod / d.total_area, np.nan)
    d["yield_local_t_per_acre"] = np.where(d.local_area > 0, d.local_prod / d.local_area, np.nan)
    d["yield_hyv_t_per_acre"] = np.where(d.hyv_incl_pajam_area > 0, d.hyv_incl_pajam_prod / d.hyv_incl_pajam_area, np.nan)
    d["yield_total_kg_per_ha"] = d.yield_total_t_per_acre * 1000 / ACRE_TO_HA
    d["area_ha"] = d.total_area * ACRE_TO_HA
    d = d.merge(nat, on="year_start", how="left")
    d["area_share_of_national"] = d.total_area / d.nat_total_area
    d["prod_share_of_national"] = d.total_prod / d.nat_total_prod
    d["yield_vs_national"] = d.yield_total_t_per_acre / d.nat_yield_total_t_per_acre
    g = d.groupby(ent, sort=False)
    assert (g.year_start.diff().dropna() == 1).all()
    for c in ["total_area", "total_prod", "yield_total_t_per_acre", "hyv_area_share"]:
        s = g[c]
        d[f"{c}_lag1"] = s.shift(1); d[f"{c}_lag2"] = s.shift(2)
        d[f"{c}_roll3_prev_mean"] = s.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    for c in ["total_area", "total_prod", "yield_total_t_per_acre"]:
        d[f"{c}_yoy_pct"] = np.where(d[f"{c}_lag1"] > 0, (d[c] / d[f"{c}_lag1"] - 1) * 100, np.nan)
    d["entity_hist_mean_yield"] = g["yield_total_t_per_acre"].transform(lambda x: x.shift(1).expanding().mean())
    d["t_index"] = d.year_start - d.year_start.min()
    d["decade"] = (d.year_start // 10) * 10
    d["post_liberation"] = (d.year_start >= 1971).astype(int)
    d["target_total_prod_next_year"] = g["total_prod"].shift(-1)
    d["target_yield_next_year"] = g["yield_total_t_per_acre"].shift(-1)
    d["entity_id"] = d[ent].astype(str).agg("|".join, axis=1)
    return d

# national context per year (from the tables' own 'Bangladesh' rows, cleaned)
nr = df[df.row_type == "national_total"].copy()
nat = pd.DataFrame({"year_start": nr.year_start, "nat_total_area": nr.total_area_clean, "nat_total_prod": nr.total_prod_clean,
                    "nat_hyv_share_area": (nr.hyv_area + nr.pajam_area.fillna(0)) / nr.total_area_clean})
nat["nat_yield_total_t_per_acre"] = nat.nat_total_prod / nat.nat_total_area
nat = nat.drop_duplicates("year_start")

dist = df[df.row_type == "district"].copy()
dist["pajam_area"] = dist.pajam_area.fillna(0); dist["pajam_prod"] = dist.pajam_prod.fillna(0)
dist["hyv_area"] = dist.hyv_area.fillna(0)
ml = features(dist, ["district_system", "district_std"], nat)
ML_COLS = ["record_id", "entity_id", "fiscal_year", "year_start", "t_index", "decade", "post_liberation",
           "district_system", "district_std", "division", "legacy_group_23", "source_table",
           "local_area", "hyv_area", "pajam_area", "total_area", "area_ha", "local_prod", "hyv_prod_clean", "pajam_prod", "total_prod",
           "hyv_incl_pajam_area", "hyv_incl_pajam_prod", "hyv_area_share", "local_area_share",
           "yield_total_t_per_acre", "yield_local_t_per_acre", "yield_hyv_t_per_acre", "yield_total_kg_per_ha",
           "nat_total_area", "nat_total_prod", "nat_yield_total_t_per_acre", "nat_hyv_share_area",
           "area_share_of_national", "prod_share_of_national", "yield_vs_national",
           "total_area_lag1", "total_area_lag2", "total_area_roll3_prev_mean", "total_area_yoy_pct",
           "total_prod_lag1", "total_prod_lag2", "total_prod_roll3_prev_mean", "total_prod_yoy_pct",
           "yield_total_t_per_acre_lag1", "yield_total_t_per_acre_lag2", "yield_total_t_per_acre_roll3_prev_mean", "yield_total_t_per_acre_yoy_pct",
           "hyv_area_share_lag1", "hyv_area_share_lag2", "hyv_area_share_roll3_prev_mean", "entity_hist_mean_yield",
           "is_cultivated", "dq_flags_str", "target_total_prod_next_year", "target_yield_next_year"]
ml = ml.rename(columns={"hyv_prod_clean": "hyv_prod"})
ML_COLS = [c if c != "hyv_prod_clean" else "hyv_prod" for c in ML_COLS]
ml[ML_COLS].to_csv(f"{OUT}/aus_rice_ml_ready_district_panel.csv", index=False)

# ---------------- harmonised 23-group panel 1979-80 .. 2014-15 ----------------
base = df[(df.row_type == "district") & (df.district_system != "20-district")].copy()
base["pajam_area"] = base.pajam_area.fillna(0); base["pajam_prod"] = base.pajam_prod.fillna(0); base["hyv_area"] = base.hyv_area.fillna(0)
agg = (base.assign(local_prod_c=base.local_prod_clean, hyv_prod_c=base.hyv_prod_clean.fillna(0))
       .groupby(["legacy_group_23", "fiscal_year", "year_start"], as_index=False)
       .agg(local_area=("local_area", "sum"), hyv_area=("hyv_area", "sum"), pajam_area=("pajam_area", "sum"),
            local_prod_clean=("local_prod_c", lambda x: x.sum(min_count=len(x))), hyv_prod_clean=("hyv_prod_c", "sum"), pajam_prod=("pajam_prod", "sum"),
            n_components=("district_std", "nunique"), src_system=("district_system", "first"),
            flags=("dq_flags_str", lambda s: ";".join(sorted({f for x in s for f in x.split(";") if f})))))
agg["total_area_clean"] = agg.local_area + agg.hyv_area + agg.pajam_area
agg["total_prod_clean"] = agg.local_prod_clean + agg.hyv_prod_clean + agg.pajam_prod
agg["district_std"] = agg.legacy_group_23; agg["district_system"] = "23-group-harmonised"
agg["division"] = agg.legacy_group_23.map(G2DIV)
agg["is_aggregated_from_64"] = agg.src_system == "64-district"
agg["source_table"] = ""; agg["record_id"] = "P23|" + agg.fiscal_year + "|" + agg.legacy_group_23
agg["is_cultivated"] = agg.total_area_clean > 0; agg["dq_flags_str"] = agg.flags
pan = features(agg, ["district_system", "district_std"], nat)
pan = pan.rename(columns={"hyv_prod_clean": "hyv_prod"})
PCOLS = [c for c in ML_COLS if c in pan.columns] + ["n_components", "is_aggregated_from_64"]
pan[PCOLS].to_csv(f"{OUT}/aus_rice_panel_23groups_1979_2015.csv", index=False)

# ---------------- validation numbers for the report ----------------
rep = {}
rep["rows_master"] = len(master); rep["rows_ml"] = len(ml); rep["rows_panel"] = len(pan)
nd = dist.groupby("year_start")[["total_area_clean", "total_prod_clean"]].sum().join(nat.set_index("year_start")[["nat_total_area", "nat_total_prod"]])
nd["area_gap_pct"] = (nd.total_area_clean / nd.nat_total_area - 1) * 100
nd["prod_gap_pct"] = (nd.total_prod_clean / nd.nat_total_prod - 1) * 100
rep["nat_gap"] = nd[["area_gap_pct", "prod_gap_pct"]].round(2)
nd.to_pickle("/tmp/nat_gap.pkl")
m2 = master.copy()
rep["flag_counts"] = m2.dq_flags_str.str.split(";").explode().replace("", np.nan).dropna().value_counts()
rep["cells_expected"] = int(aud.raw_cells.apply(lambda s: len(json.loads(s))).sum())
rep["values_in_master"] = int(master[VAL].notna().sum().sum())
print(rep["rows_master"], rep["rows_ml"], rep["rows_panel"])
print(rep["flag_counts"])
print("raw cells in audit:", rep["cells_expected"], "| non-null values in master:", rep["values_in_master"])
print(rep["nat_gap"].describe())
print(rep["nat_gap"][rep["nat_gap"].abs().max(axis=1) > 1])
pd.to_pickle(rep, "/tmp/rep.pkl")
