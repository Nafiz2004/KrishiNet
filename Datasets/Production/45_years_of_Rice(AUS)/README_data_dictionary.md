# Aus Rice – Bangladesh (BBS "45 Years Agriculture Statistics", Tables 1.1.1 – 1.1.23)

Source: 28-page PDF, 23 tables, fiscal years 1969-70 → 2014-15 (46 years), area in **acres**, production in **tons** (labelled M.tons from 2006-07).
Rebuild everything with `python build_aus_dataset.py` (needs `parse_aus.py`, pdfplumber, pandas).

## Files
| File | Rows | Use |
|---|---|---|
| `aus_rice_ml_ready_district_panel.csv` | 1,397 | **Main ML file.** District-level rows only, engineered features, lags, targets. |
| `aus_rice_panel_23groups_1979_2015.csv` | 828 | 23 consistent regions × 36 years (1979-80→2014-15). 64-district years aggregated back to the 23 legacy districts, so there is no break at 2006-07. Best for long time-series models. |
| `aus_rice_master_long.csv` | 1,450 | Everything in the PDF in tidy form: districts + 7 division subtotals (2014-15) + 46 national totals. Reported values untouched, plus checks/flags. |
| `aus_rice_raw_cells_audit.csv` | 752 | Every PDF row exactly as extracted (raw cell strings, page, table). Proves nothing was lost. |

## Coverage / structure
* 1969-70 → 1978-79: **20 districts** (Tables 1.1.1-4; old names such as Bakergonj = Barisal, Chittagong H.T.)
* 1979-80 → 2005-06: **23 districts** (Tables 1.1.5-18)
* 2006-07 → 2014-15: **64 districts** (Tables 1.1.19-23)
* Variety columns: Local, HYV, Pajam (only 1977-78 → 1998-99 as separate column; before 1977-78 it is inside HYV, from 1999-00 it is inside HYV), Total.

## Lossless reconciliation
Non-blank cells in PDF = 9,764 → 9,647 values in master + 96 cells that are an exact duplicate block in Table 1.1.13 (kept in the audit file) + 21 numbers that the PDF split over two cells (re-joined). Reported Total = Local + HYV + Pajam for every row except the 1995-96 shift (below) and ±1 rounding in 11 rows of Table 1.1.23. District area sums equal the PDF's national row in all 46 years; production sums too except 1987-88.

## Problems found IN the source PDF (all flagged in `dq_flags_str`)
| Issue | What I did |
|---|---|
| Year labels mistyped: `1780-81`, `1782-83`, `2009-01`, `2001-00`, `2000-02`, `2002-03`/`2003-04` in Table 1.1.22 | Corrected from table sequence to 1980-81, 1982-83, 2009-10, 2010-11, 2011-12, 2012-13, 2013-14. Raw label kept in `fy_label_raw`. |
| Table 1.1.13 (1995-96): Pajam and Total columns missing; the 1996-97 Local/HYV block printed twice | 1995-96 Local/HYV kept; Pajam & reported Total = NULL; totals recomputed as Local+HYV (`src_1995_96_...`). Pajam for 1995-96 is unknown (small, ≤1% in neighbouring years). |
| 1992-93 is a copy of 1991-92 (all 24 rows identical) | Kept, flagged `src_1992_93_identical_to_1991_92`. Consider dropping 1992-93 or treating as missing. |
| 1987-88 Local production lost a digit (yield 0.04 t/acre vs ~0.38 normally) | Raw kept. In `local_prod_clean` / `total_prod_clean` the 21 affected districts + national row are ×10 (`adj_1987_88_local_prod_x10`; resulting yields 0.3-0.6). **This is an inference; verify against the original BBS book.** |
| Khulna 1987-88 local production (9,230) is wrong but not fixable by ×10 | Clean value set to NULL (`suspect_unrecoverable_...`). |
| Lalmonirhat 2014-15: one empty cell (HYV production) | NULL in master; set 0 in ML file (area and total are 0). |
| Cox's Bazar 2009-10, Barguna 2013-14: yield > 1.3 t/acre | Left as is, flagged `yield_outlier_review`. |

## Key columns (ML file)
* **ids**: `entity_id` (system+district; use for grouping/CV), `district_std` (spellings normalised), `district_system`, `division` and `legacy_group_23` (see note), `fiscal_year`, `year_start`, `t_index`, `decade`, `post_liberation`.
* **levels**: `local_/hyv_/pajam_/total_area` (acres), `area_ha`, `local_prod`, `hyv_prod`, `pajam_prod`, `total_prod` (tons; 1987-88 cleaned).
* **harmonised HYV**: `hyv_incl_pajam_area/prod`, `hyv_area_share`, `local_area_share` (use these – the raw HYV column changes meaning over time).
* **yields**: `yield_total_t_per_acre`, `_local_`, `_hyv_`, `yield_total_kg_per_ha` (NULL when area = 0).
* **context**: `nat_*` national totals that year, `area_share_of_national`, `prod_share_of_national`, `yield_vs_national`.
* **time features (past-only, no leakage)**: `*_lag1`, `*_lag2`, `*_roll3_prev_mean`, `*_yoy_pct`, `entity_hist_mean_yield`.
* **targets**: `target_total_prod_next_year`, `target_yield_next_year` (NULL for the last year of each entity).
* `is_cultivated` (area>0), `dq_flags_str`.

## Cautions for modelling
* `division` is taken from the PDF's own Table 1.1.23 grouping and applied back in time; `legacy_group_23` (64 → 23 mapping) is **my administrative mapping, not printed in the PDF**. It passes a continuity check at the 2005-06 → 2006-07 boundary (per-group area ratios 0.70-1.08; national total continuous).
* Lags never cross a system change (20 → 23 → 64 districts); use the 23-group panel for long histories.
* Many zeros in northern districts are genuine "not grown" values, not missing data (`is_cultivated`). Tangail is 0 in 1969-70 (not yet a separate district).
* Split by **time** (e.g. train ≤ 2005, test 2006-2014), not randomly, and drop or down-weight 1992-93 / 1995-96 / 1987-88 if you want fully clean training data.
