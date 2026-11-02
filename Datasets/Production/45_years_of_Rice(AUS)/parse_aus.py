"""Parse 'Estimates of Aus rice' tables 1.1.1 - 1.1.23 (BBS, 45 years Agriculture Statistics)
into a tidy long dataset + lossless raw-cell audit file."""
import re, json
import pdfplumber
import numpy as np
import pandas as pd

PDF = "/mnt/user-data/uploads/45_years_of_Major_Crops__AUS_rice_.pdf"

# ---- table layout (derived from the PDF headers) --------------------------------------
# raw labels are exactly as printed in the PDF; fy is the corrected fiscal year
T = {
 "1.1.1":  dict(pages=[1],       raw=["1969-70","1970-71","1971-72"], fy=["1969-70","1970-71","1971-72"], cats=["local","hyv","total"], system="20-district"),
 "1.1.2":  dict(pages=[2],       raw=["1972-73","1973-74","1974-75"], fy=["1972-73","1973-74","1974-75"], cats=["local","hyv","total"], system="20-district"),
 "1.1.3":  dict(pages=[3],       raw=["1975-76","1976-77"],           fy=["1975-76","1976-77"],           cats=["local","hyv","total"], system="20-district"),
 "1.1.4":  dict(pages=[4],       raw=["1977-78","1978-79"],           fy=["1977-78","1978-79"],           cats=["local","hyv","pajam","total"], system="20-district"),
 "1.1.5":  dict(pages=[5],       raw=["1979-80","1780-81"],           fy=["1979-80","1980-81"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.6":  dict(pages=[6],       raw=["1981-82","1782-83"],           fy=["1981-82","1982-83"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.7":  dict(pages=[7],       raw=["1983-84","1984-85"],           fy=["1983-84","1984-85"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.8":  dict(pages=[8],       raw=["1985-86","1986-87"],           fy=["1985-86","1986-87"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.9":  dict(pages=[9],       raw=["1987-88","1988-89"],           fy=["1987-88","1988-89"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.10": dict(pages=[10],      raw=["1989-90","1990-91"],           fy=["1989-90","1990-91"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.11": dict(pages=[11],      raw=["1991-92","1992-93"],           fy=["1991-92","1992-93"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.12": dict(pages=[12],      raw=["1993-94","1994-95"],           fy=["1993-94","1994-95"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.13": dict(pages=[13],      raw=["1995-96","1996-97"],           fy=["1995-96","1996-97"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.14": dict(pages=[14],      raw=["1997-98","1998-99"],           fy=["1997-98","1998-99"],           cats=["local","hyv","pajam","total"], system="23-district"),
 "1.1.15": dict(pages=[15],      raw=["1999-00","2000-01"],           fy=["1999-00","2000-01"],           cats=["local","hyv","total"], system="23-district"),
 "1.1.16": dict(pages=[16],      raw=["2001-02","2002-03"],           fy=["2001-02","2002-03"],           cats=["local","hyv","total"], system="23-district"),
 "1.1.17": dict(pages=[17],      raw=["2003-04","2004-05"],           fy=["2003-04","2004-05"],           cats=["local","hyv","total"], system="23-district"),
 "1.1.18": dict(pages=[18],      raw=["2005-06"],                     fy=["2005-06"],                     cats=["local","hyv","total"], system="23-district"),
 "1.1.19": dict(pages=[19,20],   raw=["2006-07","2007-08"],           fy=["2006-07","2007-08"],           cats=["local","hyv","total"], system="64-district"),
 "1.1.20": dict(pages=[21,22],   raw=["2008-09","2009-01"],           fy=["2008-09","2009-10"],           cats=["local","hyv","total"], system="64-district"),
 "1.1.21": dict(pages=[23,24],   raw=["2001-00","2000-02"],           fy=["2010-11","2011-12"],           cats=["local","hyv","total"], system="64-district"),
 "1.1.22": dict(pages=[25,26],   raw=["2002-03","2003-04"],           fy=["2012-13","2013-14"],           cats=["local","hyv","total"], system="64-district"),
 "1.1.23": dict(pages=[27,28],   raw=["2014-2015"],                   fy=["2014-15"],                     cats=["local","hyv","total"], system="64-district"),
}
HEAD_WORDS = re.compile(r"^(Table|Sl|Estimates|Zila|Local|HYV|Total|Pajam|Area|Name|Districts)", re.I)


def clean_cell(c):
    return "" if c is None else re.sub(r"\s+", "", str(c))


def parse_rows(page_table):
    """yield (sl, name, [numeric strings]) for each data row"""
    for row in page_table:
        cells = ["" if c is None else str(c).strip() for c in row]
        if not any(cells):
            continue
        if cells[0].startswith(("Table", "Estimates", "Zila")):
            continue
        # name = first cell with letters that is not a header word
        name = None; idx = None
        for i, c in enumerate(cells[:3]):
            flat = re.sub(r"\s+", " ", c)
            if re.search(r"[A-Za-z]", flat) and not HEAD_WORDS.match(flat):
                name, idx = flat, i; break
        if name is None:
            continue
        vals = [clean_cell(c) for c in cells[idx + 1:]]
        if idx == 0:                       # totals rows: name sits in col 0, col 1 is an empty merged cell
            while vals and vals[0] == "":
                vals = vals[1:]
        # drop trailing/embedded blanks only if row has surplus; keep positions for blank-in-middle detection
        yield cells[0] if idx else "", name, vals


def main():
    pdf = pdfplumber.open(PDF)
    records, audit = [], []
    division = None
    for tno, cfg in T.items():
        ncat = len(cfg["cats"]); nyear = len(cfg["raw"]); N = nyear * ncat * 2
        # sanity: header year labels present on page text
        for pg in cfg["pages"]:
            for r in parse_rows(pdf.pages[pg - 1].extract_tables()[0]):
                sl, name, vals = r
                if not vals or not any(re.fullmatch(r"\d+", v) for v in vals):
                    continue
                raw_vals = list(vals)
                flags = []
                if len(vals) != N:
                    nb = [v for v in vals if v != ""]
                    if len(nb) == N:                    # grid artefact: spurious empty cells
                        vals = nb; flags.append("grid_blank_dropped")
                    elif len(nb) == N + 1:              # last column split across two cells
                        vals = nb[:-2] + [nb[-2] + nb[-1]]; flags.append("last_col_cell_merged")
                    else:
                        raise ValueError(f"{tno} p{pg} {name}: expected {N} values got {len(vals)} -> {raw_vals}")
                audit.append(dict(table=tno, pdf_page=pg, sl=sl, name=name, raw_cells=json.dumps(raw_vals)))
                rtype = ("national_total" if name.lower().startswith("bangladesh")
                         else "division_subtotal" if "Division" in name else "district")
                if rtype == "district" and tno == "1.1.23":
                    pass
                for y in range(nyear):
                    rec = dict(source_table=tno, source_pdf_page=pg, fy_label_raw=cfg["raw"][y],
                               fiscal_year=cfg["fy"][y], district_raw=name, row_type=rtype,
                               district_system=cfg["system"], sl_no=sl.replace("\n", "/"))
                    for c, cat in enumerate(cfg["cats"]):
                        for k, m in enumerate(["area", "prod"]):
                            v = vals[y * ncat * 2 + c * 2 + k]
                            rec[f"{cat}_{m}"] = int(v) if v != "" else np.nan
                    rec["dq_flags"] = list(flags)
                    if any(v == "" for v in vals[y * ncat * 2:(y + 1) * ncat * 2]):
                        rec["dq_flags"].append("blank_cell_in_source")
                    records.append(rec)
    df = pd.DataFrame(records)
    aud = pd.DataFrame(audit)
    return df, aud


if __name__ == "__main__":
    df, aud = main()
    df.to_pickle("/tmp/raw_parsed.pkl"); aud.to_pickle("/tmp/audit.pkl")
    print(df.shape, aud.shape)
    print(df.groupby(["source_table", "row_type"]).size().unstack(fill_value=0))
