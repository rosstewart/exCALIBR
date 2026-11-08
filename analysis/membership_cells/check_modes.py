"""Check that calibration read the composed weights with the intended roles and mode.

For every (arm, K, dataset) in compose_report.csv: the composed layout (non-empty calibration samples in order)
gives the expected weight row of each role; the dataset's pipeline log line 'Sample indices: P=.., B=.., G=.., S=..'
must match it (None for an absent sample). Mode: standard (P and a benign reference), NU (no P/LP) or PU (no B/LB
and no Synonymous). Also reports the prior next to the standard calibration's.

Usage:  python -m analysis.membership_cells.check_modes [--out .../cells/light_c1]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

ROLE = {"P": "Pathogenic/Likely Pathogenic", "B": "Benign/Likely Benign", "G": "population", "S": "Synonymous"}
STD_CAL = {"cell_all": "medium_baseline_popall", "cell_mis": "medium_plpmis"}
SENS = Path("/data/ross/assay_calibration/missense_only_calibrations/sens_popmis_lp")


def _logged(calib_dir, ds):
    logs = glob.glob(str(calib_dir / ds / "logs" / "*_pipeline.log"))
    for f in logs:
        m = re.findall(r"Sample indices: P=(\w+), B=(\w+), G=(\w+), S=(\w+)", open(f).read())
        if m:
            return {k: (None if v == "None" else int(v)) for k, v in zip("PBGS", m[-1])}
    return None


def _prior(d, ds):
    c = sorted(glob.glob(str(d / ds / "*_calibration.json")))
    return json.load(open(c[0]))["prior"] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/ross/assay_calibration/missense_only_calibrations/cells/light_c1")
    a = ap.parse_args()
    out = Path(a.out)
    rep = pd.read_csv(out / "compose_report.csv")
    rows = []
    for r in rep[rep.status == "ok"].itertuples():
        layout = r.layout.split("|")
        expect = {k: (layout.index(v) if v in layout else None) for k, v in ROLE.items()}
        got = _logged(out / f"calib_{r.arm}_{r.K}", r.dataset)
        mode = "NU" if expect["P"] is None else ("PU" if expect["B"] is None and expect["S"] is None else "standard")
        rows.append(dict(dataset=r.dataset, arm=r.arm, K=r.K, mode=mode, layout=r.layout, expected=expect, logged=got,
                         match=(got == expect) if got is not None else None,
                         prior=_prior(out / f"calib_{r.arm}_{r.K}", r.dataset),
                         prior_std=_prior(SENS / STD_CAL[r.arm], r.dataset)))
    d = pd.DataFrame(rows)
    d.to_csv(out / "mode_check.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(d.groupby(["arm", "K", "mode"]).match.value_counts(dropna=False).to_string())
    bad = d[d.match != True]
    if len(bad):
        print("\nMISMATCHES / MISSING:"); print(bad[["dataset", "arm", "K", "layout", "expected", "logged"]].to_string(index=False))
    print("\nnon-standard modes:"); print(d[d["mode"] != "standard"][["dataset", "arm", "K", "mode", "layout", "logged", "prior", "prior_std"]].to_string(index=False))
    d["dprior"] = d.prior - d.prior_std
    print("\nprior vs standard calibration (cell_all vs all-P/LP; cell_mis vs missense refit):")
    print(d.groupby(["arm", "K"]).dprior.describe()[["count", "25%", "50%", "75%"]].round(3).to_string())


if __name__ == "__main__":
    main()
