"""CPU tests of the RFD4 design stage plumbing (ptm_rfd3.rfd4_run): inputs from a PTM spec, and collect() of a synthetic
RFD4 output into the RFD3 layout that `ptm-rfd3 stage` reads. No RFD4 model is needed."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ptm_rfd3 import rfd4_run
from ptm_rfd3 import structio as sio

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    work = tmp_path_factory.mktemp("rfd4")
    spec = {"egfr_pY1138": {"input": str(REPO / "examples" / "egfr_tail_1126_1150.pdb"), "ptm": {"A1138": "phospho-Tyr"},
                            "ptm_hotspots": ["ptm"], "ptm_conformers": 1, "contig": "70-110,/0,A1126-1150"}}
    sp = work / "spec.json"
    sp.write_text(json.dumps(spec))
    info = rfd4_run.make_inputs(sp, work / "w", lengths=[80, 95], n_confs=20)
    return work / "w", info


def test_make_inputs_writes_fixed_length_condition_specs(built):
    w, info = built
    y = yaml.safe_load(Path(info["yaml"]).read_text())
    assert sorted(y) == ["egfr_pY1138_ptm_k0_L80", "egfr_pY1138_ptm_k0_L95"]
    e = y["egfr_pY1138_ptm_k0_L80"]
    tgt = sio.read_structure(e["input"])                      # the PREPARED target (the pipeline renumbers it)
    ptr = int(tgt.res_id[tgt.res_name == "PTR"][0])
    lo, hi = int(tgt.res_id.min()), int(tgt.res_id.max())
    assert e["contig"] == f"80-80, /0, A{lo}-{hi}"
    assert e["conditions"]["C_CRD"][True] == [{"select": "sel('A')"}]
    hot = [h["select"] for h in e["conditions"]["C_HOT"][True]]
    assert any(f"/{ptr}/" in h and "O1P" in h for h in hot), hot
    assert Path(e["input"]).exists()
    assert "PTR:" in info["ptm_atoms"] and "O1P" in info["ptm_atoms"]
    assert "PTMSMC_PTM_ATOMS" in (w / "rfd4_env.sh").read_text()


def test_collect_renames_binder_to_A_and_writes_rfd3_layout(built, tmp_path):
    w, info = built
    y = yaml.safe_load(Path(info["yaml"]).read_text())
    name = "egfr_pY1138_ptm_k0_L80"
    tgt = sio.read_structure(y[name]["input"])
    binder = tgt[np.isin(tgt.atom_name, ["N", "CA", "C", "O"])].copy()   # stand-in binder: backbone copy, moved away
    binder.coord = binder.coord + 25.0
    binder.chain_id[:] = "B"
    binder.res_name[:] = "ALA"
    binder.hetero[:] = False
    from biotite import structure as struc
    design = struc.concatenate([tgt, binder])
    out = tmp_path / "rfd4_out" / f"{name}_0"
    out.mkdir(parents=True)
    sio.write_cif(design, out / f"{name}_0_generated.cif")
    dest = tmp_path / "dest"
    n = rfd4_run.collect(tmp_path / "rfd4_out", w, dest)
    assert n == 1
    meta = json.loads((dest / f"{name}_0_model_0.json").read_text())
    assert meta["specification"]["extra"]["example"] == "egfr_pY1138_ptm_k0"
    assert meta["rfd4"]["chain_map"] == {"B": "A", "A": "B"}
    assert meta["metrics"]["n_clashing.interresidue_clashes_w_sidechain"] == 0
    from ptm_rfd3 import pipeline as P
    a = P.read_cif_text(gzip.open(dest / f"{name}_0_model_0.cif.gz", "rt").read())
    assert set(a.chain_id[a.res_name == "PTR"]) == {"B"} and set(a.res_name[a.chain_id == "A"]) == {"ALA"}
    assert (dest / "ptm_build").is_dir()
