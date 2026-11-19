"""CPU tests of ptm_rfd3.ptm_physics: PTM-general group charges on real CCD residues, and the sign/ordering of the terms."""
import numpy as np
import pytest
import torch
from biotite.structure import info

from ptm_rfd3.ptm_physics import AA20, PhysicsTerms, PTMChem, group_charges, residue_graph, virtual_cb


def charges_of(code):
    a = info.residue(code)
    a = a[a.element != "H"]
    el, adj, ring = residue_graph(a, np.arange(len(a)))
    q = group_charges(el, adj, ring)
    return {str(n): round(float(c), 3) for n, c in zip(a.atom_name, q)}, float(q.sum())


@pytest.mark.parametrize("code,net", [
    ("PTR", -2.0), ("SEP", -2.0), ("TPO", -2.0),       # phospho-Tyr/Ser/Thr: dianionic monoesters
    ("TYS", -1.0),                                      # sulfo-Tyr
    ("ALY", 0.0),                                       # acetyl-Lys: amide, neutral (the charge switch vs Lys +1)
    ("M3L", 1.0), ("MLY", 1.0),                         # tri-/di-methyl-Lys: still cationic
    ("CIR", 0.0),                                       # citrulline: urea, neutral (vs Arg +1)
    ("HYP", 0.0),                                       # hydroxy-Pro (ring N neutral)
    ("NAG", 0.0),                                       # GlcNAc: amide + polyol, neutral
])
def test_group_charges_on_ccd_residues(code, net):
    q, tot = charges_of(code)
    # backbone carboxylate (C + O + OXT) of the free CCD residue adds -1 and the free backbone amine +1 for amino acids
    backbone = (q.get("O", 0) + q.get("OXT", 0) + q.get("N", 0)) if "OXT" in q else 0.0
    assert abs((tot - backbone) - net) < 1e-6, (code, q)


def test_phosphate_charge_sits_on_terminal_oxygens():
    q, _ = charges_of("PTR")
    assert q["P"] == 0 and q["OH"] == 0
    assert all(abs(q[n] + 2 / 3) < 1e-3 for n in ("O1P", "O2P", "O3P"))


def test_chem_sigma_and_polarity():
    c = PTMChem.from_lists(["O", "O", "C", "N"], [-0.67, 0.0, 0.0, 1.0])
    assert c.sigma.tolist() == [-24.0, -6.0, 16.0, -50.0]
    assert c.polar.tolist() == [True, True, False, True]


def _scene(binder_res):
    """One anionic PTM atom at the origin, K=1, binder residues on a ring 6 A away, CB pointing inward."""
    n = 8
    ang = torch.linspace(0, 2 * np.pi, n + 1)[:-1]
    ca = torch.stack([7.5 * torch.cos(ang), 7.5 * torch.sin(ang), torch.zeros(n)], -1)[None]
    cb = ca * (6.0 / 7.5)
    probs = torch.zeros(1, n, 20)
    probs[..., AA20.index(binder_res)] = 1.0
    return ca, cb, probs


def test_electrostatics_sign_and_unsat():
    P = torch.zeros(1, 1, 3)
    chem = PTMChem.from_lists(["O"], [-1.0])
    pt = PhysicsTerms()
    bur = torch.ones(1, 1)
    out = {}
    for res in ("R", "D", "L"):
        ca, cb, probs = _scene(res)
        out[res] = pt.compute(P, bur, chem, ca, cb, probs, None)
    assert out["R"]["E_elec"] < 0 < out["D"]["E_elec"] and out["L"]["E_elec"] == 0
    assert out["R"]["physics"] > out["L"]["physics"] > out["D"]["physics"]
    assert out["L"]["unsat"] > 0.5 and out["R"]["unsat"] == 0           # buried anion: no polar partner around Leu
    assert out["R"]["dG_solv"] > 0                                       # burying a charged O costs solvation


def test_neutral_ptm_has_no_electrostatics_and_apolar_burial_is_favourable():
    P = torch.zeros(1, 2, 3)
    chem = PTMChem.from_lists(["C", "C"], [0.0, 0.0])                      # e.g. the methyls of a methyl-Lys / acetyl
    ca, cb, probs = _scene("K")
    o = PhysicsTerms().compute(P, torch.ones(1, 2), chem, ca, cb, probs, None)
    assert o["E_elec"] == 0 and o["dG_solv"] < 0 and o["unsat"] == 0


def test_virtual_cb_points_away_from_neighbours():
    ca = torch.tensor([[[0.0, 0, 0], [3.8, 0, 0], [5.7, 3.3, 0]]])
    cb = virtual_cb(ca)
    assert abs(float((cb[0, 1] - ca[0, 1]).norm()) - 1.53) < 1e-4
    assert float((cb[0, 1] - ca[0, 1]) @ ((ca[0, 0] + ca[0, 2]) / 2 - ca[0, 1])) < 0
