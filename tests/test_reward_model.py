"""CPU tests of ptm_rfd3.reward_model: legacy equivalence, schedules, normalisation, config loading and overrides."""
import json
import math

import pytest
import torch

from ptm_rfd3.particles import BurialReward
from ptm_rfd3.ptm_physics import PhysicsTerms, PTMChem, PTMReward
from ptm_rfd3.reward_model import Normalize, RewardModel, Schedule, TermSpec, load


def scene(K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    ptm = torch.zeros(K, 4, 3) + 0.5 * torch.randn(K, 4, 3, generator=g)
    other = torch.randn(K, 10, 3, generator=g) * 4 + torch.tensor([8.0, 0, 0])
    binder = torch.randn(K, 30, 3, generator=g) * 6
    X = torch.cat([ptm, other, binder], 1)
    L = X.shape[1]
    m = lambda a, b: torch.tensor([a <= i < b for i in range(L)])
    probs = torch.softmax(torch.randn(K, 10, 20, generator=g), -1)
    chem = PTMChem.from_lists(["O", "O", "P", "C"], [-0.67, -0.67, 0.0, 0.0])
    ca_idx = torch.arange(14, 44, 3)
    cb_idx = ca_idx + 1
    return X, m(0, 4), m(4, 14), m(14, 44), chem, ca_idx, cb_idx, probs


@pytest.mark.parametrize("mode,physics", [("solid", False), ("solid", True), ("count", False), ("count", True)])
def test_default_config_reproduces_legacy_reward(mode, physics):
    X, P, Q, B, chem, ca, cb, probs = scene()
    env = {"PTMSMC_MODE": mode, "PTMSMC_PHYSICS": "1" if physics else "0", "PTMSMC_ALPHA": "0.7", "PTMSMC_CLASH_W": "0.2"}
    _, model = load(env)
    new = model.evaluate(X, P, Q, B, chem=chem, ca_idx=ca, cb_idx=cb, probs=probs, progress=1.0)["reward"]
    old = PTMReward(BurialReward(mode=mode, alpha=0.7, clash_w=0.2), PhysicsTerms() if physics else None).terms(
        X, P, Q, B, chem=chem, ca_idx=ca, cb_idx=cb, probs=probs)["reward"]
    assert torch.allclose(new, old, atol=1e-5)


def test_schedule_shapes():
    assert Schedule(start=0.5)(0.4) == 0 and Schedule(start=0.5)(0.6) == 1
    assert Schedule(start=0.0, stop=1.0, shape="linear")(0.25) == 0.25
    assert abs(Schedule(shape="cosine")(0.5) - 0.5) < 1e-12
    assert Schedule(shape="power", power=2)(0.5) == 0.25
    assert Schedule(shape="step", start=0.2, stop=0.6)(0.5) == 0 and Schedule(shape="step", start=0.2, stop=0.6)(0.6) == 1
    assert Schedule(start=0.2, stop=0.6, hold=False)(0.9) == 0 and Schedule(start=0.2, stop=0.6)(0.9) == 1
    assert Schedule(shape="linear", floor=0.5)(0.0) == 0.5
    with pytest.raises(ValueError):
        Schedule(shape="wiggly")


def test_normalize():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(Normalize("scale", 2.0)(x), x / 2)
    z = Normalize("zscore")(x)
    assert abs(float(z.mean())) < 1e-6 and abs(float(z.std()) - 1) < 1e-6
    assert torch.allclose(Normalize("none", clip=(1.5, 2.5))(x), torch.tensor([1.5, 2.0, 2.5]))


def test_multicomponent_sum_and_time_dependence():
    X, P, Q, B, chem, ca, cb, probs = scene()
    terms = [TermSpec("ptm_burial", 1.0), TermSpec("contact_share", 0.5),
             TermSpec("electrostatics", 0.02, schedule=Schedule(start=0.6, shape="linear"))]
    m = RewardModel(terms, BurialReward(mode="solid"), PhysicsTerms())
    kw = dict(chem=chem, ca_idx=ca, cb_idx=cb, probs=probs)
    early = m.evaluate(X, P, Q, B, progress=0.3, **kw)
    late = m.evaluate(X, P, Q, B, progress=0.8, **kw)
    assert torch.allclose(early["reward"], early["ptm_burial"] + 0.5 * early["contact_share"], atol=1e-6)
    f = (0.8 - 0.6) / 0.4
    assert torch.allclose(late["reward"], late["ptm_burial"] + 0.5 * late["contact_share"] + 0.02 * f * late["electrostatics"],
                          atol=1e-5)


def test_missing_chemistry_skips_physics_terms():
    X, P, Q, B, *_ = scene()
    m = RewardModel([TermSpec("ptm_burial"), TermSpec("electrostatics", 1.0)])
    out = m.evaluate(X, P, Q, B)                     # no chem / residues: the physics term is skipped, burial still works
    assert "electrostatics" not in out and torch.allclose(out["reward"], out["ptm_burial"])


def test_config_file_and_env_overrides(tmp_path):
    cfg = {"policy": {"scheme": "smc", "lam_max": 30, "fold_dedup": True},
           "geometry": {"mode": "solid", "ray_len": 12},
           "physics": {"a0": 20},
           "terms": [{"name": "ptm_burial", "weight": 1.0},
                     {"name": "unsat_polar", "weight": 0.3, "schedule": {"start": 0.7, "shape": "cosine"},
                      "normalize": {"mode": "scale", "scale": 2, "clip": [-1, 1]}}]}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    pol, m = load({"PTMSMC_CONFIG": str(p), "PTMSMC_W_UNSAT_POLAR": "0.5", "PTMSMC_W_COMPACTNESS": "0.1",
                   "PTMSMC_RAY_LEN": "9"})
    assert pol == {"PTMSMC_SCHEME": "smc", "PTMSMC_LAM_MAX": "30", "PTMSMC_FOLD_DEDUP": "True"}
    assert m.geometry.ray_len == 9.0 and m.physics.a0 == 20
    w = {t.name: t.weight for t in m.terms}
    assert w == {"ptm_burial": 1.0, "unsat_polar": 0.5, "compactness": 0.1}
    t = [t for t in m.terms if t.name == "unsat_polar"][0]
    assert t.schedule.shape == "cosine" and t.normalize.clip == (-1, 1)
    _, m2 = load({"PTMSMC_TERMS": "ptm_burial:1,clash:0.3"})
    assert [(t.name, t.weight) for t in m2.terms] == [("ptm_burial", 1.0), ("clash", 0.3)]
    with pytest.raises(ValueError):
        load({"PTMSMC_W_NONSENSE": "1"})


def test_rfd4_geometry_defaults():
    _, m = load({}, generator="rfd4")
    assert (m.geometry.r0, m.geometry.d0, m.geometry.ray_r, m.geometry.mode) == (8.0, 3.8, 3.0, "solid")


def test_shipped_example_configs_load():
    from pathlib import Path

    from ptm_rfd3.particles import ParticlePolicy

    files = sorted((Path(__file__).resolve().parents[1] / "examples" / "particle_guidance").glob("*.yaml"))
    assert files
    for f in files:
        pol, model = load({"PTMSMC_CONFIG": str(f)})
        ParticlePolicy.from_env({**pol, "PTMSMC_N_PARTICLES": "16"})          # every policy key is a real field
        assert model.terms and model.terms[0].name == "ptm_burial"


def test_close_contact_and_charge_partner():
    from ptm_rfd3.ptm_physics import AA20
    K = 3
    P = torch.zeros(K, 1, 3)
    # one binder residue: CA 7.9 A away, CB pointing at the PTM atom -> charge centre 4.0 A from it
    ca = torch.tensor([7.9, 0, 0]).expand(K, 1, 3)
    cb = torch.tensor([6.4, 0, 0]).expand(K, 1, 3)
    side = torch.stack([torch.tensor([[1.5, 0, 0]]), torch.tensor([[3.5, 0, 0]]), torch.tensor([[3.5, 0, 0]])])  # 1.5 A = overlap
    X = torch.cat([P, ca, cb, side], 1)
    L = X.shape[1]
    m = lambda idx: torch.tensor([i in idx for i in range(L)])
    probs = torch.zeros(K, 1, 20)
    probs[0, 0, AA20.index("R")] = 1; probs[1, 0, AA20.index("R")] = 1; probs[2, 0, AA20.index("D")] = 1
    chem = PTMChem.from_lists(["O"], [-1.0])
    model = RewardModel([TermSpec("ptm_close_contact"), TermSpec("charge_partner")])
    out = model.evaluate(X, m({0}), m(set()), m({1, 2}), chem=chem, ca_idx=torch.tensor([1]), cb_idx=torch.tensor([2]),
                         probs=probs, binder_all=m({1, 2, 3}))
    cc, cp = out["ptm_close_contact"], out["charge_partner"]
    assert cc[0] < 0 and cc[1] == 0 and cc[2] == 0                       # only the 1.5 A side-chain atom overlaps
    assert cp[1] == 1.0 and cp[2] == 0.0                                  # Arg next to an anion counts, Asp does not
    neutral = model.evaluate(X, m({0}), m(set()), m({1, 2}), chem=PTMChem.from_lists(["C"], [0.0]),
                             ca_idx=torch.tensor([1]), cb_idx=torch.tensor([2]), probs=probs)
    assert torch.all(neutral["charge_partner"] == 0)                      # neutral PTM: inert
    cation = model.evaluate(X, m({0}), m(set()), m({1, 2}), chem=PTMChem.from_lists(["N"], [1.0]),
                            ca_idx=torch.tensor([1]), cb_idx=torch.tensor([2]), probs=probs)
    assert cation["charge_partner"][2] == 1.0 and cation["charge_partner"][1] == 0.0   # Asp pairs with a cationic PTM


def test_term_params_and_rfd4_close_d(tmp_path):
    import json as _j
    p = tmp_path / "c.json"
    p.write_text(_j.dumps({"terms": [{"name": "ptm_burial"}, {"name": "ptm_close_contact", "weight": 0.5,
                                                              "params": {"close_d": 3.0}}]}))
    _, m = load({"PTMSMC_CONFIG": str(p)})
    assert m.terms[1].params == {"close_d": 3.0}
    _, m4 = load({"PTMSMC_TERMS": "ptm_burial:1,ptm_close_contact:0.5"}, generator="rfd4")
    assert m4.terms[1].params == {"close_d": 4.0}


def test_peptide_binding_terms():
    torch.manual_seed(0)
    K = 3
    tgt = torch.stack([torch.arange(10.0) * 3.8, torch.zeros(10), torch.zeros(10)], -1)       # a 10-atom peptide
    near = tgt + torch.tensor([0.0, 5.0, 0.0])                                               # binder alongside it
    half = torch.cat([near[:5], near[5:] + torch.tensor([0, 40.0, 0])])                      # binder along half of it
    far = near + torch.tensor([0.0, 60.0, 0.0])
    X = torch.stack([torch.cat([tgt, b]) for b in (near, half, far)])
    m = lambda a, b: torch.tensor([a <= i < b for i in range(20)])
    model = RewardModel([TermSpec("interface_contacts"), TermSpec("peptide_burial")], BurialReward(mode="solid"))
    out = model.evaluate(X, m(0, 1), m(1, 10), m(10, 20))
    ic, pb = out["interface_contacts"], out["peptide_burial"]
    # half: 5 atoms in contact + atom 5 at 6.28 A from the binder end, inside the 6.0-7.5 A ramp (0.81 credit)
    assert ic[0] == 1.0 and abs(float(ic[1]) - (5 + (7.5 - (3.8 ** 2 + 25) ** 0.5) / 1.5) / 10) < 1e-4 and ic[2] == 0.0
    assert pb[0] > pb[1] > pb[2] == 0.0
    _, m4 = load({"PTMSMC_TERMS": "ptm_burial:1,interface_contacts:0.5"}, generator="rfd4")
    assert m4.terms[1].params == {"d_on": 8.0}
