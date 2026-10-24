"""ptm-rfd3 command line:  build / inputs.

  python -m ptm_rfd3.cli inputs --target 1abc.cif --ptm A45=phospho-Ser --name tauS45 --out OUT \
        --binder-length 60-100 --hotspots ptm --crop-radius 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import af3_inputs, rfd3_inputs, rfd3_prep
from . import structio as sio
from . import target as T


def _sites_key_atoms(tb: T.TargetBuild):
    pts = []
    for b in tb.built:
        res = sio.residue_slice(tb.variants[0], b.key if b.kind == "replace" else b.ligand_key)
        pts.append(res.coord)
    return np.vstack(pts) if pts else None


def cmd_inputs(a):
    out = Path(a.out)
    (out / "targets").mkdir(parents=True, exist_ok=True)
    (out / "rfd3").mkdir(parents=True, exist_ok=True)
    arr = sio.read_structure(a.target)
    reqs = T.parse_requests(a.ptm)
    tb = T.build_modified_target(arr, reqs, n_confs=a.n_confs, top_k=a.top_k, seed=a.seed)
    files = T.write_outputs(tb, out / "targets", a.name, fmts=tuple(a.formats.split(",")))
    manifest = {"name": a.name, "ptm": a.ptm, "build_report": tb.report, "variants": [], "control": None}

    # PTM-site coordinates (for cropping) from the modified variant 0
    ptm_pts = []
    for r in reqs:
        res = sio.residue_slice(tb.variants[0], r.key)
        ptm_pts.append(res.coord)
    for b in tb.built:
        if b.kind == "adduct":
            ptm_pts.append(sio.residue_slice(tb.variants[0], b.ligand_key).coord)
    center = np.vstack(ptm_pts)

    def prep(arr_v, tag):
        x = arr_v
        if a.crop_radius:
            protect = {(r.key.chain, r.key.res_id) for r in reqs}
            x = rfd3_prep.crop_around(x, center, a.crop_radius, pad=a.crop_pad, protect=protect)
        pt = rfd3_prep.prepare_for_rfd3(x, keep_chains=a.chains.split(",") if a.chains else None)
        cif = out / "rfd3" / f"{a.name}_{tag}.cif"
        sio.write_cif(pt.atoms, cif)
        return pt, str(cif)

    specs = {}
    for k, v in enumerate(tb.variants):
        pt, cif = prep(v, f"ptm_k{k}")
        sites = rfd3_inputs.detect_sites(pt.atoms)
        lig = [s.code for s in sites if s.kind == "adduct"]
        for mode in a.hotspots.split(","):
            hb = mode.endswith("_hb")                       # e.g. ptm_hb = hotspots + H-bond-acceptor conditioning
            base = mode[:-3] if hb else mode
            nn = 3
            if base.startswith("patch") and base[5:].isdigit():   # patch5 = patch with 5 neighbour residues
                nn, base = int(base[5:]), "patch"
            hs = rfd3_inputs.hotspot_dict(pt.atoms, sites, mode=base, n_neighbors=nn)
            specs[f"{a.name}_ptm_k{k}_{mode}"] = rfd3_inputs.make_spec(
                cif, pt.contig_target(), hs, lig, binder_length=a.binder_length,
                hbond_acceptor=rfd3_inputs.hbond_acceptors(pt.atoms, sites) if hb else None)
        manifest["variants"].append({"k": k, "cif": cif, "segments": pt.segments, "ligand_chains": pt.ligand_chains,
                                     "sites": [s.__dict__ for s in sites],
                                     "seg_map": {f"{c}:{r}:{i}": list(v_) for (c, r, i), v_ in pt.seg_map.items()}})
    # unmodified control target (same crop) for selectivity checks
    ptc, cifc = prep(tb.unmodified, "unmod")
    manifest["control"] = {"cif": cifc, "segments": ptc.segments}
    # control spec: same epitope (anchor + parent side chain atoms that exist), no modification
    pt0, _ = prep(tb.variants[0], "ptm_k0")
    ctrl_hs = {}
    for s_ in rfd3_inputs.detect_sites(pt0.atoms):
        if s_.kind != "replace":
            continue                                   # glycan/adduct sites have no parent-side-chain analogue
        have = set(rfd3_inputs._residue_atoms(ptc.atoms, s_.chain, s_.res_id).atom_name)
        atoms = [n for n in rfd3_inputs.parent_epitope_atoms(s_) if n in have]
        if atoms:
            ctrl_hs[s_.key] = ",".join(atoms)
    if ctrl_hs:
        specs[f"{a.name}_unmod_ctrl"] = rfd3_inputs.make_spec(cifc, ptc.contig_target(), ctrl_hs, [],
                                                              binder_length=a.binder_length)
    (out / "rfd3" / "specs.json").write_text(json.dumps(specs, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({"files": files, "specs": list(specs), "resolved": tb.report["already_resolved"]}, indent=1))


def cmd_prepare(a):
    """Dry run of what `python -m ptm_rfd3.rfd3_patch inputs=SPEC` does before RFD3 starts."""
    from . import rfd3_run
    new = rfd3_run.expand_inputs(a.spec, a.out, n_confs=a.n_confs)
    print(new or f"{a.spec}: no spec uses PTM keys ({', '.join(rfd3_run.EXTRA_KEYS)}); RFD3 would read it unchanged")


def cmd_design(a):
    """`ptm-rfd3 design <rfd3 hydra args>`: the normal RFD3 CLI + PTM keys in the spec file (builds the PTMs first)."""
    import sys
    from . import rfd3_patch
    sys.argv = ["rfd3"] + a.rfd3_args
    rfd3_patch.main()


def cmd_stage(a):
    from . import stage
    from .stage import _parse_labelled
    tars = _parse_labelled(a.rfd3_tar)
    res = stage.stage(tars, a.out, top=a.top, n_seq=a.n_seq, workers=a.workers, seed=a.seed, ala_bias=a.ala_bias,
                      phos_bias=a.phos_bias, temperature=a.temperature, af3_spec=a.af3_spec, two_stage=a.two_stage, geometry_only=a.geometry_only,
                      mpnn_model=a.mpnn_model, keep_sequence=a.keep_sequence, geometry_gate=not a.no_geometry_gate,
                      exclude_uids=set(Path(a.exclude_list).read_text().split()) if a.exclude_list else None,
                      include_uids=set(Path(a.include_list).read_text().split()) if a.include_list else None)
    print(json.dumps(res, indent=1))
    if "af3_tar" in res:                                        # --geometry-only writes no AF3 inputs
        print("next: IN_TAR=%s TAR=<out.tar> sbatch scripts/slurm/af3_batch.sbatch ; then `ptm-rfd3 score --tar <out.tar> --index %s --out scores.csv`"
              % (res["af3_tar"], res["index"]))


def cmd_control(a):
    from . import stage
    print(json.dumps(stage.control_stage(a.scores, a.index, a.manifests, a.out), indent=1))


def cmd_score(a):
    from . import selectivity as SEL
    df = SEL.score_tar(a.tar, a.index)
    if df.attrs.get("n_missing"):
        print(f"WARNING: {df.attrs['n_missing']} indexed predictions have no usable AF3 output and are NOT in the scores")
    raw = str(Path(a.out).with_suffix("")) + "_raw.csv"
    df.to_csv(raw, index=False)                                           # per-prediction scores (input of `ptm-rfd3 control`)
    paired = SEL.pair_arms(df)
    if paired.attrs.get("n_unpaired"):
        print(f"note: {paired.attrs['n_unpaired']} binders have only one arm scored (expected for a two-stage screen) and are not in the pairs")
    if paired.empty:
        print(f"{len(df)} predictions scored -> {raw} (no PTM/control pairs yet: run `ptm-rfd3 control --scores {raw} ...` for stage 2)")
        return
    paired.to_csv(a.out, index=False)
    print(f"{len(paired)} binders scored -> {a.out}")
    d = paired.d_tm.values
    if len(d) > 5:
        lo, hi = SEL.bootstrap_ci(d)
        print(f"token-matched delta: median {np.median(d):+.3f} (95% CI {lo:+.3f}..{hi:+.3f}); fraction >= 0.10: {np.mean(d >= .10):.2f}")


def cmd_select(a):
    import pandas as pd
    from . import selectivity as SEL
    paired = pd.read_csv(a.scores)
    ok = SEL.select(paired, dict(d_tm=a.min_delta))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ok.to_csv(Path(a.out) / "selected.csv", index=False)
    n = SEL.export_models(ok, a.tar, a.out) if a.tar else 0
    print(f"{len(ok)} of {len(paired)} binders pass (PTM ipTM>=0.5, >=80% PTM atoms contacted, min dist>=2.2 A, pLDDT>=85, matched delta>={a.min_delta}); {n} models -> {a.out}")
    print("confirm the best with 5 AF3 seeds before trusting them (single-seed hits regress to the mean): `ptm-rfd3 seeds --scores %s --index <af3_index.json> "
          "--af3-inputs <af3_inputs.tar> --out SEEDS`, AF3, `score`, `confirm`." % a.scores)


def cmd_seeds(a):
    import pandas as pd
    from . import confirm as C
    index_rows = [r for ix in a.index for r in json.load(open(ix))]
    res = C.make_seed_inputs(pd.read_csv(a.scores), index_rows, a.af3_inputs, a.out, n_seeds=a.n_seeds, crit=dict(d_tm=a.min_delta))
    print(json.dumps(res, indent=1))
    print("next: IN_TAR=%s TAR=<seeds_out.tar> sbatch scripts/slurm/af3_batch.sbatch ; `ptm-rfd3 score --tar <seeds_out.tar> --index %s --out seed_scores.csv` ; "
          "`ptm-rfd3 confirm --scores seed_scores.csv --out confirmed.csv`" % (res["af3_tar"], res["index"]))


def cmd_confirm(a):
    import pandas as pd
    from . import confirm as C
    out = C.summarize(pd.read_csv(a.scores), n_seeds=a.n_seeds, min_worst=a.min_worst, require_engagement=a.require_engagement)
    out.to_csv(a.out, index=False)
    print(f"{int(out.confirmed.sum())} of {len(out)} binders are selective in all {a.n_seeds} seeds (worst-seed matched delta > {a.min_worst}"
          + (", PTM engagement filters in every seed" if a.require_engagement else "") + f") -> {a.out}")
    print(f"{int((out.selective_all & out.engaged_all).sum())} of them also keep the PTM engaged (>= 80% of its atoms within 4 A, closest approach >= 2.2 A) in every seed "
          "(column engaged_all; `--require-engagement` makes this the criterion)")


def main(argv=None):
    import sys
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "design":                          # everything after `design` belongs to RFD3 (also leading --options)
        return cmd_design(argparse.Namespace(rfd3_args=raw[1:]))
    ap = argparse.ArgumentParser(prog="ptm-rfd3")
    sub = ap.add_subparsers(required=True)
    q = sub.add_parser("design", help="RFD3 with PTM keys in the spec (pass the usual RFD3 hydra args, e.g. inputs=spec.json out_dir=OUT ckpt_path=...)")
    q.add_argument("rfd3_args", nargs=argparse.REMAINDER)
    q.set_defaults(fn=cmd_design)
    q = sub.add_parser("stage", help="after RFD3: geometry filter + PTM-aware MPNN + AF3 inputs (PTM arm and unmodified control)")
    q.add_argument("rfd3_tar", nargs="+", help="RFD3 output tar(s), optionally LABEL=path")
    q.add_argument("--out", required=True)
    q.add_argument("--top", type=int, default=6, help="designs per (spec, conformer) that go to sequence design")
    q.add_argument("--n-seq", type=int, default=2)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--phos-bias", action="store_true", help="add a second sequence set with the phosphate Arg/Lys bias (lowers selectivity on average, see README)")
    q.add_argument("--af3-spec", default=None, help="score all designs against this spec's built target (e.g. the PTR spec when RFD3 saw a ligand representation)")
    q.add_argument("--two-stage", action="store_true", help="AF3 inputs for the PTM arm only; run `ptm-rfd3 control` afterwards (saves ~1/3 of the AF3 time)")
    q.add_argument("--geometry-only", action="store_true", help="only the geometry/engagement statistics per run and spec (evaluation of RFD3 checkpoints; no MPNN, no AF3 inputs)")
    q.add_argument("--temperature", type=float, default=0.1, help="MPNN sampling temperature")
    q.add_argument("--mpnn-model", default="released",
                   help="LigandMPNN weights: released (upstream weights, PTM-aware parsing; default = the validated EGFR campaign), ptm-ft (PTM fine-tuned, shipped in LigandMPNN_PTM/), "
                        "stock (plain upstream LigandMPNN, --ptm_aware 0: the reference arm of the benchmarks) or a checkpoint path")
    q.add_argument("--keep-sequence", action="store_true",
                   help="use the binder sequence that is already in each design (generators that co-design sequence and structure, e.g. RFD4); no LigandMPNN")
    q.add_argument("--no-geometry-gate", action="store_true",
                   help="send EVERY design to AF3 (geometry is still computed and recorded): the 'no pre-filter' control of a generator, not the pipeline")
    q.add_argument("--exclude-list", default=None, help="text file, one design uid (label__name) per line: these designs are not staged (an independent, fresh sample)")
    q.add_argument("--include-list", default=None, help="text file, one design uid (label__name) per line: ONLY these designs are staged (the same backbones for every model of a comparison)")
    q.add_argument("--ala-bias", type=float, default=1.0, help="penalty on Ala in MPNN (PTM-aware MPNN drifts to Ala-rich sequences; best median of 5 configs in N015, n.s.)")
    q.set_defaults(fn=cmd_stage)
    q = sub.add_parser("control", help="two-stage screen, stage 2: control-arm AF3 inputs only for binders whose PTM-arm prediction passes the engagement filters")
    q.add_argument("--scores", required=True, help="per-prediction scores of the PTM-arm run (`ptm-rfd3 score` output has pairs; use selectivity.score_tar CSV)")
    q.add_argument("--index", required=True)
    q.add_argument("--manifests", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_control)
    q = sub.add_parser("score", help="token-matched PTM-vs-control selectivity of an AF3 run (+ bootstrap CI)")
    q.add_argument("--tar", required=True, nargs="+", help="AF3 output tar(s)")
    q.add_argument("--index", required=True, nargs="+", help="index json(s) of the run (stage 1 and, for a two-stage screen, the control index)")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_score)
    q = sub.add_parser("select", help="binders passing the engagement + selectivity filters; exports their AF3 models")
    q.add_argument("--scores", required=True)
    q.add_argument("--tar", default=None)
    q.add_argument("--out", required=True)
    q.add_argument("--min-delta", type=float, default=0.15)
    q.set_defaults(fn=cmd_select)
    q = sub.add_parser("seeds", help="AF3 inputs with N seeds for the single-seed hits (winner's-curse check, run before trusting a binder)")
    q.add_argument("--scores", required=True, help="paired scores (`ptm-rfd3 score` output)")
    q.add_argument("--index", required=True, nargs="+", help="index json(s) of the screen (`stage` output, + control index of a two-stage screen)")
    q.add_argument("--af3-inputs", required=True, nargs="+", help="AF3 input tar(s) of the screen (`stage` output, + control inputs)")
    q.add_argument("--out", required=True, help="output prefix")
    q.add_argument("--n-seeds", type=int, default=5)
    q.add_argument("--min-delta", type=float, default=0.15, help="single-seed token-matched delta a binder needs to be re-run")
    q.set_defaults(fn=cmd_seeds)
    q = sub.add_parser("confirm", help="per-binder summary over the AF3 seeds of `seeds` + `score`; flags binders selective in every seed")
    q.add_argument("--scores", required=True, help="`ptm-rfd3 score` output of the seed run")
    q.add_argument("--out", required=True)
    q.add_argument("--n-seeds", type=int, default=5)
    q.add_argument("--min-worst", type=float, default=0.05, help="the worst seed must exceed this token-matched delta")
    q.add_argument("--require-engagement", action="store_true", help="also require the PTM-engagement filters to hold in every seed (stricter; 4 of the 27 EGFR binders pass)")
    q.set_defaults(fn=cmd_confirm)
    q = sub.add_parser("prepare", help="expand an RFD3 spec file with PTM keys into plain RFD3 specs (builds the PTMs)")
    q.add_argument("spec")
    q.add_argument("--out", required=True, help="work dir for the built targets and the expanded specs")
    q.add_argument("--n-confs", type=int, default=300)
    q.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("inputs", help="build PTMs + write RFD3 specs")
    p.add_argument("--target", required=True)
    p.add_argument("--ptm", action="append", required=True, help="RES=NAME, e.g. A45=phospho-Ser (repeatable)")
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chains", default="", help="comma list of protein chains to keep")
    p.add_argument("--binder-length", default="60-100")
    p.add_argument("--hotspots", default="ptm", help="comma list of modes: ptm,ptm_site,patch (suffix _hb adds H-bond-acceptor conditioning)")
    p.add_argument("--crop-radius", type=float, default=0.0)
    p.add_argument("--crop-pad", type=int, default=2)
    p.add_argument("--n-confs", type=int, default=300)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--formats", default="cif,pdb", help="user-facing PTM-target formats (RFD3 always gets CIF)")
    p.set_defaults(fn=cmd_inputs)
    a = ap.parse_args(raw)
    try:
        a.fn(a)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as e:     # readable failure instead of a traceback for user errors
        print(f"ptm-rfd3 {raw[0]}: error: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
