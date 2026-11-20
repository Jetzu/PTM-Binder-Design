"""RFD4-Proteina as the design generator of the PTM pipeline: PTM spec -> RFD4 condition spec (YAML) -> designs in the
layout `ptm-rfd3 stage` reads (`<name>_model_<i>.json` + `.cif.gz`, binder = chain A, ptm_build/*/manifest.json).

    python -m ptm_rfd3.rfd4_run inputs  SPEC.json WORK_DIR [--lengths 80,95,110]   -> WORK_DIR/rfd4_specs.yaml, rfd4_env.sh
    python -m ptm_rfd3.rfd4_run collect RFD4_OUT_DIR WORK_DIR DEST_DIR                -> DEST_DIR in the RFD3 output layout

The modified target is built exactly as for RFD3 (ptm_rfd3.rfd3_run.expand_inputs: same builder, conformers, hotspot
modes); each expanded spec becomes one RFD4 example per binder length:
    contig      "<L>-<L>, /0, <the RFD3 target segments>"  (a FIXED length: particle guidance needs one token layout per batch)
    C_CRD, C_SEQ  every atom / residue of the target chains (target fixed; a PTM residue in the RFD4 vocabulary, e.g. PTR,
                  SEP, TPO, stays one NCAA token, anything else is atomized by RFD4)
    C_HOT       the RFD3 `select_hotspots` of the hotspot mode (PTM atoms, + side chain, + patch residues)
`rfd4_env.sh` exports PTMSMC_PTM_ATOMS for ptm_rfd3.rfd4_particles (the detect_sites atoms of every modification).
RFD4 keeps the input chain id for preserved segments, so the binder comes out under another letter; `collect` renames
binder -> A and the target chains -> B, C, ... (the RFD3 convention the stage step expects) and adds the clash count.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
from biotite import structure as struc

from biotite.structure.io import pdbx

from . import rfd3_inputs, rfd3_run
from . import structio as sio

log = logging.getLogger("ptm_rfd3.rfd4_run")


def _hot_selection(key: str, atoms: str) -> str:
    m = re.match(r"^([A-Za-z]+)(-?\d+)$", key)
    if not m:
        raise ValueError(f"hotspot key {key!r} is not <chain><residue number>")
    ch, rid = m.groups()
    return f"sel('{ch}/*/{rid}')" if atoms.strip().upper() == "ALL" else f"sel('{ch}/*/{rid}/[{atoms.replace(' ', '')}]')"


def _split_contig(contig: str) -> tuple[str, list[str]]:
    parts = [p.strip() for p in contig.split(",") if p.strip()]
    if "/0" not in parts:
        raise ValueError(f"contig {contig!r}: expected '<binder range>,/0,<target segments>'")
    i = parts.index("/0")
    binder, target = parts[:i], parts[i + 1:]
    if len(binder) != 1 or not re.fullmatch(r"\d+(-\d+)?", binder[0]):
        raise ValueError(f"contig {contig!r}: RFD4 path supports one designed binder segment before /0")
    return binder[0], target


def make_inputs(spec_path: str | Path, work_dir: str | Path, lengths: list[int] | None = None, n_confs: int = 300) -> dict:
    import yaml

    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    expanded = rfd3_run.expand_inputs(spec_path, work / "ptm_build", n_confs=n_confs)
    specs = json.loads(Path(expanded or spec_path).read_text())
    out, ptm_atoms, chains_of = {}, {}, {}
    for name, sp in specs.items():
        cif = str(Path(sp["input"]).resolve())
        atoms = sio.read_structure(cif)
        for s in rfd3_inputs.detect_sites(atoms):
            ptm_atoms.setdefault(s.code, [])
            ptm_atoms[s.code] += [a for a in s.ptm_atoms if a not in ptm_atoms[s.code]]
        brange, target = _split_contig(sp["contig"])
        lo, hi = (int(v) for v in (brange.split("-") + [brange])[:2])
        Ls = lengths or [(lo + hi) // 2]
        chains = sorted({re.match(r"^([A-Za-z]+)", t).group(1) for t in target if re.match(r"^[A-Za-z]", t)})
        tgt_sel = [{"select": f"sel('{c}')"} for c in chains]
        hot = [{"select": _hot_selection(k, v)} for k, v in (sp.get("select_hotspots") or {}).items()]
        for L in Ls:
            if not lo <= L <= hi:
                log.warning("%s: binder length %d is outside the spec range %s", name, L, brange)
            cond = {"C_CRD": {True: tgt_sel}, "C_SEQ": {True: tgt_sel}}
            if hot:
                cond["C_HOT"] = {True: hot}
            out[f"{name}_L{L}"] = {"input": cif, "contig": ", ".join([f"{L}-{L}", "/0"] + target), "conditions": cond}
            chains_of[f"{name}_L{L}"] = chains
    ypath = work / "rfd4_specs.yaml"
    ypath.write_text(yaml.safe_dump(out, sort_keys=False))
    env = ";".join(f"{k}:{','.join(v)}" for k, v in ptm_atoms.items())
    (work / "rfd4_env.sh").write_text(f"export PTMSMC_PTM_ATOMS='{env}'\n")
    (work / "rfd4_index.json").write_text(json.dumps({k: {"spec_key": k.rsplit("_L", 1)[0], "target_chains": chains_of[k]}
                                                         for k in out}, indent=1))
    return dict(yaml=str(ypath), n=len(out), ptm_atoms=env, expanded=expanded)


def _read_any_cif(path: Path) -> struc.AtomArray:
    if path.suffix == ".zst":
        import zstandard
        text = zstandard.ZstdDecompressor().decompress(path.read_bytes()).decode()
    elif path.suffix == ".gz":
        text = gzip.open(path, "rt").read()
    else:
        text = path.read_text()
    import io
    a = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(text)), model=1, use_author_fields=False)
    return a[a.element != "H"]


def clash_count(a: struc.AtomArray, binder_chain: str, scale: float = 0.75) -> int:
    """Binder-target heavy-atom pairs closer than scale x (sum of vdW radii) (the lab's standing RFD4 feasibility rule)."""
    b, t = a[a.chain_id == binder_chain], a[a.chain_id != binder_chain]
    if not len(b) or not len(t):
        return 0
    rb = np.array([struc.info.vdw_radius_single(e) or 1.7 for e in b.element])
    rt = np.array([struc.info.vdw_radius_single(e) or 1.7 for e in t.element])
    d = np.linalg.norm(b.coord[:, None] - t.coord[None], axis=-1)
    return int((d < scale * (rb[:, None] + rt[None])).sum())


def collect(rfd4_out: str | Path, work_dir: str | Path, dest: str | Path, pop_size: int = 0) -> int:
    """RFD4 designs -> RFD3 layout for `ptm-rfd3 stage --keep-sequence`. Returns the number of designs written.
    Output names are `<example>_b<population>_model_<row>` (as RFD3 batches): with particle-guidance logs the population
    and row come from the log (which example ids a full_simulation call held); otherwise replicate i is row i % pop_size of
    population i // pop_size (pop_size = the RFD4 batch size; 0 = one population per example). Guided designs get the
    per-particle `ptmsmc` entry (weight, lineage, reward terms) like RFD3 outputs."""
    src, work, dst = Path(rfd4_out), Path(work_dir), Path(dest)
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((work / "rfd4_index.json").read_text())
    if (work / "ptm_build").is_dir():
        shutil.copytree(work / "ptm_build", dst / "ptm_build", dirs_exist_ok=True)
    logs = {}
    for lp in sorted(Path(src).rglob("*.ptmsmc.json")):
        logs[lp.name] = json.loads(lp.read_text())
    where = {}                                                   # example id -> (population, row, per-particle info)
    for k, lg in enumerate(logs[x] for x in sorted(logs)):
        for row, eid in enumerate(lg.get("example_ids") or []):
            info = dict(scheme=lg["policy"]["scheme"], weight=float(lg["weights"][row]), lineage=int(lg["ancestry"][row]),
                        lam=float(lg.get("lam", 0.0)), **{t: float(v[row]) for t, v in lg["terms"].items()})
            where[str(eid)] = (k, row, info)
    n = 0
    files = sorted(p for p in src.rglob("*_generated.cif*") if p.suffix in (".cif", ".zst", ".gz"))
    for i, p in enumerate(files):
        did = p.name.split("_generated")[0]
        base = next((k for k in sorted(index, key=len, reverse=True) if did.startswith(k)), None)
        if base is None:
            log.warning("collect: %s matches no input example; skipped", p)
            continue
        a = _read_any_cif(p)
        tchains = set(index[base]["target_chains"])        # preserved segments keep their input chain id in RFD4
        pchains = [c for c in dict.fromkeys(a.chain_id.tolist())]
        designed = [c for c in pchains if c not in tchains]
        if len(designed) != 1:
            raise ValueError(f"{p}: cannot tell the binder chain (chains {pchains}, target {tchains})")
        ren = {designed[0]: "A"}
        nxt = iter("BCDEFGHIJKLMNOPQRSTUVWXYZ")
        for c in pchains:
            if c not in ren:
                ren[c] = next(nxt)
        a.chain_id = np.array([ren[c] for c in a.chain_id])
        order = np.argsort([0 if c == "A" else 1 for c in a.chain_id], kind="stable")
        a = a[order]
        if did in where:
            pop, row, info = where[did]
        else:
            rep_i = int(m.group(1)) if (m := re.search(r"_(\d+)$", did)) else 0
            pop, row, info = (rep_i // pop_size, rep_i % pop_size, None) if pop_size else (0, rep_i, None)
        name = f"{base}_b{pop:03d}_model_{row}"
        f = pdbx.CIFFile()
        pdbx.set_structure(f, a)
        import io
        buf = io.StringIO()
        f.write(buf)
        with gzip.open(dst / f"{name}.cif.gz", "wt") as fh:
            fh.write(buf.getvalue())
        meta = {"specification": {"extra": {"example": index[base]["spec_key"]}},
                "metrics": {"n_clashing.interresidue_clashes_w_sidechain": clash_count(a, "A")},
                "rfd4": {"source": str(p), "example": base, "design_id": did, "chain_map": ren}}
        if info is not None:
            meta["ptmsmc"] = info
        (dst / f"{name}.json").write_text(json.dumps(meta, indent=1))
        n += 1
    if logs:
        (dst / "ptmsmc_rfd4_logs.json").write_text(json.dumps(logs))
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inputs")
    a.add_argument("spec"); a.add_argument("work_dir")
    a.add_argument("--lengths", default="", help="comma-separated fixed binder lengths (default: midpoint of the contig range)")
    a.add_argument("--n-confs", type=int, default=300)
    c = sub.add_parser("collect")
    c.add_argument("rfd4_out"); c.add_argument("work_dir"); c.add_argument("dest")
    c.add_argument("--pop-size", type=int, default=0, help="RFD4 batch size (population) for unguided runs")
    args = ap.parse_args(argv)
    if args.cmd == "inputs":
        ls = [int(v) for v in args.lengths.split(",") if v.strip()] or None
        print(json.dumps(make_inputs(args.spec, args.work_dir, ls, n_confs=args.n_confs), indent=1))
    else:
        print(json.dumps({"designs": collect(args.rfd4_out, args.work_dir, args.dest, pop_size=args.pop_size)}))


if __name__ == "__main__":
    main()
