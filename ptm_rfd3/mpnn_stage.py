"""Sequence design stage: PTM-aware LigandMPNN on RFD3 backbones.

Uses the PTM-aware LigandMPNN fork that ships in this repository (`LigandMPNN_PTM/`, branch LMPNN of the lab repo); this
module only CALLS its run.py: modified residues keep their parent identity, are fixed, and their modification atoms
become ligand context. CPU is enough (small model); one process per structure, single-threaded.

Configuration (all optional):
  PTM_RFD3_LIGANDMPNN      directory with run.py (default: <repo>/LigandMPNN_PTM)
  PTM_RFD3_MPNN_PYTHON     interpreter that has LigandMPNN's requirements (default: the interpreter running ptm-rfd3)
  PTM_RFD3_MPNN_PYTHONPATH extra PYTHONPATH for that interpreter (e.g. a shim directory for missing packages)

Models (`model=`): "released" = upstream LigandMPNN weights, PTM atoms as `extras` context (what the reported EGFR campaign
used); "ptm-ft" = the PTM fine-tuned checkpoint shipped in the fork, with the `sidechain` context it was trained with;
"stock" = plain upstream LigandMPNN (`--ptm_aware 0`: no PTM handling at all), the reference for benchmarks.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MODELS = {
    "released": dict(checkpoint="model_params/ligandmpnn_v_32_010_25.pt", context="extras"),
    "ptm-ft": dict(checkpoint="model_params_ptm/ligandmpnn_ptm_ft_v2A.pt", context="sidechain"),
    "stock": dict(checkpoint="model_params/ligandmpnn_v_32_010_25.pt", context=None),     # upstream LigandMPNN as is (`--ptm_aware 0`): REFERENCE arm
}


def ligandmpnn_dir() -> Path:
    """Directory of the PTM-aware LigandMPNN (run.py); env PTM_RFD3_LIGANDMPNN or the in-repo copy."""
    d = Path(os.environ.get("PTM_RFD3_LIGANDMPNN") or Path(__file__).resolve().parents[1] / "LigandMPNN_PTM").resolve()
    if not (d / "run.py").is_file():
        raise FileNotFoundError(f"LigandMPNN not found at {d} (no run.py). Use the integrated repository checkout or set "
                                "PTM_RFD3_LIGANDMPNN to a checkout of the LMPNN branch.")
    return d


def mpnn_python() -> str:
    return os.environ.get("PTM_RFD3_MPNN_PYTHON") or sys.executable


def resolve_model(model: str = "released") -> tuple[str, str | None]:
    """(checkpoint path, ptm_context_atoms or None = not PTM-aware) for a model name; any other value is taken as a checkpoint path (context `sidechain`)."""
    spec = MODELS.get(model)
    ckpt, ctx = (str(ligandmpnn_dir() / spec["checkpoint"]), spec["context"]) if spec else (str(Path(model).resolve()), "sidechain")
    if not Path(ckpt).is_file():
        hint = " (run `bash get_model_params.sh ./model_params` inside LigandMPNN_PTM/)" if model in ("released", "stock") else ""
        raise FileNotFoundError(f"LigandMPNN checkpoint not found: {ckpt}{hint}")
    return ckpt, ctx


def phosphate_bias_json(pdb_path: str, out_json: str, aa_bias: dict[str, float] | None = None,
                        cutoff: float = 7.0, anchor_atoms=("P", "S")) -> int:
    """Per-residue bias toward Arg/Lys (salt bridge) / Tyr,His,Ser,Thr,Asn,Gln (H-bond donors) for binder (chain A)
    residues whose CB lies within `cutoff` of a phosphate/sulfate anchor atom of any modified target residue.
    Returns the number of biased residues (LigandMPNN `--bias_AA_per_residue`)."""
    import json
    import numpy as np
    from biotite.structure.io import pdb
    aa_bias = aa_bias or {"R": 1.2, "K": 0.8, "H": 0.3, "Y": 0.2, "S": 0.2, "T": 0.2, "N": 0.2, "Q": 0.2}
    a = pdb.PDBFile.read(pdb_path).get_structure(model=1)
    mod = a[(a.chain_id != "A") & np.isin(a.atom_name, list(anchor_atoms)) & a.hetero]
    if not len(mod):
        json.dump({}, open(out_json, "w"))
        return 0
    cb = a[(a.chain_id == "A") & (a.atom_name == "CA")]
    out = {}
    for r, x in zip(cb.res_id, cb.coord):
        if np.linalg.norm(mod.coord - x, axis=1).min() < cutoff + 3.0:      # CA within cutoff+3 ~ CB within cutoff
            out[f"A{int(r)}"] = dict(aa_bias)
    json.dump(out, open(out_json, "w"))
    return len(out)


_PATH_FLAGS = {"--bias_AA_per_residue", "--omit_AA_per_residue", "--bias_AA_per_residue_multi", "--omit_AA_per_residue_multi",
               "--fixed_residues_multi", "--redesigned_residues_multi", "--pdb_path_multi", "--parse_these_chains_only_multi"}


def _absolutize_extra(extra: list[str]) -> list[str]:
    """run.py runs in the LigandMPNN directory: a file given to it by a relative path must be resolved against the caller's directory first."""
    out = []
    for i, x in enumerate(extra):
        if i and extra[i - 1] in _PATH_FLAGS and os.path.exists(x):
            x = str(Path(x).resolve())
        out.append(x)
    return out


def _run_one(a):
    name, pdb, out, nseq, temp, seed, omit, extra, model = a
    extra = _absolutize_extra(list(extra or []))
    if extra and "--phos-bias" in extra:                         # sentinel (anywhere in the list): per-residue phosphate bias for this structure
        Path(out).mkdir(parents=True, exist_ok=True)
        bj = f"{out}/{name}_bias.json"
        n_b = phosphate_bias_json(pdb, bj)
        extra = [x for x in extra if x != "--phos-bias"] + (["--bias_AA_per_residue", bj] if n_b else [])
    ckpt, context = resolve_model(model)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    if os.environ.get("PTM_RFD3_MPNN_PYTHONPATH"):                      # prepended to an inherited PYTHONPATH
        env["PYTHONPATH"] = os.pathsep.join(x for x in (os.environ["PTM_RFD3_MPNN_PYTHONPATH"], os.environ.get("PYTHONPATH")) if x)
    o = f"{out}/{name}"
    cmd = [mpnn_python(), "run.py", "--model_type", "ligand_mpnn", "--checkpoint_ligand_mpnn", ckpt, "--pdb_path", pdb,
           "--out_folder", o, "--chains_to_design", "A", "--omit_AA", omit, "--number_of_batches", str(nseq),
           "--batch_size", "1", "--temperature", str(temp), "--seed", str(seed), "--save_stats", "0",
           "--pack_side_chains", "0", "--verbose", "0", "--ligand_mpnn_use_atom_context", "1",
           *(["--ptm_aware", "1", "--ptm_context_atoms", context] if context else ["--ptm_aware", "0"])] + list(extra)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ligandmpnn_dir(), env=env, capture_output=True, text=True)
    rows = []
    fa = glob.glob(f"{o}/seqs/*.fa")
    if r.returncode == 0 and fa:
        recs = open(fa[0]).read().split(">")[1:]
        for k, rec in enumerate(recs[1:]):                     # record 0 = the input sequence
            head, seq = rec.strip().split("\n")[:2]
            conf = re.search(r"overall_confidence=([0-9.]+)", head)
            rows.append({"design": name, "seq_idx": k, "binder_seq": seq.split(":")[0].strip(),
                         "mpnn_conf": float(conf.group(1)) if conf else float("nan"), "head": head})
    err = (r.stderr[-300:] or f"exit code {r.returncode}") if r.returncode else ("" if rows else "no sequences in output (return code 0)")
    return name, rows, round(time.time() - t0, 1), err


def design_sequences(pdbs: dict[str, str], out_dir: str, n_seq: int = 8, temperature: float = 0.1,
                     seed: int = 0, omit_aa: str = "C", workers: int = 8, extra_args: list[str] | None = None,
                     extra_by_name: dict[str, list[str]] | None = None, model: str = "released"):
    """pdbs: {design_name: PDB path with binder chain A + PTM-bearing target}. Returns list of row dicts.
    model: "released" | "ptm-ft" | path to a LigandMPNN checkpoint (see module docstring)."""
    out_dir = str(Path(out_dir).resolve())                            # run.py runs in the LigandMPNN directory: every path it gets must be absolute
    pdbs = {n: str(Path(p).resolve()) for n, p in pdbs.items()}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    resolve_model(model)                                              # fail before any worker starts
    jobs = [(n, p, out_dir, n_seq, temperature, seed + i, omit_aa, (extra_by_name or {}).get(n, extra_args or []), model)
            for i, (n, p) in enumerate(pdbs.items())]
    rows, errs = [], {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, r, dt, err in ex.map(_run_one, jobs):
            rows += r
            if err:
                errs[name] = err
    return rows, errs
