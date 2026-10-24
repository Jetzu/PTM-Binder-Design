"""PTM-aware RFD3 inputs: build the modified target ONCE, then hand stock RFD3 an ordinary spec.

An RFD3 input file (JSON/YAML, name -> spec) may carry these extra keys; they are consumed here and never reach RFD3:

  "ptm"             {"A1138": "phospho-Tyr", "A45": "N-GlcNAc", "A9": "KCR", "B3": "adduct:LIG@NZ:C1"}   (or a list of
                    "A1138=phospho-Tyr" strings) - modifications to place on `input`. Already-resolved ones are kept.
                    "keep" = build nothing, only normalise the modifications the input file already contains.
  "ptm_hotspots"    "ptm" | "ptm_site" | "patch" | "patch5" ... (or a list) - fill `select_hotspots` from the modification
                    atoms; explicit `select_hotspots` entries of the spec are kept and win.
  "ptm_conformers"  N (default 1) - one spec per top-N RDKit conformer (names are <name>_ptm_k0.._ptm_kN-1, plus _<mode> when several hotspot modes).
  "ptm_seed"        RDKit sampling seed (default 0).
  "covalent_bonds"  ["A45:ND2-Z:C1", ...] - extra covalent bonds the input file does not carry (CHAIN+RES:ATOM-CHAIN+RES:ATOM).
  "ptm_motifs"      GENERATED chain that carries a PTM / covalent ligand: ["phospho-Ser", {"mod": "N-GlcNAc", "seq": "LNT",
                    "pos": 2, "ss": "strand"}, ...] - each entry becomes a built 3-residue peptide (ideal geometry, modification
                    placed by the RDKit builder) used as a FIXED, indexed motif; put {m1}, {m2}, ... (or {motif}) in the contig where
                    it goes, e.g. "contig": "40-60,{m1},40-60" (the motif position follows from the length ranges). No `input` needed.
                    Several motifs are stacked apart (their own extents + 6 A, or `ptm_motif_spacing` A) so the fixed segments do not overlap.

How covalent bonds get specified (easiest first): nothing for a named modification (the registry knows the link: backbone
bonds for residue replacements, anchor-atom -> ligand atom for glycans/lipids); nothing for a modified input (its
struct_conn / bond records are read); `covalent_bonds` only for links the file lacks.

Residue numbers of the polymer are PRESERVED, so contigs and selections written against the original file stay valid;
the one exception: chains whose `chain+number` keys would exceed 4 characters (residue >= 1000) are shifted to start at 1 (RFD3
keeps its source key in a 4-character column) and your contig/selections are translated with the same offset (see report.json).
Only covalently attached ligands/glycan residues are moved to their own chain letters from the end of the alphabet
(Z, Y, ...); the mapping is printed and stored in the build report. The prepared structures, build report and a manifest
(for the AF3 scoring steps) are written to <work_dir>/<spec name>/ and reused when the same build is requested again.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np

from . import ccd, rfd3_inputs, rfd3_prep
from . import structio as sio
from . import target as T

log = logging.getLogger("ptm_rfd3.rfd3_run")

MOTIF_VERSION = 2                                  # bump when the motif geometry / placement code changes (cache key)
EXTRA_KEYS = ("ptm", "ptm_hotspots", "ptm_conformers", "ptm_seed", "ptm_n_confs", "covalent_bonds", "ptm_motifs", "ptm_motif_spacing")
_BOND_RE = re.compile(r"^\s*([A-Za-z0-9])(\d+)([A-Za-z]?):([A-Za-z0-9'\"*]+)\s*-\s*([A-Za-z0-9])(\d+)([A-Za-z]?):([A-Za-z0-9'\"*]+)\s*$")


def has_ptm_keys(spec: dict) -> bool:
    return isinstance(spec, dict) and ("ptm" in spec or "ptm_motifs" in spec)


def _what(v) -> str:
    """One modification value of the `ptm` key: a string (name / CCD code / `adduct:...` / `smiles:...`) or an object with explicit linkage:
      {"smiles": "*C(C)=O", "anchor": "NZ", "name": "ZAC"}                      custom group, `*` = attachment point (name optional)
      {"adduct": "NAG", "anchor": "ND2", "ligand_atom": "C1", "leaving": ["O1"]}  CCD ligand with explicit linkage"""
    if not isinstance(v, dict):
        return str(v)
    if "smiles" in v:
        if "anchor" not in v:
            raise ValueError(f"custom group {v!r} needs `anchor` (the residue atom the group binds to, e.g. NZ)")
        return "smiles:" + "|".join([v["smiles"], v["anchor"]] + ([v["name"]] if v.get("name") else []))
    if "adduct" in v:
        if "anchor" not in v or "ligand_atom" not in v:
            raise ValueError(f"adduct {v!r} needs `anchor` (residue atom) and `ligand_atom`")
        return f"adduct:{v['adduct']}@{v['anchor']}:{v['ligand_atom']}" + (("-" + ",".join(v["leaving"])) if v.get("leaving") else "")
    raise ValueError(f"unknown modification object {v!r}; use a name / CCD code string, or an object with `smiles` or `adduct`")


def _requests(ptm) -> list[str]:
    if ptm in ("keep", "none", None, [], {}):
        return []
    if isinstance(ptm, dict):
        return [f"{k}={_what(v)}" for k, v in ptm.items()]
    if isinstance(ptm, str):
        return [ptm]
    return [str(x) for x in ptm]


def _add_bonds(arr, bonds: list[str]):
    """Add user-declared covalent bonds ("A45:ND2-Z:C1") to a copy of `arr`."""
    from biotite import structure as struc
    arr = arr.copy()
    if arr.bonds is None:
        arr.bonds = struc.BondList(arr.array_length())
    for text in bonds:
        m = _BOND_RE.match(text)
        if not m:
            raise ValueError(f"bad covalent_bonds entry {text!r}; expected 'A45:ND2-Z1:C1' (CHAIN+RESID:ATOM-CHAIN+RESID:ATOM)")
        c1, r1, i1, a1, c2, r2, i2, a2 = m.groups()
        ends = []
        for c, r, i, a in ((c1, r1, i1, a1), (c2, r2, i2, a2)):
            sel = np.where((arr.chain_id == c) & (arr.res_id == int(r)) & (arr.atom_name == a))[0]
            if len(sel) != 1:
                raise ValueError(f"covalent_bonds {text!r}: atom {c}{r}{i}:{a} matches {len(sel)} atoms (need exactly 1)")
            ends.append(int(sel[0]))
        arr.bonds.add_bond(ends[0], ends[1], struc.BondType.SINGLE)
    return arr


def _build_key(path: Path, reqs: list[str], bonds: list[str], seed: int, n_conf: int, n_confs: int) -> str:
    st = path.stat()
    blob = json.dumps({"p": str(path.resolve()), "size": st.st_size, "mtime": int(st.st_mtime), "ptm": sorted(reqs),
                       "bonds": sorted(bonds), "seed": seed, "k": n_conf, "n": n_confs}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def build_target(spec_name: str, spec: dict, work_dir: Path, spec_dir: Path, n_confs: int = 300):
    """Build (or load from cache) the modified target for one spec. Returns (cif paths per conformer, unmodified cif, report)."""
    src = Path(spec["input"])
    if not src.is_absolute() and not src.exists() and (spec_dir / src).exists():
        src = spec_dir / src
    if not src.exists():
        raise FileNotFoundError(f"{spec_name}: input {spec['input']!r} not found")
    reqs = _requests(spec["ptm"])
    bonds = list(spec.get("covalent_bonds") or [])
    n_conf = max(int(spec.get("ptm_conformers", 1)), 1)
    seed = int(spec.get("ptm_seed", 0))
    n_confs = int(spec.get("ptm_n_confs", n_confs))
    key = _build_key(src, reqs, bonds, seed, n_conf, n_confs)
    d = (work_dir / f"{spec_name}_{key}").resolve()
    rep_p = d / "report.json"
    if rep_p.exists():
        rep = json.loads(rep_p.read_text())
        cifs = [str(d / f"ptm_k{k}.cif") for k in range(rep["n_conformers"])]
        if all(Path(c).exists() for c in cifs):
            log.warning("ptm_rfd3: reusing built target %s", d)
            return cifs, str(d / "unmod.cif"), rep
    d.mkdir(parents=True, exist_ok=True)
    arr = sio.read_structure(str(src))
    if bonds:
        arr = _add_bonds(arr, bonds)
    tb = T.build_modified_target(arr, T.parse_requests(reqs), n_confs=n_confs, top_k=n_conf, seed=seed)
    used = {str(n) for v in tb.variants[:1] for n in v.res_name}
    custom = ccd.custom_components(used)                                   # SMILES-defined groups in this target (kept for the AF3 stage)
    cifs, lig_map, variants, id_off = [], {}, [], {}
    for k, v in enumerate(tb.variants[:n_conf]):
        pt = rfd3_prep.prepare_for_rfd3(v, renumber=False, compact_ids=True)
        id_off = pt.id_offset
        c = d / f"ptm_k{k}.cif"
        sio.write_cif(pt.atoms, c)
        cifs.append(str(c))
        variants.append({"k": k, "cif": str(c), "segments": pt.segments, "ligand_chains": pt.ligand_chains,
                         "sites": [x.__dict__ for x in rfd3_inputs.detect_sites(pt.atoms)],
                         "seg_map": {f"{ch}:{rid}:{ins}": list(nv) for (ch, rid, ins), nv in pt.seg_map.items()}})
        lig_map = {f"{ch}{rid}{ins}": new for (ch, rid, ins), new in pt.lig_map.items()}   # original residue -> ligand chain
    ptc = rfd3_prep.prepare_for_rfd3(tb.unmodified, renumber=False, compact_ids=True)
    sio.write_cif(ptc.atoms, d / "unmod.cif")
    # manifest in the format of `ptm-rfd3 inputs`, so pipeline.write_af3_inputs (PTM arm + unmodified control) works on it
    (d / "manifest.json").write_text(json.dumps(
        {"name": spec_name, "ptm": reqs, "build_report": tb.report, "variants": variants, "custom_components": custom,
         "control": {"cif": str(d / "unmod.cif"), "segments": ptc.segments}}, indent=1, default=str))
    rep = {"input": str(src), "ptm": reqs, "covalent_bonds": bonds, "n_conformers": len(cifs), "id_offset": id_off,
           "ligand_chains": lig_map, "build": tb.report, "custom_components": custom}
    rep_p.write_text(json.dumps(rep, indent=1, default=str))
    log.warning("ptm_rfd3: built %s -> %s (%d conformer(s); ligand chains %s)", spec_name, d, len(cifs), lig_map or "none")
    return cifs, str(d / "unmod.cif"), rep


_TOK = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])(\d+)(?:-[A-Za-z]?(\d+))?(?![A-Za-z0-9])")
SELECT_KEYS = ("unindex", "select_fixed_atoms", "select_unfixed_sequence", "select_hotspots", "select_buried",
               "select_partially_buried", "select_exposed", "select_hbond_donor", "select_hbond_acceptor", "ligand_selection")


def translate_ids(text: str, offsets: dict[str, int]) -> str:
    """Shift 'A1126-1150' / 'A1138' tokens of a contig or selection string by the per-chain id offsets (lengths like '80-110'
    and '/0' have no chain letter and stay)."""
    def sub(m):
        c, a, b = m.group(1), int(m.group(2)), m.group(3)
        off = offsets.get(c, 0)
        return f"{c}{a - off}" + (f"-{int(b) - off}" if b else "")
    return _TOK.sub(sub, text)


def translate_spec(spec: dict, offsets: dict[str, int]) -> dict:
    """Apply the residue-id offsets of a compacted target to the fields of a user spec that name residues."""
    if not offsets:
        return spec
    out = dict(spec)
    if "contig" in out:
        out["contig"] = translate_ids(out["contig"], offsets)
    for k in SELECT_KEYS:
        v = out.get(k)
        if isinstance(v, str):
            out[k] = translate_ids(v, offsets)
        elif isinstance(v, dict):
            out[k] = {translate_ids(kk, offsets): vv for kk, vv in v.items()}
    return out


def check_ranges(spec: dict, atoms) -> None:
    """Every residue a contig / selection names must exist in the prepared input (RFD3 needs gap-free ranges)."""
    have = {(str(c), int(r)) for c, r in zip(atoms.chain_id, atoms.res_id)}
    texts = [spec.get("contig", "")] + [spec.get(k) for k in SELECT_KEYS if isinstance(spec.get(k), str)]
    for k in SELECT_KEYS:
        if isinstance(spec.get(k), dict):
            texts += list(spec[k])
    missing = []
    for t in texts:
        for m in _TOK.finditer(t or ""):
            c, a, b = m.group(1), int(m.group(2)), m.group(3)
            if not any(cc == c for cc, _r in have):
                continue                                              # token of another kind (e.g. an atom name)
            for r in range(a, int(b or a) + 1):
                if (c, r) not in have:
                    missing.append(f"{c}{r}")
    if missing:
        raise ValueError(f"contig/selections name residues that are not in the input: {sorted(set(missing))[:10]} "
                         "(RFD3 ranges must be gap-free; split the range or fix the numbering)")


def _mode(m: str) -> tuple[str, int, bool]:
    hb = m.endswith("_hb")
    base = m[:-3] if hb else m
    nn = 3
    if base.startswith("patch") and base[5:].isdigit():
        nn, base = int(base[5:]), "patch"
    if base not in ("ptm", "ptm_site", "patch"):
        raise ValueError(f"unknown ptm_hotspots mode {m!r} (ptm | ptm_site | patch | patchN, optional _hb suffix)")
    return base, nn, hb


class _Keep(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _norm_motif(m) -> dict:
    return {"mod": m} if isinstance(m, str) else dict(m)


def build_motif_input(spec_name: str, spec: dict, work_dir: Path, n_confs: int = 300):
    """Build the `ptm_motifs` of a spec: one CIF per conformer holding all motifs (chains A, B, ...; glycan/ligand chains Z, Y, ...)
    and the contig placeholder -> segment map. Returns (cifs, unmod cif, report, {placeholder: 'A1-3'})."""
    from . import motif
    motifs = [_norm_motif(m) for m in spec["ptm_motifs"]]
    if len(motifs) > 8:
        raise ValueError("at most 8 ptm_motifs per spec")
    n_conf = max(int(spec.get("ptm_conformers", 1)), 1)
    seed = int(spec.get("ptm_seed", 0))
    key = hashlib.sha1(json.dumps({"v": MOTIF_VERSION, "m": motifs, "k": n_conf, "s": seed, "n": n_confs,
                                   "gap": spec.get("ptm_motif_spacing")}, sort_keys=True).encode()).hexdigest()[:12]
    d = (work_dir / f"{spec_name}_motifs_{key}").resolve()
    segs = {}
    chains = "ABCDEFGH"
    built = []
    for i, m in enumerate(motifs):
        tb, _k = motif.build_motif(m["mod"], m.get("seq"), m.get("pos"), m.get("ss", "helix"), n_confs=n_confs,
                                   top_k=n_conf, seed=seed + i, chain=chains[i])
        built.append(tb)
        n_res = len({int(r) for r in tb.unmodified.res_id[tb.unmodified.chain_id == chains[i]]})
        segs[f"m{i + 1}"] = f"{chains[i]}1-{n_res}"
    if len(motifs) == 1:
        segs["motif"] = segs["m1"]
    d.mkdir(parents=True, exist_ok=True)
    from biotite import structure as struc
    cifs = []
    # fixed motifs must not overlap: stack them along y, each separated by its own extent (a farnesyl tail is long) + `gap` A
    gap = spec.get("ptm_motif_spacing")
    radii = [float(np.linalg.norm(tb.variants[0].coord - tb.variants[0].coord.mean(0), axis=1).max()) for tb in built]
    shifts = [0.0]
    for i in range(1, len(built)):
        shifts.append(shifts[-1] + (float(gap) if gap is not None else radii[i - 1] + radii[i] + 6.0))

    def placed(a, i):
        a = a.copy()
        a.coord = a.coord + np.array([0.0, shifts[i], 0.0], dtype=a.coord.dtype)
        return a

    def unique_ligands(parts):
        """Every motif brings its own ligand chain 'Z': give each ligand residue of the combined input a chain letter of its own."""
        pool = iter(c for c in "ZYXWVUTSRQPONM" if c not in chains)
        out = []
        for i, a in enumerate(parts):
            a = a.copy()
            lig_ch = [c for c in dict.fromkeys(a.chain_id.tolist()) if c != chains[i]]
            new = {c: next(pool) for c in lig_ch}
            a.chain_id = np.array([new.get(c, c) for c in a.chain_id.tolist()])
            out.append(a)
        return out
    n_avail = min(len(tb.variants) for tb in built)
    if n_conf > n_avail:
        log.warning("ptm_rfd3: %s: %d conformer(s) requested, the builder produced %d distinct ones", spec_name, n_conf, n_avail)
        n_conf = n_avail
    for k in range(n_conf):
        parts = unique_ligands([placed(tb.variants[k], i) for i, tb in enumerate(built)])
        arr = struc.concatenate(parts)
        pt = rfd3_prep.prepare_for_rfd3(arr, renumber=False)
        c = d / f"ptm_k{k}.cif"
        sio.write_cif(pt.atoms, c)
        cifs.append(str(c))
    ctl = rfd3_prep.prepare_for_rfd3(struc.concatenate(unique_ligands([placed(tb.unmodified, i) for i, tb in enumerate(built)])), renumber=False)
    sio.write_cif(ctl.atoms, d / "unmod.cif")
    rep = {"motifs": motifs, "segments": segs, "n_conformers": len(cifs), "ligand_chains": list(pt.ligand_chains),
           "build": [tb.report for tb in built]}
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    log.warning("ptm_rfd3: built %d PTM motif(s) for %s -> %s (segments %s)", len(motifs), spec_name, d, segs)
    return cifs, str(d / "unmod.cif"), rep, segs


def expand_spec(name: str, spec: dict, work_dir: Path, spec_dir: Path, n_confs: int = 300) -> dict[str, dict]:
    """One user spec with PTM keys -> one or more plain RFD3 specs (conformers x hotspot modes)."""
    segs = None
    if "ptm_motifs" in spec:
        if "input" in spec or "ptm" in spec:
            raise ValueError(f"{name}: `ptm_motifs` builds its own input; do not combine it with `input` / `ptm`")
        cifs, _unmod, rep, segs = build_motif_input(name, spec, work_dir, n_confs=n_confs)
    else:
        cifs, _unmod, rep = build_target(name, spec, work_dir, spec_dir, n_confs=n_confs)
    if segs is None:
        spec = translate_spec(spec, rep.get("id_offset") or {})        # chains whose keys were > 4 chars were shifted to start at 1
    modes = spec.get("ptm_hotspots") or []
    modes = [modes] if isinstance(modes, str) else list(modes)
    base = {k: copy.deepcopy(v) for k, v in spec.items() if k not in EXTRA_KEYS}
    out: dict[str, dict] = {}
    for k, cif in enumerate(cifs):
        atoms = sio.read_structure(cif)
        if k == 0 and spec.get("contig"):
            check_ranges({**spec, "contig": spec["contig"].format_map(_Keep()) if segs is not None else spec["contig"]}, atoms)
        sites = rfd3_inputs.detect_sites(atoms)
        lig = sorted({s.code for s in sites if s.kind == "adduct"})
        for mode in (modes or [None]):
            sp = copy.deepcopy(base)
            sp["input"] = cif
            sp.setdefault("dialect", 2)
            if segs is not None:                                    # generated chain: motifs are fixed, indexed segments
                contig = sp.get("contig", "")
                for ph, seg in segs.items():
                    contig = contig.replace("{" + ph + "}", seg)
                declared = [ph for ph in segs if ph != "motif"]
                counts = {ph: sp.get("contig", "").count("{" + ph + "}") for ph in segs}
                bad = [ph for ph in declared if counts[ph] + (counts.get("motif", 0) if len(declared) == 1 else 0) != 1]
                if "{" in contig or bad or (len(declared) > 1 and counts.get("motif", 0)):
                    raise ValueError(f"{name}: every declared motif must appear exactly once in the contig as {{m1}}, {{m2}}, ... "
                                     f"(or {{motif}} for a single motif); problem with {bad or sorted(segs)} in {sp.get('contig')!r}")
                sp["contig"] = contig
                sp.setdefault("select_fixed_atoms", {})
                if isinstance(sp["select_fixed_atoms"], dict):
                    for ph, seg in segs.items():
                        if ph != "motif":
                            sp["select_fixed_atoms"].setdefault(seg, "ALL")
            if mode:
                m, nn, hb = _mode(mode)
                hs = rfd3_inputs.hotspot_dict(atoms, sites, mode=m, n_neighbors=nn)
                if sp.get("select_hotspots") not in (None, {}) and not isinstance(sp.get("select_hotspots"), dict):
                    raise ValueError(f"{name}: with ptm_hotspots, select_hotspots must be a dict {{'A12': 'ALL', ...}}, got {sp['select_hotspots']!r}")
                if isinstance(sp.get("select_hotspots"), dict):
                    hs.update(sp["select_hotspots"])           # explicit entries win
                sp["select_hotspots"] = hs
                sp.setdefault("infer_ori_strategy", "hotspots")
                if hb:
                    sp["select_hbond_acceptor"] = rfd3_inputs.hbond_acceptors(atoms, sites)
            if lig and "ligand" not in sp:
                sp["ligand"] = ",".join(lig)            # RFD3 takes ONE comma-separated `ligand` field
            nm = f"{name}_ptm_k{k}" + (f"_{mode}" if len(modes) > 1 else "")      # same naming as `ptm-rfd3 inputs`
            out[nm] = sp
    return out


def load_specs(path: Path) -> dict:
    txt = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


def expand_inputs(path: str | Path, work_dir: str | Path, n_confs: int = 300) -> str | None:
    """Expand every spec of an RFD3 inputs file that has PTM keys. Returns the path of the expanded JSON, or None when
    the file is not a spec file or no spec uses PTM keys (then RFD3 gets the original file untouched)."""
    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in (".json", ".yaml", ".yml"):
        return None
    specs = load_specs(p)
    if not isinstance(specs, dict) or not any(has_ptm_keys(s) for s in specs.values()):
        return None
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    for name, sp in specs.items():
        out.update(expand_spec(name, sp, work, p.parent, n_confs=n_confs) if has_ptm_keys(sp) else {name: sp})
    dst = work / "specs_expanded.json"
    dst.write_text(json.dumps(out, indent=1))
    log.warning("ptm_rfd3: %d spec(s) -> %d expanded spec(s) in %s", len(specs), len(out), dst)
    return str(dst)
