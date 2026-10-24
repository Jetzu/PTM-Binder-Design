# PTM-aware LigandMPNN

A fork of [LigandMPNN](https://github.com/dauparas/LigandMPNN) (Dauparas et al.) that **understands post-translationally modified (PTM) residues and covalently attached ligands**. It is meant as the sequence-design stage of PTM binder design: give it a target that carries a phospho-tyrosine, an acetyl-lysine, a glycan, … and design a binder around it.

It works with the **released LigandMPNN weights, no retraining**, and ships an optional **PTM fine-tuned checkpoint** that is clearly better near modifications (+6.7 pts for phospho-residues vs stock). Inputs without any modification behave exactly like upstream with the released weights (tested).

**Contents:** [At a glance](#performance-at-a-glance) · [Why](#why-this-fork-exists) · [Installation](#installation) · [Quick start](#quick-start) · [Binder against a PTM target](#designing-a-binder-against-a-ptm-target) · [Usage cookbook](#usage-cookbook) · [What it does](#what-it-does) · [Flags](#ptm-flags) · [Output](#output-files) · [Recovery benchmarks](#does-it-help) · [Fine-tuned weights](#fine-tuned-weights) · [Validation vs stock](#validation-against-stock-ligandmpnn) · [Phospho-specific binders](#phospho-specific-binders-what-af3-says) · [Limitations](#limitations) · [Fine-tuning pipeline](#fine-tuning-pipeline) · [Reproducing](#reproducing-the-benchmarks) · [Environment variables](#environment-variables-research-scripts) · [Troubleshooting](#troubleshooting) · [Tests](#tests) · [Layout](#repository-layout) · [Credits](#credits-and-license)

---

## Performance at a glance

All numbers: held-out structures released in 2023 or later (never in the PDB snapshot the released model could have seen, homologues of the training data removed), paired comparisons with 95 % bootstrap CIs, details and caveats in the sections below. "Fine-tuned" = `model_params_ptm/ligandmpnn_ptm_ft_v2A.pt`.

| question | result |
|---|---|
| Does PTM-aware parsing alone help? | yes, near the modification: +1.4 pts native recovery for biological PTMs, +5.1 pts for phospho-residues vs stock parsing (same weights) |
| Does the fine-tune help further? | yes: vs stock **+2.3 pts (all biological PTMs), +6.7 pts (phospho-residues)** within 8 Å of a modification (full test set); binder-style setting (PTM chain fixed, partner designed) **+9.6 pts for phospho-residues** (33 structures); replicated on 3 training seeds |
| Any regression on ordinary (non-PTM) design? | **no**: LigandMPNN's own test sets +0.4 pts recovery; designability (ESMFold self-consistency) of non-PTM backbones 85.1 % vs 83.3 % (n.s.) |
| Designability of PTM-binder sequences | unchanged (98-99 % of EGFR-binder sequences fold back onto their backbone for stock, PTM-aware and fine-tuned); mean pLDDT 0.5-0.8 points lower |
| Does it design phospho-binding pockets better? | **yes**: in 16 natural phospho-peptide complexes the redesigned binder recovers the native pocket residues 66 % of the time vs 50 % for stock, with 22 % basic residues at the pocket vs 11 % (native 24.5 %) |
| Higher AF3 binding confidence for redesigned natural binders? | no measurable difference (ipTM -0.01 [-0.07, +0.04], 18 complexes; AF3 without MSA is a weak judge) - [details](#validation-against-stock-ligandmpnn) |
| Phospho-**specific** binders from the sequence designer alone? | no model does this on average (1-2 % of sequences are specific); PTM-aware models make binders engage the phosphate far more often (46 % -> 80 %) - [details](#phospho-specific-binders-what-af3-says) |
| Glycans / covalent ligands | no regression, no site-specific gain (stock already sees the glycan atoms) |
| More PTM types | lysine acetylation / methylation and oxidised Cys: no extra gain from the fine-tune (and none from sampling them more often during training) |

## Why this fork exists

Stock LigandMPNN only knows the 20 standard amino acids. Reading a structure with a modified residue goes wrong in one of two ways:

| Situation in the input | What stock LigandMPNN does |
|---|---|
| Residue ProDy calls "protein" but LigandMPNN does not know (SEP, TPO, PTR, MSE, CSO …) | becomes token `X`, **its modification atoms (e.g. the phosphate) are silently dropped**, and it is redesigned like any other residue |
| Residue ProDy does not know (HYP, MLY, M3L, ALY, …) | turned into *ligand atoms* and **removed from the chain** (a gap in the residue numbering) |
| Covalent ligand (glycan, covalent inhibitor) | atoms are context, but the residue carrying it can be redesigned away from Asn/Cys/… |

On 140 real PTM structures stock LigandMPNN dropped residues from the chain in 53 and produced `X` tokens in 93 of them. For a binder-design pipeline this is the worst case: the modification is exactly the thing the binder should recognise, and the design model cannot see it.

## Installation

Requirements: **Python 3.12** (the tested, pinned versions of `requirements_tested.txt` need >= 3.12; older Python needs unpinned numpy/scipy and was not tested), PyTorch, ProDy, SciPy, NumPy; `ml-collections` + `dm-tree` + `absl-py` only for side-chain packing (`--pack_side_chains 1`; without them `run.py` still designs sequences and says what is missing if you ask for packing). A GPU is optional: design runs on CPU in seconds to a few minutes per structure and uses a CUDA GPU automatically when PyTorch sees one. The versions below were resolved in a clean virtual environment in which **all 25 tests pass** (`requirements_tested.txt`); the upstream pins in `requirements.txt` are older and not what this fork was tested with.

```bash
git clone https://github.com/Jetzu/PTM-Binder-Design.git && cd PTM-Binder-Design/LigandMPNN_PTM

python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel; for a GPU: `pip install torch` (CUDA build) or see https://pytorch.org
pip install -r requirements_tested.txt                # numpy, scipy, ProDy, ml-collections, dm-tree, absl-py, biopython (torch is installed above)
bash get_model_params.sh ./model_params               # released LigandMPNN weights (~120 MB, not in git)
```

* The PTM fine-tuned checkpoint is **in the repository** (`model_params_ptm/ligandmpnn_ptm_ft_v2A.pt`, 10.5 MB); nothing extra to download.
* The vendored `openfold` code was patched (`np.int` -> `int`) so side-chain packing works with NumPy >= 1.24.
* ProDy 2.6.1 has no prebuilt wheel for every platform on Python 3.12: pip then compiles it and needs a C compiler (e.g. `gcc`) and the Python headers.

## Quick start

After [installation](#installation):

```bash
python run.py --model_type ligand_mpnn \
    --pdb_path examples/1SHA.cif \
    --out_folder ./outputs/ptm_example \
    --pack_side_chains 1 --seed 1
```

`examples/1SHA.cif` is a slice of PKA with a phospho-threonine (TPO 197) and a phospho-serine (SEP 338). The run prints what it found and what it will do (excerpt; the full log also lists the residues to be redesigned / fixed and one `Type:` line per ligand-context atom, and side-chain packing prints a harmless PyTorch `UserWarning` from the vendored openfold code):

```
Modified residues (PTM) found:
  index 11: TPO -> parent THR, 4 modification atoms, fixed
  index 23: SEP -> parent SER, 4 modification atoms, fixed
The number of ligand atoms parsed is equal to: 46
```

Input can be **PDB or mmCIF** (`.pdb`, `.cif`, `.cif.gz`); chain ids are the author chain ids, like in PDB files. With a covalent glycan the log looks like this:

```
Covalent protein-ligand links found:
  A78 ASN:ND2 -- NAG:C1:C1 1.45 A
```

To use the fine-tuned weights (recommended for PTM targets) add `--checkpoint_ligand_mpnn model_params_ptm/ligandmpnn_ptm_ft_v2A.pt` - see [Fine-tuned weights](#fine-tuned-weights).

Side-chain packing (`--pack_side_chains 1`) needs `ml-collections`, `dm-tree` and `absl-py` (all in `requirements_tested.txt`).

## Designing a binder against a PTM target

The PTM sits on the **target**; the **binder** is what you design. Mark the target chain as fixed by listing only the binder chain in `--chains_to_design`. The repository ships a small example, `examples/1SHA.cif`: an SH2 domain (chain A) in complex with a phospho-tyrosine peptide (chain B, carries PTR). Here the domain plays the binder and is redesigned around the phosphopeptide:

```bash
python run.py --model_type ligand_mpnn \
    --pdb_path examples/1SHA.cif \
    --chains_to_design A \
    --checkpoint_ligand_mpnn model_params_ptm/ligandmpnn_ptm_ft_v2A.pt \
    --pack_side_chains 1 --number_of_packs_per_design 4 \
    --batch_size 8 --number_of_batches 4 --temperature 0.1 \
    --out_folder outputs/binder
```

For your own complex (e.g. binder backbones from RFdiffusion3 around a modified target) replace `--pdb_path` and `--chains_to_design`; chain ids are the author chain ids of the file. **Runtime:** this command (32 sequences x 4 side-chain packs, 103 designed residues) takes about 30 s on an idle 48-core CPU node and a few seconds on a GPU; without `--pack_side_chains` it is a few seconds, a single-sequence smoke test (`--batch_size 1 --number_of_batches 1`) about 40 s including start-up. Cost grows with chain length and with the number of sequences and packs; on a shared or busy node PyTorch can oversubscribe the cores and the command may take many minutes: set `OMP_NUM_THREADS=8` (40 s instead of > 15 min on a busy 64-core login node), and use `python -u` if you redirect stdout, otherwise the log stays empty until the end. The log lists what was recognised (here `PTR -> parent TYR ... fixed`, designed residues A2-A104, fixed residues B201-B205) and the FASTA header carries `ptm_fixed=B201:PTR`.

What happens: the target chain is kept as is, the PTM residue on it is read as its parent type (PTR -> Tyr in the sequence) **and its phosphate/side-chain atoms are given to the model as ligand-like context**, so the binder residues around the modification are designed knowing it is there. Everything else is stock LigandMPNN (`--omit_AA`, `--bias_AA`, `--fixed_residues`, `--symmetry_residues`, … all work; see [README_upstream.md](README_upstream.md)).

Rule of thumb: use `ligand_mpnn` (the only model that uses atom context). `protein_mpnn` / `soluble_mpnn` still parse PTMs correctly but cannot see the modification atoms.

## Usage cookbook

Residue ids everywhere are `<chain><number>` as in the input file (author chain ids and numbering; insertion codes as `A52A`). All commands assume the installation above and add `--model_type ligand_mpnn`; append `--checkpoint_ligand_mpnn model_params_ptm/ligandmpnn_ptm_ft_v2A.pt` to use the fine-tuned weights. `--seed N` makes a run reproducible; `--batch_size` x `--number_of_batches` = number of sequences.

| I want to ... | command (besides `--pdb_path <file> --out_folder out`; `<file>` = `.pdb`, `.cif` or `.cif.gz`) |
|---|---|
| design everything except the modified residues (default) | nothing extra - PTM and covalently modified residues are fixed automatically |
| design only some residues, keep everything else | `--redesigned_residues "A12 A15 B3"` (only the listed residues are designed; modified residues stay fixed unless listed) |
| keep some residues, design the rest | `--fixed_residues "A1 A2 A3"` |
| design one chain around a fixed PTM target | `--chains_to_design B` (see the binder example above) |
| redesign a modified residue itself | list it in `--redesigned_residues "E197"`: it is designed as a standard residue and loses its modification atoms (the log says "redesigned (modification dropped)") |
| change what the model sees of a modification | `--ptm_context_atoms sidechain`, `extras` or `none` (default `sidechain`) |
| behave exactly like upstream LigandMPNN | `--ptm_aware 0` |
| many structures in one call | `--pdb_path_multi inputs.json` (JSON `{"path.pdb": ""}`; per-structure `--fixed_residues_multi` / `--redesigned_residues_multi` as in [README_upstream.md](README_upstream.md)) |
| score the log-probabilities of a given sequence | `python score.py --model_type ligand_mpnn --autoregressive_score 1 --use_sequence 1 --batch_size 1 --number_of_batches 10` (writes `out/<name>.pt`; PTM flags work as in `run.py`) |
| side-chain conformations of the designs | `--pack_side_chains 1 --number_of_packs_per_design 4` (needs `ml-collections`, `dm-tree`, `absl-py`) |

**Check that the modification was recognised** - the log lists every modified residue and what is done with it, e.g. for `examples/1SHA.cif`:

```
Modified residues (PTM) found:
  index 11: TPO -> parent THR, 4 modification atoms, fixed
  index 23: SEP -> parent SER, 4 modification atoms, fixed
```

and the FASTA header of every output file carries `ptm_fixed=E197:TPO;E338:SEP`. If a modification you expect is missing from that list, the residue name is not in the table of 1125 CCD components (see [What it does](#what-it-does)); `--ptm_auto_detect 1` (default) still keeps unknown peptide-bonded residues as token `X`.

**Typical PTM-binder workflow** (the PTM-RFD3 pipeline uses exactly this): generate binder backbones around the modified target -> `run.py ... --chains_to_design <binder>` with the fine-tuned checkpoint (8-32 sequences per backbone, `--temperature 0.1`) -> fold the sequences with the modified and the unmodified target and keep only designs that are confidently predicted with the modification (see [Phospho-specific binders](#phospho-specific-binders-what-af3-says) for why this screen matters; the sequence model alone does not make binders phospho-specific).

## What it does

Pipeline in `ptm_utils.py` + `data_utils.parse_PDB`:

1. **Recognise modified residues.** A table of **1125 CCD components** (every L-peptide-linking component with a standard parent, built from the wwPDB `components.cif.gz` by `ptm_data/build_modified_residues.py`; includes SEC/PYL) maps e.g. SEP→Ser, TPO→Thr, PTR→Tyr, MSE→Met, HYP→Pro, MLY/M3L/ALY→Lys, CSO→Cys. Such a residue joins the backbone graph and the sequence **as its parent type**, so chain continuity and neighbour residues are right. A residue with an `CA` anchor but missing backbone atoms stays a (masked) residue instead of opening a gap.
2. **Auto-detect unknown residues.** A residue missing from the table that is peptide-bonded to the chain (needs N, CA, C; followed to the end of arbitrarily long runs) is kept as token `X` (`--ptm_auto_detect`).
3. **Find covalent links.** A ligand heavy atom within `r_i + r_j + 0.45 Å` (Cordero covalent radii) of a protein **side-chain** atom marks that residue as covalently modified. Metals and hydrogens are ignored; links through backbone atoms (N, CA, C, O, OXT; e.g. ACE/NME caps) never freeze a residue.
3b. **Find protein-protein cross-links** (`--detect_crosslinks`, on by default in `run.py`/`score.py`). Covalent bonds between two protein residues beyond the peptide bond - the Lys-Gly isopeptide of ubiquitin / SUMO, lanthionine, … - mark both residues as covalently modified. Standard Cys-Cys disulfides, backbone-backbone contacts (also head-to-tail macrocycle closures), metals and hydrogens are ignored. Checked on K48- and K63-linked diubiquitin (1AAR, 3NS8, 2JF5; Lys NZ - Gly C at ~1.3 Å). Cross-links are listed in the log under the same heading as ligand links ("Covalent protein-ligand links found").
4. **Freeze what must not change.** PTM and covalently modified residues are **fixed** (a modification only exists on its residue type). An explicit `--redesigned_residues` list overrides this; a redesigned PTM residue loses its modification atoms.
5. **Show the modification to the model.** The side-chain atoms of fixed PTM residues are appended to the ligand atoms that neighbouring residues see (`--ptm_context_atoms sidechain`, default). Atoms that already reach the model through `--ligand_mpnn_use_side_chain_context` are not counted twice.
6. **Write results faithfully.** Fixed PTM residues are written to `packed/*.pdb` **verbatim** (original residue name, all atoms, coordinates, HETATM flag).

Two additions in `model_utils.py` prepare the fine-tuning (see below): a per-residue modification class and a per-context-atom flag. Both are **zero-initialised, so the released weights give bit-identical outputs** (tested against the code before the change, including seeded sampling).

## PTM flags

Available in `run.py` and `score.py`.

| flag | default | meaning |
|---|---|---|
| `--ptm_aware` | `1` | parse modified residues as parents, keep their atoms, find covalent links. `0` = exact upstream behaviour |
| `--ptm_auto_detect` | `1` | also keep unknown peptide-bonded residues (token `X`) |
| `--detect_crosslinks` | `1` | detect covalent bonds between protein residues (ubiquitin/SUMO isopeptide, lanthionine); both residues are fixed. Disulfides are not affected |
| `--fix_ptm_residues` | `1` | do not redesign modified residues (ignored when `--redesigned_residues` is given) |
| `--fix_covalent_residues` | `1` | do not redesign residues that carry a covalently bound ligand |
| `--ptm_context_atoms` | `sidechain` | atoms of fixed PTM residues shown as context: `sidechain` (whole side chain), `extras` (only the atoms the parent residue lacks: phosphate, SE, methyl/acetyl groups), `none` (parent identity only) |

## Output files

Same layout as upstream (`seqs/`, `backbones/`, `packed/`, optional `stats/`), with these differences when modifications exist:

* FASTA header gets `ptm_fixed=E197:TPO;E338:SEP`; sequences show the **parent** letter (T, S). Without `--seed` the header shows the randomly drawn seed.
* File names follow the input name; for mmCIF inputs the extension stays in the name (`seqs/complex.cif.fa`, `backbones/complex.cif_1.pdb`, `packed/complex.cif_packed_1_1.pdb`). `design_contrast.py` writes `seqs/<name>.fa` only.
* `packed/*.pdb`: fixed PTM residues verbatim; designed residues packed as usual.
* `backbones/*.pdb`: PTM residues carry their original residue name (backbone atoms only).

```
>ptm_tpo_sep, T=0.1, seed=1, num_res=26, num_ligand_res=14, use_ligand_context=True, ligand_cutoff_distance=8.0, batch_size=1, number_of_batches=1, model_path=./model_params/ligandmpnn_v_32_010_25.pt, ptm_fixed=E197:TPO;E338:SEP
```

## Does it help?

Honest summary: **yes near the modification, nowhere else, and parent identity alone does nothing** - the gain comes from showing the model the modification atoms. Numbers are *native-sequence recovery* (fraction of designed residues that equal the native residue, T = 0.1, 64 samples per structure) - a proxy; there is **no wet-lab or AF2/AF3-based validation** of designed sequences yet.

**Setup.** All residues are designed except PTM / covalent residues; recovery is pooled over designed residues whose CA is within 8 Å of a modification atom (`near8`). Compared against stock parsing with the *same* positions fixed. CIs: paired bootstrap over structures.

**Held-out test set** (released weights; 798 structures with PTMs released **in 2023 or later** - never in the PDB snapshot the released model could have seen - UniProt-deduplicated; "bio PTM" = phospho, acetyl/methyl-lysine, HYP, oxidised Cys, …; large entries timed out):

| subset (entries) | stock parsing | PTM-aware, `sidechain` | difference [95 % CI] |
|---|---|---|---|
| all biological PTMs (285) | 0.630 | **0.644** | +0.014 [+0.008, +0.020] |
| phospho-residues (80) | 0.424 | **0.474** | **+0.051** [+0.034, +0.068] |
| lysine modifications (59) | 0.611 | 0.625 | +0.014 [+0.005, +0.022] |
| oxidised / modified Cys (103) | 0.539 | 0.555 | +0.016 [+0.006, +0.026] |
| MSE-only (164) | 0.505 | 0.531 | +0.026 |
| hydroxyproline (36) | 0.845 | 0.835 | **−0.010** [−0.017, −0.002] |
| **control:** residues > 16 Å from any modification | 0.541 | 0.541 | ±0.000 |

Binder-like setting (the PTM-bearing chain fixed, all other chains designed): phospho-residues 0.489 → 0.552 (+0.063 [+0.028, +0.098], 333 positions); all biological PTMs +0.001 (not significant).

`ptm_context_atoms` variants (all biological PTMs, same set): `none` 0.617, `extras` 0.634, `sidechain` 0.644 (stock 0.630) - parent identity alone is *worse* than stock, the modification atoms are what helps. An earlier 140-structure benchmark (3 sampling seeds) agrees: near-site +2.0 pts [+1.5, +2.6], phospho +5.2 pts.


## Fine-tuned weights

`model_params_ptm/ligandmpnn_ptm_ft_v2A.pt` (10.5 MB, in git) is LigandMPNN `v_32_010_25` fine-tuned on PTM / covalent-modification contexts. Use it like any other checkpoint: `--checkpoint_ligand_mpnn model_params_ptm/ligandmpnn_ptm_ft_v2A.pt` (with `--model_type ligand_mpnn`). It expects the PTM-aware parsing of this fork (the default), so the modification atoms are in the context it was trained with.

**Training data** (all from the AF2-2021-12 mmCIF snapshot, i.e. entries released up to Dec 2021): 12,171 PDB entries with a modified residue (6.9k of them MSE-only, 514 with phospho-residues), 4,775 entries with covalent glycans and 18,703 ordinary entries from LigandMPNN's own training list as replay. Examples are 256-residue crops centred on a PTM, glycan or ligand contact; PTM / covalent residues are fixed, the rest designed, with a mix of "design everything", "design the neighbourhood" and "design one chain" masks. Phospho entries are sampled x4, MSE-only x0.3, repeated proteins are down-weighted (1/sqrt(cluster size)), residues near a PTM count x3 in the loss. Everything homologous (>=30 % sequence identity, plus exact short-chain matches, incl. superseded PDB ids) to the test set or to LigandMPNN's own validation/test entries was removed. Weights only, EMA 0.999, lr 5e-5, 8k steps (best-by-validation checkpoint at step 5500), ~0.45 H100-hours.

**Result** - paired difference in native-sequence recovery of the designed residues within 8 Å of a modification, on the **full held-out 2023+ test set** (all 800 structures, 285 biological-PTM entries; shipped checkpoint = training seed 0), 95 % bootstrap CI over structures:

| subset (entries) | vs stock LigandMPNN | vs released weights + PTM-aware context |
|---|---|---|
| all biological PTMs (285) | **+2.3 pts** [+1.6, +3.0] | +0.9 [+0.4, +1.4] |
| phospho-residues (80) | **+6.7 pts** [+4.9, +8.7] | +1.6 [+0.8, +2.5] |
| hydroxyproline (36) | +0.5 [-0.8, +2.0] (n.s.; the -1 pt deficit of the PTM-aware parsing alone is gone) | +1.5 [+0.2, +2.9] |
| lysine modifications (59) | +2.2 [+1.1, +3.3] | +0.8 [0.0, +1.7] |
| oxidised / modified Cys (103) | +1.5 [+0.5, +2.5] | -0.1 [-0.9, +0.7] (n.s.) |
| MSE-only (164) | +3.4 [+2.9, +3.9] | +0.8 [+0.5, +1.2] |

**Replication:** two further training seeds were evaluated on the 598 structures of up to 450 residues (the scope of the first evaluation round; 200 biological-PTM entries, 57 with phospho-residues) and agree with the seed-0 model on that subset: gain over released weights + PTM-aware context +1.2 / +1.3 / +1.3 pts (biological PTMs) and +2.4 / +2.5 / +2.4 pts (phospho-residues); on that shorter subset the seed-0 phospho gain over stock is +8.4 pts, larger than the +6.7 of the full set - the full-set number is the one to quote.

Binder-style setting (PTM chain fixed, other chains designed; 33 phospho and 144 biological-PTM structures have a designed partner chain): vs stock phospho-residues **+9.6 pts** [+5.5, +13.5], all biological PTMs +1.4 [+0.4, +2.4]; vs released + PTM-aware context +3.3 [+1.3, +5.1] and +1.3 [+0.3, +2.3]; lysine / Cys modifications no gain.

**Read this before relying on it**
* The gain near modifications comes from the PTM training data, not from generic continued training. Control: the same recipe trained on replay data only (no PTM or glycan entries) gains +0.08 pts near modifications over released + PTM-aware context (n.s., [-0.21, +0.40]) against +0.90 for the shipped model (v2A - control: +0.82 [+0.36, +1.29]; phospho +1.18, HYP +1.63), while both improve residues > 16 Å from any modification by the same +0.5 pts: the far-from-PTM improvement is generic, the near-PTM one is not. Absolute levels on the full test set (all 285 biological-PTM entries): released + PTM-aware context 0.644, replay-only control 0.645, shipped 0.653.
* The modification atoms are used by the released weights already (stripping them costs 2.6 pts near the site, 4.9 for phospho); the fine-tune does not increase that dependence (+0.14 [-0.24, +0.53]). Its gain is therefore about better recovery given the modification atoms, not about using them more.
* The fine-tune does not help lysine modifications or oxidised Cys beyond what the PTM-aware context already gives, and sampling those classes 4-5x more often during training did not change that.
* Native-sequence recovery is a proxy, and it is close to its ceiling: free-running recovery near modifications is 0.653 (shipped model, all 285 biological-PTM test entries), while the same model scores 0.684 when it is shown the native identities of the residues decoded before (an oracle the sampler never has); sampling at T=0.1 equals near-greedy decoding (0.654), and inference-only refinement sweeps add at most +0.2 pts. In a 22-structure smoke test, LASErMPNN (0.623), ProteinMPNN (0.614) and Caliby (0.454) did not exceed stock LigandMPNN (0.647) near modifications. Hydroxyproline (collagen) entries hold 28 % of the near-modification positions and 46 % of the gain; without them the gain over released + PTM-aware context is +0.7 pts overall.
* Side-chain atoms of the fixed residues as extra context (`--ligand_mpnn_use_side_chain_context 1`) do not help near modifications (-0.3 pts; the fine-tuned weights were not trained with it).
* No designs have been folded or tested in the lab (see the AF3 section for the computational specificity tests).
* No regression elsewhere: on LigandMPNN's own test sets (425 structures: small molecules, metals, nucleotides) recovery near ligands is +0.5 to +0.7 pts and overall +0.4 pts (CIs exclude 0; nucleotides neutral).
* Extra PTM class / atom-flag inputs (see below) did not help (+1.1 vs +1.2 pts) and freezing the core network and training only the context modules was weaker (+0.6), so the shipped checkpoint is the plain weights-only fine-tune; its extra inputs are zero and have no effect.
* A first fine-tune with a constant learning rate and no EMA overfitted the redundant training pool after ~2.5k steps; a version with stronger noise / dropout and heavier phospho sampling was worse than the released weights. The recipe above is what survived.

## Limitations

* **Hydroxyproline (collagen) is slightly worse than stock** (−1 pt, 36 test entries). Not explained: adding backbone atoms to the context does not fix it.
* **D-amino acids** and other non-L-peptide components are only picked up by auto-detect (token `X`); the virtual C-beta assumes L chirality.
* Cross-links between protein residues are detected only by distance (`--detect_crosslinks`); disulfides are deliberately left designable.
* **PTM-aware design does not make binders phospho-specific on its own** (see [Phospho-specific binders](#phospho-specific-binders-what-af3-says)): it makes them engage the modification, but specificity is rare for every model and mostly a backbone question.
* "Verbatim" output covers the parsed conformer (ProDy keeps altloc A; zero-occupancy atoms are skipped unless `--parse_atoms_with_zero_occupancy 1`).
* A ligand residue *named like water* (HOH/WAT/…) is ignored, as upstream does.
* The released LigandMPNN weights have never seen PTM-specific training data: the benefit of the PTM-aware parsing comes from correct inputs (+5 pts phospho); the fine-tuned checkpoint adds PTM-specific training on top (+1.6 pts phospho over the PTM-aware released weights). Expect no miracles for modifications with few contacts.
* Decoding order: fixed residues are decoded first *on average* (LigandMPNN draws `(chain_mask+1e-4)·|randn|`); in ~0.2 % of orders a designed residue can precede a fixed one. Inherited from upstream.
* Fixed, but worth knowing: a symmetry group that mixes fixed and designed residues used to hand the fixed member's identity to designed partners (and ignore `--omit_AA`); the sampler now keeps them separate.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: ml_collections` (or `--pack_side_chains 1 needs ml-collections ...`) | only side-chain packing needs it: `pip install -r requirements_tested.txt`, or drop `--pack_side_chains` |
| an error about a missing `./model_params/*.pt` file | the released weights are not in git: `bash get_model_params.sh ./model_params` (run from the repository root; `--checkpoint_ligand_mpnn` takes any other path) |
| the modified residue is not listed under "Modified residues (PTM) found" | it is not in `ptm_data/modified_residues.json` (1125 CCD components) or has no backbone atoms; check the log for `Covalent protein-ligand links` and the `X` handling described in [What it does](#what-it-does) |
| a residue was redesigned that should have stayed | modified / covalently linked residues are fixed by default; for any other residue list it in `--fixed_residues`. `--redesigned_residues` makes everything *not listed* fixed |
| output sequence has the parent letter (T for TPO) | intended: the sequence shows the parent residue type; the modified residue is kept verbatim in `packed/*.pdb` and named in the FASTA header `ptm_fixed=` |
| results differ between runs | set `--seed`; without it every run draws a new seed (printed in the FASTA header) |
| `UserWarning ... tensor_utils.py ... non-tuple sequence for multidimensional indexing` during side-chain packing | harmless warning from the vendored openfold code with PyTorch >= 2.9 (results unaffected) |
| results look worse with the fine-tuned checkpoint on a non-PTM input | it is tuned for PTM contexts (no regression on ordinary inputs in the benchmarks above, but also no benefit); use the released weights for PTM-free proteins |

## Repository layout

```
run.py, score.py            entry points (upstream) + PTM flags
ptm_utils.py                modified-residue table lookup, polymer split, covalent links, cross-links, freezing policy, PTM flags
data_utils.py, model_utils.py   upstream, extended (mmCIF input, PTM parsing, zero-init PTM inputs)
ptm_data/                   modified_residues.json (1125 CCD components) + builder from components.cif.gz
examples/1SHA.cif           SH2 domain (chain A) with a phospho-tyrosine peptide (chain B, PTR) - the example used below
model_params_ptm/           ligandmpnn_ptm_ft_v2A.pt - PTM fine-tuned checkpoint (10.5 MB)
requirements_tested.txt     package versions verified in a clean environment
README_upstream.md          the original LigandMPNN documentation (all upstream flags and examples)
```

## Credits and license

Fork of [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN); the sequence-design models, weights and training data lists are theirs. **Licence:** the upstream code stays under MIT (`LICENSE`); the additions of this fork, our modifications and the PTM fine-tuned weights are under the [PolyForm Noncommercial License 1.0.0](LICENSE-PTM-NONCOMMERCIAL) — see [`NOTICE.md`](NOTICE.md) for which files fall under which. Please cite the original work:

```
@article{dauparas2023atomic,
  title={Atomic context-conditioned protein sequence design using LigandMPNN},
  author={Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and An, Linna and Anishchenko, Ivan and Glasscock, Cameron and Baker, David},
  journal={Biorxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022}
}
```

The PTM-aware parsing, freezing policy, covalent-link detection, mmCIF input were developed with Claude (Anthropic) and cross-reviewed by Codex (OpenAI).
