"""ColabFold MMseqs2 server MSAs for AF3 inputs (login node only: compute nodes have no internet).

`fetch_msa(seq)` caches the a3m by sequence hash. PTMs do not change the MSA (the query row keeps the
parent letter; KCR/BTK/DA2 need X, see af3_inputs.AF3_X_CODES).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.colabfold.com"
CACHE = Path(os.environ.get("PTM_RFD3_MSA_CACHE") or Path.home() / ".cache" / "ptm_rfd3" / "msa")


def _http(url, data=None, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "ptm-rfd3-msa/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as exc:                                  # noqa: BLE001
            print("  http error", exc, "retry", k, flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("http failed " + url)


def fetch_msa(seq: str, name: str = "q", cache: Path = CACHE, verbose: bool = True) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(seq.encode()).hexdigest()[:16]
    p = cache / f"{h}.a3m"
    if p.exists():
        return p.read_text()
    q = urllib.parse.urlencode({"q": f">{name}\n{seq}\n", "mode": "env"}).encode()
    while True:
        r = json.loads(_http(f"{API}/ticket/msa", data=q))
        if r.get("status") in ("PENDING", "RUNNING", "COMPLETE"):
            break
        time.sleep(30)
    tid = r["id"]
    while r["status"] != "COMPLETE":
        if r["status"] in ("ERROR", "UNKNOWN"):
            raise RuntimeError(f"server status {r}")
        time.sleep(5)
        r = json.loads(_http(f"{API}/ticket/{tid}"))
    blob = _http(f"{API}/result/download/{tid}")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        parts = []
        for m in tf.getmembers():
            if m.name.endswith(".a3m"):
                parts.append(tf.extractfile(m).read().decode().replace("\x00", ""))
    text = _combine(parts, seq)
    p.write_text(text)
    if verbose:
        print(f"MSA {h}: {text.count('>')} sequences", flush=True)
    return text


def _combine(parts: list[str], seq: str) -> str:
    """Merge uniref + environmental a3m files into one a3m whose first record is the query."""
    seen, out = set(), []
    for t in parts:
        name = None
        for ln in t.splitlines():
            if ln.startswith(">"):
                name = ln
            elif ln.strip() and name is not None:
                key = ln.replace("-", "").upper()
                if key not in seen:
                    seen.add(key)
                    out.append(f"{name}\n{ln.strip()}\n")
                name = None
    # make sure the first record is exactly the query
    out = [o for o in out if o.split("\n")[1].replace("-", "").upper() != seq.upper()]
    return f">query\n{seq}\n" + "".join(out)
