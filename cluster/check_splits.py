"""Print an md5 of every split's (train, val, test) indices per target x protocol; compare across machines."""
import hashlib, importlib, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
from bmd.data import TARGETS, load_dataset
from bmd.splits import make_splits
tr = importlib.import_module("02_train")
for t in TARGETS:
    for p in ("random", "doi", "temporal", "kfold", "material"):
        df = load_dataset(t, dated_only=p == "temporal")
        g = tr.material_key(df) if p == "material" else df["doi_group"].values
        n = 3 if p == "temporal" else 5
        h = hashlib.md5()
        for a, b, c in make_splits(g, "doi" if p == "material" else p, n, 0.2, 0.15, 42, dates=df["pub_date"].values):
            for x in (a, b, c):
                h.update(np.asarray(x, dtype=np.int64).tobytes())
        print(t, p, h.hexdigest())
