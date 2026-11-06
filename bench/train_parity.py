#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   # Exactly the LightGBM that lightgbm3-sys bundles (its lightgbm/VERSION.txt).
#   # A different version is allowed to train a different model, and then this
#   # could not tell a version change from a bug in the port.
#   "lightgbm==4.7.0",
#   "numpy>=1.26",
#   "pandas>=2.3.3",
#   # train.py is imported whole, so its imports have to resolve.
#   "matplotlib>=3.10.6",
#   "scikit-learn>=1.7.2",
#   "seaborn>=0.13.2",
#   "shap>=0.48.0",
# ]
# ///
"""Check that psychic's Rust trainer builds the same model as train.py.

    ./bench/train_parity.py [features.csv]

psychic trains in-process (`src/training/`), as a port of train.py. This runs
both on one features CSV - by default the one the last retrain wrote - and
compares the trees in the two model.txt files. They must be identical, byte for
byte; the parameters section is allowed to differ only in `num_threads`.

train.py is run single-threaded. psychic's LightGBM is built without OpenMP, so
it sums histograms in one fixed order; the Python wheel uses every core, which
changes the last bit of some sums and so of some leaf values. Measured on 80k
rows, that moved scores by at most 4e-14 and no ranking - real, but not a
difference in the port.

Run it after changing train.py, `src/training/`, or the LightGBM version. The
LightGBM parameters are written down twice, in train.py and in `src/training/`,
and this is what notices when they drift apart.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path("/tmp/psychic-train-parity")
DEFAULT_DATA_DIR = Path.home() / ".local/share/psychic"


def load_train_py():
    spec = importlib.util.spec_from_file_location("train_py", REPO / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_rust(csv_path, out_dir):
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=REPO, check=True)
    subprocess.run(
        [
            str(REPO / "target/release/psychic"),
            "internal",
            "train",
            str(csv_path),
            str(out_dir),
        ],
        check=True,
    )


def train_python(csv_path, out_dir, schema_path):
    train_py = load_train_py()
    make_params = train_py.make_params
    train_py.make_params = lambda constraints: {
        **make_params(constraints),
        "num_threads": 1,
    }
    shutil.copy(schema_path, out_dir / "feature_schema.json")
    sys.argv = [
        "train.py",
        str(csv_path),
        str(out_dir / "model"),
        "--data-dir",
        str(out_dir),
    ]
    train_py.main()


def split_model(path):
    """The trees, and the parameters section, of a LightGBM model file."""
    text = path.read_text()
    trees, _, rest = text.partition("\nparameters:\n")
    parameters, _, _ = rest.partition("\nend of parameters\n")
    return trees, parameters.splitlines()


def main():
    csv_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR / "features.csv"
    )
    schema_path = csv_path.parent / "feature_schema.json"
    if not csv_path.exists() or not schema_path.exists():
        sys.exit(
            f"Need {csv_path} and {schema_path} beside it; `psychic retrain` writes both."
        )

    shutil.rmtree(WORK, ignore_errors=True)
    rust_dir, python_dir = WORK / "rust", WORK / "python"
    rust_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)

    print(f"== Rust trainer on {csv_path}")
    train_rust(csv_path, rust_dir)
    print(f"\n== train.py on {csv_path}")
    train_python(csv_path, python_dir, schema_path)

    rust_trees, rust_params = split_model(rust_dir / "model.txt")
    python_trees, python_params = split_model(python_dir / "model.txt")
    rust_stats = json.loads((rust_dir / "model_stats.json").read_text())
    python_stats = json.loads((python_dir / "model_stats.json").read_text())

    print("\n== Comparison")
    print(
        f"best iteration: rust {rust_stats['best_iteration']}, python {python_stats['best_iteration']}"
    )
    param_diff = [
        (r, p)
        for r, p in zip(rust_params, python_params, strict=True)
        if r != p and not r.startswith("[num_threads:")
    ]
    if rust_trees != python_trees or param_diff:
        for r, p in param_diff:
            print(f"parameter differs: rust {r}  python {p}")
        if rust_trees != python_trees:
            print(
                f"trees differ: diff {rust_dir / 'model.txt'} {python_dir / 'model.txt'}"
            )
        sys.exit(1)
    print(f"identical: {rust_trees.count('Tree=')} trees, byte for byte")


if __name__ == "__main__":
    main()
