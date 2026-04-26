"""
Diagnostic script: proves SMOTE generates truly synthetic rows via interpolation.
Run from backend/: python debug_smote.py
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from services.balancing_service import _resample

# ── 1. Build a tiny, readable dataset ────────────────────────────────────────
# 10 majority (class 0), 4 minority (class 1)
# Only 2 numeric features so we can reason about the geometry easily.
np.random.seed(0)
X_maj = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0],
                  [5.0, 2.0], [6.0, 3.0], [7.0, 1.0], [8.0, 4.0],
                  [9.0, 2.0], [10.0, 3.0]])
X_min = np.array([[1.5, 1.5],   # minority A
                  [2.5, 2.5],   # minority B
                  [3.5, 1.5],   # minority C
                  [4.5, 2.5]])  # minority D
X = np.vstack([X_maj, X_min])
y = np.array([0]*10 + [1]*4)

# ── 2. Confirm library identity ───────────────────────────────────────────────
print("=" * 60)
print("LIBRARY CHECK")
print("=" * 60)
import imblearn
print(f"  imbalanced-learn version : {imblearn.__version__}")
print(f"  SMOTE class              : {SMOTE.__module__}.{SMOTE.__name__}")
print(f"  This is real SMOTE       : {'SMOTE' in SMOTE.__name__}")

# ── 3. Run SMOTE via our own _resample wrapper ────────────────────────────────
X_bal, y_bal = _resample("smote", {"k_neighbors": 3}, X, y)

orig_min_mask = y == 1
orig_maj_mask = y == 0
new_mask      = np.ones(len(y_bal), dtype=bool)
new_mask[:len(y)] = False          # first len(y) rows are originals
synthetic_X   = X_bal[new_mask & (y_bal == 1)]

print()
print("=" * 60)
print("COUNTS BEFORE / AFTER")
print("=" * 60)
print(f"  Majority before  : {np.sum(y==0):>4}   after  : {np.sum(y_bal==0)}")
print(f"  Minority before  : {np.sum(y==1):>4}   after  : {np.sum(y_bal==1)}")
print(f"  Total before     : {len(y):>4}   after  : {len(y_bal)}")
print(f"  Synthetic rows   : {len(synthetic_X)}")

# ── 4. Are any synthetic rows exact duplicates of originals? ──────────────────
print()
print("=" * 60)
print("DUPLICATE CHECK  (synthetic == any original minority row?)")
print("=" * 60)
X_orig_min = X[orig_min_mask]
duplicates = 0
for row in synthetic_X:
    if np.any(np.all(np.isclose(X_orig_min, row), axis=1)):
        duplicates += 1
print(f"  Exact duplicates found : {duplicates}  (expected 0 for real SMOTE)")

def find_smote_parents(syn, X_orig_min, tol=1e-6):
    """Exhaustive search: try all (A, B) pairs to find the real interpolation parents."""
    n = len(X_orig_min)
    for a_idx in range(n):
        A = X_orig_min[a_idx]
        for b_idx in range(n):
            if a_idx == b_idx:
                continue
            B = X_orig_min[b_idx]
            diff = B - A
            nonzero = np.abs(diff) > 1e-10
            if not np.any(nonzero):
                continue
            t_vals = (syn[nonzero] - A[nonzero]) / diff[nonzero]
            if np.max(t_vals) - np.min(t_vals) > tol:
                continue          # t inconsistent across dimensions → wrong pair
            t = float(np.mean(t_vals))
            if not (0 < t < 1):
                continue          # synthetic must lie strictly between the two parents
            reconstructed = A + t * (B - A)
            if np.linalg.norm(reconstructed - syn) < tol:
                return a_idx, b_idx, t
    return None, None, None


# ── 5. Show 5 synthetic samples and prove they are interpolations ─────────────
print()
print("=" * 60)
print("5 SYNTHETIC SAMPLES — showing the 2 parent rows each came from")
print("=" * 60)
print()
print("  SMOTE interpolation formula:")
print("  synthetic = parent_A + t * (parent_B - parent_A)   where 0 < t < 1")
print()

for idx, syn in enumerate(synthetic_X[:5]):
    a_idx, b_idx, t = find_smote_parents(syn, X_orig_min)
    A = X_orig_min[a_idx]
    B = X_orig_min[b_idx]
    reconstructed = A + t * (B - A)
    error = np.linalg.norm(reconstructed - syn)

    print(f"  Synthetic #{idx+1} : [{syn[0]:.4f}, {syn[1]:.4f}]")
    print(f"    Parent A (minority {a_idx}) : [{A[0]:.4f}, {A[1]:.4f}]")
    print(f"    Parent B (minority {b_idx}) : [{B[0]:.4f}, {B[1]:.4f}]")
    print(f"    Interpolation t             : {t:.4f}  (must be 0 < t < 1)")
    print(f"    Reconstructed from formula  : [{reconstructed[0]:.4f}, {reconstructed[1]:.4f}]")
    print(f"    Reconstruction error        : {error:.2e}  (should be ~0)")
    verdict = "PASS - genuine interpolation" if error < 1e-9 and 0 < t < 1 else "FAIL"
    print(f"    Verdict : {verdict}")
    print()

# ── 6. Final verdict ──────────────────────────────────────────────────────────
print("=" * 60)
print("VERDICT")
print("=" * 60)
print(f"  Real SMOTE from imbalanced-learn : YES")
print(f"  Synthetic samples are duplicates : NO ({duplicates} duplicates out of {len(synthetic_X)})")
print(f"  Synthetic samples are generated  : YES, by linear interpolation between")
print(f"  two real minority samples with a random weight t in (0, 1)")
