import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
from scipy.stats import entropy as sp_entropy, ks_2samp
from scipy.ndimage import laplace
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SEED = 42
np.random.seed(SEED)

DATA_DIR = "data"
STAT_KEYS = ["mean", "std", "entropy", "sharpness"]


def load_images_recursive(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(dirpath, f))
    return paths


def load_images_flat(root, split="train"):
    paths = []
    for cls in sorted(os.listdir(os.path.join(root, split))):
        d = os.path.join(root, split, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(d, f))
    return paths


def extract_stats(path):
    img = Image.open(path).convert("L") # read the image and convert it to gray scale (.convert("L"))
    arr = np.array(img, dtype=np.float32)
    return {
        "mean": arr.mean(),
        "std": arr.std(),
        "entropy": sp_entropy(np.maximum(arr.ravel(), 1e-12)),
        "sharpness": laplace(arr).var(),    
    }


def stats_matrix(stats):
    return np.array([[s[k] for k in STAT_KEYS] for s in stats])


def compute_mahalanobis(X_id, X_ood):
    scaler = StandardScaler()
    X_id_n = scaler.fit_transform(X_id)
    X_ood_n = scaler.transform(X_ood)

    mu = X_id_n.mean(axis=0)
    cov_inv = pinv(np.cov(X_id_n, rowvar=False))

    d_id = np.array([mahalanobis(x, mu, cov_inv) for x in X_id_n])
    d_ood = np.array([mahalanobis(x, mu, cov_inv) for x in X_ood_n])
    return d_id, d_ood


def subsample(paths, n, rng):
    if len(paths) <= n:
        return paths
    idx = rng.choice(len(paths), n, replace=False)
    return [paths[i] for i in idx]

def plot_comparison(X_id, X_ood, d_id, d_ood, path="datasets_comparison.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, k in enumerate(STAT_KEYS):
        axes[i].hist(X_id[:, i], bins=50, alpha=0.5, label="CheXpert (ID)", density=True)
        axes[i].hist(X_ood[:, i], bins=50, alpha=0.5, label="ChestXray (OOD)", density=True)
        axes[i].set_title(k)
        axes[i].legend(fontsize=8)

    axes[4].hist(d_id, bins=50, alpha=0.5, label="CheXpert (ID)", density=True)
    axes[4].hist(d_ood, bins=50, alpha=0.5, label="ChestXray (OOD)", density=True)
    axes[4].set_title("Mahalanobis distance")
    axes[4].legend(fontsize=8)

    fig.delaxes(axes[5])

    plt.suptitle(path, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    logging.info(f"Saved {path}")



if __name__ == "__main__":
    rng = np.random.RandomState(SEED)

    chex_paths = load_images_recursive(os.path.join(DATA_DIR, "CheXpert", "valid"))
    chest_paths = load_images_flat(os.path.join(DATA_DIR, "ChestXray"), "train")
    chd_paths = load_images_recursive(os.path.join(DATA_DIR, "CHD-CXR", "train"))

    logging.info(f"CheXpert: {len(chex_paths)} samples")
    logging.info(f"ChestXray: {len(chest_paths)} samples")
    logging.info(f"CHD-CXR: {len(chd_paths)} samples")


    chest_paths = subsample(chest_paths, len(chex_paths), rng)
    chd_paths = subsample(chd_paths, len(chex_paths), rng)
    logging.info(f"ChestXray subsampled to: {len(chest_paths)} samples")
    logging.info(f"CHD-CXR subsampled to: {len(chd_paths)} samples")

    logging.info("Extracting stats...")
    X_id = stats_matrix([extract_stats(p) for p in chex_paths])
    X_ood = stats_matrix([extract_stats(p) for p in chest_paths])
    X_ood_chd = stats_matrix([extract_stats(p) for p in chd_paths])

    d_id, d_ood = compute_mahalanobis(X_id, X_ood)
    d_id, d_ood_chd = compute_mahalanobis(X_id, X_ood_chd)

    logging.info("=" * 60)
    logging.info("MAHALANOBIS DISTANCE")
    logging.info(f"  CheXpert (ID):   mean={d_id.mean():.4f}  std={d_id.std():.4f}  median={np.median(d_id):.4f}")
    logging.info(f"  ChestXray (OOD): mean={d_ood.mean():.4f}  std={d_ood.std():.4f}  median={np.median(d_ood):.4f}")
    logging.info(f"  CHD-CXR (OOD): mean={d_ood_chd.mean():.4f}  std={d_ood_chd.std():.4f}  median={np.median(d_ood_chd):.4f}")

    ks, pv = ks_2samp(d_id, d_ood)
    ks2, pv2 = ks_2samp(d_id, d_ood_chd)
    logging.info(f"  KS test: stat={ks:.4f}  p-value={pv:.2e}")
    logging.info(f"  KS test: stat={ks2:.4f}  p-value={pv2:.2e}")

    logging.info("")    
    logging.info("PER-STATISTIC MEANS")
    for i, k in enumerate(STAT_KEYS):
        m1, s1 = X_id[:, i].mean(), X_id[:, i].std()
        m2, s2 = X_ood[:, i].mean(), X_ood[:, i].std()
        m3, s3 = X_ood_chd[:, i].mean(), X_ood_chd[:, i].std()
        logging.info(f"  {k:>10s}:  ID mu={m1:.2f} sig={s1:.2f}  |  OOD Chest mu={m2:.2f} sig={s2:.2f} |  OOD CHD mu={m3:.2f} sig={s3:.2f}")

    plot_comparison(X_id, X_ood, d_id, d_ood, path="CheX vs Chest")
    plot_comparison(X_id, X_ood_chd, d_id, d_ood_chd, path="CheX vs CHD")
