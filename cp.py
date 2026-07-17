from argparse import Namespace
import math
import torch
from torch.utils.data import DataLoader, Subset
from data_utils import build_dataset
import logging
import warnings
warnings.filterwarnings("ignore")

# Logging config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("cp.log")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s"
)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# Set device to 'cuda' if available otherwise use cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_args(seed):
    """Synthetic args for building the chexpert dataset"""

    defaults = {
        "dataset": "chex",
        "data_pct": 1.0,
        "dataset_cat": 1,
        "dataset_seed": seed,
        "include_lateral": False,
        "norm_type": "default",
        "input_size": 224,
        "resize_size": 256,
        "crop_type": "rc",
        "aug_type": "aff",
        "scale_min": 0.08,
        "color_jitter": 0.2,
        "rot": 10,
        "iwm_blur_prob": 0.2,
        "iwm_noise_prob": 0.0,
        "iwm_noise_range": (0.05, 0.2),
        "batch_size": 8,
        "shuffle_seed": seed,
        # cp related
        "alpha": 0.1,
        "calib_fraction": 0.7,
    }

    return Namespace(**defaults)


def build_chex_dataset(args):
    dataset = build_dataset(args, split="test")
    logger.info(
        f"step 1: loaded {type(dataset).__name__} validation set with {len(dataset)} samples")
    sample = dataset[0]
    return dataset


def load_model():
    from load_model import model

    logger.info("step 2: model loaded")
    return model


def make_full_masks(batch_size, num_patches, device):
    full_mask = torch.arange(num_patches, device=device).unsqueeze(
        0).repeat(batch_size, 1)
    masks_enc = [full_mask]
    masks_pred = [full_mask]
    return masks_enc, masks_pred


def compute_embedding_scores(model, imgs):
    imgs = imgs.to(device)

    num_patches = model.encoder.patch_embed.num_patches
    masks_enc, masks_pred = make_full_masks(imgs.size(0), num_patches, device)
    aug_params = torch.zeros(imgs.size(0), 5, device=device)

    with torch.inference_mode():
        # h -> target embeddings
        h = model.forward_target(imgs, masks_enc, masks_pred, 1, "avg_ln")
        # z_enc -> context encoder output
        z_enc = model.encoder(imgs, masks_enc)
        # z -> predictor output -> predicted target embeddings
        z = model.forward_context_with_z(
            z_enc, aug_params, masks_enc, masks_pred)

    scores = (z - h).pow(2).mean(dim=(1, 2))
    return scores, z, h


def forward_pass(model, dataset):
    sample_imgs = torch.stack([dataset[0][0], dataset[1][0]], dim=0)
    scores, z, h = compute_embedding_scores(model, sample_imgs)

    logger.info(f"step 3: imgs shape = {tuple(sample_imgs.shape)}")
    logger.info(f"step 3: l2 gap = {scores.mean().item():.6f}")
    return z, h


def split_for_conformal(dataset, calib_fraction, seed):
    num_samples = len(dataset)
    num_calib = int(round(num_samples * calib_fraction))
    num_calib = max(1, min(num_calib, num_samples - 1))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(num_samples, generator=generator).tolist()
    calib_indices = perm[:num_calib]
    test_indices = perm[num_calib:]
    return Subset(dataset, calib_indices), Subset(dataset, test_indices)


def collect_scores(model, dataset, batch_size, shuffle_seed=None):
    if shuffle_seed is None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        generator = torch.Generator()
        generator.manual_seed(shuffle_seed)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator)
    all_scores = []
    for batch in loader:
        imgs = batch[0]
        scores, _, _ = compute_embedding_scores(model, imgs)
        all_scores.append(scores.detach().cpu())
    return torch.cat(all_scores, dim=0)


def conformal_quantile(scores, alpha):
    scores = scores.detach().cpu().float().sort().values
    num_scores = scores.numel()
    q_level = math.ceil((num_scores + 1) * (1 - alpha))
    q_level = max(1, min(q_level, num_scores))
    return scores[q_level - 1].item()


def vanilla_cp(model, dataset, args):
    calib_dataset, test_dataset = split_for_conformal(
        dataset, args.calib_fraction, args.shuffle_seed)
    calib_scores = collect_scores(
        model, calib_dataset, args.batch_size, shuffle_seed=args.shuffle_seed + 1)
    q_hat = conformal_quantile(calib_scores, args.alpha)

    test_scores = collect_scores(
        model, test_dataset, args.batch_size, shuffle_seed=args.shuffle_seed + 2)
    accepted = test_scores <= q_hat
    coverage = accepted.float().mean().item()

    logger.info(f"step 4: calibration size = {len(calib_dataset)}")
    logger.info(f"step 4: test size = {len(test_dataset)}")
    logger.info(f"step 4: alpha = {args.alpha:.3f}")
    logger.info(f"step 4: q_hat = {q_hat:.6f}")
    logger.info(f"step 4: coverage = {coverage:.3f}")
    logger.info("*" * 200)
    # logger.info(f"step 4: calibration score mean = {calib_scores.mean().item():.6f}")
    # logger.info(f"step 4: test score mean = {test_scores.mean().item():.6f}")
    # logger.info(
    #     f"step 4: first 5 test scores = {[round(v, 6) for v in test_scores[:5].tolist()]}")
    return q_hat, calib_scores, test_scores, accepted


if __name__ == "__main__":
    seeds = [42]
    for seed in seeds:
        args = build_args(seed)
        dataset = build_chex_dataset(args)
        model = load_model()
        forward_pass(model, dataset)
        vanilla_cp(model, dataset, args)