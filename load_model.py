import torch
from argparse import Namespace

from models import init_jepa_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT = "chexworld_pretrained.tar"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
saved = ckpt["args"]
saved = vars(saved) if not isinstance(saved, dict) else dict(saved)

defaults = {
    "model": "vit_base",
    "input_size": 224,
    "patch_size": 16,
    "drop_path": 0.0,
    "stop_grad_conv1": False,
    "stop_grad_norm1": False,
    "pred_emb_dim": 384,
    "pred_depth": 6,
    "policy_dim": 5,
    "iwm_disable": False,
    "unify_embed": False,
    "cond_type": "feat",
    "mae_init_weights": False,
    "mask_type": "multi_multiblock",
    "ssl_type": "iwm_dual_easy",
    "pretrained": "",
    "rel_pos_disable": False,
    "reverse_pred": False,
    "extra_loss_weight": 1.0,
    "extra_mean": True,
    "reg_weight": 0.0,
    "loss_type": "l2",
    "target_last_k": 1,
    "target_norm_type": "avg_ln",
    "extra_global_scale": [0.3, 1.0],
    "min_overlap": -1.0,
}

for k, v in defaults.items():
    saved.setdefault(k, v)

args = Namespace(**saved)

model = init_jepa_model(args, device=device, ssl_type=args.ssl_type)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
print(model)