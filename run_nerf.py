"""
Simplified NeRF Training Script — Single-GPU, Blender Synthetic Dataset.

Implements the *complete* NeRF pipeline from:
    Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields
    for View Synthesis", ECCV 2020.

Key architecture preserved:
  • Positional encoding  (γ) for position (L=10) and direction (L=4)
  • 8-layer, 256-wide coarse MLP + identical fine MLP
  • Stratified coarse sampling  (N_c = 64 points per ray)
  • Hierarchical fine sampling  (N_f = 128 importance-weighted points)
  • Alpha-compositing volume rendering equation
  • Exponential learning-rate decay

Removed (engineering overhead, not part of the paper's contribution):
  • Multi-GPU / DDP
  • NDC coordinate transforms  (forward-facing scenes only)
  • LLFF / DeepVoxels / LINEMOD dataset loaders
  • configargparse boilerplate
"""

import os
import time
import numpy as np
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from run_nerf_helpers import (
    NeRF, get_embedder, get_rays, get_rays_np, sample_pdf,
    img2mse, mse2psnr, to8b,
)
from load_blender import load_blender_data


# ═════════════════════════════════════════════════════════════════════════════
#  Hyperparameters — all NeRF paper defaults, consolidated in one place
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Experiment ──
    "expname":       "lego",
    "basedir":       "./logs",
    "datadir":       "./data/nerf_synthetic/lego",

    # ── Network architecture ──
    "netdepth":      8,           # layers in coarse MLP
    "netwidth":      256,         # hidden units per layer
    "netdepth_fine": 8,           # layers in fine MLP
    "netwidth_fine": 256,         # hidden units in fine MLP
    "multires":      10,          # log2 of max positional-encoding freq (xyz)
    "multires_views": 4,          # log2 of max positional-encoding freq (dir)
    "use_viewdirs":  True,        # condition colour on viewing direction

    # ── Sampling ──
    "N_samples":     64,          # coarse samples per ray (N_c in the paper)
    "N_importance":  128,         # fine samples per ray   (N_f in the paper)
    "perturb":       1.0,         # stratified sampling jitter (1=on, 0=off)
    "raw_noise_std": 0.0,         # noise regularisation on σ during training

    # ── Training ──
    "N_iters":       15_000,      # total training iterations (early-stop limit)
    "N_rand":        1024,        # rays per gradient step
    "lrate":         5e-4,        # initial Adam learning rate
    "lrate_decay":   500,         # LR decay: every 500k steps → ×0.1
    "precrop_iters": 500,         # train on centre crop for first N iters
    "precrop_frac":  0.5,         # fraction of image used for centre crop

    # ── Blender dataset ──
    "half_res":      True,        # load at 400×400 instead of 800×800
    "white_bkgd":    True,        # composite RGBA onto white background
    "testskip":      8,           # load 1/N of test & val images

    # ── Batching ──
    "chunk":         1024 * 32,   # max rays rendered in parallel
    "netchunk":      1024 * 64,   # max pts sent through MLP in parallel

    # ── Logging ──
    "i_print":       100,
    "i_weights":     5000,
    "i_testset":     5000,
    "i_video":       10000,
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


# ═════════════════════════════════════════════════════════════════════════════
#  Core rendering functions
# ═════════════════════════════════════════════════════════════════════════════

def batchify(fn, chunk):
    """Split a big tensor through *fn* in chunks to avoid OOM."""
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk):
    """Positional-encode inputs, then evaluate the MLP *fn*."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


# ── Volume rendering equation (Eq. 3 in the paper) ──────────────────────────

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Convert raw MLP outputs (rgb, σ) to rendered colour via alpha compositing.

    Implements the discrete approximation of the volume rendering integral:
        C(r) = Σ_i  T_i · (1 - exp(-σ_i · δ_i)) · c_i
    where T_i = exp(-Σ_{j<i} σ_j · δ_j)  is the accumulated transmittance.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
    )
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)

    # T_i = cumprod(1 - α) — exclusive prefix product (Eq. 3)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1
    )[:, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
    )
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


# ── Hierarchical volume sampling  (coarse → fine) ───────────────────────────

def render_rays(
    ray_batch, network_fn, network_query_fn, N_samples,
    retraw=False, perturb=0.0, N_importance=0, network_fine=None,
    white_bkgd=False, raw_noise_std=0.0, verbose=False,
):
    """Render a batch of rays using the two-pass hierarchical strategy.

    Pass 1 (coarse): Stratified sampling of N_samples points along each ray.
    Pass 2 (fine):   Importance sampling of N_importance additional points,
                     guided by the coarse network's density predictions,
                     then re-evaluating all (N_samples + N_importance) points
                     through the fine network.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    # ── Pass 1: stratified (coarse) sampling ──
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Evaluate coarse network
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd,
    )

    # ── Pass 2: importance (fine) sampling ──
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0)
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd,
        )

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)

    return ret


# ── Top-level render interface ───────────────────────────────────────────────

def batchify_rays(rays_flat, chunk, **kwargs):
    """Render rays in minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk, rays=None, c2w=None, near=0.0, far=1.0,
           use_viewdirs=False, **kwargs):
    """Render an image (from c2w) or a batch of rays.

    Returns:  rgb_map, disp_map, acc_map, extras_dict
    """
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near_t = near * torch.ones_like(rays_d[..., :1])
    far_t = far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near_t, far_t], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, savedir=None):
    """Render a sequence of poses (e.g. a spiral) and return stacked images."""
    H, W, focal = hwf
    rgbs, disps = [], []

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            imageio.imwrite(
                os.path.join(savedir, "{:03d}.png".format(i)), to8b(rgbs[-1])
            )

    return np.stack(rgbs, 0), np.stack(disps, 0)


# ═════════════════════════════════════════════════════════════════════════════
#  Model construction
# ═════════════════════════════════════════════════════════════════════════════

def create_nerf(cfg):
    """Instantiate coarse + fine NeRF MLPs, embedders, and optimizer."""

    # Positional encoding for 3D position (L=10 → 63-dim)
    embed_fn, input_ch = get_embedder(cfg["multires"])

    # Positional encoding for viewing direction (L=4 → 27-dim)
    embeddirs_fn, input_ch_views = get_embedder(cfg["multires_views"])

    output_ch = 5 if cfg["N_importance"] > 0 else 4
    skips = [4]

    # Coarse network
    model = NeRF(
        D=cfg["netdepth"], W=cfg["netwidth"],
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=cfg["use_viewdirs"],
    ).to(device)
    grad_vars = list(model.parameters())

    # Fine network (same architecture, independent weights)
    model_fine = None
    if cfg["N_importance"] > 0:
        model_fine = NeRF(
            D=cfg["netdepth_fine"], W=cfg["netwidth_fine"],
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=cfg["use_viewdirs"],
        ).to(device)
        grad_vars += list(model_fine.parameters())

    # Closure that wires embedders → MLP for a single query
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=cfg["netchunk"],
    )

    optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["lrate"], betas=(0.9, 0.999))

    # ── Checkpoint loading ──
    start = 0
    ckpt_dir = os.path.join(cfg["basedir"], cfg["expname"])
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = [
        os.path.join(ckpt_dir, f)
        for f in sorted(os.listdir(ckpt_dir))
        if f.endswith(".tar")
    ]
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path, weights_only=False)
        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    # ── Render kwargs ──
    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb":          cfg["perturb"],
        "N_importance":     cfg["N_importance"],
        "network_fine":     model_fine,
        "N_samples":        cfg["N_samples"],
        "network_fn":       model,
        "use_viewdirs":     cfg["use_viewdirs"],
        "white_bkgd":       cfg["white_bkgd"],
        "raw_noise_std":    cfg["raw_noise_std"],
        "ndc":              False,          # synthetic scenes, no NDC
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


# ═════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═════════════════════════════════════════════════════════════════════════════

def train():
    cfg = CONFIG

    # ── Load Blender data ──
    images, poses, render_poses, hwf, i_split = load_blender_data(
        cfg["datadir"], cfg["half_res"], cfg["testskip"]
    )
    i_train, i_val, i_test = i_split
    print(f"Loaded blender: images {images.shape}, render_poses {render_poses.shape}")

    near, far = 2.0, 6.0

    # Composite RGBA onto white background
    if cfg["white_bkgd"]:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1],
    ])

    # ── Create model ──
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(cfg)
    global_step = start

    render_kwargs_train.update({"near": near, "far": far})
    render_kwargs_test.update({"near": near, "far": far})

    render_poses = torch.Tensor(render_poses).to(device)
    poses = torch.Tensor(poses).to(device)

    # ── Training ──
    N_iters = cfg["N_iters"] + 1
    N_rand = cfg["N_rand"]

    print(f"Training for {cfg['N_iters']} iterations")
    print(f"TRAIN views: {i_train}")
    print(f"TEST  views: {i_test}")
    print(f"VAL   views: {i_val}")

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # ── Sample random rays from one training image ──
        img_i = np.random.choice(i_train)
        target = torch.Tensor(images[img_i]).to(device)
        pose = poses[img_i, :3, :4]

        rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))

        # Centre cropping for early iterations (stabilises training)
        if i < cfg["precrop_iters"]:
            dH = int(H // 2 * cfg["precrop_frac"])
            dW = int(W // 2 * cfg["precrop_frac"])
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                    indexing="ij",
                ),
                -1,
            )
            if i == start:
                print(
                    f"[Config] Centre cropping {2*dH}×{2*dW} enabled "
                    f"until iter {cfg['precrop_iters']}"
                )
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, H - 1, H),
                    torch.linspace(0, W - 1, W),
                    indexing="ij",
                ),
                -1,
            )

        coords = torch.reshape(coords, [-1, 2])
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_coords = coords[select_inds].long()

        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]

        # ── Forward pass ──
        rgb, disp, acc, extras = render(
            H, W, K, chunk=cfg["chunk"], rays=batch_rays,
            verbose=i < 10, retraw=True, **render_kwargs_train,
        )

        # ── Loss: MSE on fine + coarse outputs ──
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0

        loss.backward()
        optimizer.step()

        # ── Exponential learning-rate decay ──
        decay_rate = 0.1
        decay_steps = cfg["lrate_decay"] * 1000
        new_lrate = cfg["lrate"] * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        # ── Logging ──
        if i % cfg["i_print"] == 0:
            tqdm.write(f"[TRAIN] Iter: {i}  Loss: {loss.item():.5f}  PSNR: {psnr.item():.2f}")

        # ── Save checkpoint ──
        if i % cfg["i_weights"] == 0:
            path = os.path.join(cfg["basedir"], cfg["expname"], f"{i:06d}.tar")
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
                    "network_fine_state_dict": render_kwargs_train["network_fine"].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print("Saved checkpoint at", path)

        # ── Render test set ──
        if i % cfg["i_testset"] == 0 and i > 0:
            testsavedir = os.path.join(
                cfg["basedir"], cfg["expname"], f"testset_{i:06d}"
            )
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(
                    torch.Tensor(poses[i_test]).to(device),
                    hwf, K, cfg["chunk"], render_kwargs_test, savedir=testsavedir,
                )
            print("Saved test set")

        # ── Render spiral video ──
        if i % cfg["i_video"] == 0 and i > 0:
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses, hwf, K, cfg["chunk"], render_kwargs_test,
                )
            moviebase = os.path.join(
                cfg["basedir"], cfg["expname"],
                f"{cfg['expname']}_spiral_{i:06d}_",
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4",
                to8b(disps / np.max(disps)), fps=30, quality=8,
            )

        global_step += 1


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    train()
