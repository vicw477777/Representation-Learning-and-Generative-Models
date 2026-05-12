import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import math
import numpy as np
import einops
from pathlib import Path
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from DIT import DiT

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 256
DEPTH = 4
HEADS = 4
MLP_RATIO = 4.0
CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 40
LR = 3e-4
T = 500
NUM_CLASSES = 10          # MNIST digit classes (used in Parts 2b and 2c)
CFG_DROPOUT = 0.1         # label-drop probability for classifier-free guidance
device = "cuda"


# ============================================================================
# DDPM UTILS
# ============================================================================


def get_ddpm_schedule(T):
    betas = torch.linspace(1e-4, 0.03, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def forward_diffusion(x0, t, alphas_cumprod):
    """
    Closed-form forward diffusion (DDPM eq. 4):
        q(x_t | x_0) = N(x_t ; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    so:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    where eps ~ N(0, I).
    """
    ### YOUR CODE STARTS HERE ###
    # Gather alpha_bar values for the current batch of timesteps
    # alphas_cumprod shape: (T,)  |  t shape: (N,)
    alpha_bar_t = alphas_cumprod[t]                        # (N,)
    # Reshape for broadcasting with (N, C, H, W) images
    alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)

    noise = torch.randn_like(x0)                           # eps ~ N(0,I)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    ### YOUR CODE ENDS HERE ###
    return xt, noise


@torch.no_grad()
def sample_ddpm(net, T, bsz, betas, alphas, alphas_cumprod,
                num_snapshots=10, labels=None, cfg_scale=1.0):
    """
    Reverse diffusion (DDPM Algorithm 2).

    Args:
        labels      – (bsz,) integer class labels for conditional generation,
                      or None for unconditional generation.
        cfg_scale   – guidance scale w >= 1.  w=1 means no CFG.
    """
    net.eval()

    # sample the initial noise
    x = torch.randn(bsz, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # Identify which timesteps to save for the visualization grid
    snapshot_indices = torch.linspace(T - 1, 0, num_snapshots).long()
    snapshots = []

    for t in reversed(range(T)):
        ### YOUR CODE STARTS HERE ###
        # ------------------------------------------------------------------ #
        # DDPM reverse step  (Algorithm 2, Ho et al. 2020)                   #
        # ------------------------------------------------------------------ #
        t_batch = torch.full((bsz,), t, dtype=torch.long, device=device)

        # Retrieve noise-schedule scalars for this timestep
        beta_t        = betas[t]                      # scalar
        alpha_t       = alphas[t]                     # scalar
        alpha_bar_t   = alphas_cumprod[t]             # scalar

        # Predict the noise
        if labels is not None and cfg_scale > 1.0:
            # Classifier-free guidance: combine conditional and unconditional predictions
            pred_noise = net.forward_with_cfg(x, t_batch, labels, cfg_scale)
        elif labels is not None:
            pred_noise = net(x, t_batch, labels)
        else:
            pred_noise = net(x, t_batch)

        # Compute the mean of x_{t-1} given x_t (DDPM eq. 11)
        coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean  = (1.0 / torch.sqrt(alpha_t)) * (x - coeff * pred_noise)

        # Add noise only for t > 0
        if t > 0:
            z     = torch.randn_like(x)
            # posterior variance: beta_t (simplified, as in Ho et al.)
            sigma = torch.sqrt(beta_t)
            x     = mean + sigma * z
        else:
            x     = mean
        ### YOUR CODE ENDS HERE ###

        if t in snapshot_indices:
            snapshots.append(x.cpu())

    # Return shape: (num_snapshots, bsz, C, H, W)
    return torch.stack(snapshots)


def visualize_forward_diffusion(dataloader, alphas_cumprod, n_steps=10):
    # Get a batch of real images
    images, _ = next(iter(dataloader))
    images = images[:8]

    # Select timesteps to show (0 to T-1)
    indices = torch.linspace(0, T - 1, n_steps).long()

    cols = []
    for t in indices:
        t_batch = torch.full((images.shape[0],), t, dtype=torch.long)
        # Apply the forward diffusion
        xt, _ = forward_diffusion(images.to(device), t_batch.to(device), alphas_cumprod)
        cols.append(xt.cpu())

    # Stack and rearrange: (Steps, Batch, C, H, W) -> (Batch * Steps, C, H, W)
    result = torch.stack(cols, dim=1)
    result = einops.rearrange(result, "b t c h w -> (b t) c h w")

    grid = vutils.make_grid(result, nrow=n_steps, normalize=True, value_range=(-1, 1))

    plt.figure(figsize=(15, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.savefig("images/forward_diffusion_process.png")
    plt.show()


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def train_unconditional(net, dataloader, betas, alphas, alphas_cumprod,
                        epochs=EPOCHS, lr=LR):
    """Part 2(a): Unconditional DDPM training."""
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    loss_history = []

    print("=== Part 2(a): Unconditional Training ===")
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            bsz = images.shape[0]

            t = torch.randint(0, T, (bsz,), device=device).long()
            xt, noise = forward_diffusion(images, t, alphas_cumprod)

            pred_noise = net(xt, t)
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if (epoch + 5) % 5 == 0:
            num_samples = 8
            num_steps = 10
            traj = sample_ddpm(net, T, num_samples, betas, alphas, alphas_cumprod, num_steps)
            grid_ready = einops.rearrange(traj, "s b c h w -> (b s) c h w")
            grid = vutils.make_grid(grid_ready, nrow=num_steps, normalize=True, value_range=(-1, 1))
            vutils.save_image(grid, f"images/unconditional_evolution_epoch_{epoch+1}.png")

    return loss_history


def train_conditional(net, dataloader, betas, alphas, alphas_cumprod,
                      epochs=EPOCHS, lr=LR, cfg_dropout=0.0):
    """
    Part 2(b)/(c): Conditional DDPM training.
    If cfg_dropout > 0, randomly drops class labels during training (for CFG).
    """
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    loss_history = []

    label_str = "CFG" if cfg_dropout > 0 else "Conditional"
    print(f"=== {label_str} Training (dropout={cfg_dropout}) ===")
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = images.shape[0]

            # Label dropout for CFG: replace some labels with num_classes (null token)
            if cfg_dropout > 0:
                drop_mask = torch.rand(bsz, device=device) < cfg_dropout
                labels = torch.where(drop_mask, torch.full_like(labels, NUM_CLASSES), labels)

            t = torch.randint(0, T, (bsz,), device=device).long()
            xt, noise = forward_diffusion(images, t, alphas_cumprod)

            pred_noise = net(xt, t, labels)
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if (epoch + 5) % 5 == 0:
            num_samples = 8
            num_steps = 10
            sample_labels = torch.arange(num_samples, device=device) % NUM_CLASSES
            traj = sample_ddpm(
                net, T, num_samples, betas, alphas, alphas_cumprod,
                num_steps, labels=sample_labels
            )
            grid_ready = einops.rearrange(traj, "s b c h w -> (b s) c h w")
            grid = vutils.make_grid(grid_ready, nrow=num_steps, normalize=True, value_range=(-1, 1))
            suffix = "cfg" if cfg_dropout > 0 else "cond"
            vutils.save_image(grid, f"images/{suffix}_evolution_epoch_{epoch+1}.png")

    return loss_history


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    Path("images").mkdir(exist_ok=True)

    # 1. Data Setup
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # Noise schedule (shared by all parts)
    betas, alphas, alphas_cumprod = get_ddpm_schedule(T)

    # Visualize the forward process before training
    visualize_forward_diffusion(dataloader, alphas_cumprod)

    # ========================================================================
    # PART 2(a): Unconditional Diffusion Model
    # ========================================================================
    net_uncond = DiT(
        input_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        hidden_size=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        mlp_ratio=MLP_RATIO,
        num_classes=0,         # 0 → no class conditioning
        learn_sigma=False,
    ).to(device)
    print(f"Unconditional model parameters: {sum(p.numel() for p in net_uncond.parameters()):,}")

    loss_history_uncond = train_unconditional(
        net_uncond, dataloader, betas, alphas, alphas_cumprod
    )

    # Final samples
    print("Generating final unconditional samples...")
    trajectory_uncond = sample_ddpm(net_uncond, T, 16, betas, alphas, alphas_cumprod)
    final_grid = vutils.make_grid(trajectory_uncond[-1], nrow=4, normalize=True, value_range=(-1, 1))
    vutils.save_image(final_grid, "images/2a_dit_mnist_final.png")

    # Denoising visualisation (last 8 images, all 10 snapshots)
    traj_vis = sample_ddpm(net_uncond, T, 8, betas, alphas, alphas_cumprod, num_snapshots=10)
    grid_ready = einops.rearrange(traj_vis, "s b c h w -> b s c h w")
    grid_ready = einops.rearrange(grid_ready, "b s c h w -> (b s) c h w")
    denoising_grid = vutils.make_grid(grid_ready, nrow=10, normalize=True, value_range=(-1, 1))
    vutils.save_image(denoising_grid, "images/2a_denoising_process.png")

    # Loss curve
    plt.figure()
    plt.plot(loss_history_uncond)
    plt.title("Part 2(a) Unconditional Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("images/2a_loss_curve.png")
    plt.close()

    # ========================================================================
    # PART 2(b): Conditional Diffusion Model
    # ========================================================================
    # num_classes=NUM_CLASSES enables the label embedder in DiT
    net_cond = DiT(
        input_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        hidden_size=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        mlp_ratio=MLP_RATIO,
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.0,   # no dropout in the purely conditional case
        learn_sigma=False,
    ).to(device)
    print(f"Conditional model parameters: {sum(p.numel() for p in net_cond.parameters()):,}")

    loss_history_cond = train_conditional(
        net_cond, dataloader, betas, alphas, alphas_cumprod
    )

    # Generate 5 images per digit (10 digits × 5 = 50 images)
    print("Generating conditional samples (5 per digit)...")
    cond_samples_all = []
    for digit in range(NUM_CLASSES):
        labels_d = torch.full((5,), digit, dtype=torch.long, device=device)
        traj_d = sample_ddpm(net_cond, T, 5, betas, alphas, alphas_cumprod,
                              num_snapshots=2, labels=labels_d)
        cond_samples_all.append(traj_d[-1])  # final denoised images (t=0)
    cond_samples_all = torch.cat(cond_samples_all, dim=0)   # (50, 1, H, W)
    grid_cond = vutils.make_grid(cond_samples_all, nrow=5, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid_cond, "images/2b_conditional_samples.png")

    # Loss curve
    plt.figure()
    plt.plot(loss_history_uncond, label="Unconditional")
    plt.plot(loss_history_cond, label="Conditional")
    plt.title("Part 2(b) Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("images/2b_loss_curve.png")
    plt.close()

    # ========================================================================
    # PART 2(c): Classifier-Free Guidance
    # ========================================================================
    # Train with label dropout (CFG_DROPOUT fraction of labels replaced with null)
    # num_classes+1 embedding entries: 0..9 for real digits, 10 for null/uncond.
    net_cfg = DiT(
        input_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        hidden_size=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        mlp_ratio=MLP_RATIO,
        num_classes=NUM_CLASSES,
        class_dropout_prob=CFG_DROPOUT,  # enables null-token embedding
        learn_sigma=False,
    ).to(device)
    print(f"CFG model parameters: {sum(p.numel() for p in net_cfg.parameters()):,}")

    loss_history_cfg = train_conditional(
        net_cfg, dataloader, betas, alphas, alphas_cumprod,
        cfg_dropout=CFG_DROPOUT
    )

    # Loss curve
    plt.figure()
    plt.plot(loss_history_cfg, label="CFG Training")
    plt.title("Part 2(c) CFG Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("images/2c_loss_curve.png")
    plt.close()

    # Generate samples with different guidance scales
    print("Generating CFG samples with different guidance values...")
    guidance_scales = [1, 5, 10, 15, 20]
    digits_to_show = list(range(NUM_CLASSES))   # show all 10 digits

    for w in guidance_scales:
        rows = []
        for digit in digits_to_show:
            labels_d = torch.full((5,), digit, dtype=torch.long, device=device)
            traj_d = sample_ddpm(
                net_cfg, T, 5, betas, alphas, alphas_cumprod,
                num_snapshots=2, labels=labels_d, cfg_scale=float(w)
            )
            rows.append(traj_d[-1])
        all_imgs = torch.cat(rows, dim=0)   # (50, 1, H, W)
        grid_w = vutils.make_grid(all_imgs, nrow=5, normalize=True, value_range=(-1, 1))
        vutils.save_image(grid_w, f"images/2c_cfg_scale_{w}.png")
        print(f"Saved CFG scale={w} samples.")

    print("Done! Check the 'images' folder.")