import matplotlib.pyplot as plt
import numpy as np

# def plot_loss_curves(results, model_name: str, save_path=None):
#     train_loss = results["train_loss"]
#     test_loss = results["test_loss"]
#     epochs = range(1, len(train_loss) + 1)

#     fig, ax = plt.subplots(figsize=(9, 5.5))

#     # Plot curves
#     ax.plot(
#         epochs, train_loss,
#         label="Training Loss",
#         color="#377eb8",
#         linewidth=2.5,
#         marker="o",
#         markersize=4
#     )
#     ax.plot(
#         epochs, test_loss,
#         label="Test Loss",
#         color="#e76f51",
#         linewidth=2.5,
#         marker="s",
#         markersize=4
#     )

#     # Titles and labels
#     ax.set_title(
#         " ".join(model_name.split("_")).title(),
#         fontsize=16,
#         fontweight="bold",
#         pad=14
#     )
#     ax.set_xlabel("Epoch", fontsize=12, labelpad=8)
#     ax.set_ylabel("Loss", fontsize=12, labelpad=8)

#     # Grid and axes styling
#     ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
#     ax.set_axisbelow(True)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     # Legend
#     ax.legend(
#         frameon=True,
#         fancybox=True,
#         framealpha=0.9,
#         fontsize=10
#     )

#     ax.tick_params(axis="both", labelsize=10)
#     fig.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=300, bbox_inches="tight")

#     plt.show()

def plot_loss_curves(results, model_name: str, save_path=None):
    with plt.style.context("seaborn-v0_8-darkgrid"):
        train_loss = results["train_loss"]
        test_loss = results["test_loss"]
        epochs = range(1, len(train_loss) + 1)

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.plot(epochs, train_loss, label="Train Loss",
                color="#56B4E9", linewidth=2.5, marker="o", markersize=4)
        ax.plot(epochs, test_loss, label="Test Loss",
                color="#E69F00", linewidth=2.5, marker="s", markersize=4)

        ax.set_title(" ".join(model_name.split("_")).title(),
                     fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()



def visualize_dic(
    initial, 
    deformed, 
    displacement, 
    stride=12,
    scale=0.15, 
    title="DIC Visualization"
):

    def to_numpy(x):
        if hasattr(x, "detach"):  # PyTorch tensor
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        return np.squeeze(x)

    initial = to_numpy(initial)
    deformed = to_numpy(deformed)
    displacement = to_numpy(displacement)

    if displacement.ndim != 3 or displacement.shape[-1] != 2:
        raise ValueError(
            "Displacement must have shape (H, W, 2), ordered (u_x, u_y)."
        )

    if displacement.shape[:2] != initial.shape:
        raise ValueError("Displacement and image dimensions must match.")

    ux = displacement[..., 0]
    uy = displacement[..., 1]

    H, W = initial.shape
    y, x = np.mgrid[0:H:stride, 0:W:stride]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(initial, cmap="gray")
    axes[0].set_title("Initial Speckle Pattern")
    axes[0].axis("off")

    axes[1].imshow(deformed, cmap="gray")
    axes[1].set_title("Deformed Speckle Pattern")
    axes[1].axis("off")

    axes[2].imshow(initial, cmap="gray")
    q = axes[2].quiver(
        x, y,
        ux[::stride, ::stride],
        -uy[::stride, ::stride],
        np.sqrt(ux[::stride, ::stride]**2 + uy[::stride, ::stride]**2),
        cmap="turbo",
        angles="xy",
        scale_units="xy",
        scale=scale,
        pivot="tail"
    )
    axes[2].set_title("Displacement Field")
    axes[2].set_xlim(0, W - 1)
    axes[2].set_ylim(H - 1, 0)
    axes[2].set_aspect("equal")
    axes[2].axis("off")
    fig.colorbar(q, ax=axes[2], label="Displacement magnitude")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()