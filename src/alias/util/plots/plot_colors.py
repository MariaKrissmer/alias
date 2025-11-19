# plot_color_preview.py
import matplotlib.pyplot as plt
import numpy as np
from .color_definition import COLORMAPS, CATEGORICAL_PALETTES

def plot_colormaps(colormaps):
    n = len(colormaps)
    fig, axes = plt.subplots(n, 1, figsize=(8, 0.6 * n))
    if n == 1:
        axes = [axes]

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    for ax, (name, cmap) in zip(axes, colormaps.items()):
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_axis_off()
        ax.set_title(name, fontsize=10, loc='left')

    plt.tight_layout()
    plt.show()

def plot_categorical_palettes(palettes):
    n = len(palettes)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.2 * n))
    if n == 1:
        axes = [axes]

    for ax, (name, colors) in zip(axes, palettes.items()):
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=10, loc='left')

    plt.tight_layout()
    plt.show()

# Run visual previews only when script is executed directly
if __name__ == "__main__":
    plot_colormaps(COLORMAPS)
    plot_categorical_palettes(CATEGORICAL_PALETTES)
