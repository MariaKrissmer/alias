# util/plots/pub_style.py
import matplotlib.pyplot as plt
import seaborn as sns

def set_pub_style():
    sns.set_context("paper", font_scale=1)
    sns.set_style("white")  # No grid

    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
