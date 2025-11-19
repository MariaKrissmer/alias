import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
import matplotlib.colors as mcolors

# Load base colormap
tab20c = plt.get_cmap('tab20c')

def get_tab20c_group(start_idx):
    return [tab20c(i / 19) for i in range(start_idx, start_idx + 4)]


# Define color groups
teal = get_tab20c_group(0)
orange = get_tab20c_group(4)
green = get_tab20c_group(8)
slate = get_tab20c_group(12)
soft_reds = ['#a63737', '#c35252', '#d87474', '#eb9d9d']
dark_red, light_red = soft_reds[0], soft_reds[-1]

# Exported named color groups
COLOR_GROUPS = {
    "teal": teal,
    "orange": orange,
    "slate": slate,
    "soft_reds": soft_reds,
}

# Custom colormaps
COLORMAPS = {
    "Teal": LinearSegmentedColormap.from_list('teal', teal),
    "Orange": LinearSegmentedColormap.from_list('orange', orange),
    "Slate": LinearSegmentedColormap.from_list('slate', slate),
    "Red": LinearSegmentedColormap.from_list('soft_red', soft_reds),
    "Teal_r": LinearSegmentedColormap.from_list('teal_r', teal[::-1]),
    "Orange_r": LinearSegmentedColormap.from_list('orange_r', orange[::-1]),
    "Green_r": LinearSegmentedColormap.from_list('green_r', green[::-1]),
    "Slate_r": LinearSegmentedColormap.from_list('slate_r', slate[::-1]),
    "Red_r": LinearSegmentedColormap.from_list('soft_red_r', soft_reds[::-1]),

    # Heatmap gradients
    "Heatmap: Teal–White–Red": LinearSegmentedColormap.from_list(
        "teal_white_red", [teal[0], '#ffffff', dark_red]
    ),
    "Heatmap: Slate–White–Red": LinearSegmentedColormap.from_list(
        "slate_white_red", [slate[0], '#ffffff', dark_red]
    ),
    "Heatmap: Teal–White–Orange": LinearSegmentedColormap.from_list(
        "teal_white_orange", [teal[0], '#ffffff', orange[0]]
    ),
    "Heatmap: Slate–White–Orange": LinearSegmentedColormap.from_list(
        "slate_white_orange", [slate[0], '#ffffff', orange[0]]
    ),
    "Timemap: Teal–Orange": LinearSegmentedColormap.from_list(
        "teal_orange", [teal[0], orange[0]]
    ),
    "Timemap: Slate–Orange": LinearSegmentedColormap.from_list(
        "slate_orange", [orange[0], orange[3], slate[3], slate[0]]
    ),
    "CM: Teal–White": LinearSegmentedColormap.from_list(
        "teal_white", ['#ffffff', teal[0]]
    ),
    "CM: Red–White": LinearSegmentedColormap.from_list(
        "red_white", ['#ffffff', dark_red]
    ),
    "CM: Orange–White": LinearSegmentedColormap.from_list(
        "orange_white", ['#ffffff', orange[0]]
    ),
}

def get_gradient(start_hex, end_hex, n=5):
    cmap = LinearSegmentedColormap.from_list("temp", [start_hex, end_hex])
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]

# Build tabc1 to tabc4 (categorical palettes)
tabc = {}
all_groups = [teal, orange, slate, soft_reds]

for i in range(1, 5):
    step_colors = []
    for group in all_groups:
        step = min(i, len(group))
        step_colors.extend(group[:step])
    tabc[f"tabc{i}"] = step_colors

# Export for external use
CATEGORICAL_PALETTES = tabc


# Step 1: Get all 20 tab20 colors
tab20 = list(plt.get_cmap('tab20').colors)

# Step 2: Remove colors at index 5 and 6
filtered_colors = [c for i, c in enumerate(tab20) if i not in (4, 5)]

# Step 3: Convert to hex
hex_colors = [mcolors.to_hex(c) for c in filtered_colors]

tab20_new = [mcolors.to_hex(c) for c in filtered_colors]


# Boolean color mapping (stored separately, not a palette)
TRUE_FALSE_COLORS = {
    True: tabc["tabc1"][0],   # teal[0]
    False: tabc["tabc1"][1],  # orange[0]
}

CATEGORICAL_PALETTES["tab20_new"] = tab20_new
