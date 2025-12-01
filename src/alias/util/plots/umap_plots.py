import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from .pub_style import set_pub_style
from .color_definition import CATEGORICAL_PALETTES, COLORMAPS

class UMAPCellPlotter:
    def __init__(
        self, 
        palette_name="tab20_new", 
        colormap_name="Red_r",
        simmap_name="Heatmap: Teal–White–Red",
        timemap_name="Timemap: Slate–Orange",
        histmap_name='tabc1',
        genexpmap_name='Teal_r2',
        point_size=10,
        annotate_centroids=True,
        add_umap_arrows=True,
        font_size=8
    ):
        set_pub_style()
        self.palette = CATEGORICAL_PALETTES.get(palette_name, sns.color_palette("tab20"))
        self.colormap = COLORMAPS.get(colormap_name, plt.get_cmap("viridis"))
        self.simmap = COLORMAPS.get(simmap_name, plt.get_cmap("RdBu_r"))
        self.timemap = COLORMAPS.get(timemap_name)
        self.histmap = CATEGORICAL_PALETTES.get(histmap_name)
        self.genexpmap = COLORMAPS.get(genexpmap_name, plt.get_cmap('viridis'))
        self.point_size = point_size
        self.annotate_centroids = annotate_centroids
        self.add_umap_arrows = add_umap_arrows
        self.font_size = font_size

        self.font_settings = {
            "fontsize": self.font_size,
            "fontweight": "normal",
            "family": "Helvetica"
        }
        self.title_font_settings = {
            **self.font_settings,
            "fontsize": self.font_size + 1
        }
        self.large_title_font_settings = {
            **self.font_settings,
            "fontsize": self.font_size + 2
        }


    def _draw_umap_arrows(self, ax, arrow_size='small', arrow_len=None, dx=None, dy=None,
                      arrowstyle=None, label_offset_x=None, label_offset_y=None):
        # Presets for 'small' and 'large'
        size_presets = {
            'small': {
                'arrow_len': 0.18,
                'dx': 0.01,
                'dy': 0.01,
                'arrowstyle': '-|>,head_length=0.4,head_width=0.2',
                'label_offset_x': 0.03,
                'label_offset_y': 0.03,
            },
            'large': {
                'arrow_len': 0.15,
                'dx': 0.005,
                'dy': 0.005,
                'arrowstyle': '-|>,head_length=0.6,head_width=0.3',
                'label_offset_x': 0.01,
                'label_offset_y': 0.01,
            }
        }

        # Get defaults from preset
        preset = size_presets.get(arrow_size, size_presets['small'])

        arrow_len = arrow_len if arrow_len is not None else preset['arrow_len']
        dx = dx if dx is not None else preset['dx']
        dy = dy if dy is not None else preset['dy']
        arrowstyle = arrowstyle if arrowstyle is not None else preset['arrowstyle']
        label_offset_x = label_offset_x if label_offset_x is not None else preset['label_offset_x']
        label_offset_y = label_offset_y if label_offset_y is not None else preset['label_offset_y']

        true_origin = (0.02, 0.02)
        start_x = true_origin[0] - dx
        start_y = true_origin[1] - dy

        # X-axis arrow
        arrow_h = FancyArrowPatch(
            posA=(start_x, true_origin[1]),
            posB=(start_x + arrow_len, true_origin[1]),
            transform=ax.transAxes,
            mutation_scale=10,
            linewidth=1,
            color='black',
            arrowstyle=arrowstyle
        )
        ax.add_patch(arrow_h)
        ax.text(
            start_x + arrow_len / 2,
            true_origin[1] - label_offset_y,
            'UMAP1', ha='center', va='top',
            transform=ax.transAxes,
            fontsize=self.font_size
        )

        # Y-axis arrow
        arrow_v = FancyArrowPatch(
            posA=(true_origin[0], start_y),
            posB=(true_origin[0], start_y + arrow_len),
            transform=ax.transAxes,
            mutation_scale=10,
            linewidth=1,
            color='black',
            arrowstyle=arrowstyle
        )
        ax.add_patch(arrow_v)
        ax.text(
            true_origin[0] - label_offset_x,
            start_y + arrow_len / 2,
            'UMAP2', ha='right', va='center', rotation=90,
            transform=ax.transAxes,
            fontsize=self.font_size
        )

    
    def plot_cells(
        self,
        df=None,
        annotation_column=None,
        continuous_color_column=None,
        time_color_column=None,
        gene_exp_color_column=None, 
        output_path=None,
        annotate_centroids_df=None,
        title=None,
        side_by_side_dfs=None,
        side_by_side_titles=None,
        side_by_side_centroids=None,
        debug_color_check=False
    ):
        import matplotlib.gridspec as gridspec

        if side_by_side_dfs is not None:
            n_plots = len(side_by_side_dfs)
            fig = plt.figure(figsize=(6.5 * n_plots, 6))  # Slightly wider to fit all
            gs = gridspec.GridSpec(1, n_plots + 1, width_ratios=[1]*n_plots + [0.25], wspace=0.05)
            axes = [fig.add_subplot(gs[i]) for i in range(n_plots)]
            legend_ax = fig.add_subplot(gs[-1])

            # Gather and sort all unique labels
            all_labels = sorted(set().union(*[
                df_i[annotation_column].dropna().unique()
                for df_i in side_by_side_dfs
            ]))
            palette_dict = {label: self.palette[i % len(self.palette)] for i, label in enumerate(all_labels)}

            for i, (df_i, ax) in enumerate(zip(side_by_side_dfs, axes)):
                local_title = side_by_side_titles[i] if side_by_side_titles else None
                local_centroids = side_by_side_centroids[i] if side_by_side_centroids else None

                df_i[annotation_column] = pd.Categorical(
                    df_i[annotation_column],
                    categories=all_labels,
                    ordered=True
                )

                sns.scatterplot(
                    data=df_i[df_i[annotation_column].notna()],
                    x="UMAP1", y="UMAP2",
                    hue=annotation_column,
                    palette=palette_dict,
                    s=self.point_size,
                    alpha=0.7,
                    linewidth=0,
                    ax=ax,
                    legend=False
                )

                if self.annotate_centroids and local_centroids is not None:
                    for _, row in local_centroids.iterrows():
                        label = row.get('cell_type', None)
                        if label:
                            ax.plot(row["UMAP1"], row["UMAP2"], marker='o', color='black', markersize=3.16)
                            ax.text(row["UMAP1"], row["UMAP2"], label,
                                    fontsize=self.font_size, color='black',
                                    ha='left', va='bottom')

                if self.add_umap_arrows:
                    self._draw_umap_arrows(ax, arrow_size='large')


                ax.set_xticks([]); ax.set_yticks([])
                sns.despine(ax=ax, left=True, bottom=True)
                ax.set_xlabel(""); ax.set_ylabel("")
                if local_title:
                    ax.set_title(local_title, **self.title_font_settings)

            # Final legend
            handles, labels = axes[-1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend_ax.legend(by_label.values(), by_label.keys(),
                            loc='center', frameon=False, fontsize=self.font_size)
            legend_ax.axis('off')
            fig.tight_layout()

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()
            else:
                return fig, axes

        else:

            # === CONTINUOUS OR TIME COLOR ===
            if continuous_color_column or time_color_column:
                
                if time_color_column:
                    fig, ax = plt.subplots(figsize=(3.35, 3))
                elif continuous_color_column:
                    fig, ax = plt.subplots(figsize=(3.35, 3))
                else:
                    fig, ax = plt.subplots(figsize=(3.35, 3))
                
                color_col = continuous_color_column or time_color_column
                cmap = self.simmap if continuous_color_column else self.timemap
                vmin = -1 if continuous_color_column else 0
                vmax = 1 if continuous_color_column else 1

                # Create side colorbar axis
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)

                scatter = ax.scatter(
                    df["UMAP1"], df["UMAP2"],
                    c=df[color_col],
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=self.point_size / 2,
                    alpha=0.7,
                    linewidth=0
                )

                cbar = fig.colorbar(scatter, cax=cax)
                ticks = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{v:.0f}" for v in ticks])
                cbar.set_label(color_col.replace("_", " ").title())
                
            elif gene_exp_color_column:
                
                fig, ax = plt.subplots(figsize=(3.35, 3))

                color_col = gene_exp_color_column
                cmap = self.genexpmap
                vmin = 0
                vmax = 6

                # Create side colorbar axis
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)

                scatter = ax.scatter(
                    df["UMAP1"], df["UMAP2"],
                    c=df[color_col],
                    cmap=cmap,
                    vmin=vmin,vmax=vmax,
                    s=self.point_size / 2,
                    alpha=0.7,
                    linewidth=0
                )

                cbar = fig.colorbar(scatter, cax=cax)
                ticks = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{v:.0f}" for v in ticks])
                cbar.set_label(color_col.replace("_", " ").title())
                

            # === CATEGORICAL ANNOTATION ===
            elif annotation_column:
                
                fig, ax = plt.subplots(figsize=(4.5, 3))
                
                labels = sorted(df[annotation_column].dropna().unique())
                palette_dict = {label: self.palette[i % len(self.palette)] for i, label in enumerate(labels)}
                df[annotation_column] = pd.Categorical(df[annotation_column], categories=labels, ordered=True)

                scatter = sns.scatterplot(
                    data=df[df[annotation_column].notna()],
                    x="UMAP1", y="UMAP2",
                    hue=annotation_column,
                    palette=palette_dict,
                    s=self.point_size / 2,
                    alpha=0.7,
                    linewidth=0,
                    ax=ax,
                    legend='full'  # allow handles to be created
                )

                # Remove the seaborn legend from the main plot
                if ax.get_legend():
                    ax.get_legend().remove()

                # Fetch handles and labels from the Axes
                handles, legend_labels = ax.get_legend_handles_labels()

                # Create side axis for legend
                divider = make_axes_locatable(ax)
                legend_ax = divider.append_axes("right", size="30%", pad=0.1)

                legend_ax.legend(
                    handles, legend_labels,
                    loc='upper left',
                    frameon=False,
                    fontsize=self.font_size
                )
                legend_ax.axis('off')

            # === CENTROID LABELS ===
            if self.annotate_centroids and annotate_centroids_df is not None:
                texts = []
                for _, row in annotate_centroids_df.iterrows():
                    label = row.get('cell_type', None)
                    if label:
                        ax.plot(row["UMAP1"], row["UMAP2"], marker='o', color='black', markersize=3.16)
                        texts.append(ax.text(
                            row["UMAP1"], row["UMAP2"],
                            label,
                            fontsize=self.font_size,
                            color='black',
                            ha='left', va='bottom'
                        ))
                adjust_text(
                    texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                    only_move={'points': 'none', 'text': 'xy'}
                )

            # === UMAP ARROWS ===
            if self.add_umap_arrows:
                if time_color_column:
                    self._draw_umap_arrows(ax, arrow_size='small')
                elif continuous_color_column:
                    self._draw_umap_arrows(ax, arrow_size='small')
                elif gene_exp_color_column:
                    self._draw_umap_arrows(ax, arrow_size='small')
                else:
                    self._draw_umap_arrows(ax, arrow_size='small')
                

            # === Final formatting ===
            
            ax.set_xticks([]); ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)
            ax.set_xlabel(""); ax.set_ylabel("")
            if title:
                ax.set_title(title, **self.title_font_settings)

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()
            else:
                return fig, ax

        
    def plot_weights(
        self,
        df,
        weight_matrix,
        output_path,
        gene_column="gene",
        coord_columns=("UMAP1", "UMAP2"),
        centroid_column="cell_type"
    ):
        # Prepare merged dataframe
        df = df.merge(weight_matrix, left_on=gene_column, right_index=True, how='left')
        celltypes = weight_matrix.columns
        num_celltypes = len(celltypes)
        num_cols = 3
        num_rows = -(-num_celltypes // num_cols)  # ceiling division

        # Define size per subplot
        base_size = 4  # Width and height per subplot in inches (square)

        fig_width = num_cols * base_size
        fig_height = num_rows * base_size

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        axes = axes.flatten()
        
        # Fixed color limits for all scatter plots
        vmin, vmax = 0, 1

        for i, celltype in enumerate(celltypes):
            ax = axes[i]
            ax.set_title(celltype, **self.font_settings)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)

            valid = df[gene_column].notna()
            sc = ax.scatter(
                x=df.loc[valid, coord_columns[0]],
                y=df.loc[valid, coord_columns[1]],
                c=df.loc[valid, celltype],
                cmap=self.colormap,
                s=self.point_size,
                alpha=0.8,
                edgecolor='none',
                #vmin=vmin,
                #vmax=vmax
            )

            if self.annotate_centroids:
                centroids = df[df[centroid_column] == celltype]
                for _, row in centroids.iterrows():
                    ax.scatter(
                        row[coord_columns[0]], row[coord_columns[1]],
                        color='black', s=50, edgecolor='white', zorder=3
                    )
                    ax.text(
                        row[coord_columns[0]], row[coord_columns[1]],
                        row[centroid_column], color='black', fontsize=self.font_size,
                        ha='left', va='bottom', fontweight='bold'
                    )

            if self.add_umap_arrows:
                self._draw_umap_arrows(ax, arrow_size='small')
            
            cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
            #cbar.set_ticks([0, 0.5, 1])
            #cbar.set_ticklabels(['0', '0.5', '1'])
            cbar.set_label('Weight', fontsize=self.font_size)

        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        

    def plot_cm(
        self,
        true_labels,
        pred_labels,
        output_path,
        title="Confusion Matrix",
        normalize=True
    ):
        """
        Plots a confusion matrix heatmap comparing true and predicted labels.

        Parameters:
        - true_labels (array-like): Ground truth labels.
        - pred_labels (array-like): Predicted labels.
        - output_path (str or Path): File path to save the confusion matrix figure.
        - title (str): Title for the plot.
        - normalize (bool): Whether to normalize the confusion matrix.
        - cmap (str): Matplotlib colormap name.
        """
        
        cmap = self.colormap
        
        true_labels = pd.Series(true_labels).astype(str)
        pred_labels = pd.Series(pred_labels).astype(str)

        # Unified label set for consistent axis order
        labels = sorted(set(true_labels) | set(pred_labels))

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)

        # Normalize if requested
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # handle divide-by-zero cases

        # Adjust figure size based on number of labels
        width_per_label = 0.7
        height_per_label = 0.7
        fig_width = max(6, width_per_label * len(labels))
        fig_height = max(6, height_per_label * len(labels))

        plt.figure(figsize=(fig_width, fig_height))
        ax = sns.heatmap(
            cm,
            annot=True,
            square=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Proportion" if normalize else "Count", "shrink": 0.6,"aspect": 20},
            annot_kws={"size": self.font_size},
            vmax=1.0,
            vmin=0,
        )
        
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.5, 1])
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1.5)
        
        plt.xlabel("Predicted Label", fontsize=self.font_size)
        plt.ylabel("True Label", fontsize=self.font_size)
        plt.title(title, **self.font_settings)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        
        
    def plot_similarity_histogram(self, df, label, bins, output_path=None, title=None):
        """
        Plot a histogram comparing similarity scores for a target group vs. others.

        Args:
            df (pd.DataFrame): DataFrame with 'similarity' and 'group' columns.
            label (str): Label of the group to highlight (e.g., 'CD4 T cells').
            output_path (Path or str, optional): Path to save the plot.
            title (str, optional): Title of the plot.
        """
        
        fig, ax = plt.subplots(figsize=(3, 2.5))

        ax.hist(
            df.loc[df["group"] == "other", "similarity"],
            bins=bins,
            alpha=0.5,
            label="other",
            color='grey',
            density=True,
            range=(0, 1)
        )

        ax.hist(
            df.loc[df["group"] == label, "similarity"],
            bins=bins,
            alpha=0.5,
            label=label,
            color=self.histmap[1],
            density=True,
            range=(0, 1)
        )

        ax.set_xlabel("Similarity")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.set_title(title or label)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', direction='out', length=4)
        ax.legend(frameon=False, loc='upper left')
        #ax.grid(False, linestyle='--', linewidth=0.5, alpha=0.3)

        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
            
    def plot_similarity_histogram_grouped(
        self,
        df: pd.DataFrame,
        annotation_column: str,
        bins: int,
        output_path: Path | str = None,
        title: str = None,
    ):
        """
        Plot a histogram of similarity scores for all cell types using Matplotlib.
        """

        fig, ax = plt.subplots(figsize=(5, 4))

        # Sorted list of cell types
        labels = sorted(df[annotation_column].dropna().unique())
        palette_dict = {label: self.palette[i % len(self.palette)] for i, label in enumerate(labels)}

        # Plot each histogram
        for ct in labels:
            data = df[df[annotation_column] == ct]["sim_score"]
            ax.hist(
                data,
                bins=bins,
                alpha=0.5,
                density=True,
                range=(-1, 1),
                label=ct,
                color=palette_dict[ct],
                histtype="stepfilled",
            )

        ax.set_xlabel("Similarity")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.set_title(title or "Similarity Score Distribution")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', direction='out', length=4)
        ax.legend(
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=(len(labels) + 1) // 3,
            fontsize=self.font_size
        )

        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
            
    def plot_roc(
        self,
        roc_df,
        output_path=None,
        title=None
    ):
        """
        Plot ROC curve using a DataFrame with 'fpr', 'tpr', and optionally 'label' or 'auc' columns.

        Parameters:
        - roc_df (pd.DataFrame): DataFrame with columns:
            'fpr' (False Positive Rate), 
            'tpr' (True Positive Rate), 
            and optionally 'label' (str) or 'auc' (float) for legend.
        - output_path (str or Path, optional): Path to save the figure.
        - title (str, optional): Title for the plot.
        """

        fig, ax = plt.subplots(figsize=(3, 2.5))
        set_pub_style()

        # Group by label if present, else treat all as one ROC curve
        if 'label' in roc_df.columns:
            groups = roc_df.groupby('label')
        else:
            groups = [(None, roc_df)]

        for label, group in groups:
            fpr = group['fpr'].values
            tpr = group['tpr'].values
            auc = group['auc'].iloc[0] if 'auc' in group.columns else None

            legend_label = f"{label}" if label else ""
            if auc is not None:
                legend_label += f"AUC = {auc:.3f}"

            ax.plot(fpr, tpr, label=legend_label, color=self.palette[0])

        # Diagonal line for random guess
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
              
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", **self.font_settings)
        ax.set_ylabel("True Positive Rate", **self.font_settings)
        if title:
            ax.set_title(title, **self.title_font_settings)

        ax.legend(loc="lower right", fontsize=self.font_size, frameon=False)
        #sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            return fig, ax
        
        
    def plot_similarity_heatmap(
        self,
        sim_df: pd.DataFrame,
        output_path,
        title="Similarity Heatmap",
        row_labels = None,
        col_labels = None
    ):
        """
        Plots a square-annotated similarity heatmap with fixed width and dynamic height.

        Parameters:
        - sim_df (pd.DataFrame): DataFrame where rows are cell types and columns are diseases.
        - output_path (str or Path): File path to save the heatmap figure.
        - title (str): Title for the plot.
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cell_size = 0.6  # Try increasing this
        fig_width = cell_size * sim_df.shape[1]
        fig_height = cell_size * sim_df.shape[0]

        plt.figure(figsize=(fig_width, fig_height))
        ax = sns.heatmap(
            sim_df,
            annot=True,
            fmt=".2f",
            cmap=self.colormap,
            square=True,
            vmax=1.0,
            vmin=-1.0,
            cbar_kws={
                "label": "Similarity",
                "shrink": 0.6,
                "aspect": 20
            },
            annot_kws={"size": self.font_size}  # Set annotation font size here
        )
        
        plt.title(title, **self.title_font_settings)
        plt.xlabel("Functionality Descriptions", fontsize=self.font_size)
        plt.ylabel("Cell Types", fontsize=self.font_size)
        
        if col_labels is not None:
            ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=self.font_size)
        else:
            plt.xticks(rotation=45, ha='right', fontsize=self.font_size)

        if row_labels is not None:
            ax.set_yticklabels(row_labels, rotation=0, fontsize=self.font_size)
        else:
            plt.yticks(rotation=0, fontsize=self.font_size)
        
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(0.5)
          
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


    def plot_clustering_performance(
        self,
        df: pd.DataFrame,
        output_path: Path | str = None,
        title: str = "Clustering Performance",
        x_column: str = "Embedding",
        methods: list[str] = None,
        kind: str = "line",  # "line" or "bar"
    ):
        """
        Plot clustering performance across different embedding sizes.

        Args:
            df (pd.DataFrame): DataFrame containing x_column and method columns.
            output_path (Path or str, optional): Path to save figure.
            title (str, optional): Plot title.
            x_column (str, optional): Column to use as the x-axis (default: 'Embedding').
            methods (list[str], optional): Subset of method columns to plot.
            kind (str, optional): Plot type: 'line' (default) or 'bar'.
        """
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 4))

        # Extract x-values (convert embedding labels like '10_something' -> 10 if possible)
        x = df[x_column].copy()
        if x.dtype == object:
            extracted = df[x_column].astype(str).str.extract(r"(\d+)")[0]
            if extracted.notna().all():
                x = extracted.astype(int)
        df["_x"] = x

        # Methods
        if methods is None:
            methods = [c for c in df.columns if c not in [x_column, "_x"]]

        palette_dict = {m: self.palette[i % len(self.palette)] for i, m in enumerate(methods)}

        if kind == "line":
            # line plot
            for method in methods:
                ax.plot(
                    df["_x"],
                    df[method],
                    marker="o",
                    linewidth=1.5,
                    label=method,
                    color=palette_dict[method],
                )

        elif kind == "bar":
            # grouped bar plot
            bar_width = 0.8 / len(methods)
            x_vals = np.arange(len(df["_x"]))
            for i, method in enumerate(methods):
                ax.bar(
                    x_vals + i * bar_width,
                    df[method],
                    width=bar_width,
                    label=method,
                    color=palette_dict[method],
                    alpha=0.8,
                )
            ax.set_xticks(x_vals + bar_width * (len(methods) - 1) / 2)
            ax.set_xticklabels(df["_x"].astype(str), rotation=45)

        # Labels & style
        ax.set_xlabel(x_column, **self.font_settings)
        ax.set_ylabel("Score", **self.font_settings)
        ax.set_title(title, **self.title_font_settings)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", direction="out", length=4)

        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(methods),
            fontsize=self.font_size,
        )

        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
            
    def plot_pseudotime_scatter_time(
        self,
        expr_ranks,
        llm_ranks,
        time,
        output_path: Path | str = None,
        title: str = "Pseudotime Ordering Comparison (Time Colored)"
    ):
        """
        Scatter plot of expression-based vs. LLM-based pseudotime ranks,
        colored by continuous 'time'.
        """
        fig, ax = plt.subplots(figsize=(4.5, 4))

        scatter = ax.scatter(
            expr_ranks,
            llm_ranks,
            c=time,
            cmap=self.timemap if self.timemap is not None else plt.get_cmap("viridis"),
            alpha=0.7,
            s=self.point_size/2
        )

        ax.plot([1, len(expr_ranks)], [1, len(expr_ranks)], 'r--', linewidth=1)
        ax.set_xlabel("Expression-based pseudotime rank", **self.font_settings)
        ax.set_ylabel("LLM-based pseudotime rank", **self.font_settings)
        ax.set_title(title, **self.title_font_settings)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(scatter, cax=cax)
        cb.set_label("Time", **self.font_settings)

        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', direction='out', length=4)

        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

    def plot_pseudotime_scatter_celltypes(
        self,
        expr_ranks,
        llm_ranks,
        cell_types,
        output_path: Path | str = None,
        title: str = "Pseudotime Ordering Comparison (Cell Type Colored)"
    ):
        """
        Scatter plot of expression-based vs. LLM-based pseudotime ranks,
        colored by categorical cell type.
        """
        fig, ax = plt.subplots(figsize=(4, 4.2))

        # Generate categorical palette mapping
        labels = sorted(pd.Series(cell_types).dropna().unique())
        palette_dict = {ct: self.palette[i % len(self.palette)] for i, ct in enumerate(labels)}

        sns.scatterplot(
            x=expr_ranks,
            y=llm_ranks,
            hue=cell_types,
            palette=palette_dict,
            alpha=0.7,
            s=self.point_size,
            ax=ax,
            linewidth=0
        )

        # Red diagonal line (perfect agreement)
        ax.plot([1, len(expr_ranks)], [1, len(expr_ranks)], 'r--', linewidth=1, label="Perfect agreement")

        ax.set_xlabel("Expression-based pseudotime rank", **self.font_settings)
        ax.set_ylabel("LLM-based pseudotime rank", **self.font_settings)
        ax.set_title(title, **self.title_font_settings)

        # Legend styling (below plot, horizontal layout)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            title="Cell type",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),   # place below the plot
            frameon=False,
            fontsize=self.font_size,
            ncol=min(4, len(labels))       # spread into multiple columns
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', direction='out', length=4)

        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
                
    def plot_distribution_difference(
        self,
        df=None,
        x_column="mean_diff",
        y_column="-log10p",
        label_column="cell_type",
        pval_column="mw_p",
        significance_threshold=0.05,
        count_column=None,  # NEW: column specifying size scaling
        title="Distribution difference across cell types",
        xlabel="Mean Similarity Difference (disease - other)",
        ylabel="-log10(p-value)",
        output_path=None,
        annotate=True,
        size_scale=0.2   # NEW: scaling factor for points
    ):
        """
        Plot a distribution difference (volcano-like) plot, highlighting significant points,
        with non-overlapping labels using adjustText. Point size can scale with counts.
        """
        fig, ax = plt.subplots(figsize=(4, 3))

        # Mask significant points
        sig_mask = df[pval_column] < significance_threshold

        # Determine sizes
        if count_column is not None:
            sizes = df[count_column] * size_scale
        else:
            sizes = np.full(len(df), self.point_size)

        # Plot non-significant points
        ax.scatter(
            df.loc[~sig_mask, x_column],
            df.loc[~sig_mask, y_column],
            color="grey",
            s=sizes[~sig_mask],
            alpha=0.7,
            label=f"ns (p>={significance_threshold})"
        )

        # Plot significant points
        ax.scatter(
            df.loc[sig_mask, x_column],
            df.loc[sig_mask, y_column],
            color="red",
            s=sizes[sig_mask],
            alpha=0.7,
            label=f"sig (p<{significance_threshold})"
        )

        # Annotate points if requested
        texts = []
        if annotate:
            for _, row in df.iterrows():
                texts.append(
                    ax.text(
                        row[x_column],
                        row[y_column],
                        str(row[label_column]),
                        fontsize=self.font_size,
                        ha="left",
                        va="bottom",
                        color='black'
                    )
                )
            # Adjust labels to avoid overlap
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                only_move={'points': 'none', 'text': 'xy'}
            )

        # Significance threshold line
        ax.axhline(-np.log10(significance_threshold), color="k", ls="--", lw=1)

        # Labels and title
        ax.set_xlabel(xlabel, **self.font_settings)
        ax.set_ylabel(ylabel, **self.font_settings)
        ax.set_title(title, **self.title_font_settings)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
        else:
            return fig, ax
