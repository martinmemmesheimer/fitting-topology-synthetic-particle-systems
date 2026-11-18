import colorcet as cc
import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d

#Change path to the location of the database
def main(study_folder, study_name, figure_save_folder):
    conn = sqlite3.connect(os.path.join(study_folder, f"{study_name}.db"))

    df_rsa = pd.read_sql_query("SELECT * FROM rsa_runs", conn)
    df_reconstructions = pd.read_sql_query("SELECT * FROM reconstructions", conn)
    df_images = pd.read_sql_query("SELECT * FROM images", conn)
    df_optim = pd.read_sql_query("SELECT * FROM optimization_runs", conn)
    df_metrics = pd.read_sql_query("SELECT * FROM metrics", conn)
    df_distributions = pd.read_sql_query("SELECT * FROM distribution_values", conn)

    conn.close()

    df_metrics_with_optim_data = df_metrics.merge(
        df_optim[["optimization_id", "pd_dims"]],
        left_on="entity_id",
        right_on="optimization_id",
        how="left"
    )

    df_mean_metrics = (
        df_metrics_with_optim_data.groupby(
            ["parent_image_id", "entity_type", "pd_dims", "metric", "submetric"],
            dropna=False
        )["value"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["parent_image_id", "metric", "submetric"])
    )

    def format_entity_type(row, entity_name_map):

        entity_type = row["entity_type"]

        if entity_type != "optimization":
            return entity_name_map[entity_type]

        pd_dims = row["pd_dims"].strip("[]").replace(" ", "")

        return rf"Optimization ($S=\{{{pd_dims}\}}$)"

    entity_name_map = {
        "image": "Original Image",
        "rsa": "RSA",
        "reconstruction": "Reconstruction",
    }

    sort_order = {
        "Original Image": 0,
        "RSA": 1,
        "Reconstruction": 2,
        "Optimization ($S=\\{0\\}$)": 3,
        "Optimization ($S=\\{1\\}$)": 4,
        "Optimization ($S=\\{2\\}$)": 5,
        "Optimization ($S=\\{0,1\\}$)": 6,
        "Optimization ($S=\\{0,2\\}$)": 7,
        "Optimization ($S=\\{1,2\\}$)": 8,
        "Optimization ($S=\\{0,1,2\\}$)": 9,
    }

    for name, group in df_mean_metrics.groupby(["metric", "parent_image_id", "submetric"], dropna=False):
        if "graph" in name[0]:
            continue
        group["Image Type"] = group.apply(format_entity_type, axis=1, entity_name_map=entity_name_map)
        group["sort_key"] = group["Image Type"].map(sort_order)
        group = group.sort_values("sort_key").drop(columns="sort_key")
        group.insert(0, "Image Type", group.pop("Image Type"))
        group = group.drop(columns=["parent_image_id", "entity_type", "pd_dims", "metric", "submetric", "std"])

        print("Group:", name)
        print(group)
        print()

    df_distributions_with_optim_data = df_distributions.merge(
        df_optim[["optimization_id", "pd_dims"]],
        left_on="entity_id",
        right_on="optimization_id",
        how="left"
    )

    def plot_feature_histograms(df, feature, color_palette, label_map, feature_map, title_map, n_bins=15, savepath=None):
        
        superscript_map = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

        df = df.copy()
        df = df[df["feature"] == feature]

        scale_exp = int(np.floor(np.log10(df["value"].median())))
        df["scaled_value"] = df["value"] / (10 ** scale_exp)
        binrange=(df["scaled_value"].min(), df["scaled_value"].quantile(0.998))

        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams["font.family"] = "DejaVU Sans"

        df = df[
            ~(
                (df["entity_type"] == "optimization") &
                (df["pd_dims"] != "[0, 1, 2]")
            )
        ].copy()

        df["Image Type"] = df["entity_type"].map(label_map)

        df["n_entity_ids"] = (
            df.groupby(["parent_image_id", "entity_type", "pd_dims"], dropna=False)["entity_id"]
            .transform("nunique")
        )

        df["weight"] = 1 / df["n_entity_ids"]

        df["subplot_title"] = df["parent_image_id"].map(title_map)

        g = sns.FacetGrid(
            df,
            col="subplot_title",
            col_wrap=2,
            sharex=False,
            sharey=False,
            height=2,
            aspect=2
        )

        g.map_dataframe(
            sns.histplot,
            x="scaled_value",
            hue="Image Type",
            hue_order=list(label_map.values()),
            bins=n_bins,
            binrange=binrange,
            multiple="dodge",
            stat="count",
            weights="weight",
            shrink=0.9,
            palette=color_palette,
            edgecolor=None,
            linewidth=0,
            alpha=1.0,
            legend=False
            )

        xmin, xmax = min(ax.get_xlim()[0] for ax in g.axes.flat), max(ax.get_xlim()[1] for ax in g.axes.flat)
        ymin, ymax = min(ax.get_ylim()[0] for ax in g.axes.flat), max(ax.get_ylim()[1] for ax in g.axes.flat)
        g.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        g.set_titles("{col_name}")
        g.set_axis_labels("", "")
        g.figure.supxlabel(f"{feature_map[feature]} (in Pixels x 10{str(scale_exp).translate(superscript_map)})", y=0.015)
        g.figure.supylabel("Number of Particles", x=-0.005)
        g.figure.suptitle(f"{feature_map[feature]} Distributions", fontweight="bold", fontsize=12, y=1.025)

        handles = [
            mpatches.Patch(color=palette[i], label=label)
            for i, label in enumerate(label_map.values())
        ]

        g.figure.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.075)
        )

        if savepath is not None:
            g.savefig(savepath, dpi=800)

    n_bins = 12
    palette = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']

    label_map = {
        "image": "Real Image",
        "rsa": "RSA",
        "reconstruction": "Reconstruction",
        "optimization": "Optimization"    
    }

    feature_map = {
        "particle_sizes": "Particle Volume",
        "particle_surface_area": "Particle Surface Area"
    }

    title_map = {
        "S-S2_4-1": "RC1",
        "S-S2_4-3": "RC2",
        "S-S2_4-4": "RC3",
        "S-S2_4-6": "RC4"
    }

    os.makedirs(figure_save_folder, exist_ok=True)

    plot_feature_histograms(df_distributions_with_optim_data, "particle_sizes", n_bins=n_bins, color_palette=palette, label_map=label_map, feature_map=feature_map, title_map=title_map, savepath=os.path.join(figure_save_folder, "concrete_histogram_particle_sizes.png"))
    plot_feature_histograms(df_distributions_with_optim_data, "particle_surface_area", n_bins=n_bins, color_palette=palette, label_map=label_map, feature_map=feature_map, title_map=title_map, savepath=os.path.join(figure_save_folder, "concrete_histogram_particle_surface_area.png"))

    optimization_times =  df_optim.groupby(["image_id", "pd_dims"], dropna=False)["time_optimization_graph"].agg(["mean", "std"])

    for name, group in optimization_times.groupby("pd_dims"):
        print(f"Optimization Time: {name}")
        print(group)

if __name__ == "__main__":
    study_folder = "simulation_study_results_concrete/logs"
    study_name = "study_concrete"
    figure_save_folder = "example_renderings"

    main(study_folder, study_name, figure_save_folder)