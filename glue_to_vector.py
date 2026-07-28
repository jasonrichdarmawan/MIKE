# %%

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %%

wide = pd.read_csv("./output/glue_eval.csv", delimiter=";")
id_cols = ["model", "dataset", "method"]
edit_cols = [c for c in wide.columns if c not in id_cols]

long = wide.melt(
    id_vars=id_cols,
    value_vars=edit_cols,
    var_name="edit_num",
    value_name="f1_new",
)

long["edit_num"] = pd.to_numeric(long["edit_num"])

models = wide["model"].unique()
datasets = wide["dataset"].unique()
methods = wide["method"].unique()
color_map = {
    method: plt.get_cmap("tab20")(index % 20)
    for index, method in enumerate(methods)
}

# One chart for each (dataset, model) pair.  y-axes are intentionally not
# shared because the score ranges may differ even for the same dataset.
fig, axes = plt.subplots(
    nrows=len(datasets),
    ncols=len(models),
    figsize=(4 * len(models), 2.8 * len(datasets)),
    sharex=True,
    squeeze=False,
)

for row, dataset in enumerate(datasets):
    for col, model in enumerate(models):
        ax = axes[row, col]
        subset = long[(long["dataset"] == dataset) & (long["model"] == model)]

        for method, method_data in subset.groupby("method", sort=False):
            method_data = method_data.sort_values("edit_num")
            ax.plot(
                method_data["edit_num"],
                method_data["f1_new"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=method,
                color=color_map[method],
            )

        if row == 0:
            ax.set_title(model)
        if col == 0:
            ax.set_ylabel(f"{dataset}\nF1")
        if row == len(datasets) - 1:
            ax.set_xlabel("Number of edits")
        ax.grid(axis="y", alpha=0.3)

# Use the same fixed method-to-color mapping in the legend and every subplot.
handles = [
    Line2D([0], [0], color=color_map[method], marker="o", linewidth=1.5, label=method)
    for method in methods
]
fig.legend(handles=handles, loc="lower center", ncol=min(len(methods), 7), frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("./output/glue_eval_grid.pdf", bbox_inches="tight")
plt.show()

# %%