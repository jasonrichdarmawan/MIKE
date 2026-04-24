import ast
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

tqdm.pandas()
sns.set_theme()
sns.set_style("whitegrid")


def save_plot(path: str, title=None, xlabel: str=None, ylabel: str=None, legend=True):
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legend:
        plt.legend()
    if title is not None:
        plt.title(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_matrix(matrix: np.array, labels: pd.Index, x_label: str, y_label: str, vmax=None):
    """
    Caveat:
    - Dark color represents low values (close to 0)
    - Light color represents high values (close to vmax)
    """
    with plt.rc_context(rc={"font.size": 42}):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20)
        if vmax is not None:
            im = ax.imshow(matrix, origin='lower', vmin=0, vmax=vmax)
        else:
            im = ax.imshow(matrix, origin='lower')
        divider = make_axes_locatable(ax)
        
        cax = divider.append_axes('right', size='5%', pad=0.5)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label("Percentage of Cases", fontsize=30)
        cbar.ax.tick_params(labelsize=26)
        
        ax.set_xlabel(x_label, fontsize=36, labelpad=18)
        ax.set_ylabel(y_label, fontsize=36, labelpad=18)
        
        max_ticks = 12
        label_count = len(labels)
        step = max(1, int(np.ceil(label_count / max_ticks)))
        ticks = np.unique(np.append(np.arange(0, label_count, step), label_count - 1)).astype(int)
        tick_labels = [labels[i] for i in ticks]
        ax.set_xticks(ticks, labels=tick_labels)
        ax.set_yticks(ticks, labels=tick_labels)
        ax.tick_params(axis='both', which='major', labelsize=28)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')


def plot_layer_matrix(generations: pd.DataFrame, target_classification: str | bool, path: str):
    """
    Example:
    - target_classification: "e2" / "e3"
    - if target_classification is True:
      - backpatching (i.e. activation patching)
    - if target_classification is str:
      - entity_description
    
    generations.groupby(["source_layer", "target_layer"])["classification"].value_counts(normalize=False)[14]:
    target_layer | classification
    0            | other            2
    ...          |
    29           | e3               1
                 | other            1
    ...          |
    generations.groupby(["source_layer", "target_layer"])["classification"].value_counts(normalize=True)[14]:
    target_layer | classification
    0            | other            1.0
    ...          |
    29           | e3               0.5
                 | other            0.5
    ...
    """
    success_matrix = (
        generations \
            .groupby(["source_layer", "target_layer"])["classification"]
            .value_counts(normalize=True)
            .unstack("classification")
            .fillna(0)
            * 100
    )

    if target_classification not in success_matrix.columns:
        with sns.plotting_context("paper", font_scale=1.5):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"No '{target_classification}' data", ha="center", va="center")
            ax.set_axis_off()
            save_plot(path, legend=False)
        return

    matrix = success_matrix[target_classification].unstack("target_layer").fillna(0)
    plot_matrix(matrix.to_numpy(), matrix.columns, "Target Layer", "Source Layer")
    save_plot(path, legend=True)


def get_entity_counts(df: pd.DataFrame):
    """
    Example:
    df["e1_e2_entity_layer"]:
    0 | NaN
    1 | NaN
    df["last_e2_entity_layer"]:
    0 | NaN
    1 | NaN
    df["last_e2_entity_layer"].count() == 0
    df["e1_e3_entity_layer"]:
    0 | NaN
    1 | NaN
    df["last_e3_entity_layer"]:
    0 | [14]
    1 | [14]
    df["last_e3_entity_layer"].count() == 2
            e2  e3
    e1      0.0 0.0
    last    0.0 1.0
    """
    return pd.DataFrame({
        "e2": {
            "e1": df["e1_e2_entity_layer"].count() / len(df),
            "last": df["last_e2_entity_layer"].count() / len(df),
        },
        "e3": {
            "e1": df["e1_e3_entity_layer"].count() / len(df), # expected to be 0.0, if not, then it indicates that this is a shortcut knowledge, where the model can directly decode the third hop entity from the first hop entity without going through the second hop entity.
            "last": df["last_e3_entity_layer"].count() / len(df),
        }
    })


def get_sublayer_counts(df: pd.DataFrame, target: str):
    """
    df["attention_prediction_projection_layers"]:
    0 | [29,30]
    1 | [31]
    df["attention_prediction_projection_layers"].explode().mean() == 30.0
    df["mlp_prediction_projection_layers"]:
    0 | NaN
    1 | NaN
    """
    return pd.DataFrame({
        "proportion": {
            "attention": df[f"attention_{target}_projection_layers"].count() / len(df),
            "mlp": df[f"mlp_{target}_projection_layers"].count() / len(df)
        },
        "mean layer": {
            "attention": df[f"attention_{target}_projection_layers"].explode().mean(),
            "mlp": df[f"mlp_{target}_projection_layers"].explode().mean()
        }
    })


def get_stage_layers(df: pd.DataFrame):
    return pd.DataFrame({
        "stage": [
            "e1_e2_entity_layer", 
            "info_prop", 
            "e1_e3_entity_layer",
            "last_e3_entity_layer", 
            "mlp_prediction_projection_layers",
            "attention_prediction_projection_layers"
        ],
        "proportion": [
            df["e1_e2_entity_layer"].count() / len(df),
            df["info_prop"].count() / len(df),
            df["e1_e3_entity_layer"].count() / len(df),
            df["last_e3_entity_layer"].count() / len(df),
            df["attention_prediction_projection_layers"].count() / len(df),
            df["mlp_prediction_projection_layers"].count() / len(df),
        ],
        "layer": [
            df["e1_e2_entity_layer"].mean(), 
            df["info_prop"].mean(), 
            df["e1_e3_entity_layer"].mean(),
            df["last_e3_entity_layer"].mean(),
            df["attention_prediction_projection_layers"].mean(),
            df["mlp_prediction_projection_layers"].mean(),
        ],
    })


def plot_stage_boxplots(correct_df: pd.DataFrame, incorrect_df: pd.DataFrame, path: str):
    """
    Note:
    - "..._entity_layer": generated by `generate_entity_description.py` entity description generation experiment
    - "..._projection_layers": generated by `project_sublayer.py` projection experiment
    - "..._attention_knockout_layers": generated by `knockout_attention.py` attention knockout experiment
    
    Caveat:
    - col "{source}_{target}_entity_layer": Layer where the target hop entity is decoded
      - e.g "e1_e2_entity_layer": Layer where the second hop entity is decoded when the activation of the first hop entity is patched in to another prompt.
    - col "{source}_{target}_knockout_layers": Layers where the target hop entity is no longer decoded after knocking out attention from the source hop entity position to the target hop entity position
      - e.g "last_e1_attention_knockout_layers": \
          Setting: Knockout attention from the prompt's last token position to the first hop entity position \
          Meaning: Layers where the attention is knocked out and the generated tokens changed or become incorrect, indicating that the model can no longer decode the first hop entity from these layers after the knockout, thus these layers are critical for decoding the first hop entity.
    - col "{sublayer}_{target}_projection_layers": Layers where the target hop entity is decoded from the sublayer's projection
      - e.g "attention_e2_projection_layers": \
          Setting: Project the attention sublayer output to the vocabulary space
          Meaning: Layers where the second hop entity is decoded from the attention sublayer's projection
    - col "info_prop": Mean layer where information is propagated from the first hop entity to the second hop entity, calculated as the mean of
      - "last_e1_attention_knockout_layers"
      - "attention_e2_projection_layers"
      - "last_e2_entity_layer"
    """
    correct_df = correct_df.copy()
    incorrect_df = incorrect_df.copy()

    preprocess_cols = [
        "e1_e2_entity_layer",
        "last_e1_attention_knockout_layers",
        "attention_e2_projection_layers",
        "last_e2_entity_layer",
        "e1_e3_entity_layer",
        "last_e3_entity_layer",
        "attention_prediction_projection_layers",
        "mlp_prediction_projection_layers",
    ]

    for df in [correct_df, incorrect_df]:
        for col in preprocess_cols:
            df[col] = df[col].apply(lambda x: min(x) if type(x) is list and len(x) > 0 else np.nan)

    info_prop_cols = [
        "last_e1_attention_knockout_layers",
        "attention_e2_projection_layers",
        "last_e2_entity_layer",
    ]
    correct_df["info_prop"] = correct_df[info_prop_cols].mean(axis=1, skipna=True)
    cols = [
        "e1_e2_entity_layer",
        "info_prop",
        "e1_e3_entity_layer",
        "last_e3_entity_layer",
        "attention_prediction_projection_layers",
        "mlp_prediction_projection_layers"
    ]
    correct_data = pd.melt(
        correct_df, 
        value_vars=cols,
        value_name='value',
        var_name='stage',
    )
    correct_data["Two-Hop Query Answer"] = "Correct"
    print("Correct setting stages:")
    print(get_stage_layers(correct_df))

    incorrect_df["info_prop"] = incorrect_df[info_prop_cols].mean(axis=1, skipna=True)
    incorrect_data = pd.melt(
        incorrect_df, 
        value_vars=cols,
        value_name='value',
        var_name='stage',
    )
    incorrect_data["Two-Hop Query Answer"] = "Incorrect"
    print("Incorrect setting stages:")
    print(get_stage_layers(incorrect_df))

    combined = pd.concat([correct_data, incorrect_data]).reset_index()
    labels = {
        "e1_e2_entity_layer": "1st Hop Resolved",
        "info_prop": "Info Propagation",
        "last_e3_entity_layer": "2nd Hop Resolved",
        "attention_prediction_projection_layers": "Prediction from Attention",
        "mlp_prediction_projection_layers": "Prediction from MLP",
    }
    combined = combined.replace(labels)

    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots()
        hatches = ['//', '..']
        sns.boxplot(x="stage", y="value", data=combined, hue="Two-Hop Query Answer", gap=0.1, ax=ax)
        ax.set_xlabel(None)
        ax.set_xticklabels(labels=labels.values(), rotation=45, ha="right", rotation_mode='anchor')
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, title="Two-Hop Query Answer")

        patches = [patch for patch in ax.patches if type(patch) is matplotlib.patches.PathPatch]
        h = np.repeat(hatches, (len(patches) // len(hatches)))
        for patch, hatch in zip(patches, h):
            patch.set_hatch(hatch)
            fc = patch.get_facecolor()
            patch.set_edgecolor(fc)
            patch.set_facecolor('none')

        for lp, hatch in zip(ax.legend().get_patches(), hatches):
            lp.set_hatch(hatch)
            fc = lp.get_facecolor()
            lp.set_edgecolor(fc)
            lp.set_facecolor('none')

        sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, title="Two-Hop Query Answer")

        save_plot(f"{path}/stages_boxplot.pdf", ylabel="Layer", legend=False)


def plot_series(series: pd.Series, label: str, layers: int, dropna=True, ax=None, linestyle="-", x_offset: float = 0.0, marker: str | None = None):
    data = series.value_counts(normalize=True, dropna=dropna).sort_index().reindex(range(layers), fill_value=0) * 100
    x = np.arange(layers, dtype=float) + x_offset
    if ax is not None:
        return sns.lineplot(x=x, y=data.to_numpy(), ax=ax, label=label, linestyle=linestyle, marker=marker)
    return sns.lineplot(x=x, y=data.to_numpy(), label=label, linestyle=linestyle, marker=marker)


def plot_min_entity_description_layers(df: pd.DataFrame, path: str, layers: int):
    """
    Example:
    df["e1_e2_entity_layer"]:
    id      |   e1_e2_entity_layer
    36468   |   NaN
    71021   |   NaN
    70093   |   [12, 13]
    """
    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots()
        offsets = {
            "e1_e2": -0.18,
            "last_e2": -0.06,
            "e1_e3": 0.06,
            "last_e3": 0.18,
        }
        plot_series(
            series=df["e1_e2_entity_layer"].apply(lambda x: min(x) if type(x) is list else np.nan),
            label="$e_2$ decoded from $t_1$",
            layers=layers,
            dropna=False,
            ax=ax,
            linestyle="-",
            x_offset=offsets["e1_e2"],
            marker="o",
        )
        plot_series(
            series=df["last_e2_entity_layer"].apply(lambda x: min(x) if type(x) is list else np.nan),
            label="$e_2$ decoded from $t_2$",
            layers=layers,
            dropna=False,
            ax=ax,
            linestyle=":",
            x_offset=offsets["last_e2"],
            marker="s",
        )
        plot_series(
            series=df["e1_e3_entity_layer"].apply(lambda x: min(x) if type(x) is list else np.nan),
            label="$e_3$ decoded from $t_1$",
            layers=layers,
            dropna=False,
            ax=ax,
            linestyle="--",
            x_offset=offsets["e1_e3"],
            marker="^",
        )
        plot_series(
            series=df["last_e3_entity_layer"].apply(lambda x: min(x) if type(x) is list else np.nan),
            label="$e_3$ decoded from $t_2$",
            layers=layers,
            dropna=False,
            ax=ax,
            linestyle="-.",
            x_offset=offsets["last_e3"],
            marker="D",
        )
        step = max(1, layers // 12)
        ticks = np.unique(np.append(np.arange(0, layers, step), layers - 1)).astype(int)
        ax.set_xticks(ticks)
        sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
        save_plot(path, xlabel="Layer", ylabel="Percentage of Cases", legend=False)



def calc_backpatching_oracle_success(df: pd.DataFrame, source: str):
    return df[f"{source}_original_activation_patching_layer"].count() / len(df)


def calc_backpatching_success(layers: dict[str, pd.DataFrame]):
    return pd.DataFrame({
        "Correct": {
            "e1": calc_backpatching_oracle_success(layers["correct"], "e1"),
            "last": calc_backpatching_oracle_success(layers["correct"], "last"),
        },
        "Incorrect": {
            "e1": calc_backpatching_oracle_success(layers["incorrect"], "e1"),
            "last": calc_backpatching_oracle_success(layers["incorrect"], "last"),
        }
    })


def load_layers(path: str):
    df = pd.read_csv(path, index_col=0)
    for col in df.columns:
        if col not in ["id", "composition_correct"]:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if type(x) is not float else x)
    return df


def load_backpatching(path):
    df = pd.read_csv(path, index_col=0)
    df.loc[df["source_layer"] <= df["target_layer"], "classification"] = "masked"
    return df.copy()


def load_entity_description(path: str, source: str, seeds=1):
    """
    Caveat:
    - Biran et al. [1] entity description generaton experiment' require to classify generations from 3 csv files. However, 1 csv file size is approximately 9.05 GB.

    [1] Biran et al., "Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries," in EMNLP 2024
    """
    gens = []
    for seed in range(seeds):
        csv_path = f"{path}/{source}_entity_description_s{seed}_classified.csv"
        gens.append(pd.read_csv(csv_path, index_col=0))
    generations = pd.concat(gens).drop_duplicates(
        subset=["id", "source_layer", "target_layer", "classification"]).sort_values(
        ["id", "source_layer", "target_layer"])
    return generations


def main(args):
    layer_count = {
        "gpt2": 12,
        "meta-llama/Llama-2-7b-hf": 32,
        "meta-llama/Llama-2-13b-hf": 40,
        "meta-llama/Meta-Llama-3-8B": 32,
        "meta-llama/Meta-Llama-3-70B": 80,
        "Qwen/Qwen3-4B-Instruct-2507": 36,
    }
    
    layers = {}
    for model in args.models:
        print(f"Processing {model} stages")
        dataset_path = f"{args.datasets_dir}/{model}"
        plots_path = f"{args.output_path}/{model}"

        layers[model] = {
            "correct": load_layers(f"{dataset_path}/composition_correct/layers.csv"),
            "incorrect": load_layers(f"{dataset_path}/composition_incorrect/layers.csv")
        }
        print(f"Correct setting example count: {len(layers[model]['correct'])}")
        print(f"Incorrect setting example count: {len(layers[model]['incorrect'])}")
        plot_stage_boxplots(layers[model]["correct"], layers[model]["incorrect"], plots_path)
        print(f"Correct setting prediction in sublayers:\n"
              f"{get_sublayer_counts(layers[model]['correct'], 'prediction')}")
        print(f"Incorrect setting prediction in sublayers:\n"
              f"{get_sublayer_counts(layers[model]['incorrect'], 'prediction')}")
        plot_min_entity_description_layers(pd.concat([layers[model]['correct'], layers[model]["incorrect"]]),
                                           f"{plots_path}/min_entity_description_layers.pdf",
                                           layer_count[model])

    entity_descriptions = {}
    for model in args.models:
        print(f"Processing {model} entity descriptions")
        dataset_path = f"{args.datasets_dir}/{model}"
        plots_path = f"{args.output_path}/{model}"

        entity_descriptions[model] = {
            "e1": load_entity_description(f"{dataset_path}/composition_correct/entity_description", "e1"),
            "last": load_entity_description(f"{dataset_path}/composition_correct/entity_description", "last")
        }
        print(f"Correct setting entity description counts:\n{get_entity_counts(layers[model]['correct'])}")
        print(f"Incorrect setting entity description counts:\n{get_entity_counts(layers[model]['incorrect'])}")
        plot_layer_matrix(entity_descriptions[model]["e1"], "e2", f"{plots_path}/entity_description_e1_e2.pdf")
        plot_layer_matrix(entity_descriptions[model]["last"], "e2", f"{plots_path}/entity_description_last_e2.pdf")
        plot_layer_matrix(entity_descriptions[model]["last"], "e3", f"{plots_path}/entity_description_last_e3.pdf")

    backpatching = {}
    for model in args.models:
        print(f"Processing {model} backpatching")
        dataset_path = f"{args.datasets_dir}/{model}"
        plots_path = f"{args.output_path}/{model}"

        backpatching[model] = {
            "correct": {
                "e1": load_backpatching(f"{dataset_path}/composition_correct/activation_patching/e1_original_activation_patching_classified.csv"),
                "last": load_backpatching(f"{dataset_path}/composition_correct/activation_patching/last_original_activation_patching_classified.csv")
            },
            "incorrect": {
                "e1": load_backpatching(f"{dataset_path}/composition_incorrect/activation_patching/e1_original_activation_patching_classified.csv"),
                "last": load_backpatching(f"{dataset_path}/composition_incorrect/activation_patching/last_original_activation_patching_classified.csv")
            }
        }
        plot_layer_matrix(backpatching[model]["incorrect"]["e1"], True, f"{plots_path}/backpatching_e1.pdf")
        plot_layer_matrix(backpatching[model]["incorrect"]["last"], True, f"{plots_path}/backpatching_last.pdf")
        print(f"Backpatching success:")
        print(calc_backpatching_success(layers[model]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasets_dir")
    parser.add_argument("output_path")
    parser.add_argument("-m", "--models", nargs="+", required=True, choices=[
        "gpt2",
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B"])
    main(parser.parse_args())