import os
import sys
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


METRICS = ["jaccard_score", "micro_f1", "macro_f1"]
METRIC_NAMES = ["Jaccard", "Mic F1", "Mac F1"]
MODELS = [
    "google--gemma-7b",
    "allenai--OLMo-7B",
    "meta-llama--Llama-2-13b-chat-hf",
    "meta-llama--Llama-2-70b-chat-hf",
    "gpt-3.5-turbo",
]
DATASETS = ["semeval", "goemotions"]
DATASET_NAMES = ["SemEval", "GoEmotions"]
MARKERS = ["o", "P", "*"]
SHOTS = [5, 15, 25]


# compare performance of random retrieval with same shot prior and ground truth
def improvement_over_prior(analyses_folder, basename=None):

    fig, ax = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(10, 5),
        sharey=True,
        sharex=True,
    )

    for i, metric in enumerate(METRICS):
        for j, dataset in enumerate(DATASETS):
            for k, model in enumerate(MODELS):
                try:
                    df = pd.read_csv(
                        os.path.join(
                            analyses_folder,
                            f"{dataset}-{model}",
                            f"{metric}.csv",
                        ),
                        index_col=0,
                    )
                except FileNotFoundError:
                    continue

                y_yhat = [
                    float(df[f"{shot}s"]["Ground Truth"].split("\n±")[0])
                    for shot in SHOTS
                ]
                y_yprior = [
                    float(
                        df[f"Proxy-prior-{shot}s"]["Ground Truth"].split("\n±")[
                            0
                        ]
                    )
                    for shot in SHOTS
                ]
                improvement = [
                    100 * (y_yhat[i] - y_yprior[i]) / (y_yprior[i] + 1e-6)
                    for i in range(len(y_yhat))
                ]

                if j == k == 0:
                    ax[j][k].plot(
                        SHOTS,
                        improvement,
                        label=METRIC_NAMES[i],
                        marker=MARKERS[i],
                        linestyle="--",
                    )
                else:
                    ax[j][k].plot(
                        SHOTS,
                        improvement,
                        marker=MARKERS[i],
                        linestyle="--",
                    )

                ax[j][k].hlines(
                    0, -10, 40, colors="black", linestyle="--", linewidth=0.5
                )
                if j == 0:
                    model = model.split("--")[1] if "--" in model else model
                    ax[j][k].set_title(model)

                if k == 0:
                    ax[j][k].set_ylabel(
                        DATASET_NAMES[j], rotation=70, labelpad=13
                    )

                ax[j][k].set_xticks(SHOTS)
                ax[j][k].set_ylim(-60, 60)
                ax[j][k].set_xlim(0, 30)
                ax[j][k].yaxis.set_major_formatter(mtick.PercentFormatter())
                # ax[j][k].grid()

    fig.suptitle("")
    fig.text(0.568, 0.964, "Improvement over Prior", ha="center", fontsize=17)
    fig.text(0.568, 0.0, "Shot", ha="center", fontsize=14)
    fig.legend(
        loc="center left",
        fontsize=9,
        bbox_to_anchor=(-0.005, 0.49),
        # frameon=False,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"improvement-over-prior.pdf",
        )
    )
    plt.clf()


# compare performance of random retrieval with best shot prior and ground truth
def improvement_over_best_prior(analyses_folder, basename=None):

    fig, ax = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(10, 5),
        sharey=True,
        sharex=True,
    )

    for i, metric in enumerate(METRICS):
        for j, dataset in enumerate(DATASETS):
            for k, model in enumerate(MODELS):
                try:
                    df = pd.read_csv(
                        os.path.join(
                            analyses_folder,
                            f"{dataset}-{model}",
                            f"{metric}.csv",
                        ),
                        index_col=0,
                    )
                except FileNotFoundError:
                    continue

                yprior = [
                    float(
                        df[f"Proxy-prior-{shot}s"]["Ground Truth"].split("\n±")[
                            0
                        ]
                    )
                    for shot in SHOTS
                ]
                argmax = yprior.index(max(yprior))
                prior_max_shot = SHOTS[argmax]

                y_yhat = [
                    float(df[f"{shot}s"]["Ground Truth"].split("\n±")[0])
                    for shot in SHOTS
                ]
                y_yprior = float(
                    df[f"Proxy-prior-{prior_max_shot}s"]["Ground Truth"].split(
                        "\n±"
                    )[0]
                )
                improvement = [
                    100 * (y_yhat[i] - y_yprior) / (y_yprior + 1e-6)
                    for i in range(len(y_yhat))
                ]

                if j == k == 0:
                    ax[j][k].plot(
                        SHOTS,
                        improvement,
                        label=METRIC_NAMES[i],
                        marker=MARKERS[i],
                        linestyle="--",
                    )
                else:
                    ax[j][k].plot(
                        SHOTS,
                        improvement,
                        marker=MARKERS[i],
                        linestyle="--",
                    )

                ax[j][k].hlines(
                    0, -10, 40, colors="black", linestyle="--", linewidth=0.5
                )
                if j == 0:
                    model = model.split("--")[1] if "--" in model else model
                    ax[j][k].set_title(model, fontsize=11)

                if k == 0:
                    ax[j][k].set_ylabel(
                        DATASET_NAMES[j], rotation=70, labelpad=13
                    )

                ax[j][k].set_xticks(SHOTS)
                ax[j][k].set_ylim(-60, 60)
                ax[j][k].set_xlim(0, 30)
                ax[j][k].yaxis.set_major_formatter(mtick.PercentFormatter())
                # ax[j][k].grid()

    fig.suptitle("")
    fig.text(
        0.548,
        0.964,
        "Improvement over Best Task Prior",
        ha="center",
        fontsize=17,
    )
    fig.text(0.548, 0.004, "Shot", ha="center", fontsize=13)
    fig.legend(
        loc="center left",
        fontsize=9,
        bbox_to_anchor=(-0.045, 0.49),
        # frameon=False,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"improvement-over-best-prior.pdf",
        ),
        bbox_inches="tight",
    )
    plt.clf()


# compare performance of random retrieval with same-shot prior and ground truth
def strength(analyses_folder, basename=None):

    # labels = ["sim$(\hat{f}, f)$", "sim$(\hat{f}, \hat{f}_{\mathcal{D}})$"]
    labels = ["Ground Truth", "Prior"]
    metric_names = ["JS", "Mic", "Mac"]

    results = {
        dataset: {model: {metric: [] for metric in METRICS} for model in MODELS}
        for dataset in DATASETS
    }

    height_range = [float("inf"), -float("inf")]

    for dataset in DATASETS:
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for metric in METRICS:

                try:
                    df = pd.read_csv(
                        os.path.join(
                            analyses_folder,
                            f"{dataset}-{model}",
                            f"{metric}.csv",
                        ),
                        index_col=0,
                    )
                except FileNotFoundError:
                    results[dataset][model][metric] = {
                        "y_yhat": (None, None),
                        "yhat_yprior": (None, None),
                    }
                    continue

                y_yhat = [
                    float(df[f"{shot}s"]["Ground Truth"].split("\n±")[0])
                    for shot in SHOTS
                ]
                y_yhat_std = [
                    float(df[f"{shot}s"]["Ground Truth"].split("\n±")[1])
                    for shot in SHOTS
                ]
                yhat_yprior = [
                    float(
                        df[f"{shot}s"][f"Proxy-prior-{shot}s"].split("\n±")[0]
                    )
                    for shot in SHOTS
                ]
                yhat_yprior_std = [
                    float(
                        df[f"{shot}s"][f"Proxy-prior-{shot}s"].split("\n±")[1]
                    )
                    for shot in SHOTS
                ]

                results[dataset][model][metric] = {
                    "y_yhat": (y_yhat, y_yhat_std),
                    "yhat_yprior": (yhat_yprior, yhat_yprior_std),
                }

                for quant in results[dataset][model][metric]:
                    height_range[0] = min(
                        height_range[0],
                        min(results[dataset][model][metric][quant][0]),
                    )
                    height_range[1] = max(
                        height_range[1],
                        max(results[dataset][model][metric][quant][0]),
                    )

    n_results = 2  # y_yhat and yhat_yprior

    # bar plot comparing the strength of the prior to the ground truth
    # compare for each dataset, all the models for every metric at different shots
    fig, ax = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(13, 5),
        sharey=True,
        sharex=True,
    )

    barwidth = 0.09
    colors = ["limegreen", "orange"]

    for j, dataset in enumerate(DATASETS):
        for k, model in enumerate(MODELS):
            for i, metric in enumerate(METRICS):
                # nested_x_value is y_yhat and yhat_yprior
                # x_value is SHOT

                for l, quant in enumerate(results[dataset][model][metric]):

                    pos = np.arange(len(SHOTS)) * (
                        ((n_results + 1) * len(METRICS) + 2) * barwidth
                    )

                    if results[dataset][model][metric][quant][0] is not None:
                        if i == j == k == 0:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                label=labels[l],
                                color=colors[l],
                                edgecolor="black",
                            )
                        else:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                color=colors[l],
                                edgecolor="black",
                            )

                    if l == n_results - 1 and j == len(DATASETS) - 1:
                        for po in pos:
                            ax[j][k].text(
                                po
                                # move as bars as there are nested x values + gap between nested x values
                                + i * barwidth * (n_results + 1)
                                # move to the middle of current bars
                                # (we dont consider gap part of bars for text purposes)
                                + n_results * barwidth / 2
                                # adjust because text is not a point (assume about one bar)
                                # + starting point is after first bar
                                - 0.8 * barwidth,  #  - offset_y_axis_labels[i],
                                -0.205
                                * (height_range[1] - min(height_range[0], 0)),
                                metric_names[i],
                                size=10,
                                fontweight="bold",
                                rotation=60,
                                ha="left",
                                rotation_mode="anchor",
                            )
                if j == 0:
                    model_name = (
                        model.split("--")[1] if "--" in model else model
                    )
                    ax[j][k].set_title(model_name)

                if k == 0:
                    ax[j][k].set_ylabel(
                        DATASET_NAMES[j], rotation=70, labelpad=15, fontsize=14
                    )

                ax[j][k].grid(axis="y", linestyle="dashed")

                ax[j][k].set_xticks(
                    # move to middle of whole current section of nested_x_values + gap for each
                    # y_axis value, adjust one bar back because starting after the first bar
                    barwidth * (len(METRICS) * (n_results + 1) / 2 - 1)
                    + pos
                    # - offset_x_axis_labels
                )
                ax[j][k].set_xticklabels(SHOTS)
                for label in ax[j][k].get_xticklabels():
                    label.set_y(
                        label.get_position()[1] - 0.08
                    )  # Adjust the offset value as needed

                ax[j][k].set_ylim(top=height_range[1] * 1.1, bottom=0)

    fig.suptitle("")
    # fig.legend(
    #     loc="center left",
    #     fontsize=12,
    #     bbox_to_anchor=(-0.01, 0.52),
    #     # ncol=2,
    #     frameon=False,
    # )
    fig.legend(
        loc="upper center",
        fontsize=12,
        bbox_to_anchor=(0.59, 1.025),
        ncol=n_results,
        frameon=False,
    )
    # set x label for entire figure
    fig.text(
        0.42,
        0.966,
        "Similarity to",
        ha="center",
        fontsize=16,
    )
    fig.text(0.53, 0.004, "Shot", ha="center", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"prior-strength.pdf",
        )
    )
    plt.clf()


def consistency(analyses_folder, basename=None):

    # labels = ["sim$(\hat{f}, f)$", "sim$(\hat{f}, \hat{f}_{\mathcal{D}})$"]
    labels = ["ICL", "Prior (examples)", "Prior (labels)"]
    metric_names = ["JS", "Mic", "Mac"]

    results = {
        dataset: {model: {metric: [] for metric in METRICS} for model in MODELS}
        for dataset in DATASETS
    }

    height_range = [float("inf"), -float("inf")]

    for dataset in DATASETS:
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for metric in METRICS:
                try:
                    df = pd.read_csv(
                        os.path.join(
                            analyses_folder,
                            f"{dataset}-{model}",
                            f"{metric}_individual.csv",
                        ),
                        index_col=0,
                    )
                except FileNotFoundError:
                    results[dataset][model][metric] = {
                        "icl": (None, None),
                        "prior": (None, None),
                        "prior_sedl": (None, None),
                    }
                    continue

                icl = [df["metric_mean"][f"{shot}s"] for shot in SHOTS]
                icl_std = [df["metric_std"][f"{shot}s"] for shot in SHOTS]
                prior = [
                    df["metric_mean"][f"Prior-{shot}s-traindev"]
                    for shot in SHOTS
                ]
                prior_std = [
                    df["metric_std"][f"Prior-{shot}s-traindev"]
                    for shot in SHOTS
                ]
                prior_sedl = [
                    df["metric_mean"][f"Prior-{shot}s-traindev-sedl"]
                    for shot in SHOTS
                ]
                prior_sedl_std = [
                    df["metric_std"][f"Prior-{shot}s-traindev-sedl"]
                    for shot in SHOTS
                ]

                results[dataset][model][metric] = {
                    "icl": (icl, icl_std),
                    "prior": (prior, prior_std),
                    "prior_sedl": (prior_sedl, prior_sedl_std),
                }

                for quant in results[dataset][model][metric]:
                    height_range[0] = min(
                        height_range[0],
                        min(results[dataset][model][metric][quant][0]),
                    )
                    height_range[1] = max(
                        height_range[1],
                        max(results[dataset][model][metric][quant][0]),
                    )

    n_results = 3  # icl, prior, prior_sedl

    # bar plot comparing the strength of the prior to the ground truth
    # compare for each dataset, all the models for every metric at different shots
    fig, ax = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(13, 5),
        sharey=True,
        sharex=True,
    )

    barwidth = 0.09
    colors = ["limegreen", "skyblue", "orange"]

    for j, dataset in enumerate(DATASETS):
        for k, model in enumerate(MODELS):
            for i, metric in enumerate(METRICS):
                # nested_x_value is y_yhat and yhat_yprior
                # x_value is SHOT

                for l, quant in enumerate(results[dataset][model][metric]):

                    pos = np.arange(len(SHOTS)) * (
                        ((n_results + 1) * len(METRICS) + 2) * barwidth
                    )

                    if results[dataset][model][metric][quant][0] is not None:
                        if i == j == k == 0:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                label=labels[l],
                                color=colors[l],
                                edgecolor="black",
                            )
                        else:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                color=colors[l],
                                edgecolor="black",
                            )

                    if l == n_results - 1 and j == len(DATASETS) - 1:
                        for po in pos:
                            ax[j][k].text(
                                po
                                # move as bars as there are nested x values + gap between nested x values
                                + i * barwidth * (n_results + 1)
                                # move to the middle of current bars
                                # (we dont consider gap part of bars for text purposes)
                                + n_results * barwidth / 2
                                # adjust because text is not a point (assume about one bar)
                                # + starting point is after first bar
                                - 0.8 * barwidth,  #  - offset_y_axis_labels[i],
                                -0.205
                                * (height_range[1] - min(height_range[0], 0)),
                                metric_names[i],
                                size=10,
                                fontweight="bold",
                                rotation=60,
                                ha="left",
                                rotation_mode="anchor",
                            )
                if j == 0:
                    model_name = (
                        model.split("--")[1] if "--" in model else model
                    )
                    ax[j][k].set_title(model_name)

                if k == 0:
                    ax[j][k].set_ylabel(
                        DATASET_NAMES[j], rotation=70, labelpad=15, fontsize=14
                    )

                ax[j][k].grid(axis="y", linestyle="dashed")

                ax[j][k].set_xticks(
                    # move to middle of whole current section of nested_x_values + gap for each
                    # y_axis value, adjust one bar back because starting after the first bar
                    barwidth * (len(METRICS) * (n_results + 1) / 2 - 1)
                    + pos
                    # - offset_x_axis_labels
                )
                ax[j][k].set_xticklabels(SHOTS)
                for label in ax[j][k].get_xticklabels():
                    label.set_y(
                        label.get_position()[1] - 0.08
                    )  # Adjust the offset value as needed

                ax[j][k].set_ylim(top=height_range[1] * 1.1, bottom=0)

    fig.suptitle("")
    # fig.legend(
    #     loc="center left",
    #     fontsize=12,
    #     bbox_to_anchor=(-0.01, 0.52),
    #     # ncol=2,
    #     frameon=False,
    # )
    fig.legend(
        loc="upper center",
        fontsize=12,
        bbox_to_anchor=(0.605, 1.025),
        ncol=n_results,
        frameon=False,
    )
    # set x label for entire figure
    fig.text(
        0.345,
        0.966,
        "Consistency of",
        ha="center",
        fontsize=16,
    )
    fig.text(0.533, 0, "Shot", ha="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"consistency.pdf",
        )
    )
    plt.clf()


def reinforcement(analyses_folder, basename=None):

    # labels = ["sim$(\hat{f}, f)$", "sim$(\hat{f}, \hat{f}_{\mathcal{D}})$"]
    labels = ["ICL", "Prior (examples) Prompt", "Prior (labels) Prompt"]
    metric_names = ["JS", "Mic", "Mac"]

    results = {
        dataset: {model: {metric: [] for metric in METRICS} for model in MODELS}
        for dataset in DATASETS
    }

    height_range = [float("inf"), -float("inf")]

    for dataset in DATASETS:
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for metric in METRICS:
                try:
                    df = pd.read_csv(
                        os.path.join(
                            analyses_folder,
                            f"{dataset}-{model}",
                            f"{metric}_individual.csv",
                        ),
                        index_col=0,
                    )
                except FileNotFoundError:
                    results[dataset][model][metric] = {
                        "icl": (None, None),
                        "prior": (None, None),
                        "prior_sedl": (None, None),
                    }
                    continue

                icl = [df["metric_mean"][f"{shot}s"] for shot in SHOTS]
                icl_std = [df["metric_std"][f"{shot}s"] for shot in SHOTS]
                prior = [
                    df["metric_mean"][f"Prior-{shot}s-prompt-{shot}s"]
                    for shot in SHOTS
                ]
                prior_std = [
                    df["metric_std"][f"Prior-{shot}s-prompt-{shot}s"]
                    for shot in SHOTS
                ]
                prior_sedl = [
                    df["metric_mean"][f"Prior-{shot}s-prompt-{shot}s-sedl"]
                    for shot in SHOTS
                ]
                prior_sedl_std = [
                    df["metric_std"][f"Prior-{shot}s-prompt-{shot}s-sedl"]
                    for shot in SHOTS
                ]

                results[dataset][model][metric] = {
                    "icl": (icl, icl_std),
                    "prior": (prior, prior_std),
                    "prior_sedl": (prior_sedl, prior_sedl_std),
                }

                for quant in results[dataset][model][metric]:
                    height_range[0] = min(
                        height_range[0],
                        min(results[dataset][model][metric][quant][0]),
                    )
                    height_range[1] = max(
                        height_range[1],
                        max(results[dataset][model][metric][quant][0]),
                    )

    n_results = 3  # icl, prior, prior_sedl

    # bar plot comparing the strength of the prior to the ground truth
    # compare for each dataset, all the models for every metric at different shots
    fig, ax = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(13, 5),
        sharey=True,
        sharex=True,
    )

    barwidth = 0.09
    colors = ["limegreen", "skyblue", "orange"]

    for j, dataset in enumerate(DATASETS):
        for k, model in enumerate(MODELS):
            for i, metric in enumerate(METRICS):
                # nested_x_value is y_yhat and yhat_yprior
                # x_value is SHOT

                for l, quant in enumerate(results[dataset][model][metric]):

                    pos = np.arange(len(SHOTS)) * (
                        ((n_results + 1) * len(METRICS) + 2) * barwidth
                    )

                    if results[dataset][model][metric][quant][0] is not None:
                        if i == j == k == 0:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                label=labels[l],
                                color=colors[l],
                                edgecolor="black",
                            )
                        else:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                color=colors[l],
                                edgecolor="black",
                            )

                    if l == n_results - 1 and j == len(DATASETS) - 1:
                        for po in pos:
                            ax[j][k].text(
                                po
                                # move as bars as there are nested x values + gap between nested x values
                                + i * barwidth * (n_results + 1)
                                # move to the middle of current bars
                                # (we dont consider gap part of bars for text purposes)
                                + n_results * barwidth / 2
                                # adjust because text is not a point (assume about one bar)
                                # + starting point is after first bar
                                - 0.8 * barwidth,  #  - offset_y_axis_labels[i],
                                -0.205
                                * (height_range[1] - min(height_range[0], 0)),
                                metric_names[i],
                                size=10,
                                fontweight="bold",
                                rotation=60,
                                ha="left",
                                rotation_mode="anchor",
                            )
                if j == 0:
                    model_name = (
                        model.split("--")[1] if "--" in model else model
                    )
                    ax[j][k].set_title(model_name)

                if k == 0:
                    ax[j][k].set_ylabel(
                        DATASET_NAMES[j], rotation=70, labelpad=20, fontsize=14
                    )

                ax[j][k].grid(axis="y", linestyle="dashed")

                ax[j][k].set_xticks(
                    # move to middle of whole current section of nested_x_values + gap for each
                    # y_axis value, adjust one bar back because starting after the first bar
                    barwidth * (len(METRICS) * (n_results + 1) / 2 - 1)
                    + pos
                    # - offset_x_axis_labels
                )
                ax[j][k].set_xticklabels(SHOTS)
                for label in ax[j][k].get_xticklabels():
                    label.set_y(
                        label.get_position()[1] - 0.08
                    )  # Adjust the offset value as needed

                ax[j][k].set_ylim(top=height_range[1] * 1.1, bottom=0)

    fig.suptitle("")
    # fig.legend(
    #     loc="center left",
    #     fontsize=12,
    #     bbox_to_anchor=(-0.01, 0.52),
    #     # ncol=2,
    #     frameon=False,
    # )
    fig.legend(
        loc="upper center",
        fontsize=12,
        bbox_to_anchor=(0.65, 1.025),
        ncol=n_results,
        frameon=False,
    )
    # set x label for entire figure
    fig.text(
        0.34,
        0.966,
        "Consistency of",
        ha="center",
        fontsize=16,
    )
    fig.text(0.557, 0, "Shot", ha="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"reinforcement.pdf",
        )
    )
    plt.clf()


def holistic_consistency(analyses_folder, basename=None):

    # labels = ["sim$(\hat{f}, f)$", "sim$(\hat{f}, \hat{f}_{\mathcal{D}})$"]
    labels = [
        "ICL",
        "Prior",
        "Prior (labels)",
        "Prior Prompt",
        "Prior (labels) Prompt",
    ]
    metric_names = ["JS", "Mic"]
    shots = [5, 25]

    results = {
        dataset: {
            model: {metric: [] for metric in METRICS[:-1]} for model in MODELS
        }
        for dataset in DATASETS
    }

    height_range = [float("inf"), -float("inf")]

    for dataset in DATASETS:
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for metric in METRICS[:-1]:
                try:
                    df = pd.read_csv(
                        os.path.join(
                            analyses_folder,
                            f"{dataset}-{model}",
                            f"{metric}_individual.csv",
                        ),
                        index_col=0,
                    )
                except FileNotFoundError:
                    results[dataset][model][metric] = {
                        "icl": (None, None),
                        "prior": (None, None),
                        "prior_sedl": (None, None),
                        "prior_prompt": (None, None),
                        "prior_prompt_sedl": (None, None),
                    }
                    continue

                icl = [df["metric_mean"][f"{shot}s"] for shot in shots]
                icl_std = [df["metric_std"][f"{shot}s"] for shot in shots]
                prior = [
                    df["metric_mean"][f"Prior-{shot}s-traindev"]
                    for shot in shots
                ]
                prior_std = [
                    df["metric_std"][f"Prior-{shot}s-traindev"]
                    for shot in shots
                ]
                prior_sedl = [
                    df["metric_mean"][f"Prior-{shot}s-traindev-sedl"]
                    for shot in shots
                ]
                prior_sedl_std = [
                    df["metric_std"][f"Prior-{shot}s-traindev-sedl"]
                    for shot in shots
                ]
                prior_prompt_std = [
                    df["metric_std"][f"Prior-{shot}s-prompt-{shot}s"]
                    for shot in shots
                ]
                prior_prompt_sedl = [
                    df["metric_mean"][f"Prior-{shot}s-prompt-{shot}s-sedl"]
                    for shot in shots
                ]
                prior_prompt_sedl_std = [
                    df["metric_std"][f"Prior-{shot}s-prompt-{shot}s-sedl"]
                    for shot in shots
                ]

                results[dataset][model][metric] = {
                    "icl": (icl, icl_std),
                    "prior": (prior, prior_std),
                    "prior_sedl": (prior_sedl, prior_sedl_std),
                    "prior_prompt": (prior, prior_prompt_std),
                    "prior_prompt_sedl": (
                        prior_prompt_sedl,
                        prior_prompt_sedl_std,
                    ),
                }

                for quant in results[dataset][model][metric]:
                    height_range[0] = min(
                        height_range[0],
                        min(results[dataset][model][metric][quant][0]),
                    )
                    height_range[1] = max(
                        height_range[1],
                        max(results[dataset][model][metric][quant][0]),
                    )

    n_results = 5  # icl, prior, prior_sedl, prior_prompt, prior_prompt_sedl

    # bar plot comparing the strength of the prior to the ground truth
    # compare for each dataset, all the models for every metric at different shots
    fig, ax = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(15, 5),
        sharey=True,
        sharex=True,
    )

    barwidth = 0.09
    colors = [
        "limegreen",
        "orange",
        "skyblue",
        "violet",
        "red",
    ]

    for j, dataset in enumerate(DATASETS):
        for k, model in enumerate(MODELS):
            for i, metric in enumerate(METRICS[:-1]):
                for l, quant in enumerate(results[dataset][model][metric]):
                    # nested_x_value is y_yhat and yhat_yprior
                    # x_value is SHOT

                    pos = np.arange(len(shots)) * (
                        ((n_results + 1) * len(METRICS[:-1]) + 2) * barwidth
                    )

                    if results[dataset][model][metric][quant][0] is not None:
                        if i == j == k == 0:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                label=labels[l],
                                color=colors[l],
                                edgecolor="black",
                            )
                        else:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                color=colors[l],
                                edgecolor="black",
                            )

                    if l == n_results - 1 and j == len(DATASETS) - 1:
                        for po in pos:
                            ax[j][k].text(
                                po
                                # move as bars as there are nested x values + gap between nested x values
                                + i * barwidth * (n_results + 1)
                                # move to the middle of current bars
                                # (we dont consider gap part of bars for text purposes)
                                + n_results * barwidth / 2
                                # adjust because text is not a point (assume about one bar)
                                # + starting point is after first bar
                                - 0.8 * barwidth,  #  - offset_y_axis_labels[i],
                                0.13
                                * (height_range[1] - min(height_range[0], 0)),
                                metric_names[i],
                                size=10,
                                fontweight="bold",
                                rotation=60,
                                ha="left",
                                rotation_mode="anchor",
                            )
                if j == 0:
                    model_name = (
                        model.split("--")[1] if "--" in model else model
                    )
                    ax[j][k].set_title(model_name)

                if k == 0:
                    ax[j][k].set_ylabel(
                        DATASET_NAMES[j], rotation=70, labelpad=20, fontsize=14
                    )

                ax[j][k].grid(axis="y", linestyle="dashed")

                ax[j][k].set_xticks(
                    # move to middle of whole current section of nested_x_values + gap for each
                    # y_axis value, adjust one bar back because starting after the first bar
                    barwidth * (len(METRICS[:-1]) * (n_results + 1) / 2 - 1)
                    + pos
                    # - offset_x_axis_labels
                )
                ax[j][k].set_xticklabels(shots)
                for label in ax[j][k].get_xticklabels():
                    label.set_y(
                        label.get_position()[1] - 0.08
                    )  # Adjust the offset value as needed

                ax[j][k].set_ylim(
                    top=height_range[1] * 1.03, bottom=height_range[0] * 0.97
                )

    fig.suptitle("")
    # fig.legend(
    #     loc="center left",
    #     fontsize=12,
    #     bbox_to_anchor=(-0.01, 0.52),
    #     # ncol=2,
    #     frameon=False,
    # )
    fig.legend(
        loc="upper center",
        fontsize=12,
        bbox_to_anchor=(0.585, 1.025),
        ncol=n_results,
        frameon=False,
    )
    # set x label for entire figure
    fig.text(
        0.245,
        0.966,
        "Consistency of",
        ha="center",
        fontsize=16,
    )
    fig.text(0.535, 0, "Shot", ha="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"holistic-consistency.pdf",
        )
    )
    plt.clf()


def reinforcement_accuracy(analyses_folder, basename=None):

    # labels = ["sim$(\hat{f}, f)$", "sim$(\hat{f}, \hat{f}_{\mathcal{D}})$"]
    labels = ["ICL", "Prior Prompt", "Prior (labels) Prompt"]
    metric_names = ["JS", "Mic"]

    results = {
        dataset: {
            model: {metric: [] for metric in METRICS[:-1]} for model in MODELS
        }
        for dataset in DATASETS
    }

    height_range = [float("inf"), -float("inf")]

    for dataset in DATASETS:
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for metric in METRICS[:-1]:
                try:
                    df = pd.read_csv(
                        os.path.join(
                            analyses_folder,
                            f"{dataset}-{model}",
                            f"{metric}.csv",
                        ),
                        index_col=0,
                    )
                except FileNotFoundError:
                    results[dataset][model][metric] = {
                        "y_yhat": (None, None),
                        "yhat_yprior": (None, None),
                        "yhat_yprior_sedl": (None, None),
                    }
                    continue

                y_yhat = [
                    float(df[f"{shot}s"]["Ground Truth"].split("\n±")[0])
                    for shot in SHOTS
                ]
                y_yhat_std = [
                    float(df[f"{shot}s"]["Ground Truth"].split("\n±")[1])
                    for shot in SHOTS
                ]

                prior_examples = [
                    float(
                        df[f"Prior-{shot}s-prompt-{shot}s"][
                            f"Prior-{shot}s-traindev"
                        ].split("\n±")[0]
                    )
                    for shot in SHOTS
                ]
                prior_examples_std = [
                    float(
                        df[f"Prior-{shot}s-prompt-{shot}s"][
                            f"Prior-{shot}s-traindev"
                        ].split("\n±")[1]
                    )
                    for shot in SHOTS
                ]

                prior_labels = [
                    float(
                        df[f"Prior-{shot}s-prompt-{shot}s-sedl"][
                            f"Prior-{shot}s-traindev-sedl"
                        ].split("\n±")[0]
                    )
                    for shot in SHOTS
                ]
                prior_labels_std = [
                    float(
                        df[f"Prior-{shot}s-prompt-{shot}s-sedl"][
                            f"Prior-{shot}s-traindev-sedl"
                        ].split("\n±")[1]
                    )
                    for shot in SHOTS
                ]

                results[dataset][model][metric] = {
                    "y_yhat": (y_yhat, y_yhat_std),
                    "yhat_yprior": (prior_examples, prior_examples_std),
                    "yhat_yprior_sedl": (prior_labels, prior_labels_std),
                }

                for quant in results[dataset][model][metric]:
                    height_range[0] = min(
                        height_range[0],
                        min(results[dataset][model][metric][quant][0]),
                    )
                    height_range[1] = max(
                        height_range[1],
                        max(results[dataset][model][metric][quant][0]),
                    )

    n_results = 3  # y_yhat, yhat_yprior, yhat_yprior_sedl

    # bar plot comparing the strength of the prior to the ground truth
    # compare for each dataset, all the models for every metric at different shots
    fig, ax = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(15, 5),
        sharey=True,
        sharex=True,
    )

    barwidth = 0.09
    colors = ["limegreen", "violet", "red"]

    for j, dataset in enumerate(DATASETS):
        for k, model in enumerate(MODELS):
            for i, metric in enumerate(METRICS[:-1]):
                # nested_x_value is y_yhat and yhat_yprior
                # x_value is SHOT

                for l, quant in enumerate(results[dataset][model][metric]):

                    pos = np.arange(len(SHOTS)) * (
                        ((n_results + 1) * len(METRICS[:-1]) + 2) * barwidth
                    )

                    if results[dataset][model][metric][quant][0] is not None:
                        if i == j == k == 0:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                label=labels[l],
                                color=colors[l],
                                edgecolor="black",
                            )
                        else:
                            ax[j][k].bar(
                                pos
                                + i * barwidth * (n_results + 1)
                                + l * barwidth,
                                results[dataset][model][metric][quant][0],
                                barwidth,
                                yerr=results[dataset][model][metric][quant][1],
                                color=colors[l],
                                edgecolor="black",
                            )

                    if l == n_results - 1 and j == len(DATASETS) - 1:
                        for po in pos:
                            ax[j][k].text(
                                po
                                # move as bars as there are nested x values + gap between nested x values
                                + i * barwidth * (n_results + 1)
                                # move to the middle of current bars
                                # (we dont consider gap part of bars for text purposes)
                                + n_results * barwidth / 2
                                # adjust because text is not a point (assume about one bar)
                                # + starting point is after first bar
                                - 0.8 * barwidth,  #  - offset_y_axis_labels[i],
                                0.15
                                * (height_range[1] - min(height_range[0], 0)),
                                metric_names[i],
                                size=10,
                                fontweight="bold",
                                rotation=60,
                                ha="left",
                                rotation_mode="anchor",
                            )
                if j == 0:
                    model_name = (
                        model.split("--")[1] if "--" in model else model
                    )
                    ax[j][k].set_title(model_name)

                if k == 0:
                    ax[j][k].set_ylabel(
                        DATASET_NAMES[j], rotation=70, labelpad=20, fontsize=14
                    )

                ax[j][k].grid(axis="y", linestyle="dashed")

                ax[j][k].set_xticks(
                    # move to middle of whole current section of nested_x_values + gap for each
                    # y_axis value, adjust one bar back because starting after the first bar
                    barwidth * (len(METRICS[:-1]) * (n_results + 1) / 2 - 1)
                    + pos
                    # - offset_x_axis_labels
                )
                ax[j][k].set_xticklabels(SHOTS)
                for label in ax[j][k].get_xticklabels():
                    label.set_y(
                        label.get_position()[1] - 0.08
                    )  # Adjust the offset value as needed

                ax[j][k].set_ylim(
                    top=height_range[1] * 1.1, bottom=height_range[0] * 0.97
                )

    fig.suptitle("")
    # fig.legend(
    #     loc="center left",
    #     fontsize=12,
    #     bbox_to_anchor=(-0.01, 0.52),
    #     # ncol=2,
    #     frameon=False,
    # )
    fig.legend(
        loc="upper center",
        fontsize=12,
        bbox_to_anchor=(0.617, 1.025),
        ncol=n_results,
        frameon=False,
    )
    fig.text(
        0.353,
        0.966,
        "Proxy Performance of",
        ha="center",
        fontsize=16,
    )
    fig.text(0.531, 0, "Shot", ha="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"reinforcement-accuracy.pdf",
        )
    )
    plt.clf()


def baselines(analyses_folder, log_folder, basename=None):

    def read_scores(experiment_folder, model, shot=25):
        with open(
            os.path.join(
                experiment_folder,
                f"{model}-similarity-retrieval-{shot}-shot_0",
                "aggregated_metrics.yml",
            )
        ) as fp:
            data = yaml.safe_load(fp)[""]

        def process_metric(metric):
            if "+" in metric:
                return float(metric.split("+")[0])
            elif "±" in metric:
                return float(metric.split("±")[0])

        return {
            model: {
                "JS": process_metric(data["test_jaccard_score"]),
                "Mic": process_metric(data["test_micro_f1"]),
                "Mac": process_metric(data["test_macro_f1"]),
            }
        }

    results = {"SemEval": {}, "GoEmotions": {}}
    height_range = {
        "GoEmotions": [float("inf"), -float("inf")],
        "SemEval": [float("inf"), -float("inf")],
    }

    models = [
        "google--gemma-7b",
        "allenai--OLMo-7B",
        "meta-llama--Llama-2-13b-chat-hf",
        "meta-llama--Llama-2-70b-chat-hf",
    ]

    api_models = [
        "gpt-3.5-turbo",
        "gpt-4-1106-preview",
    ]

    for dataset in ["GoEmotions", "SemEval"]:
        for model in models:
            results[dataset].update(
                read_scores(os.path.join(log_folder, dataset), model)
            )

        for model in api_models:
            results[dataset].update(
                read_scores(os.path.join(log_folder, dataset + "OpenAI"), model)
            )

    results["SemEval"].update(
        {"Demux": {"JS": 0.612, "Mic": 0.723, "Mac": 0.581}}
    )
    results["GoEmotions"].update(
        {"Demux": {"JS": 0.661, "Mic": 0.702, "Mac": 0.692}},
    )

    for dataset in results:
        for model in results[dataset]:
            for metric in results[dataset][model]:
                if results[dataset][model][metric] is not None:
                    height_range[dataset][0] = min(
                        height_range[dataset][0],
                        results[dataset][model][metric],
                    )
                    height_range[dataset][1] = max(
                        height_range[dataset][1],
                        results[dataset][model][metric],
                    )

    n_results = 7

    # bar plot comparing the strength of the prior to the ground truth
    # compare for each dataset, all the models for every metric at different shots
    fig, ax = plt.subplots(
        len(DATASETS),
        1,
        figsize=(6, 3),
        sharex=True,
    )

    barwidth = 0.04
    colors = [
        "skyblue",
        "limegreen",
        "orange",
        "violet",
        "red",
        "peru",
        "dimgray",
    ]

    for j, dataset in enumerate(results):
        for k, model in enumerate(results[dataset]):
            # for i, metric in enumerate(results[dataset][model]):
            # nested_x_value is models
            # x_value is metric

            pos = np.arange(len(METRICS)) * (n_results + 3) * barwidth

            if results[dataset][model]["JS"] is not None:
                if j == 0:
                    ax[j].bar(
                        pos + k * barwidth,
                        results[dataset][model].values(),
                        barwidth,
                        label=model.split("--")[1] if "--" in model else model,
                        color=colors[k],
                        edgecolor="black",
                    )
                else:
                    ax[j].bar(
                        pos + k * barwidth,
                        results[dataset][model].values(),
                        barwidth,
                        color=colors[k],
                        edgecolor="black",
                    )

        ax[j].grid(axis="y", linestyle="dashed")

        ax[j].set_ylabel(
            DATASET_NAMES[j], rotation=70, labelpad=13, fontsize=10
        )

        ax[j].set_xticks(
            # move to middle of whole current section of nested_x_values + gap for each
            # y_axis value, adjust one bar back because starting after the first bar
            barwidth * ((n_results + 1) / 2 - 1)
            + pos
        )

        ax[j].set_xticklabels(["Jaccard Score", "Micro F1", "Macro F1"])

        ax[j].set_ylim(
            top=height_range[dataset][1] * 1.03,
            bottom=height_range[dataset][0] * 0.97,
        )

    fig.suptitle("")
    plt.subplots_adjust(top=0.7)
    fig.legend(
        loc="upper center",
        fontsize=8,
        bbox_to_anchor=(0.55, 1.05),
        ncol=4,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"baselines.pdf",
        ),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    script = sys.argv[1]
    globals()[script](*sys.argv[2:])
