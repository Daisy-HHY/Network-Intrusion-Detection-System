from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metric_comparison(metrics_df: pd.DataFrame, output_path: Path):
    ax = metrics_df.plot(x="model", y=["accuracy", "precision", "recall", "f1"], kind="bar")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=200)
    plt.close(ax.figure)


def plot_confusion(cm, labels, output_path: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(frame: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=frame, x="importance", y="feature", orient="h")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
