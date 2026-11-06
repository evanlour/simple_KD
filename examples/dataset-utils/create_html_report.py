import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

HISTORY_PATH = "/mnt/storage/Datasets_results/CIFAR100_vit_distillation_results_deit_tiny_KD/history.csv"
PREDICTIONS_PATH = (
    "/mnt/storage/Datasets_results/CIFAR100_vit_train_results/predictions.parquet"
)


if __name__ == "__main__":
    # Example of reading values from train history
    df = pd.read_csv(HISTORY_PATH)
    cols = [df["teacher_train_accuracy"], df["student_train_accuracy"]]
    names = ["Teacher", "Student"]
    x = df.index

    fig1 = go.Figure()
    for column, name in zip(cols, names):
        fig1.add_trace(
            go.Scatter(
                x=x,
                y=column,
                mode="lines",
                name=name,
            )
        )

    fig1.update_layout(
        title={
            "text": "Teacher vs student train comparison",
            "font": {"size": 35, "color": "black", "family": "Arial Black"},
        },
        xaxis_title={"text": "Epochs", "font": {"size": 25, "family": "Arial Black"}},
        yaxis_title={
            "text": "Accuracy(%)",
            "font": {"size": 25, "family": "Arial Black"},
        },
        legend={"title": "Traces", "font": {"size": 18, "family": "Arial Black"}},
        template="plotly_white",
    )
    fig1.show()

    # Example of plotting the predictions of a model
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    cm = confusion_matrix(predictions["ground_truth"], predictions["prediction"])

    fig2 = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=np.arange(0, 100),
            y=np.arange(0, 100),
            colorscale="Blues",
            colorbar={"title": {"text": "Samples", "font": {"size": 25}}},
        )
    )

    fig2.update_layout(
        title={
            "text": "Confusion matrix",
            "font": {"size": 35, "color": "black", "family": "Arial Black"},
        },
        xaxis_title={
            "text": "Predictions",
            "font": {"size": 25, "family": "Arial Black"},
        },
        yaxis_title={
            "text": "Ground truth",
            "font": {"size": 25, "family": "Arial Black"},
        },
        yaxis={"autorange": "reversed"},
        template="plotly_white",
    )
    fig2.show()
