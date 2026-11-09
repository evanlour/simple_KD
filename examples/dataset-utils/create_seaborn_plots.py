import pandas as pd
from sklearn.metrics import confusion_matrix

from simple_kd.visualization import plot_cm, plot_lines

HISTORY_PATH = "/mnt/storage/Datasets_results/CIFAR100_vit_distillation_results_deit_tiny_KD/history.csv"
PREDICTIONS_PATH = (
    "/mnt/storage/Datasets_results/CIFAR100_vit_train_results/predictions.parquet"
)


if __name__ == "__main__":
    # Example of reading values from train history
    df = pd.read_csv(HISTORY_PATH)
    values = [df["teacher_train_accuracy"], df["student_train_accuracy"]]
    fig1 = plot_lines(
        values,  # type: ignore
        title="Teacher vs student train comparison",
        xlabel="Epochs",
        ylabel="Accuracy(%)",
        savename="./res/train_lines.png",
    )

    # Example of plotting the predictions of a model
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    cm = confusion_matrix(predictions["ground_truth"], predictions["prediction"])
    fig2 = plot_cm(
        cm,
        classes=100,
        title="Confusion matrix",
        xlabel="Predictions",
        ylabel="Ground truth",
        savename="./res/confusion_matrix.png",
    )
