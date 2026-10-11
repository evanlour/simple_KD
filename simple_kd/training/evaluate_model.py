import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from simple_kd.utils import InMemoryDataset
from pathlib import Path

def evaluate_model(model: torch.nn.Module,
        images: Path | np.ndarray,
        labels: Path | np.ndarray,
        batch_size:int,
        augs: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        classes:int = 1000,
        save_folder: Path | None = None,
        save_prefix: str = "") -> dict:
    """Tests the given model on the given dataset. If the model is trained on a subset of the dataset, provide the class number.

    Args:
        model (torch.nn.Module): The model that will be tested.
        images (str | np.ndarray): The test images. Accepts either a numpy array, or the location to one.
        labels (str | np.ndarray): The test labels. Accepts either a numpy array, or the location to one.
        batch_size (int): The batch size to be used.
        augs (torch.nn.Module): The augmentations that will be applied to the model. Usually conversion to float and normalization.
        device (torch.device): The device where the testing will take place.
        criterion (torch.nn.Module): The criterion for loss calculation.
        classes (int, optional): Set to the number of the model's trained classes. If the number is lower than of the class amount, samples with a class label equal or above that will be removed. Defaults to 1000.
        save_folder (str, optional): If not none, saves test metrics to this directory. Defaults to None.
    Returns:
        ret_data(dict): Dictionary containing the labels and predictions, along with accuracy, loss, precision and recall data.
    """

    # Setting up the test dataset
    if isinstance(images, Path):
        images = np.load(images)
    if isinstance(labels, Path):
        labels = np.load(labels)

    assert type(images) is np.ndarray, "Something went wrong, images is not a numpy array!"
    assert type(labels) is np.ndarray, "Something went wrong, labels is not a numpy array!"
    # If the model is trained on a subset of classes, subtract them
    mask = labels < classes
    images = images[mask]
    labels = labels[mask]

    test_dataset = InMemoryDataset(images, labels, augmentations=augs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                pin_memory_device=str(device))

    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device) # type: ignore

            outputs = model(inputs)

            # Loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) #type: ignore

            # Accuracy
            test_total += labels.size(0) #type: ignore
            test_correct += (predicted == labels).sum().item() #type: ignore

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_loss /= test_total
    test_accuracy = 100. * test_correct / test_total

    precision = float(precision_score(all_labels, all_preds, average="weighted", zero_division=0))    #type: ignore
    recall = float(recall_score(all_labels, all_preds, average="weighted", zero_division=0))          #type: ignore
    f1_test_score = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))       #type: ignore
    precision_macro = float(precision_score(all_labels, all_preds, average="macro", zero_division=0)) #type: ignore
    recall_macro = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))       #type: ignore
    f1_test_score_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))    #type: ignore

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1_score": f1_test_score,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1_score": f1_test_score_macro,
    }

    if save_folder is not None:
        json_name = f"{save_prefix}_dataset_test_scores.json" if save_prefix else "dataset_test_scores.json"
        with open(save_folder / json_name, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}%")
    print(f"Test Precision (Weighted): {precision:.4f}")
    print(f"Test Recall (Weighted): {recall:.4f}")
    print(f"Test F1 Score (Weighted): {f1_test_score:.4f}")
    print(f"Test Precision (Macro): {precision_macro:.4f}")
    print(f"Test Recall (Macro): {recall_macro:.4f}")
    print(f"Test F1 Score (Macro): {f1_test_score_macro:.4f}")

    if save_folder is not None:
        cm = confusion_matrix(all_labels, all_preds)
        normalized_cm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9) # Better for imbalanced classes

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            normalized_cm,
            cmap="Blues",
            cbar=True,
        )
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_name = f"{save_prefix}_confusion_matrix.png" if save_prefix else "confusion_matrix.png"
        plt.savefig(save_folder / cm_name)
        plt.close()

        df_preds = pd.DataFrame({
            "ground_truth": all_labels,
            "prediction": all_preds,
            "correct": [x == y for x, y in zip(all_labels, all_preds)]
        })
        parquet_name = f"{save_prefix}_predictions.parquet" if save_prefix else "predictions.parquet"
        df_preds.to_parquet(save_folder / parquet_name, index=False)

    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "empty_cache"):
        torch.accelerator.empty_cache() #type: ignore

    return metrics
