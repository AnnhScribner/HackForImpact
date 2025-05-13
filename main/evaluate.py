# evaluate.py
# Anna Scribner, Michael Gilbert, Muskan Gupta, Roger Tang

import argparse
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import glob
import os

from model import get_model
from custom_dataset import get_dataset


def load_latest_model(model, device, weights_folder):
    """
    Loads the most recent model checkpoint from the given folder.

    Args:
        model: The initialized model object.
        device: The torch device (CPU or CUDA).
        weights_folder (str): Path to the folder containing model weights.

    Returns:
        model: The model with loaded weights.
    """
    list_of_files = glob.glob(os.path.join(weights_folder, 'model_epoch_*.pth'))
    if not list_of_files:
        raise FileNotFoundError(f"No model files found in {weights_folder}.")

    latest_file = max(list_of_files, key=os.path.getctime)
    checkpoint = torch.load(latest_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model from: {latest_file}")
    return model


def evaluate_model(test_data_dir, device, criterion, weights_folder):
    """
    Evaluates the model on the test dataset and computes metrics.

    Args:
        test_data_dir (str): Path to test dataset.
        device: The torch device (CPU or CUDA).
        criterion: Loss function.
        weights_folder (str): Path to trained model weights.

    Returns:
        avg_test_loss (float)
        accuracy (float)
        precision (float)
        recall (float)
        f1 (float)
    """
    test_data = get_dataset(test_data_dir)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=6, pin_memory=True)

    model = get_model(device)
    model = load_latest_model(model, device, weights_folder)
    model.eval()

    test_loss = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="🔍 Evaluating", unit="batch") as progress_bar:
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)

                test_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({"Loss": loss.item()})
                progress_bar.update()

    # Metrics
    all_labels = np.array(all_labels)
    all_predicted = np.array(all_predicted)
    avg_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted, average='macro')
    recall = recall_score(all_labels, all_predicted, average='macro')
    f1 = f1_score(all_labels, all_predicted, average='macro')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predicted)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(conf_mat=cm, figsize=(10, 10), show_absolute=True, show_normed=True)

    # Create metrics folder if it doesn’t exist
    os.makedirs("./metrics", exist_ok=True)
    plt.savefig("./metrics/Custom_Evaluation.png")
    plt.show()

    return avg_test_loss, accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on a dataset.')
    parser.add_argument('test_data_dir', type=str, help='Directory path for test data.')
    parser.add_argument('--weights_folder', type=str, default='./models', help='Folder path for model weights.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()

    # Run evaluation
    avg_test_loss, accuracy, precision, recall, f1 = evaluate_model(
        args.test_data_dir, device, criterion, args.weights_folder
    )

    # Print final metrics
    print("\n📊 Evaluation Results:")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
