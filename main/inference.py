
"""
This module handles image classification inference using a pre-trained model.

It loads the latest model checkpoint from the weights folder, processes the input image,
and returns the predicted label ("Real" or "AI Generated") along with the confidence score.
"""

import aidetector

def run_inference(image_path):
    """
    Runs inference on a single image and returns the predicted label and accuracy.

    Args:
        image_path (str): The full path to the image file to be analyzed.

    Returns:
        tuple:
            - predicted_label (str): Either "Real" or "AI Generated"
            - accuracy (float): Confidence score (between 0 and 1) of the prediction
    """
    # Define the path where the model weights are stored
    weights_folder = "./models"

    # Call the prediction pipeline from aidetector.py
    predicted_label, probabilities = aidetector.main(image_path, weights_folder)

    # Extract the higher probability as the confidence
    prob_real = probabilities[0][0].item()
    prob_fake = probabilities[0][1].item()
    accuracy = max(prob_real, prob_fake)

    return predicted_label, accuracy
