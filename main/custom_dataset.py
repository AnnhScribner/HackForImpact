# Anna Scribner, Michael Gilbert, Muskan Gupta, Roger Tang

from torchvision import datasets, transforms
import shutil
import os

def get_transform():
    """
    Returns the transformation pipeline for preprocessing images.
    - Resizes to 200x200
    - Converts to tensor
    - Normalizes with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def is_valid_file(filename):
    """
    Checks if a file has a valid image extension.

    Args:
        filename (str): File name or path.

    Returns:
        bool: True if valid image format, False otherwise.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return filename.endswith(valid_extensions)

def remove_ipynb_checkpoints(data_dir):
    """
    Deletes .ipynb_checkpoints directory inside a dataset folder if it exists.
    (Sometimes created by Jupyter Notebooks.)
    """
    checkpoint_path = os.path.join(data_dir, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)

def get_dataset(data_dir):
    """
    Loads and returns an ImageFolder dataset with preprocessing applied.

    Args:
        data_dir (str): Path to dataset root folder (should contain subfolders like /real and /fake)

    Returns:
        torchvision.datasets.ImageFolder: Preprocessed image dataset.
    """
    remove_ipynb_checkpoints(data_dir)
    transform = get_transform()
    dataset = datasets.ImageFolder(root=data_dir, transform=transform, is_valid_file=is_valid_file)
    return dataset
