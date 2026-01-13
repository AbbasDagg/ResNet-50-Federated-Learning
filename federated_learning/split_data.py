import torch
from torch.utils.data import random_split
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import re
from torch.utils.data import DataLoader, Subset
import os


def resolve_data_root(preferred: str | None = None) -> str:
    """Resolve a shared/local data root for CIFAR10.

    Priority:
    1) explicit `preferred` value
    2) env var `CIFAR10_ROOT`
    3) project-level `data` folder (parent of this file)
    """
    if preferred:
        root = preferred
    else:
        root = os.environ.get("CIFAR10_ROOT")
        if not root:
            root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
            )
    os.makedirs(root, exist_ok=True)
    return root


def split_cifar10_data(num_clients: int = 4, data_root: str | None = None):
    """Splits CIFAR-10 data among a specified number of clients deterministically.

    Args:
        num_clients (int): Number of clients.
    Returns:
        tuple[list[DataLoader], DataLoader]: List of client DataLoaders, and test DataLoader.
    """
    # Transforms
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Resolve dataset root and load datasets
    root = resolve_data_root(data_root)
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test
    )

    # Deterministic split
    N = len(train_dataset)
    client_size = N // num_clients

    client_loaders = []

    for i in range(num_clients):
        start = i * client_size
        end = start + client_size
        subset = Subset(train_dataset, list(range(start, end)))
        loader = DataLoader(subset, batch_size=64, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return client_loaders, test_loader


def get_client_data_loader(
    client_id: str, num_clients, data_root: str | None = None
) -> tuple[DataLoader, DataLoader]:
    client_loaders, test_loader = split_cifar10_data(
        num_clients=num_clients, data_root=data_root
    )
    indx = int(re.search(r"site-(\d+)", client_id).group(1)) - 1
    print(indx)
    return client_loaders[indx], test_loader
