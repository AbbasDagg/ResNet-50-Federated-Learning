import torch
from torch.utils.data import random_split
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import re
from torch.utils.data import DataLoader, Subset

data_path = "./data"


def split_cifar10_data(num_clients: int = 4):
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

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
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


def get_client_data_loader(client_id: str, num_clients) -> tuple[DataLoader, DataLoader]:
    client_loaders, test_loader = split_cifar10_data(num_clients=num_clients)
    indx = int(re.search(r"site-(\d+)", client_id).group(1)) - 1
    print(indx)
    return client_loaders[indx], test_loader
