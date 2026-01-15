import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import re
from torch.utils.data import DataLoader, Subset
import numpy as np

def split_cifar10_data(data_path: str, num_clients: int = 4):
    """Splits CIFAR-10 data among a specified number of clients deterministically.

    Args:
        data_path (str): Root folder containing CIFAR-10 data (downloaded or to be downloaded).
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
    labels = np.array(train_dataset.targets)
    client_indices = [[] for _ in range(num_clients)]

    for label in range(10):
        idx = np.where(labels == label)[0]
        idx = np.sort(idx)
        splits = np.array_split(idx, num_clients)
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())

    # Create DataLoaders for each client
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return client_loaders, test_loader


def get_client_data_loader(
    client_id: str, num_clients: int, data_path: str
) -> tuple[DataLoader, DataLoader]:
    client_loaders, test_loader = split_cifar10_data(
        data_path=data_path, num_clients=num_clients
    )
    indx = int(re.search(r"site-(\d+)", client_id).group(1)) - 1
    print(indx)
    return client_loaders[indx], test_loader
