
import torch
from torch.utils.data import random_split
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
data_path = './data'

def split_cifar10_data(num_clients: int = 4)-> tuple[list[DataLoader], datasets.CIFAR10]:
    """Splits CIFAR-10 data among a specified number of clients.

    Args:
        num_clients (int): The number of clients to split the data for.
    Returns:
        tuple[list[DataLoader], datasets.CIFAR10]: A tuple containing a list of DataLoaders for each client and the test dataset.
        test dataset will be common for all clients.
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_path,train=False, download=True, transform=transform_test)

    client_size = len(train_dataset) // num_clients
    g = torch.Generator().manual_seed(42)
    clients = random_split(train_dataset, [client_size] * num_clients, generator=g)
    client_loaders = []
    for i in range(num_clients):
        loader = DataLoader(
            clients[i],
            batch_size=64,
            shuffle=True
        )
        client_loaders.append(loader)
    return client_loaders, test_dataset
