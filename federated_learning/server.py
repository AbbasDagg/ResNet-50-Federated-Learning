import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from RESNET_50 import get_resnet50_model


def get_test_loader(data_path: str, batch_size: int = 64) -> DataLoader:
    """Load CIFAR-10 test dataset for server evaluation."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    test_dataset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


def evaluate_global_model(model_weights: dict, test_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate global model on test set."""
    model = get_resnet50_model()
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss
