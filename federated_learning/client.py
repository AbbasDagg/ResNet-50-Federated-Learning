import torch
import torch.nn as nn
import torch.optim as optim
from RESNET_50 import get_resnet50_model
import argparse
import nvflare.client as flare
from split_data import get_client_data_loader
import os


def get_client_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, required=True, help="Client identifier")
    parser.add_argument("--num_clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to save/load the model (optional; defaults to /tmp/nvflare/models/<client_id>_cifar_net.pth)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Root folder for CIFAR10 dataset (optional; env CIFAR10_ROOT also supported)",
    )
    return parser.parse_args()


def run_client(args: argparse.Namespace):
    # get local parameters
    print("Starting client...")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = args.lr
    epochs = args.epochs
    client_id = args.client_id
    # Resolve model path: prefer arg, else env NVFLARE_MODEL_ROOT, else /tmp
    if args.model_path:
        model_path = args.model_path
    else:
        model_root = os.environ.get("NVFLARE_MODEL_ROOT", "/tmp/nvflare/models")
        os.makedirs(model_root, exist_ok=True)
        model_path = os.path.join(model_root, f"{client_id}_cifar_net.pth")

    train_loader, test_loader = get_client_data_loader(
        client_id, num_clients=args.num_clients, data_root=args.data_root
    )
    print("Data loaders obtained.")
    print(f"{DEVICE}, {lr}, {epochs}, {model_path}, {client_id}")
    resnet = get_resnet50_model()
    criterion = nn.CrossEntropyLoss()

    def evaluate(input_weights):
        resnet50 = get_resnet50_model()
        resnet50.load_state_dict(input_weights)
        resnet50.to(DEVICE)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                # calculate outputs by running images through the network
                outputs = resnet50(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct // total

    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        client_id = flare.get_site_name()
        print(
            f"({client_id}) current_round={input_model.current_round}, total_rounds={input_model.total_rounds}"
        )
        resnet.load_state_dict(input_model.params)
        resnet.to(DEVICE)
        optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=0.9)

        steps = epochs * len(train_loader)
        for epoch in range(epochs):
            running_loss = 0.0
            for _, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                optimizer.zero_grad()
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(
                f"Client {client_id}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}"
            )
        print(f"({client_id}) Finished Training")

        accuracy = evaluate(resnet.state_dict())
        torch.save(resnet.state_dict(), model_path)

        output_model = flare.FLModel(
            params=resnet.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )

        flare.send(output_model)


if __name__ == "__main__":
    client_args = get_client_args()
    run_client(client_args)
