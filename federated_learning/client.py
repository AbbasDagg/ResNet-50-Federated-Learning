import torch
import torch.nn as nn
import torch.optim as optim
from RESNET_50 import get_resnet50_model
import argparse
import nvflare.client as flare
from split_data import get_client_data_loader

# from nvflare.client.tracking import SummaryWriter
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter as TBWriter

# Use GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_client_message(client_id: str, message: str) -> None:
    """
    Function to print client messages with a consistent format.
    """
    print(f"[{client_id}] - {message}")


def get_client_args() -> argparse.Namespace:
    """
    Parse command line arguments for the federated learning client.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, required=True, help="Client identifier")
    parser.add_argument("--num_clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--lr_min",
        type=float,
        default=0.001,
        help="Minimum learning rate for LR scheduler",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--use_lr_scheduler",
        type=str,
        help="Whether to use learning rate scheduler",
    )
    parser.add_argument(
        "--workspace_path",
        type=Path,
        required=True,
        help="Path to workspace directory",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to shared data directory (defaults to workspace_path/../data)",
    )
    return parser.parse_args()


def run_client(args: argparse.Namespace) -> None:
    # Unpack arguments
    lr = args.lr
    epochs = args.epochs
    workspace_path: Path = args.workspace_path
    data_path: Path = args.data_path
    client_id = args.client_id
    lr_min = args.lr_min
    use_lr_scheduler = True if args.use_lr_scheduler.lower() == "true" else False

    print_client_message(client_id, f"Client Started Using device: {DEVICE}")

    train_loader, test_loader = get_client_data_loader(
        client_id, num_clients=args.num_clients, data_path=str(data_path)
    )
    print_client_message(client_id, "Data loaders obtained.")
    print_client_message(
        client_id,
        (
            f"num_clients={args.num_clients}, "
            f"lr={lr}, lr_min={lr_min}, "
            f"use_lr_scheduler={use_lr_scheduler}, "
            f"epochs={epochs}, workspace_path={workspace_path}, "
            f"data_path={data_path}"
        ),
    )

    resnet = get_resnet50_model()
    criterion = nn.CrossEntropyLoss()

    summary_writer = TBWriter(log_dir=workspace_path / f"tensorboard_logs/{client_id}")

    def evaluate(input_weights):
        resnet50 = get_resnet50_model()
        resnet50.load_state_dict(input_weights)
        resnet50.to(DEVICE)
        resnet50.eval()
        per_label_acc = {}
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                # calculate outputs by running images through the network
                outputs = resnet50(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for label in labels.unique():
                    per_label_correct: int = (
                        (predicted[labels == label] == label).sum().item()
                    )
                    per_label_acc[label.item()] = per_label_acc.get(label.item(), [0, 0])
                    per_label_acc[label.item()][0] += per_label_correct
                    per_label_acc[label.item()][1] += (labels == label).sum().item()
        avg_loss = total_loss / len(test_loader)
        return (
            100 * correct / total,
            {k: 100 * (v[0] / max(1, v[1])) for k, v in per_label_acc.items()},
            avg_loss,
        )

    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        client_id = flare.get_site_name()
        print_client_message(
            client_id,
            f"current_round={input_model.current_round}, total_rounds={input_model.total_rounds}",
        )

        accuracy, _per_label_accuracy, avg_loss = evaluate(input_model.params)
        print_client_message(
            client_id,
            f"Received model accuracy {accuracy:.2f}, loss {avg_loss:.4f}",
        )
        summary_writer.add_scalar(
            tag="global/global_model_accuracy",
            scalar_value=accuracy,
            global_step=input_model.current_round + 1,
        )
        summary_writer.add_scalar(
            tag="global/global_model_loss",
            scalar_value=avg_loss,
            global_step=input_model.current_round + 1,
        )

        resnet.load_state_dict(input_model.params)
        resnet.to(DEVICE)
        optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        optimizer.param_groups[0]["initial_lr"] = lr

        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=input_model.total_rounds,
                last_epoch=input_model.current_round - 1,
                eta_min=lr_min,
            )
            if input_model.current_round > 0:
                scheduler.step()
            print_client_message(
                client_id, f"Current lr: {optimizer.param_groups[0]['lr']:.6f}"
            )

        steps = epochs * len(train_loader)
        for epoch in range(epochs):
            resnet.train()
            running_loss = 0.0
            for _, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                optimizer.zero_grad()
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            training_loss = running_loss / len(train_loader)
            print_client_message(
                client_id,
                f"Epoch {epoch+1}/{epochs}, Loss: {training_loss:.4f}",
            )

            training_step = input_model.current_round * epochs + (epoch + 1)
            summary_writer.add_scalar(
                tag="local_training_loss",
                scalar_value=training_loss,
                global_step=training_step,
            )

        print_client_message(client_id, f"Finished Training")

        local_accuracy, local_per_label_accuracy, local_avg_loss = evaluate(
            resnet.state_dict()
        )
        accuracy_step = input_model.current_round + 1

        # log metrics to TensorBoard
        summary_writer.add_scalar(
            tag="local_accuracy", scalar_value=local_accuracy, global_step=accuracy_step
        )
        summary_writer.add_scalar(
            tag="local_test_loss", scalar_value=local_avg_loss, global_step=accuracy_step
        )

        print_client_message(client_id, f"Per label accuracy: {local_per_label_accuracy}")
        for class_id, acc in local_per_label_accuracy.items():
            summary_writer.add_scalar(
                tag=f"local_per_label_accuracy/class_{class_id}",
                scalar_value=acc,
                global_step=accuracy_step,
            )

        torch.save(resnet.state_dict(), workspace_path / "cifar_net.pth")

        output_model = flare.FLModel(
            params=resnet.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )

        flare.send(output_model)


if __name__ == "__main__":
    client_args = get_client_args()
    run_client(client_args)
