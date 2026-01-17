from pathlib import Path
from RESNET_50 import get_resnet50_model
from server import get_test_loader, evaluate_global_model
from schema.fl_config import FLConfig, aggregation_methods
from pydantic import ValidationError
import yaml
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
import os
import argparse
import torch


def federated_learning_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Learning with NVFlare and PyTorch"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the federated learning configuration file",
    )

    return parser.parse_args()


def load_fl_config(config_path: str) -> FLConfig:
    path = Path(config_path)
    with path.open("r") as f:
        config_dict = yaml.safe_load(f) or {}
    try:
        fl_config = FLConfig.model_validate(config_dict)
        return fl_config
    except ValidationError as e:
        raise SystemExit(f"Invalid FL config:\n{e}")


def runner():
    print("Starting Federated Learning Job...")
    fl_args = federated_learning_arg_parser()

    fl_config = load_fl_config(fl_args.config)

    server_config = fl_config.server
    n_clients = len(server_config.clients.keys())
    num_rounds = server_config.num_rounds
    aggregation_method = server_config.aggregation_method

    clients = fl_config.server.clients
    # Setup paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_path = os.path.join(repo_root, "workspace")
    data_path = os.path.join(repo_root, "data")

    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    # Server evaluation setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_loader(data_path)
    initial_model = get_resnet50_model()

    # Evaluate initial model
    print("Evaluating initial model...")
    initial_accuracy, initial_loss = evaluate_global_model(
        initial_model.state_dict(), test_loader, device
    )
    print(f"Initial Model - Accuracy: {initial_accuracy:.2f}%, Loss: {initial_loss:.4f}")

    Agg_method = aggregation_methods[aggregation_method.lower()]
    job = Agg_method(
        name=aggregation_method.lower(),
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=initial_model,
    )

    print(f"{aggregation_method.lower()} job created.")
    # Add clients
    train_script = "federated_learning/client.py"
    for i, (client_id, client_config) in enumerate(clients.items()):
        print(f"Adding client {client_id}")
        executor = ScriptRunner(
            script=train_script,
            script_args=f"--client_id site-{i+1} --num_clients {n_clients} --lr {client_config.lr} --batch_size {client_config.batch_size} --epochs {client_config.epochs} --workspace_path {workspace_path} --data_path {data_path}",
        )
        job.to(executor, f"site-{i + 1}")

    print("job-config is at ", workspace_path)
    job.simulator_run(workspace=workspace_path)

    # Evaluate final global model after training
    print("\n" + "=" * 60)
    print("Evaluating Final Global Model")
    print("=" * 60)
    final_model_path = os.path.join(
        workspace_path, "server/simulate_job/app_server/best_FL_global_model.pt"
    )
    if os.path.exists(final_model_path):
        final_model_data = torch.load(final_model_path)
        final_model_weights = final_model_data["model"]
        final_accuracy, final_loss = evaluate_global_model(
            final_model_weights, test_loader, device
        )
        print(
            f"Final Global Model - Accuracy: {final_accuracy:.2f}%, Loss: {final_loss:.4f}"
        )
        print(f"Improvement: {final_accuracy - initial_accuracy:+.2f}%")
    else:
        print("Final model not found at expected path")
    print("=" * 60)


if __name__ == "__main__":
    runner()
