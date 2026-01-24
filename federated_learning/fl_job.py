from pathlib import Path
from RESNET_50 import get_resnet50_model
from server import get_test_loader, evaluate_global_model
from schema.fl_config import FLConfig, aggregation_methods
from pydantic import ValidationError
import yaml
from nvflare.job_config.script_runner import ScriptRunner
import os
import argparse
import torch
from datetime import datetime
import json


def federated_learning_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Learning with NVFlare and PyTorch"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the federated learning configuration file",
    )

    return parser.parse_args()


def load_fl_config(config_path: Path) -> FLConfig:
    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f) or {}
    try:
        fl_config = FLConfig.model_validate(config_dict)
        return fl_config
    except ValidationError as e:
        raise SystemExit(f"Invalid FL config:\n{e}")


def get_run_path(workspace_path: str, config_path: str) -> str:
    """Generate a unique run path based on the current date and config name."""
    date_name = datetime.now().strftime("%Y%m%d_%H%M")
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    full_path = os.path.join(workspace_path, f"{config_name}_{date_name}")
    os.makedirs(full_path, exist_ok=True)
    return full_path


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

    run_path = get_run_path(workspace_path, fl_args.config)
    print(f"Run path created at: {run_path}")

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
            script_args=f"--client_id site-{i+1} --num_clients {n_clients} --lr {client_config.lr} --batch_size {client_config.batch_size} --epochs {client_config.epochs} --workspace_path {run_path} --data_path {data_path}",
        )
        job.to(executor, f"site-{i + 1}")

    print("job-config is at ", run_path)
    job.simulator_run(workspace=run_path)

    # Evaluate final global model after training
    print("\n" + "=" * 60)
    print("Evaluating Final Global Model")
    print("=" * 60)
    final_model_path = os.path.join(
        run_path, "server/simulate_job/app_server/best_FL_global_model.pt"
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

    # Save config used for this run
    with open(os.path.join(run_path, "used_fl_config.json"), "w") as f:
        json.dump(fl_config.model_dump(), f, indent=2)
    print(f"FL config saved to {os.path.join(run_path, 'used_fl_config.json')}")


if __name__ == "__main__":
    runner()
