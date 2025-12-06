from RESNET_50 import get_resnet50_model

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
import os
import argparse
from split_data import split_cifar10_data

def federated_learning_arg_parser()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated Learning with NVFlare and PyTorch")
    parser.add_argument(
        "--n_clients", type=int, default=4, help="Number of clients to participate in federated learning"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=2, help="Number of communication rounds for federated learning"
    )
    return parser.parse_args()


def runner():
    print("Starting Federated Learning Job...")
    fl_args = federated_learning_arg_parser()
    n_clients = fl_args.n_clients
    num_rounds = fl_args.num_rounds
    split_cifar10_data(num_clients=n_clients)
    train_script = "federated_learning/client.py"

    job = FedAvgJob(name="fedavg", n_clients=n_clients, num_rounds=num_rounds, initial_model=get_resnet50_model())
    print("FedAvgJob created.")
    # Add clients
    for i in range(n_clients):
        print(f"Adding client site-{i+1}")
        executor = ScriptRunner(
            script=train_script, script_args=f"--client_id site-{i+1}"
        )
        job.to(executor, f"site-{i + 1}")

    file_path = os.path.abspath(__file__)
    workspace_path = os.path.dirname(os.path.dirname(file_path)) + "/workspace"
    job.simulator_run(workspace=workspace_path)

if __name__ == "__main__":
    runner()
