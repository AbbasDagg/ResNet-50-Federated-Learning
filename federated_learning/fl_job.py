from RESNET_50 import get_resnet50_model
from server import get_test_loader, evaluate_global_model

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
        "--n_clients",
        type=int,
        default=4,
        help="Number of clients to participate in federated learning",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=2,
        help="Number of communication rounds for federated learning",
    )
    return parser.parse_args()


def runner():
    print("Starting Federated Learning Job...")
    fl_args = federated_learning_arg_parser()
    n_clients = fl_args.n_clients
    num_rounds = fl_args.num_rounds
    train_script = "federated_learning/client.py"

    # Setup paths
    file_path = os.path.abspath(__file__)
    workspace_path = os.path.dirname(os.path.dirname(file_path)) + "/workspace"
    data_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), "data")
    
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
    
    # Server evaluation setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_loader(data_path)
    initial_model = get_resnet50_model()
    
    # Evaluate initial model
    print("Evaluating initial model...")
    initial_accuracy, initial_loss = evaluate_global_model(initial_model.state_dict(), test_loader, device)
    print(f"Initial Model - Accuracy: {initial_accuracy:.2f}%, Loss: {initial_loss:.4f}")

    job = FedAvgJob(
        name="fedavg",
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=initial_model,
    )
    print("FedAvgJob created.")
    # Add clients
    for i in range(n_clients):
        print(f"Adding client site-{i+1}")
        executor = ScriptRunner(
            script=train_script, script_args=f"--client_id site-{i+1}"
        )
        job.to(executor, f"site-{i + 1}")

    print("job-config is at ", workspace_path)
    job.simulator_run(workspace=workspace_path)
    
    # Evaluate final global model after training
    print("\n" + "="*60)
    print("Evaluating Final Global Model")
    print("="*60)
    final_model_path = os.path.join(workspace_path, "server/simulate_job/app_server/best_FL_global_model.pt")
    if os.path.exists(final_model_path):
        final_model_data = torch.load(final_model_path)
        final_model_weights = final_model_data['model']        
        final_accuracy, final_loss = evaluate_global_model(final_model_weights, test_loader, device)
        print(f"Final Global Model - Accuracy: {final_accuracy:.2f}%, Loss: {final_loss:.4f}")
        print(f"Improvement: {final_accuracy - initial_accuracy:+.2f}%")
    else:
        print("Final model not found at expected path")
    print("="*60)


if __name__ == "__main__":
    runner()
