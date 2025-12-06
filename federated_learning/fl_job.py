from RESNET_50 import get_resnet50_model

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
from split_data import split_cifar10_data
import os
if __name__ == "__main__":
    n_clients = 5
    num_rounds = 2
    client_loaders, test_loader = split_cifar10_data(num_clients=n_clients)
    train_script = "src/client.py"

    job = FedAvgJob(name="fedavg", n_clients=n_clients, num_rounds=num_rounds, initial_model=get_resnet50_model())
    
    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args=f"--train_data {client_loaders[i]} --test_data {test_loader}"
        )
        job.to(executor, f"site-{i + 1}")

    file_path = os.path.abspath(__file__)
    workspace_path = os.path.dirname(os.path.dirname(file_path)) + "/workspace"
    job.simulator_run(workspace=workspace_path)
