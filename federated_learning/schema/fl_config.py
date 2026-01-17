from pydantic import BaseModel, field_validator
from typing import Dict
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob

aggregation_methods = {"fedavg": FedAvgJob}


class Client(BaseModel):
    lr: float = 0.01
    epochs: int = 5
    batch_size: int = 64


class Server(BaseModel):
    num_rounds: int
    aggregation_method: str = "fedavg"
    clients: Dict[str, Client]

    @field_validator("aggregation_method")
    @classmethod
    def validate_aggregation_method(cls, v: str) -> str:
        v = v.lower()
        if v not in aggregation_methods.keys():
            raise ValueError(
                f"aggregation_method must be one of {sorted(aggregation_methods.keys())}"
            )
        return v


class FLConfig(BaseModel):
    server: Server
