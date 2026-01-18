import argparse
import torchvision.datasets as datasets
from pathlib import Path

Default_DATASET_PATH = "./data"


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Datasets")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Default_DATASET_PATH,
        help="Path to save the downloaded dataset",
    )
    args = parser.parse_args()
    return args


def download_datasets(data_path: Path) -> None:
    path = data_path / "cifar-10-batches-py"
    if path.exists():
        print(f"CIFAR10 dataset already exists at {data_path}. Skipping download.")
        return
    print(f"Downloading CIFAR10 dataset to {data_path}...")
    datasets.CIFAR10(root=data_path, train=True, download=True)
    datasets.CIFAR10(root=data_path, train=False, download=True)
    print("CIFAR10 dataset downloaded.")


def main():
    args = arg_parse()
    download_datasets(args.data_path)


if __name__ == "__main__":
    main()
