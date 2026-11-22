import os

import torch
from torch.utils.data import TensorDataset, DataLoader

from functions.data_loader import load_market

ROOT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_DIR, "data/heston", "fixed")
RUN_ID = "20261120_181909_1e1e00d7"

train_path = (
    os.path.join(DATA_PATH, "train", f"market_{RUN_ID}.pt")
)

test_path = (
    os.path.join(DATA_PATH, "test", f"market_{RUN_ID}.pt")
)

train = load_market(train_path, "train")
test = load_market(test_path, "test")

print("Train:", train["stock"].shape)
print("Test:", test["stock"].shape)

metadata = train["metadata"]
print(f"\n[Meta Data]")
print("Run ID:", metadata["run_id"])
print("Seed:", metadata["seed"])
print("Market config:", metadata["config"]["market"])
print("Heston config:", metadata["config"]["heston"])

dataset = TensorDataset(
    train["stock"],
    train["variance"],
    train["variance_swap"],
    train["payoff"],
)

train_loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True,
)

S, V, A, payoff = next(iter(train_loader))
print(f"\n[Batched Data]")
print("Stock:", S.shape)
print("Variance:", V.shape)
print("Variance swap:", A.shape)
print("Payoff:", payoff.shape)


print(f"\n\n==================================== [전체 Train 데이터] ====================================\n\n")


for key, value in train.items():
    if isinstance(value, torch.Tensor):
        print(f"{key} - {value.shape}, {value.dtype}")
    elif isinstance(value, dict):
        print(f"{key} - ", {
            k: v.shape if isinstance(v, torch.Tensor) else type(v).__name__
            for k, v in value.items()
        })

print("\n================================================\n")

for key, value in train.items():
    print(f"{key=}")
    if isinstance(value, torch.Tensor):
        print(value[[0]])
        print()
    elif isinstance(value, dict):
        for k, v in value.items():
            print(f"{k} - ", end="")
            if isinstance(value, torch.Tensor):
                print(v[0])
            else:
                print(v)
        print()

