from pathlib import Path
import torch

def load_market(data_path, split="train"):
    data_path = Path(data_path)
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    data = torch.load(
        data_path,
        map_location="cpu",
        weights_only=True,
    )

    # 저장된 데이터가 올바른 실험에 속하는지 확인
    metadata = data["metadata"]
    if metadata["run_id"] not in data_path.stem:
        raise ValueError("run_id mismatch")

    if metadata["split"] != split:
        raise ValueError("split mismatch")

    # 시장 데이터의 기본 구조 검증
    S = data["stock"]
    V = data["variance"]
    A = data["variance_swap"]
    I = data["integrated_variance"]
    payoff = data["payoff"]

    if not (S.shape == V.shape == A.shape == I.shape):
        raise ValueError("Market path shapes do not match")

    if payoff.shape != (S.shape[0],):
        raise ValueError("Invalid payoff shape")

    if data["time_to_maturity"].shape != (S.shape[1],):
        raise ValueError("Invalid maturity shape")

    return data