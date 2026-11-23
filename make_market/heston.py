import torch
import secrets
import os

from pathlib import Path
from datetime import datetime
from uuid import uuid4
from copy import deepcopy

from utils.file_utils import load_yaml, save_yaml


def sample_heston_params(
    heston_config,
    n_paths,
    generator,
    device,
):
    mode = heston_config["mode"]

    params = {}

    for name, value in heston_config["fixed"].items():

        if mode == "fixed":

            params[name] = torch.full(
                (n_paths,),
                value,
                dtype=torch.float32,
                device=device,
            )

        elif mode == "random":

            low, high = heston_config["ranges"][name]

            params[name] = low + (high - low) * torch.rand(
                n_paths,
                generator=generator,
                device=device,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return params

@torch.no_grad()
def simulate_heston(
    market_config,
    params,
    generator,
    device,
):
    S0 = market_config["S0"]

    T = (
        market_config["maturity_days"]
        / market_config["days_per_year"]
    )

    n_steps = market_config["n_steps"]
    substeps = market_config["substeps"]

    n_paths = len(params["V0"])

    dt = T / (n_steps * substeps)

    # Heston parameters
    V0 = params["V0"]
    kappa = params["kappa"]
    theta = params["theta"]
    xi = params["xi"]
    rho = params["rho"]

    # Allocate tensors
    S = torch.empty(
        (n_paths, n_steps + 1),
        device=device,
    )

    V = torch.empty_like(S)
    I = torch.empty_like(S)

    # Initial state
    s = torch.full(
        (n_paths,),
        S0,
        dtype=torch.float32,
        device=device,
    )

    v_raw = V0.clone()
    integrated_var = torch.zeros_like(V0)

    S[:, 0] = s
    V[:, 0] = V0
    I[:, 0] = integrated_var

    rho_scale = torch.sqrt(1 - rho.square())

    # Simulation
    for t in range(n_steps):

        for _ in range(substeps):

            # Independent Brownian shocks
            z1 = torch.randn(
                n_paths,
                generator=generator,
                device=device,
            )

            z2 = torch.randn(
                n_paths,
                generator=generator,
                device=device,
            )

            # Correlated Brownian shocks
            z_S = z1
            z_V = rho * z1 + rho_scale * z2

            # Full truncation Euler
            v = v_raw.clamp(min=0)
            sqrt_vdt = torch.sqrt(v * dt)

            # Integrated variance
            integrated_var = integrated_var + v * dt

            # Stock price
            s = s * torch.exp(
                -0.5 * v * dt
                + sqrt_vdt * z_S
            )

            # Variance
            v_raw = (
                v_raw
                + kappa * (theta - v) * dt
                + xi * sqrt_vdt * z_V
            )

        S[:, t + 1] = s
        V[:, t + 1] = v_raw.clamp(min=0)
        I[:, t + 1] = integrated_var

    return S, V, I

def calculate_variance_swap(V, I, params, tau):

    kappa = params["kappa"].unsqueeze(1)
    theta = params["theta"].unsqueeze(1)

    # 미래 분산 예측하여 만기 손익 계산
    expected_remaining_variance = (
        theta * tau
        + (V - theta)
        / kappa
        * (-torch.expm1(-kappa * tau))
    )

    return I + expected_remaining_variance


def calculate_call_payoff(S, K):
    return torch.relu(S[:, -1] - K)


@torch.no_grad()
def generate_market(config):
    """
    Heston 모형으로 시장 경로와 콜옵션 만기 지급액을 생성합니다.

    경로별 주가, 순간분산, 누적분산을 시뮬레이션합니다. 
    각 관측 시점의 누적분산과 예상 잔여분산을 더해 분산스왑의 모형 가격을 계산합니다.

    Args:
        config (dict): 시장 생성 설정
            - market: 초기 주가, 콜옵션 행사가, 만기 일수, 연간 일수, 관측 스텝 수, 관측 스텝당 내부 계산 횟수
            - heston: Heston 모수의 생성 모드와 고정값입니다. 'random' 모드에서는 모수별 추출 범위도 지정
            - dataset: 생성할 경로 수와 난수 시드입니다. 시드가 'None'이면 새 시드 생성

    Returns:
        tuple[dict, dict]: (market, metadata) 튜플
            - market: 
                - stock, variance, integrated_variance, variance_swap은 각각 (n_paths, n_steps + 1) 형태의 텐서
                - payoff는 경로당 하나이며, 잔존 경로당 하나이며, time_to_maturity는 모든 경로에 공통인 연 단위 벡터
                - params에는 경로별 Heston 모수
            - metadata: 입력 설정의 복사본, 실제 사용한 시드, 경로 수, 실행 장치, 자료형, PyTorch 버전, 시뮬레이션 방법, 생성 시각, 스키마 버전
"""

    device = DEVICE

    market_config = config["market"]
    heston_config = config["heston"]
    dataset_config = config["dataset"]

    n_paths = dataset_config["n_paths"]

    seed = dataset_config["seed"]

    if seed is None:
        seed = secrets.randbits(63)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # 1. Generate Heston parameters
    params = sample_heston_params(
        heston_config=heston_config,
        n_paths=n_paths,
        generator=generator,
        device=device,
    )

    # 2. Simulate market paths
    S, V, I = simulate_heston(
        market_config=market_config,
        params=params,
        generator=generator,
        device=device,
    )

    # 3. Time to maturity
    T = (
        market_config["maturity_days"]
        / market_config["days_per_year"]
    )

    time = torch.linspace(
        0,
        T,
        market_config["n_steps"] + 1,
        device=device,
    )

    tau = T - time

    # 4. Variance swap
    A = calculate_variance_swap(
        V=V,
        I=I,
        params=params,
        tau=tau,
    )

    # 5. Option payoff
    payoff = calculate_call_payoff(
        S=S,
        K=market_config["K"],
    )

    # 6. Market dataset
    market = {
        "stock": S,
        "variance": V,
        "integrated_variance": I,
        "variance_swap": A,
        "payoff": payoff,
        "params": params,
        "time_to_maturity": tau,
    }

    # 7. Generation metadata
    metadata = {
        "config": deepcopy(config),
        "seed": seed,
        "n_paths": n_paths,
        "device": str(device),
        "dtype": str(S.dtype),
        "torch_version": str(torch.__version__),
        "simulation_method": "full_truncation_euler",
        "created_at": datetime.now().astimezone().isoformat(),
        "schema_version": 1,
    }

    return market, metadata


def split_market(market, train_ratio, seed):

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    n_paths = market["stock"].shape[0]
    device = market["stock"].device

    n_train = int(n_paths * train_ratio)

    if n_train == 0 or n_train == n_paths:
        raise ValueError("Both splits must contain paths")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    indices = torch.randperm(
        n_paths,
        generator=generator,
        device=device,
    )

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    def select(indices):

        return {
            "stock": market["stock"][indices],

            "variance": market["variance"][indices],

            "integrated_variance":
                market["integrated_variance"][indices],

            "variance_swap":
                market["variance_swap"][indices],

            "payoff": market["payoff"][indices],

            "params": {
                name: values[indices]
                for name, values in market["params"].items()
            },

            "time_to_maturity": market["time_to_maturity"],

            "path_indices": indices,
        }

    return select(train_indices), select(test_indices)


def to_cpu(data):

    if isinstance(data, torch.Tensor):
        return data.detach().cpu()

    if isinstance(data, dict):
        return {
            key: to_cpu(value)
            for key, value in data.items()
        }

    return data


def save_dataset(train, test, metadata, data_dir):

    data_dir = Path(data_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    metadata = deepcopy(metadata)

    metadata["run_id"] = run_id

    metadata["split_sizes"] = {
        "train": len(train["stock"]),
        "test": len(test["stock"]),
    }

    # Save train and test
    for split, dataset in [
        ("train", train),
        ("test", test),
    ]:

        save_dir = data_dir / split
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"market_{run_id}.pt"

        save_data = {
            **dataset,
            "metadata": {
                **metadata,
                "split": split,
                "split_n_paths": len(dataset["stock"]),
            },
        }

        torch.save(
            to_cpu(save_data),
            save_path,
        )

        print(
            f"{split.upper()}: "
            f"{len(dataset['stock']):,} paths "
            f"-> {save_path}"
        )

    # Save readable generation metadata
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_dir / f"{run_id}.yaml"

    save_yaml(metadata, metadata_path)

    return run_id


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    config_path = os.path.join(BASE_DIR, "make_market/configs/heston_config.yaml")
    config = load_yaml(config_path)

    # Generate entire market
    market, metadata = generate_market(config)

    # Split seed
    split_seed = secrets.randbits(63)

    metadata["split_seed"] = split_seed

    # Train / Test split
    train, test = split_market(
        market=market,
        train_ratio=config["dataset"]["train_ratio"],
        seed=split_seed,
    )

    # Save dataset
    run_id = save_dataset(
        train=train,
        test=test,
        metadata=metadata,
        data_dir=os.path.join(BASE_DIR, config["dataset"]["data_dir"], config["mode"]),
    )

    print(f"\nCompleted: {run_id}")