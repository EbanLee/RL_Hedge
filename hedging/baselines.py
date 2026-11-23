import math

import torch


def no_hedge(stock: torch.Tensor) -> torch.Tensor:
    """모든 시점에서 Hedge(stock) 자산 보유량 0"""
    return torch.zeros_like(stock[:, :-1])


def bs_delta_hedge(
    stock: torch.Tensor,
    variance: torch.Tensor,
    time_to_maturity: torch.Tensor,
    *,
    strike: float,
) -> torch.Tensor:
    """
    블랙-숄즈(Black–Scholes) 콜옵션 델타를 계산하여 헤지 보유량 결정

    Args:
        stock: 기초자산 가격 경로
        variance: 각 시점의 순간 분산(변동성의 제곱)
        time_to_maturity: 각 시점별 잔존 만기(연 단위)
        strike: 옵션 행사가(K)

    Returns:
        각 리밸런싱 시점의 블랙-숄즈 델타(보유 비중) 텐서
        만기 시점을 제외한 (n_paths, n_steps - 1) 형태이며 값은 (0, 1) 범위
    
    """
    if stock.ndim != 2 or stock.shape[1] < 2 or variance.shape != stock.shape:
        raise ValueError("stock and variance must have matching path and time axes")
    if time_to_maturity.shape != (stock.shape[1],):
        raise ValueError("time_to_maturity must have one value per price date")
    if not math.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be finite and positive")

    remaining_time = time_to_maturity[:-1]
    if not (torch.isfinite(remaining_time) & (remaining_time > 0)).all():
        raise ValueError("time to maturity must be positive on all trading dates")

    spot = stock[:, :-1]
    current_variance = variance[:, :-1].clamp_min(1e-8)
    total_variance = current_variance * remaining_time
    d1 = (spot.log() - math.log(strike) + 0.5 * total_variance) / total_variance.sqrt()
    return 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))

def no_trade_band_hedge(
    target_positions: torch.Tensor,
    *,
    band_width: float,
) -> torch.Tensor:
    """Keep the prior holding inside the band, or trade to its nearest edge."""
    if target_positions.ndim != 2 or target_positions.shape[1] == 0:
        raise ValueError("target_positions must have shape (n_paths, n_trade_dates)")
    if not target_positions.is_floating_point():
        raise TypeError("target_positions must be floating point")
    if not math.isfinite(band_width) or band_width < 0:
        raise ValueError("band_width must be finite and nonnegative")

    positions = torch.empty_like(target_positions)
    previous = torch.zeros_like(target_positions[:, 0])

    for step, target in enumerate(target_positions.unbind(dim=1)):
        previous = torch.clamp(
            previous,
            min=target - band_width,
            max=target + band_width,
        )
        positions[:, step] = previous

    return positions
