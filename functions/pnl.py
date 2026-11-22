import math

import torch


def compute_pnl(
    stock: torch.Tensor,
    positions: torch.Tensor,
    payoff: torch.Tensor,
    *,
    premium: float = 0.0,
    cost_rate: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Hedge 자산 수익과 거래비용을 고려하여 전체 PnL 계산.
    
    Args:
        stock: Hedge 자산의 가격
        positions: Hedge 자산의 보유량
        payoff: 만기에 지급할 금액
        premium: 옵션 판매한 금액 (수익)
        cost_rate: 거래 수수료 비율

    Returns:
        다음 dict 반환:
            - hedge_gain: hedge 자산의 총 수익
            - transaction_cost: 총 거래비용
            - net_pnl: 총 수익 (옵션 판매 비용 - 만기 지급 금액 + Hedge 수익 - 총 거래비용)
    """
    if stock.ndim != 2 or stock.shape[0] == 0 or stock.shape[1] < 2:
        raise ValueError("stock must have at least one path and two prices per path")
    trade_prices = stock[:, :-1]
    if positions.shape != trade_prices.shape:
        raise ValueError("positions must have one holding per path and price interval")
    if payoff.shape != (stock.shape[0],):
        raise ValueError("payoff must have one value per path")
    if not math.isfinite(premium):
        raise ValueError("premium must be finite")
    if not math.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("cost_rate must be finite and nonnegative")

    price_changes = stock[:, 1:] - trade_prices     # 각 시차별 주가의 변화
    hedge_gain = (positions * price_changes).sum(dim=1)     # 내가 가진 헤지 주식의 시점별 수익의 총합

    # 시점별 헤지 자산의 포지션 변화
    position_changes = torch.cat(
        (positions[:, :1], positions[:, 1:] - positions[:, :-1]), dim=1
    )
    traded_notional = (trade_prices * position_changes.abs()).sum(dim=1)    # 각 시점별 주식을 사고판 전체 비용 (시점별 주가 * 시점별 변동 주식량)
    closing_notional = stock[:, -1] * positions[:, -1].abs()    # 마지막 시점에 헤지 주식 전체 금액 (주가 * 주식수)
    transaction_cost = cost_rate * (traded_notional + closing_notional)     # cost_rate * 모든 시점에 대해 거래한 총 비용 (총 거래 수수료)

    return {
        "hedge_gain": hedge_gain,
        "transaction_cost": transaction_cost,
        "net_pnl": premium - payoff + hedge_gain - transaction_cost,
    }
