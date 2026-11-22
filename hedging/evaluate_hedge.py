import math
from argparse import ArgumentParser
from pathlib import Path

import torch

from hedging.baselines import bs_delta_hedge, no_hedge
from functions.pnl import compute_pnl
from functions.data_loader import load_market


def summarize_pnl(
    net_pnl: torch.Tensor, transaction_cost: torch.Tensor
) -> tuple[float, float, float, float]:
    """Report mean P&L, its standard deviation, loss ES95, and mean cost."""
    # Weight the last observation by its share of the worst 5% of outcomes.
    tail_mass = net_pnl.numel() / 20
    tail_count = math.ceil(tail_mass)
    tail_losses = (-net_pnl).topk(tail_count).values
    last_weight = tail_mass - (tail_count - 1)
    es95 = (tail_losses[:-1].sum() + last_weight * tail_losses[-1]) / tail_mass
    return (
        net_pnl.mean().item(),
        net_pnl.std(unbiased=False).item(),
        es95.item(),
        transaction_cost.mean().item(),
    )


@torch.no_grad()
def main() -> None:
    parser = ArgumentParser(description="Compare stock hedge baselines on one Heston run.")
    parser.add_argument("--run_id", required=True, help="ID of the saved market run")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "heston" / "fixed",
        help="Directory containing the train and test subdirectories",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--cost_rates",
        type=float,
        nargs="+",
        default=[0.0, 0.0005, 0.001],
        help="Proportional stock transaction cost rates. 0.001 is 10 basis points.",
    )
    parser.add_argument(
        "--premium",
        type=float,
        help="Option premium. Defaults to the mean payoff of the training paths.",
    )
    args = parser.parse_args()

    filename = f"market_{args.run_id}.pt"
    premium = args.premium
    if premium is None and args.split == "test":
        train_path = args.data_dir / "train" / filename
        premium = load_market(train_path, split="train")["payoff"].mean().item()

    data_path = args.data_dir / args.split / filename
    market = load_market(data_path, split=args.split)
    stock = market["stock"]
    payoff = market["payoff"]
    if premium is None:
        premium = payoff.mean().item()

    strike = market["metadata"]["config"]["market"]["K"]
    positions_by_strategy = {
        "No Hedge": no_hedge(stock),
        "BS Delta Hedge": bs_delta_hedge(
            stock, market["variance"], market["time_to_maturity"], strike=strike
        ),
    }

    print(f"Run {args.run_id} | {args.split} | {len(stock):,} paths | K={strike:g}")
    print(f"Premium per option {premium:.6f}")
    print("Cost Rate uses stock trade value. Percentages in parentheses use the option premium.")
    if premium <= 0:
        print("Premium-based percentages require a positive premium.")
    metric_width = 21
    metric_names = ("Mean P&L", "P&L std", "Loss ES95", "Mean cost")
    metric_headers = "  ".join(f"{label:^{metric_width}}" for label in metric_names)
    print(f"{'Strategy':<16} {'Cost Rate':>10}  {metric_headers}".rstrip())
    for cost_rate in args.cost_rates:
        for name, positions in positions_by_strategy.items():
            pnl = compute_pnl(
                stock, positions, payoff, premium=premium, cost_rate=cost_rate
            )
            mean, std, es95, mean_transaction_cost = summarize_pnl(
                pnl["net_pnl"], pnl["transaction_cost"]
            )
            metrics = []
            for amount in (mean, std, es95, mean_transaction_cost):
                percent = f"{amount / premium:.2%}" if premium > 0 else "N/A"
                value = f"{amount:>10.6f} ({percent})"
                metrics.append(f"{value:<{metric_width}}")
            print(f"{name:<16} {cost_rate:>10.2%}  {'  '.join(metrics)}".rstrip())
        print()


if __name__ == "__main__":
    main()
