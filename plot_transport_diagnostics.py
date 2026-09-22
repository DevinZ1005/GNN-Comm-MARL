"""Plot saved transport diagnostics without rerunning the policy (needs matplotlib)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.diagnostic).read_text())
    summaries = data["summaries"]
    horizon = data["original_env_config"]["max_steps"]
    success_radius = data["original_env_config"].get("success_radius", 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for summary in summaries:
        trace = data["trajectories"][str(summary["episode_seed"])]
        color = "#237b69" if summary["extended"]["success"] else "#ad414b"
        axes[0].plot([r["step"] for r in trace],
                     [r["payload_distance"] for r in trace], color=color, alpha=0.35, lw=1)
    axes[0].axvline(horizon, color="black", ls="--", lw=1, label="Training horizon")
    axes[0].axhline(success_radius, color="#237b69", ls=":", label="Success threshold")
    axes[0].set(xlabel="Environment steps", ylabel="Payload distance to goal",
                title="Same policy given extra time")
    axes[0].legend(frameon=False)
    colors = ["#237b69" if s["extended"]["success"] else "#ad414b" for s in summaries]
    axes[1].scatter([s["original"]["final_distance"] for s in summaries],
                    [s["extended"]["final_distance"] for s in summaries], c=colors, s=28)
    limit = max(max(s[h]["final_distance"] for s in summaries) for h in ("original", "extended")) * 1.05
    axes[1].plot([0, limit], [0, limit], color="gray", ls="--", lw=1)
    axes[1].set(xlabel=f"Distance at {horizon} steps (or success)",
                ylabel="Distance with extra time (or success)",
                title="Above diagonal: extra time ended farther away")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    counts = data["aggregate"]
    checkpoint = Path(data["checkpoint"]).stem
    fig.suptitle(f"{checkpoint} | Development episodes: {counts['episodes']} | "
                 f"Successes: {counts['original_successes']} to {counts['extended_successes']}", fontsize=12)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
