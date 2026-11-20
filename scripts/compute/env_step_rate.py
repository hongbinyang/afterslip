"""Measure single-environment control-step throughput of the Can task (milestone 0.7).

Random actions, no rendering, one environment in one process; includes episode resets.
Results: docs/setup.md §6.6.

    python scripts/compute/env_step_rate.py
"""

from __future__ import annotations

import time

import numpy as np

from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask

EPISODES = 4


def main() -> None:
    task = RobosuiteCanTask(CanTaskConfig())
    rng = np.random.default_rng(0)
    low, high = task.benchmark_env.action_spec
    task.reset(seed=0)
    for _ in range(50):  # warm-up
        task.step(rng.uniform(low, high) * 0.3)
    steps, start = 0, time.perf_counter()
    for seed in range(EPISODES):
        task.reset(seed=seed)
        for _ in range(task.horizon):
            task.step(rng.uniform(low, high) * 0.3)
            steps += 1
    seconds = time.perf_counter() - start
    print(f"{steps} control steps in {seconds:.2f} s = {steps / seconds:.0f} steps/s")


if __name__ == "__main__":
    main()
