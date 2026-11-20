"""Open-loop replay of the robomimic Can demonstrations (docs/design.md §10, milestone 0.2b).

For every demo: load its recorded initial state, apply its recorded actions without
feedback, then apply the benchmark's own success check. Also reports how far the
end-effector drifts from the recorded positions. Works with robosuite 1.4.1 (this
project's environment) and 1.5.1 (the version the data was recorded with); see README.md.

    python scripts/demo_replay/replay.py <low_dim_v15.hdf5> --robosuite 1.4.1
    python scripts/demo_replay/replay.py <low_dim_v15.hdf5> --robosuite 1.5.1
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import robosuite as suite


def make_env(version: str, env_args: dict):
    common = dict(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    if version == "1.5.1":
        # The dataset's own settings (composite BASIC controller wrapping OSC_POSE).
        kwargs = {**env_args["env_kwargs"], **common, "ignore_done": True}
        return suite.make(env_args["env_name"], **kwargs)
    # This project's settings (afterslip.tasks.robosuite_can.CanTaskConfig) with ignore_done.
    controller = suite.load_controller_config(default_controller="OSC_POSE")
    return suite.make(
        "PickPlaceCan",
        robots="Panda",
        controller_configs=controller,
        control_freq=20,
        ignore_done=True,
        **common,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("--robosuite", choices=["1.4.1", "1.5.1"], required=True)
    args = parser.parse_args()
    assert suite.__version__ == args.robosuite, f"installed robosuite is {suite.__version__}"

    with h5py.File(args.dataset, "r") as f:
        env = make_env(args.robosuite, json.loads(f["data"].attrs["env_args"]))
        names = sorted(f["data"].keys(), key=lambda n: int(n[5:]))
        failed, drift = [], []
        for name in names:
            demo = f[f"data/{name}"]
            env.reset()
            env.sim.set_state_from_flattened(demo["states"][0])
            env.sim.forward()
            recorded = demo["obs/robot0_eef_pos"][()]
            worst = 0.0
            for t, action in enumerate(demo["actions"][()]):
                obs, *_ = env.step(action)
                if t + 1 < len(recorded):
                    worst = max(worst, float(np.abs(obs["robot0_eef_pos"] - recorded[t + 1]).max()))
            drift.append(worst)
            if not env._check_success():
                failed.append(name)

    print(f"robosuite {suite.__version__}: success {len(names) - len(failed)}/{len(names)}")
    print(f"failed: {failed}")
    median, p95, worst = np.median(drift), np.percentile(drift, 95), np.max(drift)
    print(
        f"end-effector drift per demo [m]: median {median:.4f}, 95th pct {p95:.4f}, max {worst:.4f}"
    )


if __name__ == "__main__":
    main()
