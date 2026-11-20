# Demonstration replay check

Evidence for using the robomimic v1.5 Can demonstrations (recorded with robosuite 1.5.1) in this project's robosuite 1.4.1 environment (docs/design.md §10, milestone 0.2b). `replay.py` restores each demo's recorded initial state, applies its recorded actions open loop, and reports the benchmark's success and the end-effector drift from the recording.

Not part of `make test`: a full run takes several minutes and needs the downloaded dataset (`make data-can`, file `outputs/data/raw/can_ph_low_dim_v15.hdf5`).

## Running it

robosuite 1.4.1, in the project environment:

```bash
python scripts/demo_replay/replay.py outputs/data/raw/can_ph_low_dim_v15.hdf5 --robosuite 1.4.1
```

robosuite 1.5.1, in a separate throwaway virtual environment (never the project environment), on the project's Python 3.8:

```bash
uv venv --python "$(conda info --base)/envs/afterslip/bin/python" /tmp/rs151
printf 'pyobjc-core<11\npyobjc-framework-Quartz<11\npyobjc-framework-ApplicationServices<11\npyobjc-framework-Cocoa<11\n' > /tmp/rs151-constraints.txt
uv pip install --python /tmp/rs151/bin/python -c /tmp/rs151-constraints.txt robosuite==1.5.1 mujoco==3.2.3 numpy==1.24.4 h5py
/tmp/rs151/bin/python -I scripts/demo_replay/replay.py outputs/data/raw/can_ph_low_dim_v15.hdf5 --robosuite 1.5.1
```

The pyobjc constraint is needed on macOS only. Resolved versions in the tested 1.5.1 environment: robosuite 1.5.1, mujoco 3.2.3, mink 0.0.13, qpsolvers 4.8.2, numpy 1.24.4, h5py 3.11.0.

## Results (2026-11-19, macOS arm64, 200 demos)

| robosuite | Settings | Open-loop success (two runs) | End-effector drift per demo: median / 95th pct / max |
|---|---|---|---|
| 1.4.1 (this project) | `CanTaskConfig`: Panda, OSC_POSE defaults, 20 Hz | 193/200, 193/200 | 4.5–4.6 / 13.5–14.5 / 34.5–41.9 mm |
| 1.5.1 (recording version) | the dataset's own `env_args` | 187/200, 189/200 | 4.6–4.7 / 9.4–11.4 / 30.4–37.1 mm |

The failing demos differ between runs of the same version: the arm controller keeps state across episodes (found in D2), so borderline demos depend on which demo ran before. Open-loop replay therefore drifts in both versions, and the success counts are a sample, not proof of equal fidelity. The decisive check is the pretrained policy's own success in robosuite 1.4.1 (FR-POL-03); if it fails, migrating to robosuite 1.5.1 is the recorded fallback.
