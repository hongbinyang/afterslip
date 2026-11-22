# Seed registry

A reset seed fixes the random scene of an episode (where the can starts). Each purpose has its own range, and the ranges never overlap, so development work never touches the episodes used to judge results. The ranges are defined in `afterslip/seeds.py`; `tests/unit/test_seeds.py` checks they are disjoint.

| Range | Purpose | Rule |
|---|---|---|
| 0–999 | Fixtures: tests, trial runs, debugging | Free to use |
| 100000–100199 | M0 nominal validation ([protocol](m0-evaluation-protocol.md)) | Only the protocol evaluation; trial runs of the evaluation harness use fixture seeds instead |
| 200000–200199 | Final test (M3) | Reserved: never used before M3 |

Policy sampling seeds are separate: reset seed + 1000000, applied to a PyTorch generator only, never to NumPy's global random generator (which places the objects).

Demonstrations are recorded simulator states, not reset seeds, so they cannot collide with these ranges.
