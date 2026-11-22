"""Reset-seed ranges (docs/seeds.md). The ranges never overlap; a test checks this.

A reset seed fixes the random scene of an episode (object placement). Demonstrations are
recorded states, not reset seeds, so they cannot collide with these ranges.
"""

FIXTURE = range(0, 1_000)  # tests, trials, debugging
M0_VALIDATION = range(100_000, 100_200)  # M0 nominal evaluation (docs/m0-evaluation-protocol.md)
FINAL_TEST = range(200_000, 200_200)  # reserved: never used before M3

RANGES = {"fixture": FIXTURE, "m0_validation": M0_VALIDATION, "final_test": FINAL_TEST}
