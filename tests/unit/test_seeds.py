"""Seed ranges never overlap (docs/seeds.md)."""

from __future__ import annotations

import itertools

import pytest

from afterslip import seeds
from afterslip.evaluation import nominal

pytestmark = pytest.mark.unit


def test_ranges_are_disjoint():
    for (name_a, a), (name_b, b) in itertools.combinations(seeds.RANGES.items(), 2):
        assert set(a).isdisjoint(b), f"{name_a} overlaps {name_b}"


def test_evaluation_uses_validation_seeds_and_trials_use_fixture_seeds():
    assert nominal.SEEDS == tuple(range(100_000, 100_200))
    assert set(nominal.TRIAL_SEEDS) <= set(seeds.FIXTURE)
    assert set(nominal.TRIAL_SEEDS).isdisjoint(nominal.SEEDS)
