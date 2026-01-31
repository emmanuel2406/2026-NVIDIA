"""
Unit tests for LABS eval_util.

Run with: python -m pytest test_eval_util.py -v
Or: python -c "import eval_util; eval_util.test_runlength_conversion(); ..."
"""

import pytest
from pathlib import Path

from eval_util import (
    runlength_to_sequence,
    sequence_to_runlength,
    compute_energy,
    compute_merit_factor,
    validate_solution,
    parse_answers_csv,
    run_validation_for_N,
    run_full_validation,
)


@pytest.fixture
def csv_path():
    return Path(__file__).parent / "answers.csv"


class TestRunlengthConversion:
    def test_21_n3(self):
        seq = runlength_to_sequence("21")
        assert len(seq) == 3
        assert list(seq) == [1, 1, -1]

    def test_112_n4(self):
        seq = runlength_to_sequence("112")
        assert len(seq) == 4
        assert list(seq) == [1, -1, 1, 1]

    def test_311_n5(self):
        seq = runlength_to_sequence("311")
        assert len(seq) == 5
        assert list(seq) == [1, 1, 1, -1, 1]

    def test_roundtrip(self):
        for rl in ["21", "311", "141", "32111", "4221111"]:
            s = runlength_to_sequence(rl)
            rl2 = sequence_to_runlength(s)
            assert rl == rl2, f"Round-trip failed: {rl} -> {rl2}"


class TestEnergyAndMerit:
    def test_n3_sequence_21(self):
        seq = runlength_to_sequence("21")
        E = compute_energy(seq)
        F = compute_merit_factor(seq, E)
        assert E == 1
        assert abs(F - 4.5) < 1e-4

    def test_n4_sequence_112(self):
        seq = runlength_to_sequence("112")
        E = compute_energy(seq)
        F = compute_merit_factor(seq, E)
        assert E == 2
        assert abs(F - 4.0) < 1e-4

    def test_n20_sequence_5113112321(self):
        seq = runlength_to_sequence("5113112321")
        E = compute_energy(seq)
        F = compute_merit_factor(seq, E)
        assert E == 26
        assert abs(F - 7.692) < 1e-3


class TestValidation:
    def test_validate_n3(self, csv_path):
        total, passed, results = run_validation_for_N(3, csv_path)
        assert total >= 1
        for ok, msg in results:
            assert ok, msg

    def test_validate_n6_multiple_solutions(self, csv_path):
        total, passed, results = run_validation_for_N(6, csv_path)
        assert total >= 4
        for ok, msg in results:
            assert ok, msg

    def test_validate_n20(self, csv_path):
        total, passed, results = run_validation_for_N(20, csv_path)
        assert total >= 1
        for ok, msg in results:
            assert ok, msg

    def test_full_validation_all_pass(self, csv_path):
        report = run_full_validation(csv_path)
        for n, info in report.items():
            assert info['all_valid'], f"N={n}: {info['passed']}/{info['total']} passed - {info['results']}"
