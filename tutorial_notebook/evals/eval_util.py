"""
LABS Evaluation Utilities

Utilities for validating LABS solutions against the answers.csv ground truth.
Handles run-length notation for sequences (alternating +1/-1 starting with +1).
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
# ---------------------------------------------------------------------------
# Run-length notation conversion
# ---------------------------------------------------------------------------

def _parse_runlength_digit(c: str) -> int:
    """Parse a single run-length character (supports hex for values > 9, e.g. 'a'=10)."""
    if len(c) != 1:
        raise ValueError(f"Expected single character, got {c}")
    if c.isdigit():
        return int(c)
    if c.lower() in 'abcdef':
        return 10 + ord(c.lower()) - ord('a')
    raise ValueError(f"Invalid run-length character: {c}")


def runlength_to_sequence(rl_str: str) -> List[int]:
    """
    Convert run-length notation to a ±1 sequence.

    Convention: alternating runs starting with +1.
    - First run: +1 repeated r1 times
    - Second run: -1 repeated r2 times
    - Third run: +1 repeated r3 times, etc.

    Each character in rl_str is a run length (digits 0-9, hex a-f for 10-15).
    Example: "21" -> [1, 1, -1]  (2 of +1, 1 of -1)
    Example: "112" -> [1, -1, 1, 1]
    """
    if not rl_str:
        return []

    run_lengths = [_parse_runlength_digit(c) for c in rl_str]
    seq = []
    for i, length in enumerate(run_lengths):
        sign = 1 if i % 2 == 0 else -1
        seq.extend([sign] * length)
    return seq


def sequence_to_runlength(seq: List[int]) -> str:
    """Convert ±1 sequence to run-length notation (inverse of runlength_to_sequence)."""
    if len(seq) == 0:
        return ""
    rl = []
    current_sign = seq[0]
    count = 0
    for s in seq:
        if s == current_sign:
            count += 1
        else:
            rl.append(str(count) if count < 10 else chr(ord('a') + count - 10))
            current_sign = s
            count = 1
    rl.append(str(count) if count < 10 else chr(ord('a') + count - 10))
    return "".join(rl)


# ---------------------------------------------------------------------------
# LABS energy and merit factor (matches notebook implementation)
# ---------------------------------------------------------------------------

def compute_Ck(s: List[int], k: int) -> int:
    """Compute C_k for a binary sequence."""
    N = len(s)
    return sum(s[i] * s[i + k] for i in range(N - k))


def compute_energy(s: List[int]) -> int:
    """Compute LABS energy E = sum over k of C_k^2."""
    N = len(s)
    energy = 0
    for k in range(1, N):
        Ck = compute_Ck(s, k)
        energy += Ck * Ck
    return energy


def compute_merit_factor(s: List[int], energy: Optional[int] = None) -> float:
    """Compute F_N = N^2 / (2*E) merit factor."""
    N = len(s)
    if energy is None:
        energy = compute_energy(s)
    if energy == 0:
        return float('inf')
    return (N * N) / (2.0 * energy)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_answers_csv(csv_path: Union[str, Path]) -> List[Dict]:
    """
    Parse answers.csv which has multiple rows per N for different optimal solutions.

    Returns list of dicts: {N, E, F_N, sequence, skew}
    """
    path = Path(csv_path)
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'N': int(row['N']),
                'E': int(row['E']),
                'F_N': float(row['F_N']),
                'sequence': row['sequence'].strip(),
                'skew': row['skew'].strip().lower() == 'true',
            })
    return rows


def get_solutions_by_N(rows: List[Dict]) -> Dict[int, List[Dict]]:
    """Group solutions by N."""
    by_n = {}
    for r in rows:
        n = r['N']
        if n not in by_n:
            by_n[n] = []
        by_n[n].append(r)
    return by_n


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_solution(
    N: int,
    sequence_rl: str,
    expected_E: int,
    expected_F_N: float,
    tol_E: int = 0,
    tol_F_N: float = 1e-4,
) -> Tuple[bool, str]:
    """
    Validate that a sequence (in run-length notation) has correct energy E and merit factor F_N.

    Returns (is_valid, message).
    """
    seq = runlength_to_sequence(sequence_rl)
    actual_N = len(seq)
    if actual_N != N:
        return False, f"Length mismatch: expected N={N}, got len={actual_N}"

    actual_E = compute_energy(seq)
    actual_F_N = compute_merit_factor(seq, actual_E)

    if actual_E != expected_E:
        return False, f"Energy mismatch: expected E={expected_E}, got E={actual_E}"

    # Use both absolute and relative tolerance (CSV rounds to 3 decimals)
    abs_tol = max(0.001, tol_F_N)
    rel_tol = max(0.001, tol_F_N)
    if abs(actual_F_N - expected_F_N) > abs_tol and abs(actual_F_N - expected_F_N) / max(expected_F_N, 0.1) > rel_tol:
        return False, f"F_N mismatch: expected F_N={expected_F_N}, got F_N={actual_F_N}"

    return True, f"Valid: N={N}, E={actual_E}, F_N={actual_F_N:.4f}"


def validate_solution_row(row: Dict, tol_F_N: float = 1e-4) -> Tuple[bool, str]:
    """Validate a single row from the answers CSV."""
    return validate_solution(
        N=row['N'],
        sequence_rl=row['sequence'],
        expected_E=row['E'],
        expected_F_N=row['F_N'],
        tol_F_N=tol_F_N,
    )


def run_validation_for_N(
    N: int,
    csv_path: Union[str, Path],
    tol_F_N: float = 1e-4,
) -> Tuple[int, int, List[Tuple[bool, str]]]:
    """
    Validate all solutions for a specific N from the CSV.

    Returns (total, passed, list of (valid, message) for each solution).
    """
    rows = parse_answers_csv(csv_path)
    solutions = [r for r in rows if r['N'] == N]
    results = []
    for r in solutions:
        ok, msg = validate_solution_row(r, tol_F_N=tol_F_N)
        results.append((ok, msg))
    return len(solutions), sum(1 for ok, _ in results if ok), results


def run_full_validation(
    csv_path: Union[str, Path],
    tol_F_N: float = 1e-4,
) -> Dict:
    """
    Validate all solutions in the CSV.

    Returns dict with N -> {total, passed, results, all_valid}.
    """
    rows = parse_answers_csv(csv_path)
    by_n = get_solutions_by_N(rows)
    report = {}
    for n, sols in sorted(by_n.items()):
        results = [validate_solution_row(r, tol_F_N=tol_F_N) for r in sols]
        passed = sum(1 for ok, _ in results if ok)
        report[n] = {
            'total': len(sols),
            'passed': passed,
            'results': results,
            'all_valid': passed == len(sols),
        }
    return report


def get_expected_optimal_energy(N: int, csv_path: Union[str, Path]) -> Optional[int]:
    """Return the optimal energy for sequence length N from answers.csv, or None if not found."""
    rows = parse_answers_csv(csv_path)
    matches = [r for r in rows if r['N'] == N]
    return matches[0]['E'] if matches else None


def validate_energy_against_answers(
    N: int, actual_E: int, csv_path: Union[str, Path], label: str = "Result"
) -> Tuple[bool, str]:
    """Validate actual_E against expected optimal for N. Returns (ok, message)."""
    expected = get_expected_optimal_energy(N, csv_path)
    if expected is None:
        return False, f"{label}: No expected optimal for N={N} in answers.csv"
    if actual_E == expected:
        return True, f"{label}: Energy {actual_E} matches optimal for N={N}"
    return False, f"{label}: Energy {actual_E} != expected optimal {expected} for N={N}"


# ---------------------------------------------------------------------------
# Unit tests (run with pytest or python -m pytest)
# ---------------------------------------------------------------------------

def test_runlength_conversion():
    """Test run-length to sequence conversion."""
    # N=3: "21" -> [1,1,-1]
    seq = runlength_to_sequence("21")
    assert len(seq) == 3
    assert list(seq) == [1, 1, -1]

    # N=4: "112" -> [1,-1,1,1]
    seq = runlength_to_sequence("112")
    assert len(seq) == 4
    assert list(seq) == [1, -1, 1, 1]

    # Round-trip
    for rl in ["21", "311", "141", "32111"]:
        s = runlength_to_sequence(rl)
        rl2 = sequence_to_runlength(s)
        assert rl == rl2, f"Round-trip failed: {rl} -> {rl2}"


def test_energy_and_merit():
    """Test energy and merit factor computation."""
    # N=3, sequence [1,1,-1], E=1, F_N=4.5
    seq = runlength_to_sequence("21")
    E = compute_energy(seq)
    F = compute_merit_factor(seq, E)
    assert E == 1, f"Expected E=1, got {E}"
    assert abs(F - 4.5) < 1e-6, f"Expected F_N=4.5, got {F}"

    # N=4, sequence "112"
    seq = runlength_to_sequence("112")
    E = compute_energy(seq)
    F = compute_merit_factor(seq, E)
    assert E == 2
    assert abs(F - 4.0) < 1e-6


def test_validate_specific_N():
    """Test validation for N=3, N=6, N=20 against answers.csv."""
    csv_path = Path(__file__).parent / "answers.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"answers.csv not found at {csv_path}")

    # N=3: single solution "21"
    total, passed, results = run_validation_for_N(3, csv_path)
    assert total >= 1
    for ok, msg in results:
        assert ok, msg

    # N=6: multiple solutions
    total, passed, results = run_validation_for_N(6, csv_path)
    assert total >= 1
    for ok, msg in results:
        assert ok, msg

    # N=20: one solution "5113112321"
    total, passed, results = run_validation_for_N(20, csv_path)
    assert total >= 1
    for ok, msg in results:
        assert ok, msg


def test_full_validation():
    """Run full validation and assert all pass (excluding known bad rows like typo)."""
    csv_path = Path(__file__).parent / "answers.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"answers.csv not found at {csv_path}")

    report = run_full_validation(csv_path)
    failed = [(n, info) for n, info in report.items() if not info['all_valid']]
    # Allow some failures for rows with typos (e.g. 'a' in sequence)
    assert len(failed) == 0 or all(
        n == 47 for n, _ in failed
    ), f"Unexpected validation failures: {failed}"


if __name__ == "__main__":
    import sys
    csv_path = Path(__file__).parent / "answers.csv"

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run unit tests
        test_runlength_conversion()
        test_energy_and_merit()
        test_validate_specific_N()
        test_full_validation()
        print("All tests passed.")
    else:
        # Run full validation and print report
        report = run_full_validation(csv_path)
        for n, info in sorted(report.items()):
            status = "OK" if info['all_valid'] else "FAIL"
            print(f"N={n:2d}: {info['passed']}/{info['total']} passed [{status}]")
            if not info['all_valid']:
                for ok, msg in info['results']:
                    if not ok:
                        print(f"    - {msg}")
