"""
LABS Physics Tests

Unit tests verifying the physical symmetries of the LABS (Low Autocorrelation
Binary Sequence) problem. These symmetries are fundamental properties that any
valid LABS energy implementation must satisfy.
"""

from pathlib import Path

from eval_util import (
    compute_energy,
    runlength_to_sequence,
    parse_answers_csv,
)


# ---------------------------------------------------------------------------
# Symmetry helpers
# ---------------------------------------------------------------------------


def complementary_sequence(seq: list[int]) -> list[int]:
    """Flip each bit: +1 -> -1 and -1 -> +1. The complementary solution."""
    return [-x for x in seq]


def reversed_sequence(seq: list[int]) -> list[int]:
    """Reverse the sequence: read the same backwards as forwards."""
    return seq[::-1]


# ---------------------------------------------------------------------------
# Symmetry tests
# ---------------------------------------------------------------------------


def test_complementary_symmetry_same_energy():
    """
    Complementary symmetry: flipping every bit (+1 <-> -1) yields the same energy.

    For C_k(s) = sum_i s[i]*s[i+k], the complementary s' = -s gives
    C_k(s') = sum_i (-s[i])(-s[i+k]) = C_k(s), hence E(s') = E(s).
    """
    test_cases = [
        "21",      # N=3: [1, 1, -1]
        "112",     # N=4: [1, -1, 1, 1]
        "1111",    # N=4: [1, -1, -1, -1]
        "311",     # N=5
        "141",     # N=6
        "5113112321",  # N=20 optimal
    ]
    for rl in test_cases:
        seq = runlength_to_sequence(rl)
        comp = complementary_sequence(seq)
        E_orig = compute_energy(seq)
        E_comp = compute_energy(comp)
        assert E_orig == E_comp, (
            f"Complementary symmetry violated for {rl}: "
            f"E(original)={E_orig} != E(complement)={E_comp}"
        )


def test_reversal_symmetry_same_energy():
    """
    Reversal symmetry: reading the sequence backwards yields the same energy.

    Reversing the sequence s -> s_rev[i] = s[N-1-i] preserves all C_k values
    (the autocorrelation sums over the same pairs, reordered), hence E(s_rev) = E(s).
    """
    test_cases = [
        "21",      # N=3: [1, 1, -1]
        "112",     # N=4: [1, -1, 1, 1]
        "1111",    # N=4
        "311",     # N=5
        "141",     # N=6
        "5113112321",  # N=20 optimal
    ]
    for rl in test_cases:
        seq = runlength_to_sequence(rl)
        rev = reversed_sequence(seq)
        E_orig = compute_energy(seq)
        E_rev = compute_energy(rev)
        assert E_orig == E_rev, (
            f"Reversal symmetry violated for {rl}: "
            f"E(original)={E_orig} != E(reversed)={E_rev}"
        )


def test_complementary_and_reversal_combined():
    """
    Combined symmetries: complement and reversal both preserve energy.
    E(s) = E(complement(s)) = E(reverse(s)) = E(reverse(complement(s))).
    """
    seq = runlength_to_sequence("5113112321")  # N=20
    E0 = compute_energy(seq)

    comp = complementary_sequence(seq)
    rev = reversed_sequence(seq)
    rev_comp = reversed_sequence(comp)  # equivalent to complement(reverse(seq))

    assert compute_energy(comp) == E0
    assert compute_energy(rev) == E0
    assert compute_energy(rev_comp) == E0


def test_symmetry_on_answers_csv():
    """
    Verify complementary and reversal symmetry hold for all solutions in answers.csv.
    """
    csv_path = Path(__file__).parent / "answers.csv"
    if not csv_path.exists():
        return  # Skip when answers.csv not found (e.g. in minimal env)

    rows = parse_answers_csv(csv_path)
    for row in rows:
        seq = runlength_to_sequence(row["sequence"])
        expected_E = row["E"]

        comp = complementary_sequence(seq)
        rev = reversed_sequence(seq)

        E_comp = compute_energy(comp)
        E_rev = compute_energy(rev)

        assert E_comp == expected_E, (
            f"N={row['N']} complementary: E={E_comp} != expected {expected_E}"
        )
        assert E_rev == expected_E, (
            f"N={row['N']} reversed: E={E_rev} != expected {expected_E}"
        )


if __name__ == "__main__":
    tests = [
        ("Complementary symmetry (flip each bit)", test_complementary_symmetry_same_energy),
        ("Reversal symmetry (read backwards)", test_reversal_symmetry_same_energy),
        ("Complementary and reversal combined", test_complementary_and_reversal_combined),
        ("Symmetry on answers.csv", test_symmetry_on_answers_csv),
    ]
    for name, test_fn in tests:
        test_fn()
        print(f"  ✅ {name} — passed")
    print("\nAll physics symmetry tests passed.")
