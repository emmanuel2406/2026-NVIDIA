"""
DEPRECATED: This module has been merged into main.py.
Import from main instead: e.g. "from main import dcqo_flexible_circuit_v2, get_image_hamiltonian, ..."
This stub re-exports from main for backward compatibility and will be removed in a future release.
"""

import warnings

warnings.warn(
    "generate_trotterization is deprecated; import from main instead "
    "(e.g. from main import dcqo_flexible_circuit_v2, get_image_hamiltonian, ...).",
    DeprecationWarning,
    stacklevel=2,
)

from main import (
    FIXED_FIRST_TWO_PREFIX,
    calculate_energy,
    dcqo_flexible_circuit,
    dcqo_flexible_circuit_v2,
    expand_skew_symmetric_bitstring,
    expand_skew_symmetric_counts,
    get_image_hamiltonian,
    get_image_hamiltonian_skew_reduced,
    get_interactions,
    get_labs_hamiltonian,
    prepend_fixed_prefix_to_counts,
    reduce_hamiltonian_fix_first_two,
    reduce_hamiltonian_skew_symmetry,
    r_yz,
    r_yzzz,
    r_z,
    r_zz,
    r_zzz,
    r_zzzz,
)

__all__ = [
    "FIXED_FIRST_TWO_PREFIX",
    "calculate_energy",
    "dcqo_flexible_circuit",
    "dcqo_flexible_circuit_v2",
    "expand_skew_symmetric_bitstring",
    "expand_skew_symmetric_counts",
    "get_image_hamiltonian",
    "get_image_hamiltonian_skew_reduced",
    "get_interactions",
    "get_labs_hamiltonian",
    "prepend_fixed_prefix_to_counts",
    "reduce_hamiltonian_fix_first_two",
    "reduce_hamiltonian_skew_symmetry",
    "r_yz",
    "r_yzzz",
    "r_z",
    "r_zz",
    "r_zzz",
    "r_zzzz",
]
