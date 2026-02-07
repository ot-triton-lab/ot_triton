"""Triton kernels and wrappers.

Public API (eager imports):
- Common utilities: epsilon_schedule, max_diameter
- Solvers: sinkhorn_flashstyle_alternating, sinkhorn_flashstyle_symmetric
- Apply kernels: apply_plan_vec_flashstyle, apply_plan_mat_flashstyle
- Gradient kernel: sinkhorn_geomloss_online_grad_sqeuclid

Internal helpers (lazy-imported with deprecation warning):
  Import directly from their source modules instead, e.g.:
    from ot_triton.kernels.sinkhorn_flashstyle_sqeuclid import compute_bias_f
    from ot_triton.kernels._common import log_weights
"""

# Common utilities
from ot_triton.kernels._common import (
    epsilon_schedule,
    max_diameter,
)

# FlashSinkhorn solvers (shifted potential formulation)
from ot_triton.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_alternating,
    sinkhorn_flashstyle_symmetric,
)

# Apply kernels (FlashStyle, preferred)
from ot_triton.kernels.apply_flash import (
    apply_plan_vec_flashstyle,
    apply_plan_mat_flashstyle,
)

# Gradient kernel
from ot_triton.kernels.sinkhorn_triton_grad_sqeuclid import (
    sinkhorn_geomloss_online_grad_sqeuclid,
)

__all__ = [
    # Common
    "epsilon_schedule",
    "max_diameter",
    # Solvers
    "sinkhorn_flashstyle_alternating",
    "sinkhorn_flashstyle_symmetric",
    # Apply (FlashStyle)
    "apply_plan_vec_flashstyle",
    "apply_plan_mat_flashstyle",
    # Gradient
    "sinkhorn_geomloss_online_grad_sqeuclid",
]

# ---------------------------------------------------------------------------
# Deprecation bridge for internal names that were previously exported.
# These still work but emit a DeprecationWarning directing users to the
# canonical source module.
# ---------------------------------------------------------------------------
_COMPAT_MAP = {
    # _common helpers
    "log_weights": "ot_triton.kernels._common",
    # OTT-style primitives
    "apply_lse_kernel_sqeuclid": "ot_triton.kernels.sinkhorn_triton_ott_sqeuclid",
    "apply_transport_from_potentials_sqeuclid": "ot_triton.kernels.sinkhorn_triton_ott_sqeuclid",
    "update_potential": "ot_triton.kernels.sinkhorn_triton_ott_sqeuclid",
    # Apply (OTT-convention, deprecated)
    "apply_plan_vec_sqeuclid": "ot_triton.kernels.apply_ott",
    "apply_plan_mat_sqeuclid": "ot_triton.kernels.apply_ott",
    "mat5_sqeuclid": "ot_triton.kernels.apply_ott",
    # FlashSinkhorn internals
    "precompute_flashsinkhorn_inputs": "ot_triton.kernels.sinkhorn_flashstyle_sqeuclid",
    "compute_bias_f": "ot_triton.kernels.sinkhorn_flashstyle_sqeuclid",
    "compute_bias_g": "ot_triton.kernels.sinkhorn_flashstyle_sqeuclid",
    "flashsinkhorn_lse": "ot_triton.kernels.sinkhorn_flashstyle_sqeuclid",
    "flashsinkhorn_symmetric_step": "ot_triton.kernels.sinkhorn_flashstyle_sqeuclid",
    "shifted_to_standard_potentials": "ot_triton.kernels.sinkhorn_flashstyle_sqeuclid",
    "standard_to_shifted_potentials": "ot_triton.kernels.sinkhorn_flashstyle_sqeuclid",
}


def __getattr__(name):
    if name in _COMPAT_MAP:
        import importlib
        import warnings

        module_path = _COMPAT_MAP[name]
        warnings.warn(
            f"Importing {name} from ot_triton.kernels is deprecated. "
            f"Import from {module_path} directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module_path), name)
    raise AttributeError(f"module 'ot_triton.kernels' has no attribute {name!r}")
