"""Adopted cell-average profiles with six inputs and three outputs."""
import numpy as np
from .grid import Room2DFVGrid
from .profiles import compact_bump
from .geometry import ROOM_FV_ACTUATOR_CENTERS, ROOM_FV_TARGET_CENTERS, ROOM_FV_PROFILE_RADIUS

def benchmark_profiles(
    grid: Room2DFVGrid,
    *,
    quadrature_order: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized actuator and target cell-average columns."""
    actuators = np.column_stack(
        [
            compact_bump(
                grid,
                center,
                ROOM_FV_PROFILE_RADIUS,
                quadrature_order=quadrature_order,
            )
            for center in ROOM_FV_ACTUATOR_CENTERS
        ]
    )
    targets = np.column_stack(
        [
            compact_bump(
                grid,
                center,
                ROOM_FV_PROFILE_RADIUS,
                quadrature_order=quadrature_order,
            )
            for center in ROOM_FV_TARGET_CENTERS
        ]
    )
    return actuators, targets
