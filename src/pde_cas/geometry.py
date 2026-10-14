"""Geometry and physical constants for the adopted room example."""

ROOM_FV_ACTUATOR_CENTERS: tuple[tuple[float, float], ...] = (
    (0.34, 0.26),
    (1.86, 0.24),
    (2.66, 0.26),
    (0.36, 1.74),
    (1.14, 1.76),
    (2.64, 1.74),
)
ROOM_FV_TARGET_CENTERS: tuple[tuple[float, float], ...] = (
    (0.70, 0.70),
    (2.30, 1.25),
    (2.40, 0.65),
)
ROOM_FV_PROFILE_RADIUS = 0.15
ROOM_FV_DAMPING = 0.02
ROOM_FV_VELOCITY_SCALE = 0.15
ROOM_FV_SINGULAR_HORIZON = 5.0
ROOM_FV_LONG_HORIZON = 12.0
