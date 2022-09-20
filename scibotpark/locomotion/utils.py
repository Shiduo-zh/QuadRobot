

def get_default_randomization():
    """Randomize all the physical parameters."""
    param_range = {
        # The following ranges are in percentage. e.g. 0.8 means 80%.
        "mass": [0.8, 1.2],
        "inertia": [0.5, 1.5],
        "motor strength": [0.8, 1.2],
        # The following ranges are the physical values, in SI unit.
        "motor friction": [0, 0.05],  # Viscous damping (Nm s/rad).
        "latency": [0.0, 0.04],  # Time inteval (s).
        # Friction coefficient (dimensionless).
        "lateral friction": [0.5, 1.25],
        "battery": [14.0, 16.8],  # Voltage (V).
        "joint friction": [0, 0.05],  # Coulomb friction torque (Nm).
        # PD controller parameters
        "pd control": [[50, 0.4], [70, 0.8]],  # Range of the PD Controller Gain.
        "external force ratio": 1,
        "random force": [-1000, 1000],
    }
    return param_range