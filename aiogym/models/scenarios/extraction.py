from ..core import ProcessModelContract


class ExtractionModel(ProcessModelContract):
    scenario = "extraction"
    benchmark_objectives = ("tracking", "kpi", "robustness", "safety")
    display_name = "Multistage Extraction Column"
    summary = "Five-stage counter-current liquid-gas extraction column from the PC-Gym benchmark."
    n = 5
    dt_micro = 0.01
    state_names = ("CX1", "CY1", "CX2", "CY2", "CX3", "CY3", "CX4", "CY4", "CX5", "CY5")
    state_units = {name: "concentration" for name in state_names}
    state_bounds = {name: (0.0, 1.0) for name in state_names}
    action_names = ("liquid_flow_L", "gas_flow_G")
    action_kinds = {name: "pump" for name in action_names}
    output_names = ("stage_1_liquid_concentration", "stage_2_liquid_concentration", "stage_3_liquid_concentration", "stage_4_liquid_concentration", "stage_5_liquid_concentration")
    output_units = {name: "concentration" for name in output_names}
    output_bounds = {name: (0.0, 1.0) for name in output_names}
    default_y_sp = (0.0, 0.0, 0.0, 0.0, 0.0040)
    param_units = {
        "Vl": "volume", "Vg": "volume", "m": "dimensionless", "Kla": "1/s",
        "e": "dimensionless", "CX0": "concentration", "CY6": "concentration",
        "Lmin": "flow", "Lmax": "flow", "Gmin": "flow", "Gmax": "flow",
        "pump_power_max": "W", "c_max": "concentration", "t_cold": "degC", "t_amb": "degC",
    }
    param_bounds = {
        "Vl": (1.0, 20.0), "Vg": (1.0, 20.0), "m": (0.01, 20.0), "Kla": (0.1, 20.0),
        "e": (0.1, 5.0), "CX0": (0.0, 1.0), "CY6": (0.0, 1.0),
        "Lmin": (0.0, 100.0), "Lmax": (0.0, 100.0), "Gmin": (0.0, 2000.0), "Gmax": (0.0, 2000.0),
        "pump_power_max": (0.0, 10000.0), "c_max": (0.0, 10.0), "t_cold": (0.0, 40.0), "t_amb": (0.0, 45.0),
    }
    plant_regime = {"Kla": (0.6, 1.5), "CX0": (0.75, 1.25), "CY6": (0.75, 1.25)}
    economic_config = {
        "temp_band": [(None, None), (None, None), (None, None), (None, None), (0.0035, 0.0048)],
        "level_band": [],
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.0,
        "w_viol": 25.0,
    }
    input_disturbances = (
        {"name": "CX0", "event": "feed_liquid_concentration_step", "unit": "concentration", "bounds": (0.3, 0.9), "description": "liquid feed concentration"},
        {"name": "CY6", "event": "gas_feed_concentration_step", "unit": "concentration", "bounds": (0.0, 0.12), "description": "gas/extract feed concentration"},
        {"name": "Kla", "event": "mass_transfer_shift", "unit": "1/s", "bounds": (2.0, 8.0), "description": "mass-transfer coefficient"},
    )
    safety_constraints = (
        {"name": "concentration_bounds", "states": state_names, "bounds": (0.0, 1.0)},
        {"name": "cx5_quality_band", "states": ("CX5",), "bounds": (0.0035, 0.0048)},
    )

    def __init__(self):
        self.p = dict(
            Vl=5.0, Vg=5.0, m=1.0, Kla=5.0, e=2.0,
            CX0=0.60, CY6=0.050, Lmin=5.0, Lmax=10.0, Gmin=500.0, Gmax=1000.0,
            pump_power_max=[1000.0, 1000.0], c_max=1.0, t_cold=0.0, t_amb=0.0,
        )

    def _flows_from_vector(self, u):
        p = self.p
        return (
            p["Lmin"] + u[0] * (p["Lmax"] - p["Lmin"]),
            p["Gmin"] + u[1] * (p["Gmax"] - p["Gmin"]),
        )

    def _action_from_flows(self, L, G):
        p = self.p
        return [
            max(0.0, min(1.0, (L - p["Lmin"]) / max(1e-9, p["Lmax"] - p["Lmin"]))),
            max(0.0, min(1.0, (G - p["Gmin"]) / max(1e-9, p["Gmax"] - p["Gmin"]))),
        ]

    def _dynamics(self, x, u, env, ops):
        p = self.p
        CX0 = env.get("CX0", p["CX0"])
        CY6 = env.get("CY6", p["CY6"])
        Kla = env.get("Kla", p["Kla"])
        L, Gf = self._flows_from_vector(u)
        dx = []
        for i in range(5):
            CX = x[2 * i]
            CY = x[2 * i + 1]
            CXprev = CX0 if i == 0 else x[2 * (i - 1)]
            CYnext = CY6 if i == 4 else x[2 * (i + 1) + 1]
            CXeq = (ops.max(CY, 1e-9) ** p["e"]) / p["m"]
            Fn = Kla * (CX - CXeq) * p["Vl"]
            dx.extend([
                (L * (CXprev - CX) - Fn) / p["Vl"],
                (Gf * (CYnext - CY) + Fn) / p["Vg"],
            ])
        return ops.vector(dx)

    def controlled_output(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return [x[2 * i] for i in range(5)]
        return [max(0.0, min(1.0, x[2 * i])) for i in range(5)]

    def display_outputs(self, x, backend="numeric", ca=None):
        cx = [x[2 * i] for i in range(5)]
        if backend == "casadi":
            return {"levels": [], "temps": cx}
        return {"levels": [], "temps": [max(0.0, min(1.0, v)) for v in cx]}

    def outputs(self, x):
        state = self.state_vector(x)
        cx = [max(0.0, min(1.0, state[2 * i])) for i in range(5)]
        cy = [max(0.0, min(1.0, state[2 * i + 1])) for i in range(5)]
        return {
            "x": state,
            "y": list(cx),
            "levels": [],
            "temps": list(cx),
            "conc": list(cx),
            "conc_liq": cx,
            "x_liq": cx,
            "y_ext": cy,
        }

    def initial_state(self):
        return [0.431, 0.643, 0.243, 0.474, 0.097, 0.286, 0.027, 0.140, 0.007, 0.070]

    def clamp_state(self, x):
        return [max(0.0, min(1.0, v)) for v in x]

    energy_scored = False

    def energy_kw(self, u, backend="numeric", ca=None):
        return sum(u[i] * self.p["pump_power_max"][i] for i in range(2)) / 1000.0
