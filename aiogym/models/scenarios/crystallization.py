import math

from ..core import ProcessModelContract


class CrystallizationModel(ProcessModelContract):
    """Method-of-moments batch crystallizer model from clean_latest.

    State: [mu0, mu1, mu2, mu3, c].
    Action: heaters[0] is normalized aT, mapped to Tc = 30 + 10 * aT.
    """

    scenario = "crystallization"
    benchmark_objectives = ("tracking", "kpi", "robustness", "safety")
    display_name = "Batch Crystallization"
    summary = "Single-actuator crystallization benchmark with moment dynamics and Ln/CV quality targets."
    n = 1
    dt_micro = 0.02
    state_names = ("mu0", "mu1", "mu2", "mu3", "c")
    state_units = {"mu0": "moment", "mu1": "moment", "mu2": "moment", "mu3": "moment", "c": "kg/L"}
    state_bounds = {"mu0": (0.0, 1e4), "mu1": (0.0, 1e6), "mu2": (0.0, 1e8), "mu3": (0.0, 1e10), "c": (0.0, 2.0)}
    action_names = ("cooling_temperature_fraction",)
    action_kinds = {"cooling_temperature_fraction": "heater"}
    output_names = ("coefficient_variation", "mean_crystal_size")
    output_units = {"coefficient_variation": "dimensionless", "mean_crystal_size": "um"}
    output_bounds = {"coefficient_variation": (0.0, 5.0), "mean_crystal_size": (0.0, 50.0)}
    param_units = {
        "ka": "1/s", "kb": "K", "kc": "dimensionless", "kd": "dimensionless",
        "kg": "1/s", "k1": "K", "k2": "dimensionless", "a": "dimensionless",
        "b": "dimensionless", "alpha": "dimensionless", "rho": "kg/L",
        "Tc_min": "degC", "Tc_max": "degC", "Tc_nom": "degC",
        "Ln_sp": "um", "CV_sp": "dimensionless", "eps": "dimensionless",
        "B0_max": "1/s", "Ginf_scale": "dimensionless", "Ginf_max": "um/s",
    }
    param_bounds = {
        "ka": (0.0, 10.0), "kb": (-20000.0, 0.0), "kc": (0.0, 5.0), "kd": (0.0, 5.0),
        "kg": (0.0, 200.0), "k1": (-20000.0, 0.0), "k2": (0.0, 5.0), "a": (0.0, 2.0),
        "b": (0.0, 20.0), "alpha": (0.0, 20.0), "rho": (0.0, 10.0),
        "Tc_min": (0.0, 100.0), "Tc_max": (0.0, 100.0), "Tc_nom": (0.0, 100.0),
        "Ln_sp": (0.0, 50.0), "CV_sp": (0.0, 5.0), "eps": (1e-12, 1e-3),
        "B0_max": (0.0, 1.0), "Ginf_scale": (0.0, 1e-3), "Ginf_max": (0.0, 1e-2),
    }
    plant_regime = {"ka": (0.7, 1.3), "kg": (0.7, 1.3), "Ginf_scale": (0.7, 1.3)}
    economic_config = {
        "temp_band": [(None, None)],
        "level_band": [(None, None)],
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.0,
        "w_viol": 0.0,
    }
    input_disturbances = (
        {"name": "growth_factor", "event": "growth_factor_step", "unit": "multiplier", "bounds": (0.7, 1.3), "default": 1.0, "description": "growth-rate multiplier"},
        {"name": "nucleation_factor", "event": "nucleation_factor_step", "unit": "multiplier", "bounds": (0.7, 1.3), "default": 1.0, "description": "nucleation-rate multiplier"},
        {"name": "solubility_bias", "event": "solubility_bias_step", "unit": "concentration", "bounds": (-10.0, 10.0), "default": 0.0, "description": "solubility-curve offset"},
    )
    safety_constraints = (
        {"name": "moment_nonnegative", "states": state_names, "bounds": (0.0, None)},
        {"name": "cooling_temperature_bounds", "states": ("Tc",), "bounds": (30.0, 40.0)},
        {"name": "quality_targets", "states": ("Ln", "CV"), "bounds": (None, None)},
    )
    energy_scored = False
    oracle_temperature_cap = False
    supports_integral_observation = False
    randomize_common_temperatures = False

    def __init__(self):
        self.p = dict(
            ka=0.92,
            kb=-6800.0,
            kc=0.92,
            kd=1.3,
            kg=48.0,
            k1=-4900.0,
            k2=1.9,
            a=0.51,
            b=7.3,
            alpha=7.5,
            rho=2.7,
            Tc_min=30.0,
            Tc_max=40.0,
            Tc_nom=35.0,
            Ln_sp=15.0,
            CV_sp=1.0,
            eps=1e-9,
            B0_max=0.05,
            Ginf_scale=2e-6,
            Ginf_max=2e-4,
        )

    @property
    def height_max(self):
        return [1.0]

    def _finite(self, value, fallback=0.0):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return fallback
        return v if math.isfinite(v) else fallback

    def _safe_exp(self, value):
        x = self._finite(value, -60.0)
        if x < -60.0:
            return 0.0
        if x > 60.0:
            return math.exp(60.0)
        return math.exp(x)

    def _clip_numeric(self, value, lo, hi, fallback=0.0):
        return min(hi, max(lo, self._finite(value, fallback)))

    def clean_env(self, env=None):
        env = env or {}
        return {
            "growth_factor": max(0.0, self._finite(env.get("growth_factor"), 1.0)),
            "nucleation_factor": max(0.0, self._finite(env.get("nucleation_factor"), 1.0)),
            "solubility_bias": self._finite(env.get("solubility_bias"), 0.0),
        }

    def runtime_env(self, disturbance_values):
        return self.clean_env(disturbance_values)

    def action_to_tc(self, act=None):
        aT = self._action_fraction(act)
        return self.p["Tc_min"] + (self.p["Tc_max"] - self.p["Tc_min"]) * aT

    def _clean_state(self, x=None):
        state = list(x) if x is not None else self.initial_state()
        init = self.initial_state()
        caps = [1e4, 1e6, 1e8, 1e10, 2.0]
        clean = []
        for i, cap in enumerate(caps):
            v = state[i] if i < len(state) else init[i]
            clean.append(self._clip_numeric(v, 0.0, cap, init[i]))
        return clean

    def metrics(self, x=None, act=None, env=None):
        mu0, mu1, mu2, mu3, c = self._clean_state(x)
        e = self.clean_env(env)
        p = self.p
        Tc = self.action_to_tc(act)
        Tk = Tc + 273.15
        Ceq = -686.2686 + 3.579165 * Tk - 0.00292874 * Tk * Tk + e["solubility_bias"]
        S = max(0.0, self._finite(c * 1000.0 - Ceq, 0.0))
        b0_raw = p["ka"] * self._safe_exp(p["kb"] / Tk) * (S ** p["kc"]) * (max(abs(mu3), p["eps"]) ** p["kd"])
        g_raw = p["kg"] * self._safe_exp(p["k1"] / Tk) * (S ** p["k2"])
        B0 = min(p["B0_max"], max(0.0, self._finite(b0_raw * e["nucleation_factor"], 0.0)))
        Ginf = min(p["Ginf_max"], max(0.0, self._finite(g_raw * p["Ginf_scale"] * e["growth_factor"], 0.0)))
        CV = math.sqrt(max((mu2 * mu0) / (mu1 * mu1 + p["eps"]) - 1.0, 0.0))
        Ln = mu1 / (mu0 + p["eps"])
        out = dict(mu0=mu0, mu1=mu1, mu2=mu2, mu3=mu3, c=c, Tc=Tc, Tk=Tk,
                   Ceq=Ceq, S=S, B0=B0, Ginf=Ginf, CV=CV, Ln=Ln)
        out.update(e)
        return out

    def _dynamics(self, x, u, env, ops):
        p = self.p
        mu0, mu1, mu2, mu3, c = x[0], x[1], x[2], x[3], x[4]
        growth_factor = env.get("growth_factor", 1.0)
        nucleation_factor = env.get("nucleation_factor", 1.0)
        solubility_bias = env.get("solubility_bias", 0.0)
        aT = ops.max(0.0, -ops.max(-1.0, -u[0]))
        Tc = p["Tc_min"] + (p["Tc_max"] - p["Tc_min"]) * aT
        Tk = Tc + 273.15
        Ceq = -686.2686 + 3.579165 * Tk - 0.00292874 * Tk * Tk + solubility_bias
        S = ops.max(0.0, c * 1000.0 - Ceq)
        b0_raw = p["ka"] * ops.exp(p["kb"] / Tk) * (S ** p["kc"]) * (ops.max(ops.abs(mu3), p["eps"]) ** p["kd"])
        g_raw = p["kg"] * ops.exp(p["k1"] / Tk) * (S ** p["k2"])
        B0 = ops.max(0.0, -ops.max(-p["B0_max"], -(b0_raw * nucleation_factor)))
        Ginf = ops.max(0.0, -ops.max(-p["Ginf_max"], -(g_raw * p["Ginf_scale"] * growth_factor)))
        dmu0 = B0
        dmu1 = Ginf * (p["a"] * mu0 + p["b"] * mu1 * 1e-4) * 1e4
        dmu2 = 2.0 * Ginf * (p["a"] * mu1 * 1e-4 + p["b"] * mu2 * 1e-8) * 1e8
        dmu3 = 3.0 * Ginf * (p["a"] * mu2 * 1e-8 + p["b"] * mu3 * 1e-12) * 1e12
        dc = -0.5 * p["rho"] * p["alpha"] * Ginf * (p["a"] * mu2 * 1e-8 + p["b"] * mu3 * 1e-12)
        return ops.vector([dmu0, dmu1, dmu2, dmu3, dc])

    def build_state(self, x, act=None, env=None, t=0.0):
        m = self.metrics(x, act, env)
        return {
            "t": t,
            "levels": [m["CV"]],
            "temps": [m["Tc"]],
            "conc": [m["c"]],
            "moments": [m["mu0"], m["mu1"], m["mu2"], m["mu3"]],
            "crystal_metrics": [m["Ln"], m["CV"]],
            "solute_metrics": [m["c"], m["S"], m["Ceq"]],
            "crystal_temperature": [m["Tc"]],
            **m,
        }

    def outputs(self, x):
        state = self.state_vector(x)
        m = self.metrics(state, self.default_action(), self.clean_env())
        return {
            "x": state,
            "y": self.controlled_output(state),
            "levels": [m["CV"]],
            "temps": [m["Tc"]],
            "conc": [m["c"]],
            "moments": [m["mu0"], m["mu1"], m["mu2"], m["mu3"]],
            "crystal_metrics": [m["Ln"], m["CV"]],
        }

    def process_constraint_info(self, x, levels, temps, env):
        m = self.metrics(x, self.default_action(), env)
        nonfinite = any(not math.isfinite(v) for v in list(x) + [m["Ln"], m["CV"], m["Tc"], m["c"], m["S"], m["Ceq"], m["B0"], m["Ginf"]])
        negative = any(v < 0.0 for v in x)
        return {"crystallization_constraint": (10.0 if nonfinite else 0.0) + (5.0 if negative else 0.0)}

    def process_info(self, x, levels, temps, env, act=None):
        return self.metrics(x, act or self.default_action(), env)

    def common_constraint_info(self, levels, temps):
        return {"temp_high": 0.0, "temp_trip": 0.0, "level_high": 0.0, "level_low": 0.0}

    def runaway_state(self, levels, temps):
        return False

    def constraint_penalty_scales(self):
        return {"crystallization_constraint": 1.0}

    def initial_state(self):
        return [1.0, 15.0, 250.0, 4500.0, 0.90]

    def clamp_state(self, x):
        return self._clean_state(x)

    def display_outputs(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            cv = ca.sqrt(ca.fmax((x[2] * x[0]) / (x[1] * x[1] + self.p["eps"]) - 1.0, 0.0))
            ln = x[1] / (x[0] + self.p["eps"])
            return {"levels": [cv], "temps": [ln]}
        m = self.metrics(x, self.default_action(), self.clean_env())
        return {"levels": [m["CV"]], "temps": [m["Ln"]]}

    def conc(self, x):
        return [self._clean_state(x)[4]]

    def controlled_output(self, x, backend="numeric", ca=None):
        display = self.display_outputs(x, backend=backend, ca=ca)
        levels, temps = display["levels"], display["temps"]
        return [levels[0], temps[0]]

    def default_setpoint_vector(self):
        return [self.p["CV_sp"], self.p["Ln_sp"]]

    def env_setpoint_vector(self, options=None):
        options = dict(options or {})
        y_sp = self.default_setpoint_vector()
        if options.get("crystal_ln_sp") is not None:
            y_sp[1] = float(options["crystal_ln_sp"])
        if options.get("crystal_cv_sp") is not None:
            y_sp[0] = float(options["crystal_cv_sp"])
        return y_sp

    def sample_env_setpoints(self, y_sp, rng, options=None):
        options = dict(options or {})
        sampled = list(y_sp)
        if (
            options.get("crystal_random_targets")
            or options.get("randomize_setpoints")
        ):
            sampled[1] = float(rng.uniform(
                *options.get("crystal_ln_range", (10.0, 11.5))
            ))
            sampled[0] = float(rng.uniform(
                *options.get("crystal_cv_range", (0.75, 0.95))
            ))
        return sampled

    def default_action(self):
        return [0.5]

    def _action_fraction(self, act=None):
        values = self.default_action() if act is None else self.action_vector(act)
        return self._clip_numeric(values[0], 0.0, 1.0, 0.5)
