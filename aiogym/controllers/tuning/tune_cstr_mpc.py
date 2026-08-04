"""Tune the CSTR MPC baseline parameters."""
from __future__ import annotations

import itertools

from aiogym.controllers.registry import make_controller
from aiogym import make_env
from aiogym.evaluation.execution import evaluate_controller


def make_tuning_env(episode_steps: int = 400):
    return make_env(
        config={
            "scenario": "cstr",
            "reward_spec": "regulation-v1",
            "environment": {
                "control_dt": 0.5,
                "episode_steps": episode_steps,
                "auto_events": True,
                "randomize": True,
                "randomize_setpoints": False,
                "action_mode": "actuator",
                "randomize_plant": True,
                "plant_drift": True,
                "integral_obs": False,
                "terminate_on_runaway": False,
            },
        }
    )


def eval_agent(agent, episodes: int = 1):
    return evaluate_controller(agent, make_tuning_env(), episodes=episodes)


def main():
    coarse_episodes = 4
    final_episodes = 16
    pid = eval_agent(make_controller("pid", scenario="cstr"), episodes=final_episodes)
    print(f"PID track={pid['track']:.3f} return={pid['return']:.3f} constraint={pid['constraint']:.3f}")

    grid = {
        "P": [20, 40, 60],
        "move_supp": [0.005, 0.02, 0.08, 0.2],
        "cv_scale": [1.5, 3.0, 6.0, 12.0],
    }
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        config = {
            "case": "cstr-tuning",
            "policy_scope": "specialist",
            "goal": "regulation",
            "reward_spec": "regulation-v1",
            "parameters": params,
        }
        rep = eval_agent(make_controller("mpc", scenario="cstr", config=config), episodes=coarse_episodes)
        row = {**params, **rep}
        results.append(row)
        print(
            "MPC "
            + " ".join(f"{k}={params[k]}" for k in keys)
            + f" track={rep['track']:.3f} return={rep['return']:.3f} constraint={rep['constraint']:.3f}"
        )

    results.sort(key=lambda r: (r["track"], r["constraint"]))
    print("\nCOARSE TOP")
    for r in results[:12]:
        print(
            f"track={r['track']:.3f} return={r['return']:.3f} constraint={r['constraint']:.3f} "
            f"P={r['P']} move_supp={r['move_supp']} cv_scale={r['cv_scale']}"
        )

    final = []
    print("\nFINAL VERIFY")
    for r in results[:12]:
        params = {k: r[k] for k in keys}
        config = {
            "case": "cstr-tuning",
            "policy_scope": "specialist",
            "goal": "regulation",
            "reward_spec": "regulation-v1",
            "parameters": params,
        }
        rep = eval_agent(make_controller("mpc", scenario="cstr", config=config), episodes=final_episodes)
        final.append({**params, **rep})
        print(
            f"track={rep['track']:.3f} return={rep['return']:.3f} constraint={rep['constraint']:.3f} "
            f"P={params['P']} move_supp={params['move_supp']} cv_scale={params['cv_scale']}"
        )

    final.sort(key=lambda r: (r["track"], r["constraint"]))
    print("\nFINAL TOP")
    for r in final[:5]:
        print(
            f"track={r['track']:.3f} return={r['return']:.3f} constraint={r['constraint']:.3f} "
            f"P={r['P']} move_supp={r['move_supp']} cv_scale={r['cv_scale']}"
        )


if __name__ == "__main__":
    main()
