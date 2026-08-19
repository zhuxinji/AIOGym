"""Thin command-line adapters for collect, train, evaluate, and compare."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiogym.core.io import jsonable


_CONTROLLERS = ("hold", "mpc", "pid", "random")
_DESCRIPTIONS = {
    "collect": "Collect an episode-oriented Dataset.",
    "train": "Train and save one registered algorithm policy.",
    "evaluate": "Evaluate one controller or checkpoint on explicit seeds.",
    "compare": "Compare controllers and checkpoints on identical seeds.",
}


def _environment_arguments(parser, *, allow_benchmark):
    parser.add_argument(
        "scenario",
        help="registered Scenario id; inspect choices with `aiogym list scenarios`",
    )
    parser.add_argument(
        "--reward",
        help=(
            "Reward id; omit for the Scenario default"
            + ("; invalid with --benchmark" if allow_benchmark else "")
        ),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        metavar="JSON",
        help=(
            "JSON object file of model parameter overrides"
            + ("; invalid with --benchmark" if allow_benchmark else "")
        ),
    )
    if allow_benchmark:
        parser.add_argument(
            "--benchmark",
            help=(
                "fixed evaluation protocol; invalid with --reward, --parameters, "
                "--randomize, --disturbance on, --noise on, --delay on, or "
                "--fault on"
            ),
        )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="sample a new training episode on every reset",
    )
    parser.add_argument(
        "--disturbance",
        choices=("on", "off"),
        default="off",
        help="toggle scenario-owned random physical disturbances",
    )
    parser.add_argument(
        "--noise",
        choices=("on", "off"),
        default="off",
        help="toggle the default observation-noise configuration",
    )
    parser.add_argument(
        "--delay",
        choices=("on", "off"),
        default="off",
        help="toggle the default observation/action-delay configuration",
    )
    parser.add_argument(
        "--fault",
        choices=("on", "off"),
        default="off",
        help="toggle the default actuator loss-of-effectiveness fault",
    )


def _parser(command):
    parser = argparse.ArgumentParser(
        prog=f"aiogym {command}",
        description=_DESCRIPTIONS[command],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _environment_arguments(parser, allow_benchmark=command != "train")
    if command == "collect":
        parser.add_argument(
            "--controller",
            choices=_CONTROLLERS,
            default="random",
            help="controller used to collect actions",
        )
        parser.add_argument(
            "--episodes", type=int, default=1, help="number of episodes to collect"
        )
        parser.add_argument("--seed", type=int, default=0, help="first episode seed")
        parser.add_argument(
            "--max-steps", type=int, help="optional per-episode step limit"
        )
        parser.add_argument(
            "--output",
            required=True,
            type=Path,
            help="new or empty Dataset directory",
        )
    elif command == "train":
        from aiogym.workflows.algorithms import list_algorithms

        parser.add_argument(
            "algorithm", choices=list_algorithms(), help="registered algorithm id"
        )
        parser.add_argument(
            "--steps", required=True, type=int, help="positive environment-step budget"
        )
        parser.add_argument("--seed", type=int, default=0, help="training seed")
        parser.add_argument(
            "--algorithm-kwargs",
            type=Path,
            metavar="JSON",
            help="JSON object file passed to the algorithm backend",
        )
        parser.add_argument(
            "--record-every",
            type=int,
            default=500,
            help="training-curve aggregation interval in environment steps",
        )
        parser.add_argument(
            "--demonstrations",
            type=Path,
            metavar="DATASET",
            help="Dataset v2 used for behavior-cloning pretraining",
        )
        parser.add_argument(
            "--behavior-cloning-epochs",
            type=int,
            metavar="N",
            help="positive supervised epochs; required with --demonstrations",
        )
        parser.add_argument(
            "--behavior-cloning-batch-size",
            type=int,
            default=256,
            metavar="N",
            help="supervised demonstration batch size",
        )
        parser.add_argument(
            "--behavior-cloning-learning-rate",
            type=float,
            default=3e-4,
            metavar="RATE",
            help="supervised actor learning rate",
        )
        parser.add_argument(
            "--evaluate-every",
            type=int,
            help="periodically evaluate and save the best checkpoint at this interval",
        )
        parser.add_argument(
            "--evaluation-seed",
            type=int,
            default=0,
            help="seed for periodic deterministic evaluation",
        )
        parser.add_argument(
            "--output",
            required=True,
            type=Path,
            help="new or empty training directory",
        )
    elif command == "evaluate":
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--controller",
            choices=_CONTROLLERS,
            help="built-in controller to evaluate",
        )
        source.add_argument(
            "--checkpoint",
            type=Path,
            metavar="MODEL_ZIP",
            help="self-describing AIO-Gym model.zip to evaluate",
        )
        parser.add_argument(
            "--seeds",
            nargs="+",
            required=True,
            type=int,
            help=(
                "ordered reset seeds; on a Benchmark these select reproducible cases"
            ),
        )
        parser.add_argument(
            "--max-steps", type=int, help="optional per-episode step limit"
        )
        parser.add_argument(
            "--output", required=True, type=Path, help="new evaluation JSON file"
        )
    else:
        parser.add_argument(
            "--controllers",
            nargs="+",
            choices=_CONTROLLERS,
            help="built-in controllers to compare",
        )
        parser.add_argument(
            "--checkpoint",
            action="append",
            nargs=2,
            metavar=("LABEL", "MODEL_ZIP"),
            help=(
                "learned policy to compare; repeat for multiple checkpoints"
            ),
        )
        parser.add_argument(
            "--seeds",
            nargs="+",
            required=True,
            type=int,
            help=(
                "ordered reset seeds; on a Benchmark these select reproducible cases"
            ),
        )
        parser.add_argument(
            "--max-steps", type=int, help="optional per-episode step limit"
        )
        parser.add_argument(
            "--output",
            type=Path,
            help=(
                "empty output directory; optional Benchmark comparisons use "
                "runs/<scenario>/<benchmark>"
            ),
        )
    return parser


def _json_object_file(path, parser, label):
    if path is None:
        return None
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        parser.error(f"could not read {label} {path}: {error}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        parser.error(
            f"{label} is not valid JSON: {error.msg} "
            f"(line {error.lineno}, column {error.colno})"
        )
    if not isinstance(value, dict):
        parser.error(f"{label} must be a JSON object")
    return value


def _comparison_policies(args, parser, aiogym, env):
    controllers = () if args.controllers is None else tuple(args.controllers)
    if len(set(controllers)) != len(controllers):
        parser.error("--controllers must not contain duplicates")
    policies = {name: name for name in controllers}
    checkpoints = () if args.checkpoint is None else tuple(args.checkpoint)
    for label, checkpoint in checkpoints:
        if not label.strip():
            parser.error("checkpoint LABEL must be non-empty")
        if label in policies:
            parser.error(f"duplicate policy label: {label}")
        policies[label] = aiogym.load_policy(
            Path(checkpoint),
            env=env,
        )
    if len(policies) < 2:
        parser.error(
            "compare requires at least two policies from --controllers and --checkpoint"
        )
    return policies


def _comparison_output(args, env):
    if args.output is not None:
        return str(args.output.resolve())
    base_env = env.unwrapped
    scenario = base_env.scenario.id.replace("_", "-")
    return str((Path("runs") / scenario / base_env.benchmark.id).resolve())


def _success_summary(command, args, result, env):
    if command == "collect":
        return {
            "schema_version": result["schema_version"],
            "output": result["path"],
            "episodes": result["episodes"],
            "transitions": result["transitions"],
        }
    if command == "train":
        return {
            "schema_version": result["schema_version"],
            "output": result["path"],
            "algorithm": result["algorithm"],
            "actual_steps": result["actual_steps"],
            "checkpoint": result["checkpoint"],
            "best_checkpoint": result["best_checkpoint"],
            "training_curve": result["training_curve"],
            "behavior_cloning_artifact": result["behavior_cloning_artifact"],
        }
    if command == "evaluate":
        return {
            "schema_version": result["schema_version"],
            "output": str(args.output.resolve()),
            "policy": result["policy"],
            "seeds": result["seeds"],
            "ranking_metrics": result["ranking_metrics"],
            "aggregate": result["aggregate"],
        }
    metric_names = [row["name"] for row in result["ranking_metrics"]]
    return {
        "schema_version": result["schema_version"],
        "output": _comparison_output(args, env),
        "seeds": result["seeds"],
        "ranking_metrics": result["ranking_metrics"],
        "ranking": [
            {
                "policy": label,
                "metrics": {
                    name: result["evaluations"][label]["aggregate"][name]["median"]
                    for name in metric_names
                },
            }
            for label in result["ordering"]
        ],
    }


def main(command, argv=None):
    parser = _parser(command)
    args = parser.parse_args(argv)
    parameters = _json_object_file(args.parameters, parser, "parameters")
    benchmark = None if command == "train" else args.benchmark
    env = None
    evaluation_env = None
    try:
        import aiogym

        env = aiogym.make_env(
            args.scenario,
            reward=args.reward,
            parameters=parameters,
            benchmark=benchmark,
            randomize=args.randomize,
            disturbance=args.disturbance == "on",
            noise=args.noise == "on",
            delay=args.delay == "on",
            fault=args.fault == "on",
        )
        if command == "collect":
            result = aiogym.collect(
                env=env,
                policy=args.controller,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                output=args.output,
            )
        elif command == "train":
            algorithm_kwargs = _json_object_file(
                args.algorithm_kwargs,
                parser,
                "algorithm kwargs",
            )
            if args.evaluate_every is not None:
                evaluation_env = aiogym.make_env(
                    args.scenario,
                    reward=args.reward,
                    parameters=parameters,
                )
            result = aiogym.train(
                env=env,
                algorithm=args.algorithm,
                steps=args.steps,
                seed=args.seed,
                algorithm_kwargs=algorithm_kwargs,
                record_every=args.record_every,
                evaluation_env=evaluation_env,
                evaluate_every=args.evaluate_every,
                evaluation_seed=args.evaluation_seed,
                demonstrations=args.demonstrations,
                behavior_cloning_epochs=args.behavior_cloning_epochs,
                behavior_cloning_batch_size=args.behavior_cloning_batch_size,
                behavior_cloning_learning_rate=(
                    args.behavior_cloning_learning_rate
                ),
                output=args.output,
            )
        elif command == "evaluate":
            if args.checkpoint is not None:
                policy = aiogym.load_policy(
                    args.checkpoint,
                    env=env,
                )
            else:
                policy = args.controller
            result = aiogym.evaluate(
                env=env,
                policy=policy,
                seeds=args.seeds,
                max_steps=args.max_steps,
                output=args.output,
            )
        else:
            policies = _comparison_policies(args, parser, aiogym, env)
            result = aiogym.compare_policies(
                env=env,
                policies=policies,
                seeds=args.seeds,
                max_steps=args.max_steps,
                output=args.output,
            )
        summary = _success_summary(command, args, result, env)
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    finally:
        if evaluation_env is not None:
            evaluation_env.close()
        if env is not None:
            env.close()
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = ["main"]
