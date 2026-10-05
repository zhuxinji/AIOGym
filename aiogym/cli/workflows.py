"""Thin command-line adapters for collect, train, and evaluate."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from aiogym.core.io import jsonable, write_json


_CONTROLLERS = ("hold", "mpc", "pid", "random")
_DESCRIPTIONS = {
    "collect": "Collect an episode-oriented Dataset.",
    "train": "Train built-in algorithms across independent training seeds.",
    "evaluate": "Evaluate one or more controllers and checkpoints on identical seeds.",
}


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _environment_arguments(parser, *, allow_benchmark, options=None):
    parser.add_argument(
        "scenario",
        help="built-in Scenario id; inspect choices with `aiogym list scenarios`",
    )
    parser = options if options is not None else parser
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
                "--randomize, nonzero --boundary-probability, --disturbance "
                "on, --noise on, --delay on, or --fault on"
            ),
        )
    parser.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="sample a new training episode on every reset",
    )
    parser.add_argument(
        "--boundary-probability",
        type=float,
        default=0.0,
        metavar="PROBABILITY",
        help="boundary-initialization probability; requires --randomize",
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
        formatter_class=_HelpFormatter,
    )
    if command == "train":
        common = parser.add_argument_group("Training and output")
        validation = parser.add_argument_group("Validation and progress")
        environment = parser.add_argument_group("Environment and variation")
        advanced = parser.add_argument_group("Dataset, behavior cloning, and algorithm settings")
        parser.epilog = """Examples:
  aiogym train heater sac --steps 1000 --no-evaluation
  aiogym train heater sac ppo td3 --seeds 0 1 --workers 3
  aiogym train heater sac --resume-from /path/to/model.zip --steps 50000
  aiogym train heater sac ddpg ppo td3 rlpd --dataset /path/to/dataset

RLPD requires a compatible Dataset. Online SAC/DDPG/PPO/TD3 do not.
With --resume-from, --steps adds steps to the checkpoint in a new directory.
Inspect progress from another terminal with: aiogym status
"""
    else:
        environment = None
    _environment_arguments(parser, allow_benchmark=command != "train", options=environment)
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
            metavar="DIR",
            help="create a Dataset in a new or empty directory; existing data rejected",
        )
    elif command == "train":
        from aiogym.rl.algorithms import list_algorithms

        parser.set_defaults(randomize=True)
        parser.add_argument(
            "algorithm", nargs="+", choices=list_algorithms(),
            help="one or more built-in algorithm ids",
        )
        common.add_argument(
            "--steps", default=500_000, type=int, help="positive additional environment steps per task; also additive on continuation"
        )
        seeds = common.add_mutually_exclusive_group()
        seeds.add_argument(
            "--seed",
            type=int,
            help="training seed; continued training uses the checkpoint seed",
        )
        seeds.add_argument(
            "--seeds", nargs="+", type=int,
            help="independent training seeds; defaults to seed 0 for new training",
        )
        common.add_argument(
            "--workers", type=int, default=2,
            help="maximum concurrent training processes; 1 runs tasks sequentially",
        )
        common.add_argument(
            "--resume-from",
            type=Path,
            metavar="MODEL_ZIP",
            help="continue optimization from a complete checkpoint; one task only, into a new run",
        )
        advanced.add_argument(
            "--algorithm-kwargs",
            type=Path,
            metavar="JSON",
            help="JSON object file passed to the algorithm backend",
        )
        validation.add_argument(
            "--record-every",
            type=int,
            default=500,
            help="training-curve aggregation interval in environment steps",
        )
        advanced.add_argument(
            "--dataset",
            type=Path,
            metavar="DATASET",
            help="required for RLPD; optional for online algorithms only with behavior cloning",
        )
        advanced.add_argument(
            "--behavior-cloning-epochs",
            type=int,
            metavar="N",
            help="positive supervised epochs; required for Dataset-based BC",
        )
        advanced.add_argument(
            "--behavior-cloning-batch-size",
            type=int,
            default=256,
            metavar="N",
            help="supervised demonstration batch size",
        )
        advanced.add_argument(
            "--behavior-cloning-learning-rate",
            type=float,
            default=3e-4,
            metavar="RATE",
            help="supervised actor learning rate",
        )
        evaluation = validation.add_mutually_exclusive_group()
        evaluation.add_argument(
            "--evaluate-every",
            type=int,
            default=5_000,
            help="periodically evaluate and save the best checkpoint at this interval",
        )
        evaluation.add_argument(
            "--no-evaluation",
            dest="evaluate_every",
            action="store_const",
            const=None,
            help="disable periodic validation and best-checkpoint selection",
        )
        common.add_argument(
            "--output",
            type=Path,
            metavar="DIR",
            help="create a run in this exact new or empty directory (one task only); omit for automatic per-algorithm directories",
        )
    else:
        parser.add_argument(
            "--controllers",
            nargs="+",
            choices=_CONTROLLERS,
            help="built-in controllers to evaluate",
        )
        parser.add_argument(
            "--checkpoint",
            action="append",
            nargs=2,
            metavar=("LABEL", "MODEL_ZIP"),
            help=(
                "learned policy to evaluate; repeat for multiple checkpoints"
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
            metavar="DIR",
            help=(
                "write comparison.json, comparison.svg, and trajectories.npz to a new "
                "or empty directory; omit to print the full result without writing files"
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


def _evaluation_policies(args, parser, aiogym, env):
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
    if not policies:
        parser.error(
            "evaluate requires at least one policy from --controllers and --checkpoint"
        )
    return policies


def _success_summary(command, args, result):
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
            "initial_steps": result["initial_steps"],
            "added_steps": result["added_steps"],
            "actual_steps": result["actual_steps"],
            "resume_from": result["resume_from"],
            "checkpoint": result["checkpoint"],
            "best_checkpoint": result["best_checkpoint"],
            "training_curve": result["training_curve"],
            "behavior_cloning_artifact": result["behavior_cloning_artifact"],
        }
    if args.output is None:
        return result
    return {
        "schema_version": result["schema_version"],
        "output": str(args.output.resolve()),
        "seeds": result["seeds"],
        "ranking_metrics": result["ranking_metrics"],
        "ranking": [
            {
                "policy": label,
                "metrics": {
                    metric["name"]: result["evaluations"][label]["aggregate"][
                        metric["name"]
                    ][metric["aggregate"]]
                    for metric in result["ranking_metrics"]
                },
            }
            for label in result["ordering"]
        ],
    }


def _print_training_result(output, metadata):
    """Use the same artifact roles and continuation settings for every CLI run."""
    import aiogym
    from aiogym.rl.algorithms import get_algorithm
    from aiogym.workflows._metadata import ENVIRONMENT_COMPATIBILITY_FIELDS, environment_metadata

    output = Path(output).resolve()
    environment = metadata["environment"]
    scenario, algorithm = environment["scenario"], metadata["algorithm"]
    options = ["--reward", environment["reward"]]
    parameters = output / "parameters.json"
    if parameters.is_file():
        options.extend(["--parameters", str(parameters)])
    resume = ["aiogym", "train", scenario, algorithm, "--resume-from", str(output / "model.zip"),
              "--steps", "50000", "--seed", str(metadata["seed"]), *options,
              "--randomize" if environment["randomize"] else "--no-randomize",
              "--boundary-probability", str(environment["boundary_probability"]),
              "--record-every", str(metadata["record_every"])]
    for name in ("disturbance", "noise", "delay", "fault"):
        if environment[name]:
            resume.extend([f"--{name}", "on"])
    validation = metadata["evaluation"]
    if validation is None:
        resume.append("--no-evaluation")
    else:
        resume.extend(["--evaluate-every", str(validation["evaluate_every"])])
    if get_algorithm(algorithm).requires_dataset:
        resume.extend(["--dataset", metadata["dataset"]["path"]])

    checkpoint = output / "model.zip" if validation is None else output / "best/model.zip"
    print(f"\n{scenario} / {algorithm} / seed {metadata['seed']} — {metadata['actual_steps']:,} steps", file=sys.stderr)
    print(f"  Compare:  {checkpoint}" + (" (final; no validation selection)" if validation is None else " (validation best)"), file=sys.stderr)
    print(f"  Continue: {output / 'model.zip'} (final optimization state)", file=sys.stderr)
    print(f"  Curve:    {output / 'training_curve.svg'}" + (" (no validation data)" if validation is None else ""), file=sys.stderr)
    print(f"  Log:      {output / 'train.log'}", file=sys.stderr)
    # A custom reward or parameter set may be incompatible with a fixed Benchmark.
    # In that case offer an ordinary comparison retaining the trained configuration.
    comparison_options = options
    if "tracking" in aiogym.list_benchmarks(scenario):
        benchmark_env = aiogym.make_env(scenario, benchmark="tracking")
        try:
            benchmark = environment_metadata(benchmark_env)
        finally:
            benchmark_env.close()
        if all(environment[field] == benchmark[field] for field in ENVIRONMENT_COMPATIBILITY_FIELDS):
            comparison_options = ["--benchmark", "tracking"]
    compare = ["aiogym", "evaluate", scenario, *comparison_options, "--controllers", "pid",
               "--checkpoint", algorithm, str(checkpoint), "--seeds", "0", "1", "2",
               "--output", str(output / "comparison")]
    print("  Compare on 3 cases (expand seeds for a formal report):\n    " + shlex.join(compare), file=sys.stderr)
    print("  Add 50,000 steps:\n    " + shlex.join(resume), file=sys.stderr)


def main(command, argv=None, *, _managed=False):
    parser = _parser(command)
    args = parser.parse_args(argv)
    parameters = _json_object_file(args.parameters, parser, "parameters")
    benchmark = None if command == "train" else args.benchmark
    env = None
    evaluation_env = None
    try:
        import aiogym

        if command == "train":
            from ._train_batch import prepare_training, run_training_batch

            jobs = prepare_training(args)
        env = aiogym.make_env(
            args.scenario,
            reward=args.reward,
            parameters=parameters,
            benchmark=benchmark,
            randomize=args.randomize,
            boundary_probability=args.boundary_probability,
            disturbance=args.disturbance == "on",
            noise=args.noise == "on",
            delay=args.delay == "on",
            fault=args.fault == "on",
        )
        if command == "collect":
            print(f"Dataset: {args.output.resolve()} (create in new or empty directory)", file=sys.stderr)
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
            if len(jobs) > 1:
                summary = run_training_batch(
                    args, jobs, env=env,
                    parameters=parameters, algorithm_kwargs=algorithm_kwargs,
                )
                for job in jobs:
                    if job["status"] == "succeeded":
                        metadata = json.loads((Path(job["output"]) / "metadata.json").read_text(encoding="utf-8"))
                        _print_training_result(job["output"], metadata)
                print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
                return {"completed": 0, "failed": 1, "cancelled": 130}[summary["status"]]
            if args.output is None:
                args.output = Path(jobs[0]["output"])
                args.output.parent.mkdir(parents=True, exist_ok=False)
            args.algorithm = jobs[0]["algorithm"]
            args.seed = jobs[0]["seed"]
            print(f"Training {args.algorithm}: {args.output.resolve()} (create new run directory)", file=sys.stderr)
            if args.evaluate_every is not None:
                evaluation_env = aiogym.make_env(
                    args.scenario,
                    reward=args.reward,
                    parameters=parameters,
                    randomize=True,
                )
            def train_one():
                return aiogym.train(
                    env=env,
                    algorithm=args.algorithm,
                    steps=args.steps,
                    seed=args.seed,
                    algorithm_kwargs=algorithm_kwargs,
                    record_every=args.record_every,
                    evaluation_env=evaluation_env,
                    evaluate_every=args.evaluate_every,
                    dataset=args.dataset,
                    behavior_cloning_epochs=args.behavior_cloning_epochs,
                    behavior_cloning_batch_size=args.behavior_cloning_batch_size,
                    behavior_cloning_learning_rate=(
                        args.behavior_cloning_learning_rate
                    ),
                    resume_from=args.resume_from,
                    output=args.output,
                )
            if _managed:
                result = train_one()
            else:
                from ._train_job import run_training_job

                output = Path(args.output)
                if output.exists() and any(output.iterdir()):
                    raise FileExistsError(f"training output must be empty: {output}")
                output.mkdir(parents=True, exist_ok=True)
                for name, value in (("parameters", parameters), ("algorithm-kwargs", algorithm_kwargs)):
                    if value is not None:
                        write_json(output / f"{name}.json", value)
                job = {**jobs[0], "command": ["aiogym", "train", *(argv or [])]}
                result = run_training_job(job, train_one)
            _print_training_result(args.output, result)

        else:
            if args.output is not None:
                print(f"Evaluation: {args.output.resolve()} (create new report directory)", file=sys.stderr)
            policies = _evaluation_policies(args, parser, aiogym, env)
            result = aiogym.evaluate(
                env=env,
                policies=policies,
                seeds=args.seeds,
                max_steps=args.max_steps,
                output=args.output,
            )
        summary = _success_summary(command, args, result)
    except (
        OSError,
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
