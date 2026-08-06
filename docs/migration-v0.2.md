# Migrating from AIO-Gym 0.1 to 0.2

Version 0.2 replaces protocol composition with Scenario, PlantConfig, Task, and
Run.

| 0.1 concept | 0.2 replacement |
|---|---|
| Scenario + Case | `ScenarioPlugin` + `PlantConfig` + Task preset |
| Track / Goal / Anchor | explicit Task metrics and evaluation seeds |
| RewardSpec | reward owned by `TaskSpec` |
| Distribution / Curriculum | explicit workflow inputs |
| Dataset v2 | episode-oriented Dataset v3 |
| benchmark / tune / final-test | `evaluate` with explicit policy and seeds |
| run config | `train --config FILE` |

Environment construction changes from:

```python
aiogym.make_env("quadruple", case="minimum-phase", reward_spec="regulation")
```

to:

```python
aiogym.make_env("quadruple/regulation", preset="minimum-phase")
```

The common old call shape remains for one release and emits a
`DeprecationWarning`. Track loading, ranking, Claims, and artifact compatibility
are not emulated.

Dataset v2 bundles can be converted with:

```bash
python3 scripts/refactor/migrate_dataset_v2_to_v3.py OLD_DATASET NEW_DATASET
```

The converter preserves old provenance under `legacy_metadata`; it does not
invent a new validation claim. Commands removed in 0.2 print a pointer to this
guide.
