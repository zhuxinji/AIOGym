# Model and Case infrastructure

Every process model implements `ProcessModelContract`: dynamics, initial state,
actions, controlled outputs, bounds, disturbances, metadata, and optional
economics. Numerical parameters can be loaded through versioned parameter
profiles without mutating bundled models.

Cases are immutable experiment declarations under
`aiogym/models/cases/builtin/<scenario>/`. A Case must provide:

- `schema_version`, `name`, `scenario`, and `environment`;
- valid timing and observation/action settings;
- optional initialization, setpoints, disturbances, constraints, operation,
  model parameters, acceptance, and evaluation hints.

Cases cannot bind Goal, RewardSpec, or official controller defaults.
Case-specific specialist tuning metadata may be recorded under
`specialist_metadata` but is not part of a generalist Track policy contract.

`CaseSpec` stores a canonical JSON snapshot and exposes a stable SHA-256 hash.
`apply_case_overrides()` accepts only Case-owned sections, validates the result,
and never mutates the source profile.

The environment created by `make_env()` resolves settings in this order:

1. config-mode environment values;
2. selected Case environment values;
3. Scenario defaults.

Goal is derived separately from RewardSpec. Official Track validation checks
that all Cases share the declared action/observation/control-time contract.

Before adding a model or Case:

1. validate numerical parameter provenance;
2. check dimensions, finite values, bounds, and metadata;
3. verify conservation or balance equations where applicable;
4. compare numeric and symbolic dynamics when supported;
5. add deterministic reset/step and Case schedule tests;
6. add a Track only after the policy contract is stable.
