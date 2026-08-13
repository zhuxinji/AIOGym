# AGENTS.md

- Do not preserve backward compatibility. Remove obsolete paths instead of
  adding compatibility layers, fallbacks, or migrations.
- Apply these rules while forming the execution plan, before editing code. Do
  not use a post-implementation review as a substitute for designing the
  simplest compliant solution up front.
- Choose the simplest implementation that fully meets the current
  requirements. Avoid speculative abstractions, configuration, and
  indirection.
- Grow the system in layers. Start from the smallest version that works end
  to end, and add each new capability on top of a product that already
  works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall
  complexity or improve reliability. Do not reimplement common
  functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own
  implementation or adding packages. Do not assume a library lacks a
  capability without checking its documentation and types.
- Fail loudly at system boundaries. Do not add fallback behavior, silent
  defaults, broad exception handling, or alternate paths "just in case".
  Required inputs, state, dependencies, and invariants must be validated and
  must raise a precise error when missing or invalid.
- Do not use `|| true`, `try/except: pass`, permissive `getattr(..., default)`
  or `.get(..., default)`, `x or fallback`, or equivalent patterns to hide a
  failed command, missing required state, or unsupported behavior. A default
  is allowed only when it is an explicit part of the public contract.
- Long-term architecture means stable and simple boundaries with replaceable
  implementations. It does not mean implementing future capabilities before
  they are required.

## Product scope

- The required end-to-end product is:

  `make_env -> reset/step -> controller -> collect -> train ->
  save/load -> evaluate/compare`

  Runtime code outside this path is out of scope unless the current task
  explicitly requires it.

- Pipeline completeness does not include formal statistical protocols,
  bootstrap confidence intervals, success gates, artifact hash chains,
  schema migrations, resumable or milestone training, file locks,
  replay-buffer recovery, or hardware runtime safety.

- Moving complexity out of README or hiding it behind an advanced API is not
  simplification. Remove unnecessary complexity from runtime code and tests,
  or move scenario-specific experimental behavior to the owning scenario or
  experimental package.

- Validate inputs once at public boundaries. Internal layers may rely on those
  validated invariants. Do not repeat the same validation across model,
  workflow, storage, and artifact layers.

- Core may contain only behavior shared by at least two scenarios.
  Scenario-specific observation transforms, residual actions, randomization,
  calibration, hardware, and safety logic must stay in the scenario package
  or experimental code.

- A Benchmark is a fixed evaluation protocol only. Training variation must not
  change observation dimensions, action dimensions, or the physical meaning of
  an action.

- Maintain one model interface, one evaluation path, one checkpoint format,
  and one dataset format. Do not maintain parallel simple/formal or
  current/legacy paths.

- Do not add an abstraction until at least two concrete call sites need it. A
  wrapper, registry, helper, base class, or configuration object must remove
  more complexity than it adds.

- Modular means cohesive ownership, not more files, forwarding adapters,
  helper layers, protocols, or configuration objects.

- Scenario models may accept explicit parameter overrides. Keep this as a
  direct validated mapping; do not introduce a generic design system,
  parameter registry, migration layer, or configuration DSL.

- Generated files, build outputs, run artifacts, caches, and historical
  architecture reports must not be committed as active source files.

Before adding a new abstraction, answer all three questions:

1. Which two existing concrete call sites need it?
2. Which existing branches or duplicate implementations will it delete?
3. Can the same result be achieved with a direct function or dataclass?

If these questions do not have concrete answers, do not add the abstraction.
