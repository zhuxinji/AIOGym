# AIO-Gym documentation

Start with the [Quickstart](quickstart.md), then use:

- [Architecture](architecture.md) for module and data-flow boundaries;
- [Reproducibility](reproducibility.md) for the metadata that is recorded;
- [RLPD](rlpd.md) for persistent Dataset plus online-replay training;
- [External algorithms](external_algorithms.md) for the training-backend contract.

Scenario-specific documentation lives under `docs/scenarios/`, with one page
named after each Scenario id:

- [Two-input, two-output CSTR](scenarios/cstr.md);
- [Batch crystallization](scenarios/crystallization.md);
- [Multistage extraction](scenarios/extraction.md);
- [Fired heater](scenarios/heater.md);
- [Two-zone HVAC](scenarios/hvac.md);
- [Quadruple-Tank](scenarios/quadruple.md);
- [Three-Tank](scenarios/three_tank.md).

The public pipeline is:

```text
make_env -> reset/step -> controller -> collect/train -> save/load -> evaluate/compare
```
