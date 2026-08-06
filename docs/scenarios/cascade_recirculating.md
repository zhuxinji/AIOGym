# Recirculating cascade

The recirculating cascade models the V2.0 three-vessel closed hydraulic loop:
Tank 3 -> P101 -> Tank 1 -> V12 -> Tank 2 -> V23 -> Tank 3. Tank 1 has the only
2 kW heater. Tank 1 and Tank 2 have passive overflow returns to Tank 3.

The executable design baseline uses 300 x 250 x 400 mm reference vessels,
0.30 m inter-stage elevation drops, a 5-25 L/min P101 range, and a 0.37 kW pump
motor. The level-transmitter span is 0-0.5 m and is intentionally separate from
the 0.40 m physical vessel height. Later procurement pages propose conflicting
60 L Tank 1/Tank 2 and 35 L Tank 3 candidates; those values are recorded as
unresolved provenance and are not mixed into the executable baseline.

The state is `[h1, T1, h2, T2, h3, T3]`. The normalized actuator vector is
`[pump_P101, valve_V12, valve_V23, heater_H1]`. V12 and V23 use upstream liquid
head plus the 0.30 m elevation drop. P101 uses a provisional affinity-law curve
fitted to the design maximum flow, 1.7 m static lift, and 10 m full-speed head.

The field-I/O metadata declares LT101/LT201/LT301, TT101/TT201/TT301, FT12/FT23,
six LSH/LSL digital inputs, VFD/Modbus P101 control, 4-20 mA/Modbus valve
commands, SSR H1 control, and the RS485-to-TCP gateway. Flow values and switch
states are reported in full process info; the six storage states remain the
controlled-output basis.

Hardware protection semantics are:

- LSL301 blocks P101 dry-running.
- LSH101 or LSH201 blocks P101 before passive overflow onset.
- LSL101 blocks H1 dry-fire.
- the provisional 80 degC L4 threshold blocks H1.
- passive Tank 1/Tank 2 overflow is recoverable; crossing the vessel wall or
  the 100 degC simulation boundary is a hard termination.

Bundled Cases:

- `cascade-recirculating/commissioning`
- `cascade-recirculating/temperature-step`
- `cascade-recirculating/disturbance-rejection`
- `cascade-recirculating/hydraulic-commissioning`
- `cascade-recirculating/safety-recovery`

`hydraulic-commissioning` represents Phase 1 of the installation plan and sets
`heater_power=0`. The ordinary `commissioning` and `temperature-step` Cases are
Phase 2 thermal experiments after H1 is installed and its interlocks are tested.

```python
import aiogym

env = aiogym.make_env(
    "cascade-recirculating",
    case="commissioning",
    reward_spec="regulation",
)
```

The scenario has no production-economic benchmark. For new regulation work use
`cascade-recirculating-regulation-generalist-v2`; it binds the v2 L2 training
distribution, hides robustness disturbances, isolates validation/test resolved
Case identities, and uses reviewed v3 fixed anchors. The v1 regulation Track is
retained for compatibility, and the recovery diagnostic remains outside ordinary
rankings. Passive protection events are reported separately from hard process
termination.

The model remains `design-provisional`. Installed tank dimensions, standpipe and
LSH/LSL elevations, V12/V23 flow curves, the P101 VFD-frequency/flow/head curve,
heat-loss coefficients, heater efficiency, and sensor/actuator dynamics require
commissioning data before calibrated-plant claims.

The scenario is pipeline-ready at L4. Its checked-in workflow covers V2 Dataset
collection, BC/SAC/RLPD training, digest-verified checkpoint reload, validation,
benchmarking, and one-shot final-test locking. The packaged `quick` profiles are
tutorial and smoke budgets; they are not publishable multi-seed performance
baselines and do not change the `design-provisional` physical status.
