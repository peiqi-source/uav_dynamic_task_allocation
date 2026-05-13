# UAV Dynamic Task Allocation

This repository implements an event-driven UAV swarm task-level autonomous
decision system. The current engineering flow is aligned with the paper
pipeline:

1. Load UAV and target data from CSV files.
2. Build the battlefield state.
3. Screen candidate targets.
4. Select destroy-target set with Random Forest.
5. Build initial target regions with PSO.
6. Regroup targets after dynamic events with PPO, with PSO fallback.
7. Allocate UAV resources.
8. Assess resource sufficiency with Monte Carlo simulation.
9. Plan intra-cluster strike order with Attention-DQN, with heuristic fallback.
10. Simulate dynamic events, support decisions, replanning, trajectories,
    metrics and figures.

The main entry point is:

```bash
python scripts/run_full_simulation.py
```

## Quick Validation

Run from the project root:

```bash
pip install -e .

python scripts/check_config.py
python scripts/check_data.py
python scripts/check_entities.py
python scripts/check_contracts.py
python scripts/check_destroy_target_selection.py
python scripts/check_target_clustering.py
python scripts/check_resource_allocation.py
python scripts/check_monte_carlo_resource_evaluator.py
python scripts/check_strike_order_env.py
python scripts/check_support_policy.py
python scripts/check_replanning_controller.py
python scripts/check_mission_simulator.py

python scripts/run_full_simulation.py
python scripts/run_all_experiments.py
pytest -q
```

The system is designed to run on CPU. If PyTorch or model checkpoints are not
available, the main workflow falls back safely:

- RF checkpoint missing: use configured RF fallback rule.
- PPO checkpoint missing or PyTorch unavailable: fall back to PSO clustering.
- Attention-DQN checkpoint missing or PyTorch unavailable: fall back to the
  configured heuristic strike-order method.

All fallback behavior is logged.

## Scenario Experiments

Scenario configurations are stored in `configs/scenarios/`.

```bash
python scripts/run_static_scenario.py
python scripts/run_target_removed_scenario.py
python scripts/run_target_added_scenario.py
python scripts/run_uav_lost_scenario.py
python scripts/run_comprehensive_dynamic_scenario.py
python scripts/run_all_experiments.py
```

The five supported scenarios are:

- `static`: RF + PSO + Monte Carlo + Attention-DQN + MissionSimulator.
- `target_removed`: target disappearance, local release and replanning.
- `target_added`: new high-value target, RF decision, PPO regrouping,
  Monte Carlo assessment and replanning.
- `uav_lost`: attack, recon/guide and communication UAV loss, support policy
  and resource reassignment.
- `comprehensive_dynamic`: target removed, target added and UAV loss events.

Batch experiment summary:

```text
outputs/experiments/all_experiments_summary.csv
```

## Training And Evaluation

```bash
python scripts/train_destroy_target_rf.py
python scripts/compare_destroy_target_selection.py

python scripts/train_ppo_clusterer.py
python scripts/evaluate_ppo_clusterer.py

python scripts/train_strike_order_dqn.py
python scripts/evaluate_strike_order_dqn.py
```

Main checkpoints:

- `checkpoints/random_forest/destroy_target_rf.joblib`
- `checkpoints/ppo_clusterer/`
- `checkpoints/strike_order_dqn/`

## Outputs

A full simulation run writes to:

```text
outputs/full_simulation/<run_name_timestamp>/
```

Scenario experiments write to:

```text
outputs/experiments/<scenario_name>/
```

Each run stores:

- `effective_config.yaml`
- `mission_planner_summary.csv`
- `simulation/mission_simulation_log.csv`
- `simulation/mission_simulation_summary.csv`
- `simulation/mission_simulation_timeline_metrics.csv`
- `simulation/uav_trajectory_log.csv`
- `simulation/dynamic_events.csv`
- `simulation/support_decisions.csv`
- `simulation/resource_assessment.csv`
- `simulation/resource_assessment_samples.csv`
- `simulation/figures/*.png`
- `final_summary.json`

See `docs/experiment_reproduction.md` for the paper-to-code mapping and
reproduction checklist.
