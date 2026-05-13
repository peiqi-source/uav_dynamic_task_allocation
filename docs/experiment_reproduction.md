# Experiment Reproduction Guide

## Algorithm Mapping

| Paper module | Engineering module | Notes |
| --- | --- | --- |
| RF destroy-target decision | `preprocessing/destroy_target_selection.py`, `rf_destroy_selector.py` | RF checkpoint missing falls back to configured rule/top-k path. |
| PSO initial clustering | `allocation/pso_clusterer.py`, `allocation/target_clustering.py` | Default initial clustering method. |
| PPO dynamic regrouping | `envs/target_grouping_env.py`, `algorithms/rl/ppo/`, `allocation/ppo_clusterer.py` | Dynamic events prefer PPO; missing checkpoint falls back to PSO. |
| Monte Carlo resource assessment | `allocation/monte_carlo_resource_evaluator.py` | Runs after resource allocation and writes cluster-level assessment plus samples. |
| DQN / Attention-DQN strike order | `algorithms/rl/dqn/`, `planning/strike_order_planner.py` | Default method is Attention-DQN; missing checkpoint falls back to heuristic. |
| Dynamic support | `planning/support_policy.py`, `planning/replanning_controller.py` | Triggered by low Monte Carlo completion probability and dynamic events. |
| Mission simulation | `simulation/mission_simulator.py` | Produces event logs, trajectory logs, metrics and figures. |

## End-to-End Validation

Run these commands from the project root:

```bash
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
pytest -q
```

The full simulation should not fail when RF, PPO or Attention-DQN checkpoints are absent. Fallback decisions are written to logs and result metadata.

## Five Scenario Experiments

```bash
python scripts/run_all_experiments.py
```

This runs:

- `static`
- `target_removed`
- `target_added`
- `uav_lost`
- `comprehensive_dynamic`

Scenario outputs are stored in `outputs/experiments/<scenario_name>/`.

## Expected CSV Outputs

Each scenario should include:

- `mission_planner_summary.csv`
- `simulation/mission_simulation_log.csv`
- `simulation/mission_simulation_summary.csv`
- `simulation/mission_simulation_timeline_metrics.csv`
- `simulation/uav_trajectory_log.csv`
- `simulation/dynamic_events.csv`
- `simulation/support_decisions.csv`
- `simulation/resource_assessment.csv`
- `simulation/resource_assessment_samples.csv`
- `final_summary.json`

## Expected Figures

Each scenario should include PNG files under `simulation/figures/`, including:

- destroyed targets over time
- available targets over time
- available UAVs over time
- replanning count over time
- cluster completion ratio over time
- event count over time
- Monte Carlo completion distribution
- UAV spatial trajectories
- UAV status timeline
- event/replanning timeline
- final target status map

## Checkpoint Behavior

- RF missing or incompatible model: `random_forest_fallback_*`
- PPO missing checkpoint: `ppo_fallback_pso`
- Attention-DQN missing checkpoint or PyTorch unavailable: configured `fallback_method`, default `nearest_neighbor`

These behaviors are expected in a fresh CPU-only environment and are part of the engineering reproducibility design.
