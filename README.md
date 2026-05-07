# uav\_dynamic\_task\_allocation

\# UAV Dynamic Task Allocation



This project is a modular research engineering framework for UAV swarm dynamic task allocation, target grouping, reinforcement learning-based decision making, dynamic event response, and result visualization.



\## Project Structure



\- `configs/`: configuration files

\- `data/`: raw and processed data

\- `src/`: source code

\- `scripts/`: executable scripts

\- `tests/`: unit tests

\- `experiments/`: experiment records

\- `outputs/`: figures, tables, and results

\- `logs/`: running logs

\- `checkpoints/`: saved models

\- `legacy/`: original scripts before refactoring



\## Current Stage



The current stage focuses on refactoring legacy research scripts into a clean, configurable, and reproducible engineering project.



1. 数据层
CSV
  ↓
DataFrame
  ↓
UAV / Target

2. 状态层
UAV / Target
  ↓
BattlefieldState

3. 环境层
BattlefieldState + EnvConfig
  ↓
DroneBattleEnv

4. 观测层
BattlefieldState
  ↓
ObservationBuilder
  ↓
obs vector

5. 动作层
action_id
  ↓
ActionSpace
  ↓
{"uav_id": ..., "target_id": ...}

6. 奖励层
env.step(action)
  ↓
RewardCalculator
  ↓
reward / reward_breakdown

7. 经验层
obs, action_id, reward, next_obs, done
  ↓
ReplayBuffer
