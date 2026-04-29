import gym
from gym import spaces
import numpy as np

class DroneBattleEnv(gym.Env):
    def __init__(self):
        super(DroneBattleEnv, self).__init__()

        
        self.num_drones = 9
        self.num_targets = 20
        self.num_recon_drones = 3  

        self.action_space = spaces.MultiDiscrete([self.num_drones, self.num_targets])  

        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_drones * 2 + self.num_targets * 5,),
                                            dtype=np.float32)

        '''
        
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_drones*2 + self.num_recon_drones*2 + self.num_targets*5,), dtype=np.float32)
        '''

        self.state = self._init_state()

    def _init_state(self):
        drones = np.random.rand(self.num_drones, 2) * 100  
        
        targets = np.random.rand(self.num_targets, 2) * 100  
        target_exists = np.ones((self.num_targets, 1))  
        target_defense = np.random.rand(self.num_targets, 1) * 100  
        target_value = np.random.rand(self.num_targets, 1) * 100  
        return np.concatenate([drones.flatten(), targets.flatten(), target_exists.flatten(), target_defense.flatten(),
                               target_value.flatten()])

    def step(self, action):
        
        self.state = self._update_state(action)
        reward = self._compute_reward(action)
        done = self._check_done()
        return self.state, reward, done, {}

    def reset(self):
        self.state = self._init_state()
        return self.state

    def _compute_reward(self, action):
        
        reward = 0
        drone_idx, target_idx = action
        target_status_idx = self.num_drones*2 + self.num_targets*2 + target_idx
        target_defense_idx = self.num_drones*2 + self.num_targets*3 + target_idx
        target_value_idx = self.num_drones*2 + self.num_targets*4 + target_idx

        if self.state[target_status_idx] == 1:
            defense = self.state[target_defense_idx]
            value = self.state[target_value_idx]
            reward = value - defense  
            self.state[target_status_idx] = 0  

        return reward

    def _check_done(self):
        
        target_status = self.state[self.num_drones*2 + self.num_targets*2:]
        return np.all(target_status == 0)

    def _update_state(self, action):
        drone_idx, target_idx = action
        drone_position_idx = drone_idx * 2
        target_position_idx = self.num_drones*2 + target_idx * 2
        target_status_idx = self.num_drones*2 + self.num_targets*2 + target_idx

        self.state[drone_position_idx:drone_position_idx+2] = self.state[target_position_idx:target_position_idx+2]

        self._dynamic_changes()

        return self.state

    def _dynamic_changes(self):
        
        
        if np.random.rand() < 0.1:  
            new_target_idx = np.random.choice(np.where(self.state[self.num_drones*2 + self.num_targets*2:] == 0)[0])
            self.state[self.num_drones*2 + new_target_idx*2:self.num_drones*2 + (new_target_idx+1)*2] = np.random.rand(2) * 100
            self.state[self.num_drones*2 + self.num_targets*2 + new_target_idx] = 1
            self.state[self.num_drones*2 + self.num_targets*3 + new_target_idx] = np.random.rand() * 100
            self.state[self.num_drones*2 + self.num_targets*4 + new_target_idx] = np.random.rand() * 100

        if np.random.rand() < 0.05:  
            disappearing_target_idx = np.random.choice(np.where(self.state[self.num_drones*2 + self.num_targets*2:] == 1)[0])
            self.state[self.num_drones*2 + self.num_targets*2 + disappearing_target_idx] = 0

        if np.random.rand() < 0.05:  
            destroyed_drone_idx = np.random.randint(0, self.num_drones)
            self.state[destroyed_drone_idx*2:destroyed_drone_idx*2+2] = -1  

    def render(self, mode='human'):
        
        pass
