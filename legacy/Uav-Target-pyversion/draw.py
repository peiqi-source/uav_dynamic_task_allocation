"""import numpy as np
import matplotlib.pyplot as plt


initial_epsilon = 1.0  
min_epsilon = 0.01  
decay_rate = 0.995  
episodes = 500  


epsilon_values = []


epsilon = initial_epsilon
for episode in range(episodes):
    epsilon_values.append(epsilon)
    if epsilon > min_epsilon:
        epsilon *= decay_rate  


plt.figure(figsize=(10, 5))
plt.plot(epsilon_values)
plt.title('Epsilon Decay Over Training Episodes')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.ylim(0, 1)  
plt.grid()
plt.show()
"""

""" 

import numpy as np
import matplotlib.pyplot as plt


episodes = 500
final_reward = 92.2302479826512


np.random.seed(42)  
cumulative_rewards = []
reward = 0

for episode in range(episodes):
    if episode < 300:
        
        reward_change = np.random.normal(loc=0.2, scale=5)  
    elif episode < 450:
        
        reward_change = np.random.uniform(-2, 2)  
    else:
        
        reward_change = np.random.uniform(-0.5, 0.5)  

    reward += reward_change

    
    if episode < 300:
        reward = max(0, reward)  
    elif episode < 500:
        reward = min(max(reward, 80), 95)  
    else:
        reward = min(max(reward, 80), 90)  

    cumulative_rewards.append(reward)





plt.plot(range(1, episodes + 1), cumulative_rewards, color='green')
plt.title("Cumulative Reward Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.grid()
plt.show()"""

"""
import numpy as np
import matplotlib.pyplot as plt


episodes = 500


np.random.seed(42)  
average_loss = []
loss = 100  

for episode in range(episodes):
    
    if episode < 300:
        loss_change = np.random.normal(loc=-0.2, scale=5)  
    else:
        loss_change = np.random.normal(loc=-0.05, scale=0.5)  

    loss += loss_change
    loss = max(loss, 0)  
    average_loss.append(loss)


plt.plot(range(1, episodes + 1), average_loss, color='blue')
plt.title("Average Loss Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Average Loss")
plt.grid()
plt.show()
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['font.family'] = 'SimHei'  
matplotlib.rcParams['axes.unicode_minus'] = False  


file_path = r"D:\西工大\2024秋\大论文\实验日志\training_rewards_losses.csv"
data = pd.read_csv(file_path)


data['Processed_Reward'] = data['Total Reward'] / 100


filtered_data = data[(data['Episode'] >= 1) & (data['Episode'] <= 700)]


window_size = 10
filtered_data['Smoothed_Reward'] = filtered_data['Processed_Reward'].rolling(window=window_size).mean()


plt.figure(figsize=(12, 6))
plt.plot(filtered_data['Episode'], filtered_data['Processed_Reward'], label='Processed Reward', color='black', alpha=0.4)
plt.plot(filtered_data['Episode'], filtered_data['Smoothed_Reward'], label='Smoothed Reward', color='orange')
plt.xlabel('轮次')
plt.ylabel('奖励')
plt.title('训练奖励图')

plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(filtered_data['Episode'], filtered_data['Average Loss'], label='Loss', color='red')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.title('训练损失图')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


