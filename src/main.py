import numpy as np
import torch
import logging
from collections import deque
from config import Config
from environment import LEOSatEnvironment
from models import UTAgent
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TrainingMetrics: #训练指标
    def __init__(self, window_size=100):
        self.reward_history = [] #历史奖励
        self.collision_history = [] #历史碰撞率
        self.throughput_history = [] #历史吞吐量
        self.recent_rewards = deque(maxlen=window_size) #近100次奖励，用于计算最近平均奖励
        
    def update(self, reward: float, collisions: float, throughput: float):
        self.reward_history.append(reward)
        self.collision_history.append(collisions)
        self.throughput_history.append(throughput)
        self.recent_rewards.append(reward)
        
    @property
    def average_reward(self) -> float:
        return np.mean(list(self.recent_rewards)) if self.recent_rewards else 0

def train_eRACH(config, num_episodes=300):
    env = LEOSatEnvironment(config) #环境
    agents = [UTAgent(config) for _ in range(config.J)] #每个UT是一个智能体
    metrics = TrainingMetrics()

    logging.info(f"Starting training with {num_episodes} episodes")
    
    for episode in range(num_episodes): #跑300集
        episode_metrics = run_episode(env, agents) #每集200步(200个RA)
        metrics.update(
            reward=episode_metrics['total_reward'],
            collisions=episode_metrics['collision_rate'],
            throughput=episode_metrics['throughput']
        )
        
        if (episode + 1) % 10 == 0: #每10个episode记录一次日志
            logging.info(
                f"Episode {episode+1}/{num_episodes} - "
                f"Reward: {episode_metrics['total_reward']:.2f}, "
                f"Avg Reward: {metrics.average_reward:.2f}, "
                f"Collision Rate: {episode_metrics['collision_rate']:.2%}, "
                f"Throughput: {episode_metrics['throughput']:.2f}"
            )
    
    logging.info("Training completed")
    return metrics

def run_episode(env, agents):
    states = env.reset()
    transitions_for_agents = [[] for _ in range(len(agents))] #记录每个智能体交互轨迹
    total_reward = 0
    collision_count = 0
    step_count = 0
    total_throughput = 0
    done = False
    
    while not done:
        actions = [agent.select_action(states[i]) for i, agent in enumerate(agents)] #根据状态生成动作
        next_states, rewards, done = env.step(actions) #执行动作
        total_reward += sum(rewards)
        collision_count += np.sum(env.collision_info)
        total_throughput += np.sum(env.throughput)
        step_count += 1
        #对每个智能体进行轨迹记录
        for i in range(len(agents)):
            transitions_for_agents[i].append(
                (states[i], actions[i], rewards[i], next_states[i], done)
            )
        states = next_states
    
    for agent, transitions in zip(agents, transitions_for_agents): #利用轨迹更新每个智能体神经网络参数
        agent.update(transitions)
    
    return {
        'total_reward': total_reward,
        'collision_rate': collision_count / (step_count * len(agents)) if step_count > 0 else 0,
        'throughput': total_throughput / step_count if step_count > 0 else 0
    }

if __name__ == "__main__":
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    config = Config()
    metrics = train_eRACH(config)
    
    #绘制训练指标
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(metrics.reward_history, label="Total Reward", color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward Over Episodes")
    plt.grid(True)
    plt.legend()
    metrics.collision_history = [collision * 100 for collision in metrics.collision_history]
    plt.subplot(3, 1, 2)
    plt.plot(metrics.collision_history, label="Collision Rate", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Collision Rate (%)")
    plt.ylim(0, 100)
    plt.title("Collision Rate Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(metrics.throughput_history, label="Throughput", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Throughput")
    plt.ylim(0, 3)
    plt.title("Throughput Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()