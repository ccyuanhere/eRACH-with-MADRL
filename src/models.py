import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class UTAgent: #UT智能体，采用AC方法
    def __init__(self, config):
        self.actor = ActorNetwork(config.state_dim, config.action_dim)
        self.critic = CriticNetwork(config.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.gamma = config.gamma
        self.beta_e = config.beta_e

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor).detach().numpy()[0]
        return np.random.choice(len(action_probs), p=action_probs)

    def update(self, transitions): #更新网络
        actor_loss, critic_loss, entropy = self._compute_losses(transitions)
        self.actor_optimizer.zero_grad()
        (actor_loss - self.beta_e * entropy).backward() #加入熵正则化
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _compute_losses(self, transitions):
        actor_loss = 0
        critic_loss = 0
        entropy = 0
        R = 0
        
        for state, action, reward, next_state, done in reversed(transitions):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            reward_t = torch.tensor([reward], dtype=torch.float)
            #计算A、C两个网络的损失
            value = self.critic(state_t)
            next_value = 0 if done else self.critic(next_state_t).detach()
            R = reward_t + self.gamma * next_value
            advantage = (R - value).detach()
            critic_loss += (value - R.detach()).pow(2)
            probs = self.actor(state_t)
            log_prob = torch.log(probs[0, action])
            actor_loss += -log_prob * advantage
            entropy -= (probs * probs.log()).sum()

        batch_size = len(transitions) #归一化
        return (actor_loss / batch_size, 
                critic_loss / batch_size, 
                entropy / batch_size)
