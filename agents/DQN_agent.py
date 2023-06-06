import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in batch])
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.ByteTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.8, epsilon=0.8, epsilon_decay=0.9995, epsilon_min=0.01, tau=1e-3, update_target_freq=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.update_target_freq = update_target_freq
        self.printLoss = float('inf')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(state_dim, action_dim).to(self.device)
        self.target_dqn = DQN(state_dim, action_dim).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=15000)
        self.t_step = 0

    def select_action(self, state):
        
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            obs = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.dqn(obs)
            return q_values.argmax(1).item()

    # def update_target_network(self):
    #     self.target_dqn.load_state_dict(self.dqn.state_dict())
    
    # update the target network soft update
    def update_target_network_soft(self, tau):
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)
            
    
    
    def act(self, state):
        state = preprocess_state(state)  # Flatten the state
        with torch.no_grad():
            obs = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.dqn(obs)
            return q_values.argmax(1).item()
    def optimize_model(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update the target network
        self.update_target_network_soft(self.tau)
        self.printLoss = loss.item()

    def train(self, env, num_episodes, batch_size, save_path='dqn_model.pt', render=False):
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0

            while not done and not truncated:
                state = preprocess_state(state) # flatten the obs state
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                # if info['crashed']:
                    # reward = -100
                if render:
                    env.render()
                next_state = preprocess_state(next_state) # flatten the obs next state

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Optimize the model every few steps
                self.t_step = (self.t_step + 1) % self.update_target_freq
                if self.t_step == 0:
                    self.optimize_model(batch_size)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            # if episode % self.update_target_freq == 0:
            #     print("Updating target network...")
            #     self.update_target_network_soft(self.tau)

            print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}, Loss: {:.2}".format(
                episode + 1, num_episodes, total_reward, self.epsilon, self.printLoss))
        # Save the model weights
        torch.save(self.dqn.state_dict(), save_path)
        print("Successfully saved model weights at {}".format(save_path))
        if render:
            env.close()
def preprocess_state(state):
    state = state.flatten()  # Flatten the 5x5 matrix
    return state 