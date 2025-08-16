import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
import random
import json
from collections import deque, defaultdict
import time

@dataclass
class AgentState:
    """Represents the state of an AI agent."""
    id: int
    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    health: float
    energy: float
    mood: float
    relationships: Dict[int, float] = field(default_factory=dict)  # Agent ID -> relationship value
    current_task: Optional[str] = None
    inventory: Dict[str, int] = field(default_factory=dict)
    
    # Additional behavioral properties
    personality_traits: Dict[str, float] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    learned_behaviors: Dict[str, float] = field(default_factory=dict)

class AgentModel(nn.Module):
    """Neural network model for agent decision making without Ray dependency."""
    
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super(AgentModel, self).__init__()
        
        # Define neural network architecture
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Policy network (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Value network (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self._cur_value = None
    
    def forward(self, observation):
        """Forward pass through the network."""
        features = self.encoder(observation)
        
        # Get policy (action probabilities) and value
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        self._cur_value = value
        return policy_logits, value
    
    def get_action(self, observation, deterministic=False):
        """Get action from observation."""
        with torch.no_grad():
            policy_logits, value = self.forward(observation)
            
            if deterministic:
                action = torch.argmax(policy_logits, dim=-1)
            else:
                # Sample from policy distribution
                action_dist = torch.distributions.Categorical(policy_logits)
                action = action_dist.sample()
            
            return action.item(), policy_logits, value
    
    def evaluate_actions(self, observations, actions):
        """Evaluate actions for training."""
        policy_logits, values = self.forward(observations)
        action_dist = torch.distributions.Categorical(policy_logits)
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return log_probs, entropy, values

class SimpleRLTrainer:
    """Simple reinforcement learning trainer using PPO-like algorithm."""
    
    def __init__(self, model, lr=3e-4, clip_range=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Training data storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store a training transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self, gamma=0.99, lam=0.95):
        """Compute advantages using Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        last_advantage = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            advantage = delta + gamma * lam * (1 - self.dones[i]) * last_advantage
            
            advantages.insert(0, advantage)
            returns.insert(0, advantage + self.values[i])
            last_advantage = advantage
        
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
    
    def update(self, epochs=4):
        """Update the model using collected data."""
        if len(self.observations) == 0:
            return {"loss": 0.0}
        
        # Convert lists to tensors
        observations = torch.stack(self.observations)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(epochs):
            # Get current policy and value estimates
            new_log_probs, entropy, new_values = self.model.evaluate_actions(observations, actions)
            new_values = new_values.squeeze()
            
            # Compute policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Compute entropy loss (for exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear stored data
        self.clear_buffer()
        
        return {
            "loss": total_loss / epochs,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item()
        }
    
    def clear_buffer(self):
        """Clear the training buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

class AgentEnvironment:
    """Simulated environment for agent training."""
    
    def __init__(self, world_size: Tuple[float, float, float] = (100.0, 10.0, 100.0)):
        self.world_size = world_size
        self.agents = {}
        self.objects = []
        self.time_step = 0
        self.max_steps = 1000
    
    def reset(self):
        """Reset the environment."""
        self.time_step = 0
        return self.get_observations()
    
    def step(self, actions: Dict[int, int]):
        """Step the environment forward."""
        rewards = {}
        dones = {}
        infos = {}
        
        # Execute actions for each agent
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                reward = self._execute_action(agent_id, action)
                rewards[agent_id] = reward
                dones[agent_id] = self.time_step >= self.max_steps
                infos[agent_id] = {}
        
        self.time_step += 1
        observations = self.get_observations()
        
        return observations, rewards, dones, infos
    
    def _execute_action(self, agent_id: int, action: int) -> float:
        """Execute an action and return reward."""
        agent = self.agents[agent_id]
        reward = 0.0
        
        # Define action space: 0=idle, 1=move_forward, 2=turn_left, 3=turn_right, 4=interact
        if action == 1:  # Move forward
            # Move agent forward
            move_distance = 1.0
            dx = np.cos(agent.orientation[1]) * move_distance
            dz = np.sin(agent.orientation[1]) * move_distance
            
            new_pos = agent.position + np.array([dx, 0, dz])
            
            # Check bounds
            if (0 <= new_pos[0] <= self.world_size[0] and 
                0 <= new_pos[2] <= self.world_size[2]):
                agent.position = new_pos
                reward = 0.1  # Small reward for exploration
            else:
                reward = -0.5  # Penalty for hitting boundaries
        
        elif action == 2:  # Turn left
            agent.orientation[1] -= np.pi / 4
            reward = 0.05
        
        elif action == 3:  # Turn right
            agent.orientation[1] += np.pi / 4
            reward = 0.05
        
        elif action == 4:  # Interact
            # Check for nearby agents or objects
            nearby_agents = self._get_nearby_agents(agent_id, radius=5.0)
            if nearby_agents:
                reward = 0.5  # Reward for social interaction
                agent.mood = min(1.0, agent.mood + 0.1)
            else:
                reward = -0.1  # Small penalty for failed interaction
        
        # Energy and health dynamics
        agent.energy = max(0, agent.energy - 0.5)
        if agent.energy < 20:
            reward -= 0.2  # Penalty for low energy
        
        return reward
    
    def _get_nearby_agents(self, agent_id: int, radius: float) -> List[int]:
        """Get nearby agents within radius."""
        if agent_id not in self.agents:
            return []
        
        agent_pos = self.agents[agent_id].position
        nearby = []
        
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id:
                distance = np.linalg.norm(agent_pos - other_agent.position)
                if distance <= radius:
                    nearby.append(other_id)
        
        return nearby
    
    def get_observations(self) -> Dict[int, torch.Tensor]:
        """Get observations for all agents."""
        observations = {}
        
        for agent_id, agent in self.agents.items():
            # Create observation vector
            obs = np.concatenate([
                agent.position,  # 3 elements
                agent.orientation,  # 3 elements
                agent.velocity,  # 3 elements
                [agent.health / 100.0],  # 1 element (normalized)
                [agent.energy / 100.0],  # 1 element (normalized)
                [agent.mood],  # 1 element
                [len(self._get_nearby_agents(agent_id, 10.0))],  # 1 element (nearby agents count)
                [self.time_step / self.max_steps]  # 1 element (time progress)
            ])
            
            observations[agent_id] = torch.tensor(obs, dtype=torch.float32)
        
        return observations

class AgentSystem:
    """Manages AI agents and their behaviors in the virtual world without Ray dependency."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("NebulaArchitect.AgentSystem")
        self.agents: Dict[int, AgentState] = {}
        
        # Configuration with defaults
        self.system_config = config.get("agent_system", {
            "num_workers": 4,
            "batch_size": 256,
            "max_agents": 50,
            "observation_size": 13,  # Based on observation vector size
            "action_size": 5,  # idle, move_forward, turn_left, turn_right, interact
            "hidden_size": 256,
            "learning_rate": 3e-4
        })
        
        # Initialize environment and models
        self.environment = AgentEnvironment()
        self._initialize_system()
        
        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0.0,
            "average_reward": 0.0,
            "training_iterations": 0
        }
        
        self.logger.info("Agent system initialized successfully (Ray-free)")
    
    def _initialize_system(self):
        """Initialize the agent system and learning framework."""
        try:
            # Create neural network model
            self.model = AgentModel(
                obs_size=self.system_config["observation_size"],
                action_size=self.system_config["action_size"],
                hidden_size=self.system_config["hidden_size"]
            ).to(self.device)
            
            # Initialize trainer
            self.trainer = SimpleRLTrainer(
                model=self.model,
                lr=self.system_config["learning_rate"]
            )
            
            self.logger.info("Agent learning system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent system: {e}")
            raise
    
    def create_agent(self, 
                    position: np.ndarray,
                    orientation: np.ndarray,
                    initial_health: float = 100.0,
                    initial_energy: float = 100.0) -> int:
        """Create a new AI agent."""
        try:
            agent_id = len(self.agents)
            
            # Generate random personality traits
            personality_traits = {
                "curiosity": random.uniform(0.0, 1.0),
                "sociability": random.uniform(0.0, 1.0),
                "aggressiveness": random.uniform(0.0, 0.3),  # Keep low for peaceful agents
                "intelligence": random.uniform(0.5, 1.0),
                "creativity": random.uniform(0.0, 1.0)
            }
            
            agent = AgentState(
                id=agent_id,
                position=position.copy(),
                orientation=orientation.copy(),
                velocity=np.zeros(3),
                health=initial_health,
                energy=initial_energy,
                mood=0.5,  # Neutral mood
                relationships={},
                current_task=None,
                inventory={},
                personality_traits=personality_traits,
                memory=[],
                goals=["explore", "socialize", "survive"],
                learned_behaviors={}
            )
            
            self.agents[agent_id] = agent
            self.environment.agents[agent_id] = agent
            
            self.logger.info(f"Created agent {agent_id} with personality: {personality_traits}")
            return agent_id
        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            raise
    
    def remove_agent(self, agent_id: int):
        """Remove an agent from the system."""
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                if agent_id in self.environment.agents:
                    del self.environment.agents[agent_id]
                self.logger.info(f"Removed agent {agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to remove agent: {e}")
            raise
    
    def update_agent_state(self, agent_id: int, new_state: Dict[str, Any]):
        """Update an agent's state."""
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.agents[agent_id]
            for key, value in new_state.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
        except Exception as e:
            self.logger.error(f"Failed to update agent state: {e}")
            raise
    
    def get_agent_state(self, agent_id: int) -> AgentState:
        """Get the current state of an agent."""
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            return self.agents[agent_id]
        except Exception as e:
            self.logger.error(f"Failed to get agent state: {e}")
            raise
    
    def update_relationships(self, agent_id: int, other_id: int, delta: float):
        """Update the relationship value between two agents."""
        try:
            if agent_id not in self.agents or other_id not in self.agents:
                raise ValueError("One or both agents not found")
            
            # Update relationship in both directions with some variation
            self.agents[agent_id].relationships[other_id] = \
                np.clip(self.agents[agent_id].relationships.get(other_id, 0.0) + delta, -1.0, 1.0)
            
            # Other agent might have different reaction based on their personality
            other_delta = delta * (0.5 + self.agents[other_id].personality_traits.get("sociability", 0.5))
            self.agents[other_id].relationships[agent_id] = \
                np.clip(self.agents[other_id].relationships.get(agent_id, 0.0) + other_delta, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to update relationships: {e}")
            raise
    
    def get_nearby_agents(self, agent_id: int, radius: float) -> List[int]:
        """Get IDs of agents within a certain radius."""
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            return self.environment._get_nearby_agents(agent_id, radius)
        except Exception as e:
            self.logger.error(f"Failed to get nearby agents: {e}")
            raise
    
    def step_simulation(self, dt: float = 0.1):
        """Step the agent simulation forward."""
        try:
            if not self.agents:
                return
            
            # Get observations
            observations = self.environment.get_observations()
            
            # Get actions from model for each agent
            actions = {}
            transitions = {}
            
            for agent_id in self.agents.keys():
                if agent_id in observations:
                    obs = observations[agent_id].to(self.device)
                    action, policy_logits, value = self.model.get_action(obs)
                    actions[agent_id] = action
                    
                    # Store for training
                    transitions[agent_id] = {
                        'obs': obs,
                        'action': action,
                        'policy_logits': policy_logits,
                        'value': value
                    }
            
            # Step environment
            next_observations, rewards, dones, infos = self.environment.step(actions)
            
            # Store transitions for training
            for agent_id, transition in transitions.items():
                if agent_id in rewards:
                    self.trainer.store_transition(
                        obs=transition['obs'],
                        action=transition['action'],
                        reward=rewards[agent_id],
                        value=transition['value'],
                        log_prob=torch.log(transition['policy_logits'][transition['action']]),
                        done=dones.get(agent_id, False)
                    )
            
            # Update agent memories and behaviors
            self._update_agent_behaviors(rewards, dt)
            
        except Exception as e:
            self.logger.error(f"Error in simulation step: {e}")
    
    def _update_agent_behaviors(self, rewards: Dict[int, float], dt: float):
        """Update agent behaviors based on rewards and interactions."""
        for agent_id, agent in self.agents.items():
            if agent_id in rewards:
                reward = rewards[agent_id]
                
                # Update mood based on reward
                agent.mood += reward * 0.1
                agent.mood = np.clip(agent.mood, -1.0, 1.0)
                
                # Add to memory
                memory_entry = {
                    'timestamp': time.time(),
                    'position': agent.position.copy(),
                    'reward': reward,
                    'mood': agent.mood,
                    'action_taken': True
                }
                
                agent.memory.append(memory_entry)
                
                # Keep memory size manageable
                if len(agent.memory) > 100:
                    agent.memory.pop(0)
                
                # Update learned behaviors based on success
                if reward > 0.1:
                    current_behavior = agent.current_task or "explore"
                    agent.learned_behaviors[current_behavior] = \
                        agent.learned_behaviors.get(current_behavior, 0.0) + 0.1
    
    def train_agents(self, num_iterations: int = 100):
        """Train agents using the collected experience."""
        try:
            total_loss = 0
            successful_updates = 0
            
            for i in range(num_iterations):
                # Run simulation episodes to collect data
                for episode in range(5):  # Multiple episodes per iteration
                    self.environment.reset()
                    for step in range(50):  # Steps per episode
                        self.step_simulation()
                
                # Update model
                result = self.trainer.update()
                if result['loss'] > 0:
                    total_loss += result['loss']
                    successful_updates += 1
                
                # Log progress
                if i % 20 == 0 and i > 0:
                    avg_loss = total_loss / max(1, successful_updates)
                    self.logger.info(f"Training iteration {i}: Average loss = {avg_loss:.4f}")
                    
                    # Update training stats
                    self.training_stats['training_iterations'] = i
                    self.training_stats['average_loss'] = avg_loss
            
            # Final training statistics
            self.training_stats['training_iterations'] += num_iterations
            self.logger.info(f"Completed {num_iterations} training iterations")
            
        except Exception as e:
            self.logger.error(f"Failed to train agents: {e}")
            raise
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'training_stats': self.training_stats,
                'config': self.system_config
            }, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return self.training_stats.copy()
    
    def get_all_agent_states(self) -> List[Dict[str, Any]]:
        """Get state information for all agents."""
        return [
            {
                'id': agent.id,
                'position': agent.position.tolist(),
                'orientation': agent.orientation.tolist(),
                'velocity': agent.velocity.tolist(),
                'health': agent.health,
                'energy': agent.energy,
                'mood': agent.mood,
                'current_task': agent.current_task,
                'personality_traits': agent.personality_traits,
                'num_relationships': len(agent.relationships),
                'memory_size': len(agent.memory)
            }
            for agent in self.agents.values()
        ]
    
    def cleanup(self):
        """Clean up agent system resources."""
        try:
            # Clear all data structures
            self.agents.clear()
            self.environment.agents.clear()
            
            # Clear training buffer
            self.trainer.clear_buffer()
            
            # Clear GPU memory if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            self.logger.info("Agent system resources cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup agent system resources: {e}")
            raise