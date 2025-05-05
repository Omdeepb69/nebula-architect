import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

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
    relationships: Dict[int, float]  # Agent ID -> relationship value
    current_task: Optional[str]
    inventory: Dict[str, int]

class AgentModel(TorchModelV2, nn.Module):
    """Neural network model for agent decision making."""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Define neural network architecture
        self.encoder = nn.Sequential(
            nn.Linear(obs_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._cur_value = None
    
    def forward(self, input_dict, state, seq_lens):
        features = self.encoder(input_dict["obs"])
        self._cur_value = self.value_head(features)
        return self.policy_head(features), state
    
    def value_function(self):
        return self._cur_value.squeeze(1)

class AgentSystem:
    """Manages AI agents and their behaviors in the virtual world."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("NebulaArchitect.AgentSystem")
        self._initialize_system()
        self.agents: Dict[int, AgentState] = {}
        
    def _initialize_system(self):
        """Initialize the agent system and RL framework."""
        try:
            # Initialize Ray
            ray.init(ignore_reinit_error=True)
            
            # Register custom model
            ModelCatalog.register_custom_model("agent_model", AgentModel)
            
            # Initialize RL trainer
            self.trainer = PPOTrainer(
                env="agent_env",
                config={
                    "model": {
                        "custom_model": "agent_model",
                        "custom_model_config": {}
                    },
                    "framework": "torch",
                    "num_workers": self.config["system"]["num_workers"],
                    "train_batch_size": self.config["system"]["batch_size"]
                }
            )
            
            self.logger.info("Agent system initialized successfully")
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
            agent = AgentState(
                id=agent_id,
                position=position,
                orientation=orientation,
                velocity=np.zeros(3),
                health=initial_health,
                energy=initial_energy,
                mood=0.5,  # Neutral mood
                relationships={},
                current_task=None,
                inventory={}
            )
            self.agents[agent_id] = agent
            return agent_id
        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            raise
    
    def remove_agent(self, agent_id: int):
        """Remove an agent from the system."""
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
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
            
            # Update relationship in both directions
            self.agents[agent_id].relationships[other_id] = \
                self.agents[agent_id].relationships.get(other_id, 0.0) + delta
            self.agents[other_id].relationships[agent_id] = \
                self.agents[other_id].relationships.get(agent_id, 0.0) + delta
        except Exception as e:
            self.logger.error(f"Failed to update relationships: {e}")
            raise
    
    def get_nearby_agents(self, agent_id: int, radius: float) -> List[int]:
        """Get IDs of agents within a certain radius."""
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent_pos = self.agents[agent_id].position
            nearby = []
            
            for other_id, other_agent in self.agents.items():
                if other_id != agent_id:
                    distance = np.linalg.norm(agent_pos - other_agent.position)
                    if distance <= radius:
                        nearby.append(other_id)
            
            return nearby
        except Exception as e:
            self.logger.error(f"Failed to get nearby agents: {e}")
            raise
    
    def train_agents(self, num_iterations: int = 1000):
        """Train agents using reinforcement learning."""
        try:
            for i in range(num_iterations):
                result = self.trainer.train()
                if i % 100 == 0:
                    self.logger.info(f"Training iteration {i}: {result}")
        except Exception as e:
            self.logger.error(f"Failed to train agents: {e}")
            raise
    
    def cleanup(self):
        """Clean up agent system resources."""
        try:
            ray.shutdown()
            self.logger.info("Agent system resources cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup agent system resources: {e}")
            raise 