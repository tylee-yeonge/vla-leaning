# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a VLA (Vision-Language-Action) learning repository following a **top-down learning approach**. The repository contains educational materials for learning VLA implementation through hands-on practice rather than traditional bottom-up theoretical study.

## Learning Philosophy

**Top-Down vs Bottom-Up:**
- This repository embraces a top-down methodology: start with complete VLA papers/implementations, then dive into specific concepts as needed
- Goal: Get working VLA code running within 1 week, full implementation within 8 weeks
- Key principle: 30% understanding is sufficient to proceed; perfect understanding is not required

## Key Components

### Documentation
- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)**: Comprehensive 8-week learning roadmap in Korean covering:
  - Week 1: Understanding VLA overview (RT-1 paper, LeRobot demos)
  - Week 2-3: Essential concepts (Transformers, Behavioral Cloning, PyTorch)
  - Week 4-6: Building Mini VLA from scratch with PyBullet
  - Week 7-8: RT-1 re-implementation and LeRobot mastery
  - Week 9+: Isaac Sim integration for logistics applications

### Target Projects
The learning path progresses through:
1. **Mini VLA**: Simple robot arm pushing blocks (PyBullet)
2. **RT-1 Components**: FiLM layers, TokenLearner, action chunking
3. **LeRobot Integration**: Using HuggingFace's LeRobot framework
4. **Isaac Sim Projects**: Mobile manipulator for warehouse logistics

## Technical Stack

### Expected Dependencies
Based on the learning guide, projects will use:
- **PyTorch**: Deep learning framework
- **HuggingFace Transformers**: Vision Transformer (ViT) models
- **LeRobot**: HuggingFace's robotics framework
- **PyBullet**: Physics simulation (weeks 4-6)
- **Isaac Sim**: Advanced robotics simulation (weeks 9+)
- **ROS2**: Robot Operating System integration (later weeks)

### Installation Commands
```bash
# Core dependencies
pip install lerobot
pip install torch torchvision
pip install transformers
pip install pybullet

# Visualization and data collection
pip install pynput  # For teleoperation
pip install wandb   # Optional: experiment tracking
```

## Development Workflow

### Data Collection Pattern
```python
# Standard pattern for robot demonstration data
episodes = [
    {
        'obs': [img1, img2, ...],      # RGB images (224, 224, 3)
        'actions': [action1, action2, ...]  # Robot actions (joint velocities/positions)
    }
]
# Save as: demonstrations.pkl or data.pkl
```

### Model Architecture Pattern
```python
# Typical VLA structure
class VLAModel(nn.Module):
    def __init__(self):
        self.vision = ViTModel(...)      # Vision encoder
        self.language = ...              # Language encoder (optional)
        self.policy = nn.Sequential(...) # Action decoder

    def forward(self, images, instructions=None):
        vision_features = self.vision(images)
        actions = self.policy(vision_features)
        return actions
```

### Training Pattern
```python
# Behavioral Cloning = Supervised Learning for Actions
for obs, expert_action in dataloader:
    predicted_action = model(obs)
    loss = F.mse_loss(predicted_action, expert_action)
    loss.backward()
    optimizer.step()
```

## Key Concepts

### Behavioral Cloning (BC)
- Primary training method for VLA in this repository
- Supervised learning approach: learn from expert demonstrations
- Loss function: MSE between predicted and expert actions
- Simpler and more stable than Reinforcement Learning

### Vision-Language-Action Models
- **Input**: RGB images (observations) + optional language instructions
- **Output**: Robot actions (joint positions/velocities, gripper state)
- **Architecture**: Vision Transformer + Policy Head (MLP or Transformer Decoder)

### RT-1 Specific Components
When implementing RT-1:
- **FiLM layers**: Condition vision features on language embeddings
- **TokenLearner**: Reduce number of visual tokens for efficiency
- **Action Chunking**: Predict multiple future timesteps
- **EfficientNet**: Alternative vision backbone

## Common Commands

### LeRobot Visualization
```bash
# Visualize existing datasets
python -m lerobot.scripts.visualize_dataset --repo-id lerobot/pusht
```

### LeRobot Training
```bash
# Train ACT policy
python -m lerobot.scripts.train \
    --dataset lerobot/pusht \
    --policy act \
    --batch-size 32 \
    --num-epochs 100

# Train Diffusion policy
python -m lerobot.scripts.train \
    --dataset lerobot/pusht \
    --policy diffusion \
    --batch-size 32 \
    --num-epochs 100
```

### LeRobot Evaluation
```bash
python -m lerobot.scripts.eval \
    --policy act \
    --checkpoint path/to/checkpoint
```

## File Organization

Expected project structure for VLA implementations:
```
project/
├── mini_vla.py           # Model definition
├── dataset.py            # PyTorch Dataset class
├── train.py              # Training loop
├── evaluate.py           # Evaluation in simulation
├── teleop_collect.py     # Data collection via teleoperation
├── demonstrations.pkl    # Collected expert demonstrations
└── best_model.pt        # Trained model checkpoint
```

## Important Notes

### Data Format
- Images: NumPy arrays (H, W, 3) or PIL Images
- Actions: NumPy arrays, typically 7-DOF for robot arm [joint1, ..., joint7]
- Episode structure: List of dicts with 'obs' and 'actions' keys

### LeRobot Data Conversion
When converting custom data to LeRobot format:
```python
# Required fields in LeRobot dataset
data_dict = {
    'observation.image': [...],    # PIL Images
    'action': [...],               # List of action arrays
    'episode_index': [...],        # Episode number
    'frame_index': [...],          # Frame within episode
    'timestamp': [...],            # Time in seconds
}
```

### Learning Approach
- Prioritize hands-on implementation over theoretical perfection
- Run code early and often, even with limited understanding
- Iterate: implement → test → debug → improve
- Expect 0-20% success rate initially; 40-60% after refinement is good
