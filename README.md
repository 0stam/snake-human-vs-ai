# Snake: Human vs AI

A game where you can play a competitive snake match against an AI model trained with reinforcement learning.

Originally developed for CyberTech Students Research Group to showcase during university promotional events.

The model training uses Deep Q-learning with a custom implementation of a priority replay buffer and techniques for breaking endless gameplay loops. The model was trained using TensorFlow, with LiteRT (formerly TFLite) used for gameplay.

## Features

### Demo mode

When the game launches or the player is inactive, the game enters demo mode where **multiple AI-controlled snakes battle against each other**.

https://github.com/user-attachments/assets/226928d1-85a7-4854-8bfe-38eca9385897

### Gameplay

The player faces an AI-controlled snake. The first player to hit a wall or any player's snake body loses.

https://github.com/user-attachments/assets/6bcf941a-5d00-4e9b-a95e-c435a92247e6

## QoL

- **gamepad support** with automatic detection that changes on-screen movement hints
- **coyote time** (ability to change direction for a very short time after the move has started) to compensate for grid movement

## Running the game

### Requirements

Requirements for running the game (without model training) are:

- `pygame-ce`
- `tensorflow`
- `numpy`

Note: `requirements.txt` was generated on Linux and includes many dev/GPU/Linux-specific packages that may not install on Windows.

## Install and run

Create and activate a virtual environment. Then, from the project root:

```bash
pip install pygame-ce tensorflow numpy
python -m src.main
```

## Optional (Linux/full environment)

If you want the full Linux environment used in development:

```bash
pip install -r requirements.txt
```

The project was developed using Python 3.11. It will likely work with newer Python versions, but backwards compatibility, especialy library support, is not guaranteed.
The project was developed using Python 3.11. It will likely work with newer Python versions, but backward compatibility, especially library support, is not guaranteed.

## Implementation

### Reinforcement learning
- Deep Q-learning with TensorFlow
- LiteRT conversion and quantization for faster inference during gameplay
- Custom replay buffer with a dual-queue design for straightforward separation of important and average memories
- Different model architectures:
  - Simple neural network
  - Convolutional neural network
  - Simple neural network that receives the first objects in straight and diagonal directions (best performing)
- Using softmax for move selection during training to avoid looping, using the model-generated probability distribution instead of a completely random approach
- Full logs of the training process

### Simulation
- Shared logic for training and gameplay
- Game rules implementation
- Optional scoring function support for flexible training score output

### Gameplay
- Pygame-based
- Custom hierarchical UI component system
  - Automatic parent-to-children event propagation
  - Factories for common UI components
- Automatic resolution scaling
- Gameplay-specific features, such as:
  - Demo and game views
  - Gamepad support
  - Coyote time
  - Converting internal turn-based gameplay into animation
  - Flexible JSON config

## Game at display at Wrocław University of Science and Technology

<img src="https://github.com/user-attachments/assets/522439ca-a54b-4b2c-aa23-7b81568059aa" width="75%">
