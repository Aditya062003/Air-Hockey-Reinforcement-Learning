# AI Project: Air Hockey Reinforcement Learning

## Introduction

Air Hockey, implemented within the OpenAI gym, presents itself as a seemingly simple two-player hockey game. However, beneath its surface lie intricate challenges that highlight the significance of reinforcement learning in developing optimal strategies.

Our trained reinforcement learning agents outperform the basic algorithmic opponent provided by the environment, underscoring the necessity and effectiveness of reinforcement learning in surpassing predefined opponents.

We explore solutions for both discrete and continuous action spaces, employing algorithms such as Deep Deterministic Policy Gradient (DDPG) and PPO. This README provides a brief overview, and for a more in-depth understanding and access to the code implementation, refer to our GitHub repository.

<img src="DDPG/assets/demo.gif" alt="Air Hockey Demo" width="500"/>

## Algorithms Implemented

### 1. Deep Deterministic Policy Gradient (DDPG)

In our reinforcement learning scenario, the agent interacts with the environment in discrete time steps. The goal is to learn a policy that maximizes the expected return from this distribution.

#### Key Features:

- **Replay Buffer:** Enhances scalability by introducing a replay buffer.
- **Target Network:** Incorporates a separate target network for calculating target values (\(y_t\)).

#### Actor-Critic Approach:

- **Actor Function :** Defines the current policy by deterministically mapping states to specific actions.
- **Critic :** Trained using the Bellman equation.

### 2. Proximal Policy Optimization (PPO)

#### Key Features:

- GAE (Generalized Advantage Estimation) was employed as an advantage estimator.
- Policy and value networks were implemented separately, while gradient descent was applied using a joint loss function.
- Actions were transformed into a continuous domain using a Multivariate Normal Distribution.
- To constrain actions within the [-1, 1] range, the hyperbolic tangent function (\(\tanh\)) was utilized for transformation.
- Gradient clipping and input normalization were incorporated. Welford's online algorithm was employed for normalization.
- The implementation of neural networks utilized PyTorch.

## Usage

To run the code and explore the implementation, follow these steps:

1. Clone the repository: [git clone https://github.com/your-username/your-repo.git](https://github.com/Aditya062003/AIProject)
2. `cd DDPG`
3. Install dependencies: `pip install git+https://github.com/antic11d/laser-hockey-env.git`
4. Run the main script: `python ./train_agent.py`

For additional details and customization options, refer to the documentation in the repository.

## Team Members

1. Aditya Sankhla (12140060)
2. Aditya Dubey (12140100)
3. Tushar Bansal (12141680)
4. Rohit Aswani (12141390)
