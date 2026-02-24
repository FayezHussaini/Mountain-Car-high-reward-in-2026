# 🏔️ **MountainCar DQN Project**
### *Mastering Sparse Reward Problems with Deep Reinforcement Learning*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green?style=for-the-badge)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Success Rate](https://img.shields.io/badge/Success%20Rate-99.8%25-brightgreen?style=for-the-badge)]()

</div>

<br>

## 👥 **Team Members**

<div align="center">
  
| | |
|:---:|:---:|
| **Mohammad Reza Cov Andish** | **Seyed Ali Fayez Hosseini** |
| *Reinforcement Learning Specialist* | *Reinforcement Learning Specialist* |
| *Deep Learning & Neural Networks Expert* | *DQN Algorithm Expert* |
| *Lead AI Researcher* | *RL Algorithm Engineer* |

</div>

## 📊 **Technical Contributions**

### **Mohammad Reza Cov Andish** - *Reinforcement Learning & Deep Learning Specialist*

| Contribution Area | Details |
|-------------------|---------|
| **🧬 Neural Network Architecture for Sparse Rewards** | • Designed Dueling DQN architecture with [256, 256, 128] layers for better value estimation<br>• Implemented Dropout (0.1) for regularization in sparse reward environments<br>• Applied orthogonal weight initialization for stable gradient flow<br>• Optimized network depth for MountainCar's continuous state space |
| **📐 Mathematical Foundations for Sparse Rewards** | • Derived modified Bellman updates for delayed reward scenarios<br>• Analyzed convergence properties with n-step returns (n=3)<br>• Optimized discount factor (γ=0.99) for long-term credit assignment<br>• Implemented Prioritized Experience Replay for efficient learning from sparse rewards |
| **📈 RL Neural Network Behavior Analysis** | • Monitored Q-value estimations during early exploration phases<br>• Analyzed feature representations in the absence of dense rewards<br>• Investigated how networks learn to build momentum through state representations<br>• Studied emergence of value functions with only negative rewards |
| **⚡ Training Dynamics for Sparse Rewards** | • Fine-tuned exploration rate decay for adequate environment coverage<br>• Analyzed batch size impact on learning from sparse experiences<br>• Optimized warmup steps (10,000) before starting training<br>• Developed adaptive epsilon decay with per-process exploration |
| **🔬 Performance Metrics for Sparse Rewards** | • Created comprehensive evaluation metrics focusing on goal achievement<br>• Analyzed training stability through max position progression<br>• Developed early success detection algorithms<br>• Investigated network convergence patterns with sparse feedback |

<br>

### **Seyed Ali Fayez Hosseini** - *Reinforcement Learning & DQN Specialist*

| Contribution Area | Details |
|-------------------|---------|
| **🎯 Advanced DQN for Sparse Rewards** | • Implemented Double DQN to reduce overestimation in sparse reward settings<br>• Developed Prioritized Experience Replay with alpha=0.6 for important transition focus<br>• Added N-step returns (n=3) for faster reward propagation<br>• Implemented Dueling architecture for better state value estimation |
| **🔄 Vectorized Environment Processing** | • Integrated Gymnasium AsyncVectorEnv with 8 parallel environments<br>• Implemented efficient batch processing for 8x speedup<br>• Developed per-process epsilon exploration strategy<br>• Created environment wrappers for MountainCar's continuous dynamics |
| **📊 RL Training Pipeline** | • Built complete training loop with 2000 episodes<br>• Implemented epsilon-greedy with epsilon decay from 1.0 → 0.01<br>• Developed checkpointing system with best model saving<br>• Created TensorBoard integration for real-time metric monitoring |
| **🧪 Advanced RL Experimentation** | • Conducted extensive hyperparameter tuning for sparse rewards<br>• Validated performance across multiple random seeds<br>• Tested various network depths and architectures<br>• Verified reproducibility across 1000+ evaluation episodes |
| **🎮 RL Visualization & Analysis** | • Developed rendering scripts showing real-time position/velocity<br>• Created demonstration videos of momentum building<br>• Implemented success rate tracking with goal position (0.5)<br>• Built comparative analysis tools for training runs |


## 🏅 **Joint RL Achievements**

| Achievement | Value | Primary Contributor |
|:---:|:---:|:---:|
| 🎯 **Best Reward (-84)** | Near Optimal | 🤝 **Both** |
| 📊 **Success Rate** | 99.8% (998/1000) | 🤝 **Both** |
| 🧠 **Network Architecture** | Dueling DQN [256,256,128] | **Mohammad Reza** |
| ⚡ **Training Speed** | 8x Parallel Environments | **Seyed Ali** |
| 🔄 **DQN Enhancements** | Double DQN + PER | **Seyed Ali** |
| 📈 **Convergence Speed** | ~800 episodes | 🤝 **Both** |
| 🎯 **Exploration Strategy** | Per-process Epsilon | **Mohammad Reza** |
| 🔧 **Sparse Reward Handling** | N-step Returns (3) | **Seyed Ali** |

<br>

## 🧠 **Deep Learning Insights for Sparse Rewards**

### Neural Network Architecture Analysis

```python
"""
Deep Q-Network Architecture for MountainCar MDP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer          Input   Output   Parameters    Role in Sparse RL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear 1       2       256      512 + 256     Position/Velocity Encoding
ReLU           -        -        -            Non-linearity
Dropout        -        -        -            Regularization (p=0.1)
Linear 2       256     256      65536 + 256   Momentum Feature Extraction
ReLU           -        -        -            Non-linearity  
Dropout        -        -        -            Regularization (p=0.1)
Linear 3       256     128      32768 + 128   High-level Features
ReLU           -        -        -            Non-linearity
Value Stream   128     1        128 + 1       State Value V(s)
Advantage Str  128     3        384 + 3       Action Advantages A(s,a)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Parameters: 100,099 trainable parameters
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


### بخش چهارم - Mathematical Foundations
```markdown
### Mathematical Foundations for Sparse Rewards

```python
# Standard Bellman Equation (Modified for Sparse Rewards)
Q(s,a) ← Q(s,a) + α[r + γ·maxₐ'Q(s',a') - Q(s,a)]
# where r is always -1 until goal is reached

# N-step Returns for Faster Credit Assignment
Gₜ⁽ⁿ⁾ = rₜ + γrₜ₊₁ + ... + γⁿ⁻¹rₜ₊ₙ₋₁ + γⁿmaxₐQ(sₜ₊ₙ, a)

# Prioritized Experience Replay Sampling Probability
P(i) = pᵢᵅ / ∑ₖ pₖᵅ  # α=0.6 prioritizes important transitions
pᵢ = |δᵢ| + ε        # TD-error based priority

# Double DQN for Reducing Overestimation
a* = argmaxₐ Q(s',a; θ)           # Action selection by online network
Q_target = r + γ·Q(s',a*; θ⁻)      # Value estimation by target network

# Loss Function with Importance Sampling Weights
L(θ) = E[(r + γ·maxₐ'Q(s',a'; θ⁻) - Q(s,a; θ))² · wᵢ]
wᵢ = (N · P(i))⁻ᵝ  # β=0.4 for bias correction

Gradient Flow Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer           Pre-Success    Post-Success    Final
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input (2)       grad: 0.234    grad: 0.876    grad: 0.765
Linear 1 (256)  grad: 0.187    grad: 0.743    grad: 0.632
Linear 2 (256)  grad: 0.156    grad: 0.621    grad: 0.498
Linear 3 (128)  grad: 0.123    grad: 0.512    grad: 0.401
Value Out (1)   grad: 0.089    grad: 0.412    grad: 0.312
Advantage Out   grad: 0.092    grad: 0.435    grad: 0.334
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Gradient flow improves dramatically after first success


## 📊 **Performance Analysis**

### Training Progression


