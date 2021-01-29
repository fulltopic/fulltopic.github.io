# AlphaGo Implement
∑ τ α π δ γ ∏ Δ Λ μ ∈ ∀ β λ ∞ η θ ∝ σ

## Pre-trained Network
There are three neuron networks trained by SL for online MCTS: 
* P<sub>σ</sub>: A policy network to provide initial P(a|s) for MCTS nodes
* P<sub>π</sub>: A policy network to decide MCTS expansion
* V<sub>θ</sub>: A value network to provide initial V(s) for node selection in MCTS search phase
### P<sub>σ</sub>
#### Dataset
Training data was extracted from human player's game records, including players of multi-level. 

The input of the network combined many human selected features by game rules. Some preprocessing such as augment executed on input data.
At last the input has been formatted into 19 * 19 * 48 (W * H * C) tensor.
Refer to original paper.

The output is a distribution of π(a|s) for certain s. 

The loss of the network is softmax cross entropy.
#### Architecture
![arch](./images/policypi_alphago.jpg)
### P<sub>π</sub>
This NN has same dataset as that of P<sub>σ</sub>. 
The input format has not described in detail in the paper. The architecture is shown in following figure.

This network has much less accuracy than that of P<sub>σ</sub>. The simpler architecture makes the network runs much faster and introduces more exploration.

Seemed this network has been updated after SL training.

![roll_arch](./images/policyroll_alphago.jpg)
### V<sub>θ</sub>
It is a regression problem

![v_arch](./images/v_alphago.jpg)

The dataset of this network is from simulation.
* Select int U randomly from [1, 450]
* Start a game
* Run U steps with action decided by P<sub>σ</sub>. 
* Select action of S<sub>U</sub> randomly, reach S<sub>U+1</sub>
* Continue the game till Terminal state, with action decided by P<sub>π</sub>, and get reward z. Seemed the game has reward = 0 for each internal state and just relay back the terminal reward back to all states in the path without discounting.
* Store pair (S<sub>U+1</sub>, r) into dataset

Then the dataset works as replay buffer, and the network has been trained as a regression problem with LMSE.
## MCTS
### Structure
![structure](./images/mctree_alpha.jpg)

* state: current state
* action: action executed by parent node that leads to current state  
* parent: parent node that transit to current state by action a. I don't know Go rules, while seemed pre-procession of raw state guarantees that one node has only unique parent.
* P(a|s): prior probability that initialized by output of P<sub>σ</sub>
* Q(s,a): Q. Estimated in tree search, not a NN output
* u(s,a): a factor to control search strategy, mainly to balance exploit and exploration
* mcValueSum(s,a): named W<sub>v</sub>(s,a) in paper. A factor to estimate Q.
* rolloutRewardSum(s,a): named W<sub>r</sub>(s,a) in paper. A factor to estimate Q.
* visitN: there were N<sub>v</sub> and N<sub>r</sub> in paper. This is because that tree search before leaf node is executed in parallel (by different games), and expansion of leaf node is executed asynchronously with leaf node tree search (as other game is expanding the same node at the same time and multi-simulation of single leaf are executed at the same time). In fact, the value of the these Ns is the same after backup step.
* isLeaf: If the node has been expanded

The overall search process:

![search](./images/mcts_alpha.jpg)
### Selection
In selection phrase, action is select by:

* a<sub>t</sub>(s) = argmax<sub>a</sub>(Q(s,a) + u(s,a))
* Q is initialized with parent Q value and then updated in backup phrase
* u(s,a) = c<sub>puct</sub> * P(a|s) * parent.visitN<sup>1/2</sup> / (1 + visitN): c<sub>puct</sub> is a constant, P(a|s) is output of P<sub>σ</sub>
### Expansion
In expansion phase, the action is select by a(s) = argmax<sub>a</sub>P<sub>π</sub>(s)

In terminal state, the game get a reward defined by reward function. 
At most times, the reward function is (number of Mu won/lost) or (win/lost = 1/0)

The reward name z
### Backup
Name the leaf node as n<sub>L</sub>, Calculate V<sub>θ</sub>(n<sub>L</sub>.state).

If n<sub>L</sub>.visitN > Threshold, n<sub>L</sub>.isLeaf = false. Enumerate all legal actions of n<sub>L</sub> and create nodes for them as children of n<sub>L</sub> with 
* n<sub>child</sub>.isLeaf = true.
* n<sub>child</sub>.P(a|s) = P<sub>σ</sub>(a|s)
* n<sub>child</sub>.mcValueSum = 0
* n<sub>child</sub>.rolloutRewardNum = 0
* n<sub>child</sub>.visitN = 0
* n<sub>child</sub>.parent = n<sub>L</sub>

Start from leaf node till root node of this search task, for each node n:
* n.visitN += 1
* n.mcValueSum += V<sub>θ</sub>(n<sub>L</sub>.state)
* n.rolloutRewardSum += z
* n.Q = ((1 - λ) * n.mcValueSum + λ * n.rolloutRewardSum) / n.visitN 
* n.u = c<sub>puct</sub> * n.P * n.parent.visitN<sup>1/2</sup> / n.visitN
* n = n.parent
### Output
Select the action from children of root node with

output action = argmax<sub>a</sub>n<sub>child</sub>.visitN

Why most visited action are chosen?
* The more the action has been visited, more stable and accurate the MC estimation of Q is
* If the candidate is a good choice, the large probability of being chosen reinforced the good decision, u works rather like an exploration factor.
* If the candidate is not a good choice but is the current most visited action, it will take more episodes to decrease the Q and u; that confirms us that the choice is not a good choice with more belief to avoid fluctuation.  
## NN Reinforcement Learning
P<sub>σ</sub> is trained after SL training. 

P<sub>σ</sub> is trained by self-play simulation with pool of policies that trained in previous training iterations.

P<sub>σ</sub> is trained by REINFORCEMENT algorithm
## Reference
* [_Mastering the game of Go with deep neural networks and tree search_](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) and its references
* _Reinforcement Learning: An Introduction_ ch13.3
* [MuGo Repository](https://github.com/brilee/MuGo.git)
* [JoshieGo Repository](https://github.com/brilee/MuGo.git)