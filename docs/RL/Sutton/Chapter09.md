# Chapter09
## Notes
In fact, all the theoretical results for methods using function approximation presented in this part of book apply equally well to cases of partial observability.
What function approximation can't do, however, is augment the state representation with memories of past observations.
### 9.1
For RL, the target functions are nonstationary. For example, in control methods based on GPI we often seek to learn q<sub>π</sub> while π changes. 
Even if the policy remains the same, the target values of training examples are nonstationary if they are generated by boostrapping methods (DP, TD).
Methods that cannot easily handle such nonstationarity are less suitable for RL.
### 9.4
Critical to these convergence results is that states are updated according to the on-policy distribution.
For other update distributions, bootstrapping methods using function approximation may actually diverge to infinity.

MC has larger variance than TD does
## Exercises
### 9.1
[answer](https://stats.stackexchange.com/questions/215549/look-up-table-as-a-special-case-of-a-the-linear-function-approximation-reinforc)

[video](https://www.youtube.com/watch?v=UoPei5o4fps)
### 9.2
For k states, each state could be order of {0, 1, ..., n}
### 9.3
n = 2; c = {0, 1, 2}
### 9.4
Dense tiling in important dimension works. 
For example, dense stripes in requiring dimension and coarse stripes in the other dimension.
### 9.5
Suppose each tiling is represented by T<sub>i</sub> = [b<sub>0</sub>, b<sub>1</sub>, ..., b<sub>ni</sub>] (b = {0, 1}, ni = number of grids for tiling i)

The final encoding of a state is represented as X = [T<sub>0</sub>, T<sub>1</sub>, ..., T<sub>97</sub>]

X<sup>T</sup>X = T<sub>0</sub><sup>T</sup>T<sub>0</sub> + T<sub>1</sub><sup>T</sup>T<sub>1</sub> + ... + T<sub>97</sub><sup>T</sup>T<sub>97</sub>

As T is one-hot vector, T<sup>T</sup>T = 1, X<sup>T</sup>X = ∑<sub>i</sub>T<sub>i</sub><sup>T</sup>T<sub>i</sub> = 98

τ = 10

α = (τE[X<sup>T</sup>X])<sup>-1</sup> = (10 * 98)<sup>-1</sup> = 0.00102