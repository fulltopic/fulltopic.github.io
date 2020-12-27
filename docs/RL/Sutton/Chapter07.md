# Chapter07
## Exercises
### 7.1
As no update from step to step, V<sub>t+n</sub> = V<sub>t</sub> = V

δ<sub>t</sub> = R<sub>t+1</sub> + γV(S<sub>t+1</sub>) - V(S<sub>t</sub>)

H<sub>t</sub> = G<sub>t:t+n</sub> = R<sub>t+1</sub> + γR<sub>t+2</sub> + ... + γ<sup>n</sup>V(S<sub>t+n</sub>)

G<sub>t:t+n</sub> - V(S<sub>t</sub>) = R<sub>t+1</sub> + γV(S<sub>t+1</sub>) - V(S<sub>t</sub>) - γV(S<sub>t+1</sub>) + γR<sub>t+2</sub> + ... + γ<sup>n</sup>V(S<sub>t+n</sub>)

= δ<sub>t</sub> + γ(R<sub>t+2</sub> + ... + γ<sup>n-1</sup>V(S<sub>t+n</sub> - V(S<sub>t+1</sub>))

= δ<sub>t</sub> + γδ<sub>t</sub> + ... + γ<sup>n</sup>δ<sub>t+n</sub>

### 7.2
Generally, the difference between 2 algorithms are double-buffer and single-buffer.

As discussed in previous chapters, using of the updated V<sub>t+k</sub> accelerated propagation of update. Which accelerated convergence and also the bias propagation.
It is hard to determine which one is better, while updated case would be better as
* In simple case, the algorithm will converge with probability = 1, no matter which one used. So quicker one is better.
* In complicated case, the step size (alpha) is very small, updated version has very little difference to no-update version, and introduced slim bootstrapping effect. So updated version is better
### 7.3
* In small step-size cases, the problem is easy to become an MC cases, and also introduced a bias to samples: less samples for long episode, more samples for short episode. 
* In according to above reasons, the n would be smaller in smaller problems
* If return of both terminal state are 0s, the problem is in fact an (n /2) problem as walking toward each side are the same.
Make left side = -1, the agent has to learn how to escape left side. That is an n problem
### 7.4
G<sub>t:t+n</sub> = R<sub>t+1</sub> + γR<sub>t+2</sub> + ... + γ<sup>n</sup>Q(S<sub>t+n</sub>, A<sub>t+n</sub>)

δ<sub>t</sub> = G<sub>t:t+n</sub> - Q(S<sub>t</sub>, A<sub>t</sub>)

= R<sub>t+1</sub> + γQ(S<sub>t+1</sub>, A<sub>t+1</sub>) - Q(S<sub>t</sub>, A<sub>t</sub>) + γG<sub>t+1:t+n</sub> - γQ(S<sub>t+1</sub>, A<sub>t+1</sub>)

= R<sub>t+1</sub> + γQ(S<sub>t+1</sub>, A<sub>t+1</sub>) - Q(S<sub>t</sub>, A<sub>t</sub>) + γ(R<sub>t+1</sub> + γQ(S<sub>t+2</sub>, A<sub>t+2</sub>) - Q(S<sub>t+1</sub>, A<sub>t+1</sub>)) + γ<sup>2</sup>Q(S<sub>t+3</sub>, A<sub>t+3</sub>)

= ∑<sub>k=t:min(t+n,T)-1</sub>γ<sup>k-t</sup>(R<sub>k+1</sub> + γQ(S<sub>k+1</sub>, A<sub>k+1</sub>) - Q(S<sub>k</sub>, A<sub>k</sub>)
### 7.5
```
if τ ≥ 0:
    G = 0
    for k = t ... t + τ - 1:
        G = ρ(k) * (R(k+1) + γG) + (1 - ρ(k)) * V(S(k))
    V(S(τ)) += α * (G - V(S(τ))
```
