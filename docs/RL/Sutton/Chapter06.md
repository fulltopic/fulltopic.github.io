#Chapter06
## Notes
* TD learning is a combination of MC ideas and DP ideas. Like MC, TD can learn directly from raw experience without a model.
Like DP, TD update estimates based in part on other learned estimates, without waiting for final outcome
* TD is bootstrapping as its update based part on an existing estimate    
### 6.2
Advantages:
* No (environment)model required
* on-line
* Bootstrapping, so all trajectories are OK
* For fixed π, TD(0) converges to v<sub>π</sub> in the mean for constant step-size if it is sufficiently small, and decreases according to usual stochastic approximation conditions
### 6.3
MC is to match the training set, TD is to estimate the Markov process.
### 6.5
Requirements:
* All pairs of (S,A) continue to update
* Conditions required on sequence of step-size parameters
### 6.8
afterstate value functions
## Exercises
### 6.1
δ<sub>t</sub> = R<sub>t+1</sub> + γV<sub>t+1</sub>(S<sub>t+1</sub>) - V<sub>t</sub>(S<sub>t</sub>)

G<sub>t</sub> - V<sub>t</sub>(S<sub>t</sub>) = R<sub>t+1</sub> + γV<sub>t+1</sub>(S<sub>t+1</sub>) - V<sub>t</sub>(S<sub>t</sub>) 
    + γV<sub>t</sub>(S<sub>t+1</sub>) - γV<sub>t+1</sub>(S<sub>t+1</sub>)

Suppose d<sub>t</sub> = V<sub>t</sub>(S<sub>t+1</sub>) - V<sub>t+1</sub>(S<sub>t+1</sub>)

Right side = R<sub>t+1</sub> + γV<sub>t+1</sub>(S<sub>t+1</sub>) - V<sub>t</sub>(S<sub>t</sub>) + γd<sub>t</sub>

= δ<sub>t</sub> + γ(G<sub>t+1</sub> - V<sub>t</sub>(S<sub>t+1</sub>)) + γd<sub>t+1</sub>

= ∑<sub>k=(t:T-1)</sub>(γ<sup>k-t</sup>δ<sub>k</sub> + γ<sup>k-t+1</sup>d<sub>k+1</sub>)
### 6.2
Properties of some states are relatively static, for short-term planning, estimating current state value by relatively stable next state value 
provides quick and good response
### 6.3
δ<sub>t</sub> = R<sub>t+1</sub> + γV<sub>t+1</sub>(S<sub>t+1</sub>) - V<sub>t</sub>(S<sub>t</sub>)

Δ = α * δ<sub>t</sub>

All V<sub>t</sub>(S<sub>t</sub>) are initialized with the same value; γ = 1; R<sub>t</sub> = 0 for all t.

That is, before termination, there would be no update on each V<sub>t</sub>(S<sub>t</sub>). 
When episode ends in Terminal state at T, only V(S<sub>T-1</sub>) would be updated. 
So, the first episode ended with ...->A->T. As V(A) updated with negative Δ, the T is the left T.

And we get Δ = 0.1 * (0 + 1 * 0 - 0.5) = -0.05
### 6.4
Intuitively No. As the large α may cause fluctuation and may lead to converge failure, and small α had been shown in the figure.

I am not able to prove it actually.
### 6.5
After long run, values are relatively converged and stable. The large α makes Δ depend heavily on single step,
the biased approximation drives V away from true values. 

It should little to do with initiation as the estimation is near the true distribution after long run.
### 6.6
a = (0 + b) / 2

b = (a + c) / 2

c = (b + d) / 2

d = (c + e) / 2

d = (1 + d) / 2

Solve the equations
### 6.7
Δ = α * δ<sub>t</sub>

δ<sub>t</sub> = G<sub>t</sub> - V(S<sub>t</sub>) = R<sub>t+1</sub> + γV<sub>t+1</sub>(S<sub>t+1</sub>) - V<sub>t</sub>(S<sub>t</sub>)

Suppose ρ<sub>t</sub> = possibility of trajectory {s0, s1, s2, ..., st}

By behavior policy b and target policy π, 

δ<sub>t</sub>  ~ G<sup>'</sup> - V(S<sub>t</sub>)

= ρ<sup>π</sup><sub>t+1</sub> / ρ<sup>b</sup><sub>t+1</sub> * (R<sub>t+1</sub> + γV(S<sub>t+1</sub>)) - V(S<sub>t</sub>)
### 6.8
G<sub>t</sub> - Q(S<sub>t</sub>, A<sub>t</sub>) = R<sub>t+1</sub> + γG<sub>t+1</sub> - Q(S<sub>t</sub>, A<sub>t</sub>) 

= R<sub>t+1</sub> + γG<sub>t+1</sub> + γQ(S<sub>t+1</sub>, A<sub>t+1</sub>) - Q(S<sub>t</sub>, A<sub>t</sub>) - γQ(S<sub>t+1</sub>, A<sub>t+1</sub>)

= δ<sub>t</sub> + γ(G<sub>t+1</sub> - Q(S<sub>t+1</sub>, A<sub>t+1</sub>))

= δ<sub>t</sub> + γδ<sub>t+1</sub> + γ<sup>2</sup>(G<sub>t+2</sub> - Q(S<sub>t+2</sub>, A<sub>t+2</sub>))

= ∑<sub>k=(t:T-1)</sub>γ<sup>k-t</sup>δ<sub>k</sub>
### 6.11
Because that the Q(S<sub>t</sub>, A<sub>t</sub>) is always updated by max<sub>a</sub>(Q(S<sub>t+1</sub>, A<sub>t+1</sub>)),
while sometimes the policy did not choose the argmax(Q) as the next action (e.g. e-greedy)
### 6.12
Yes, they are the same
### 6.13
δ<sub>i</sub> = R + γ∑<sub>i</sub>π<sub>i</sub>(A|S)Q<sub>1-i</sub>(S,A) - Q<sub>i</sub>  (i = 0 or i = 1)
### 6.14
Define S = afterstate, i.e. how many cars are in each location, 
then learn V(s) or Q(s) by TD(0).

It speeds up convergence as 
* In DP, each policy-value process requires many iterations; in each iteration all V(s) have to be calculated; in each V(s) calculation all related V(s<sup>'</sup>)s have to be involved
* In TD, in each episode, only s<sup>'</sup> are required for update of s. The transition are just s -> s<sup>'</sup>. The randomness are introduced by policy and environment themselves, some less possible states and transitions would be ignored in learning.