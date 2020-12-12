#Chapter03
##3.3
Several factors:
* The learning object. Case1 is to learn a vehicle,
case2 is to learn the tire or road or force mechanism, case3 is to learn human driving.
* The cost of data collection, storage, computing, etc.
## 3.4
G<sub>t</sub> = ∑<sub>k=0</sub><sup>T-t-1</sup> γ<sup>k</sup>R<sub>t+k+1</sub>

For episodic case, T is limited, and G is limited, γ could be equal to 1.
## 3.5
The robot is hard to learn from local environment as all actions get reward and return as 0.
It will take rather long for exploration mechanism to escape the maze. 
In worse case, the robot would be trapped in local optimal value (=0).

The drawback of this case is that the robot has not been told that there is a solution with good reward in time.
## 3.6
It depends on rules of game. 
If all objects just took responsibility of states in front of vision system, it could be an MDP.

When there has been broken, it is not MDP as there was no state.
Once it repaired and turned on, it is still possible to access to MDP as it is an MDP.
## 3.8
q<sup>π</sup>(s, a) = E[G<sub>t</sub> | s, a] 

=  ∑<sub>s'</sub> p(s' | s, a) * E[G | s']

=  ∑<sub>s'</sub> p(s' | s, a) * V(s')

=  ∑<sub>s'</sub> p(s' | s, a) * ∑<sub>r</sub>∑<sub>a'</sub>p(r, a') * E[r + R | s', a']

=  ∑<sub>s'</sub> p(s' | s, a) * ∑<sub>r</sub>∑<sub>a'</sub>p(r, a') * r + ∑<sub>s'</sub> p(s' | s, a) * ∑<sub>r</sub>∑<sub>a'</sub>p(r, a') * E[R | s', a']

= ∑<sub>s'</sub> p(s' | s, a) * ∑<sub>r</sub>∑<sub>a'</sub>p(r, a') * r + ∑<sub>s'</sub> p(s' | s, a) * ∑<sub>r</sub>∑<sub>a'</sub>p(r, a') * q<sup>π</sup>(s', a')

= ∑π(s, a) * ∑<sub>r</sub>∑<sub>a'</sub>p(r, a') * r + ∑π(s, a) * ∑<sub>r</sub>∑<sub>a'</sub>p(r, a') * q<sup>π</sup>(s', a')

## 3.9
v<sup>π</sup>(s) = ∑<sub>a</sub>π(a|s) * ∑<sub>s',r</sub>p(s', r| s, a) * [r + γ * v<sub>π</sub>(s')]

π(a|s) = 0.25 ∀a

r = 0 ∀(s,a)

p(s',r | s, a) = 1 ∀s'

v<sub>π</sub>(s') = {2.3, 0.7, -0.4, 0.4}

γ = 0.9

v(s) = π(upper) * 1 * (0.9 * v(upper)) + ... + π(right) * 1 * (0.9 * v(right))

= 0.25 * 3 * 0.9 = 0.675 
## 3.10
G<sub>t</sub> = ∑<sub>k</sub>γ<sup>k</sup> * (R<sub>t+k+1</sub> + c)

= ∑<sub>k</sub>γ<sup>k</sup> * R<sub>t+k+1</sub> + γ<sup>k</sup> * c

Delta of G<sub>t</sub> depends merely on k. The c does not impact on relative v/q or other variables.
## 3.11
It impacts on relative values to the initial values. 
Take maze case for example, if original reward of state s is {-1, -2, -3, -4},
initial value of s is {0, 0, 0, 0}. In beginning rounds, the agent would try all actions and find -1 as best choice.
Add 4 to all rewards, reward of s is {3, 2, 1, 0}, 
the agent would more likely to be trapped in actions with reward = {0, 1, 2},
and would find the best choice of {3} after relatively longer running with exploration.

It may also extend steps of episode.
## 3.12
v(s) = ∑<sub>a</sub>π(a | s) * q(s, a)
## 3.13
q(s,a) = ∑<sub>s'</sub>p(s'|s,a)(r<sub>s->a->s'</sub> + v(s'))
## 3.14
According to figure 3.6:
* Outside the green land, driver is better
* In green land but outside the circle inside green land, putter is better.
* In circle inside green land, either is good

The better means taking action/value of corresponding action.
## 3.15
Same as above
## 3.16
q<sub>*</sub>(s,a) = ∑<sub>s'</sub>p(s'|s,a) * [r + γ * max<sub>a'</sub>q(s',a')]
(in this case, r is determined when s' determined)

S = {h, l}

A = {s, w, r}

|state|action|nextState|P|R|
|---|---|---|---|---|
|h|s|h|p0|r0|
|h|s|l|1-p0|r1|
|h|w|h|p1|r2|
|h|w|l|1-p1|r3|
|h|r|h|p2|r4|
|h|r|l|1-p2|r5|
|l|s|h|p3|r6|
|l|s|l|1-p3|r7|
|l|w|h|p4|r8|
|l|w|l|1-p4|r9|
|l|r|h|p5|r10|
|l|r|l|1-p5|r11|

q<sub>*</sub>(h,s) = p0 * [r0 + γ * max<sub>a'</sub>(q(h, s), q(h, w), q(h, r))]
    + (1 - p0) * [r1 + γ * max<sub>a'</sub>(q(l,s), q(l,w), q(l,r))]

q<sub>*</sub>(l,w) = p4 * [r8 + γ * max<sub>a'</sub>(q(h,s), q(h,w), q(h,r))]
    + (1- p4) * [r9 + γ * max<sub>a'</sub>(q(l,s), q(l,w), q(l,r))]

## 3.17
* Suppose the grid has optimal values as:
  
  |X|A0|X|B|X|
  |---|---|---|---|---|
  |X|A1|X|X|X|
  |X|A2|X|B'|X|
  |X|A3|X|X|X|
  |X|A4|X|X|X|
* By optimal state value rule: 
v<sub>*</sub>(A0) = max<sub>a</sub>∑<sub>s'</sub>p(s'|A0,a) * (r + v<sub>*</sub>(s'))
==> v<sub>*</sub>(A0) = 10 + 0.9 * A4
* By 3.2, G = ∑<sub>k</sub>(r<sub>k</sub> + γ<sup>k</sup> * R<sub>k</sub>). 
As reward inside grid is 0 for each step k, 
following the optimal policy, the best G for A4 is to go through shortest path to A0 for least k, that is to go straight forward to A0.
That is, the path is {A4, A3, ..., A0}. 
The v<sub>*</sub>(A4) = 0 + γ * v(A3) = γ<sup>2</sup> * v(A2) = ... = γ<sup>4</sup> * v(A0)
* Solve the equations: 10 = (1 - γ<sup>5</sup>) * v(A0)
==> v(A0) = 24.419
## 3.18
v<sub>*</sub>(s) = max<sub>a</sub>q(s,a)
## 3.19
q<sub>*</sub>(s,a) = ∑<sub>s',r</sub>p<sup>π</sup>(s',r|s,a) * (r + v(s'))
## 3.20
π<sub>*</sub>(a|s) = argmax<sub>a</sub>q(s,a)
## 3.21
π<sub>*</sub>(a|s) = argmax<sub>a</sub>q(s,a)

= argmax<sub>a</sub>∑<sub>s',r</sub>p<sup>π</sup>(s',r|s,a) * (r + v(s'))



