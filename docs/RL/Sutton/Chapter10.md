# Chapter10
∑ τ α π δ γ ∏ Δ Λ μ ∈ ∀ β
## Notes
To clarify:

The root cause of the difficulties with the discounted control setting is that with
function approximation we have lost the policy improvement theorem (Section 4.2). It is
no longer true that if we change the policy to improve the discounted value of one state
then we are guaranteed to have improved the overall policy in any useful sense. That
guarantee was key to the theory of our reinforcement learning control methods. With
function approximation we have lost it!
## Exercises
### 10.1
```
Loop for t = 0, 1, 2, ...:
    if NOT terminal, then:
        Take action A(t)
        Observe and store R(t+1), S(t+1)
        if S(t+1) == Terminal, then
            break
        else:
            Get A(t+1) from Pi
G = 0            
Loop for t = N, N-1, ..., 0:
    G = gamma * G + R(t)
    w = w + alpha * [G - q(S(t), A(t))] * dQ/dw
```
### 10.2
```
Loop for each step of episode:
    Take action A, observe R, S'
    If S' is terminal:
        w = w + alpha * (R - Q(S, A, w)) * dQ/dw
        Go to next episode
    w = w + alpha * (R + gamma * ∑(A')Pi(A'|S') * Q(S',A') - Q(S, A)) * dQ/dw
    Select A' for S' by Pi
    S = S'
    A = A'
```
### 10.3
Larger n means longer trajectory that impacts on current updated step. 
Longer trajectory means larger variance.
When α increases, each different trajectory drives W in different directories with large force,
and the W failed to converge.
### 10.4
```
Loop for each step:
    Choose A from S using π
    Take A, observe S', R
    Q(S,A) = Q(S,A) + alpha * (R - R(ave) + gamma * max(A')Q(S',A') - Q(S,A))
    Update R(ave) by R
    S = S'
```
### 10.5
w<sub>t+1</sub> = w<sub>t</sub> + αδ<sub>t</sub> * dV(S<sub>t</sub>, A<sub>t</sub>, w<sub>t</sub>)/dw
### 10.6
r(π) = ∑μ∑π∑p*r (10.6)

As a ring, the π is determined, that is to move along the ring by a certain directory (clock or counter-clock). So π = 1 for each state.  

As a ring and continuous task, μ = 1/3 for each state

As a deterministic environment, p = 1 for each transition.

r(π) = 1/3

v<sub>π</sub>(s) = E<sub>π</sub>[G|S]

G<sub>t</sub> = R<sub>t+1</sub> - r(π) + R<sub>t+2</sub> - r(π) + ...

Take S = B for example, for each 3 steps, the agent experienced transition of B->C->A->B.

G(B)<sub>t</sub> = R<sub>t+1</sub> - r(π) + R<sub>t+2</sub> - r(π) + R<sub>t+2</sub> - r(π) + ...

= 0 - 1/3 + 1 - 1/3 + 0 - 1/3 + ...

= 0 + R<sub>t+3</sub> - r(π) + ...

= 0

v<sub>π</sub>(s) = 0 (∀S∈{A,B,C})

### 10.7
Don't know r(π). Guess that r(π) = 1/2

v(S) = lim(γ)lim(h)γ<sub>t</sub>(E(R<sub>t+1</sub>) - r(π))

v(A) = lim(γ)lim(h)(γ<sup>0</sup>(1-1/2) + γ(0-1/2) + γ<sup>2</sup>(1-1/2) + ...)

= lim(γ)lim(h)∑<sub>t</sub>(1/2 - 1/2γ + 1/2γ<sup>2</sup> + ...)

= lim(γ)lim(h)∑<sub>t</sub>(1/2 * (-γ)<sup>t</sup>)

= lim(γ)lim(h)(0.5 * 1 / (1 - (-γ)))

= 0.25

v(B) = -v(A) = -0.25
### 10.8
For δ update, it takes q(S') and q(S) as estimated G for δ. 

(R - R(ave)) = (R - 1/3) = {-1/3, -1/3, 2/3}

δ = (R - R(ave) + q(S') - q(S)) 

e.g. δ(B) = -1/3 + ΔQ<sub>t</sub>(B)

The δ would produce more stable estimate as 
* δ update takes trajectory with more steps
* After updating, the (Q<sub>estimate</sub> - Q<sub>true</sub>) decreased, (ΔQ<sub>t</sub> - (R - R(ave)) decreased, updating of R(ave) and w more stable
### 10.9
β<sub>n</sub> = β/o<sub>n</sub>

o<sub>n</sub> = o<sub>n-1</sub> + β(1 - o<sub>n-1</sub>)