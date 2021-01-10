# Chapter13
## Exercises
### 13.1
* Suppose v(G) = 0. 
* Make p = possibility of selecting right at any state.
* make v1, v2, v3 = v(s) from left to right, i.e. v1 = v(S)

v1 = p * v2 + (-1)

v2 = p * v1 + (-1) + (1 - p) * v3 + (-1)

v3 = (1 - p) * v2 + (-1)

==>

v(S) = v1 = 3 / (2 * (p - 1)) - 1

v<sub>max</sub>(S) = -2.5 when p -> 0
### 13.2

#### 13.2.1: Page 199
η(s) = h(s) + γ∑<sub>s'</sub>η(s')∑<sub>a</sub>π(a|s')p(s|s',a)
#### 13.2.2: Page 325 Box
dJ(θ) = dv<sub>π</sub>(s0)

= ∑<sub>s</sub>(∑<sub>k=0:∞</sub>Pr(s0->s,k,π)γ<sup>k</sup>)v(s')

= ∑<sub>s</sub>(∑<sub>k=0:∞</sub>Pr(s0->s,k,π)γ<sup>k</sup>)∑<sub>a</sub>dπ(a|s)q<sub>π</sub>(s,a)

∝ ∑<sub>s</sub>μ(s)γ<sup>k</sup>∑<sub>a</sub>dπ(a|s)q<sub>π</sub>(s,a)

It means that v<sub>π</sub>(s0) = v(s') * probability of (s' could be reached from s0 in k steps).
Then each v(s') has been decayed by γ<sup>k</sup> by steps from s0
#### 13.2.3: (13.6)
dJ(θ) ∝ E<sub>π</sub>[γ<sup>k</sup>∑<sub>a</sub>dπ(a|s)q<sub>π</sub>(s,a)]

= E<sub>π</sub>[γ<sub>t</sub>∑<sub>a</sub>dπ(a|s)q<sub>π</sub>(s,a)]
### 13.3
Suppose:
* h(s,a,θ) = θ<sup>T</sup>x(s,a)
* f = e<sup>h(s,a,θ)</sup>
* π = f<sub>a</sub> / ∑<sub>b</sub>f<sub>b</sub>
* g = lnπ

dg / dθ = (dg / dπ) * (dπ / df) * (df / dh) * (dh / dθ)

dg / dπ = 1 / π

df / dh = e<sup>h</sup> = f

dh / dθ = x

(dπ / dh) = (1 / ∑<sub>b</sub>f<sub>b</sub>) * (df<sub>a</sub> / dh<sub>a</sub>) + ∑<sub>b</sub>(f<sub>a</sub> * (-1) * (∑<sub>b</sub>f<sub>b</sub>)<sup>-2</sup> * (df<sub>b</sub> / dh<sub>b</sub>))

(dπ / dh) = (1 / ∑<sub>b</sub>f<sub>b</sub>) * f<sub>a</sub> - ∑<sub>b</sub>((f<sub>a</sub> / ∑<sub>b</sub>f<sub>b</sub>) * (f<sub>b</sub> / ∑<sub>b</sub>f<sub>b</sub>))

= π<sub>a</sub> * (1 - ∑<sub>b</sub>π<sub>b</sub>)

dg / dθ = (1 / π<sub>a</sub>) * π<sub>a</sub> * (1 - ∑<sub>b</sub>π<sub>b</sub>) * x

= (1 - ∑<sub>b</sub>π<sub>b</sub>) * x

= x<sub>a</sub> - ∑<sub>b</sub>π<sub>b</sub>x<sub>b</sub>
### 13.4
∑ τ α π δ γ ∏ Δ Λ μ ∈ ∀ β λ ∞ η θ ∝ σ

d(lnπ) / d(θ<sub>μ</sub>) = d(lnπ) / d(π) * d(π) / d(μ) * d(μ) / d(θ<sub>μ</sub>)

= (1 / π) * π * (-1 / (2 * σ<sup>2</sup>)) * 2 * (a - μ) * (-1) * x

= (1 / σ<sup>2</sup>) * (a - μ) * x

d(lnπ) / d(θ<sub>σ</sub>) = (1 / π) * π * (-1 / 2) * (a - μ)<sup>2</sup> * (-2) * σ<sup>-3</sup> * 2 * σ * x

=((a - μ)<sup>2</sup> / σ<sup>-2</sup>) * x
### 13.5
#### 13.5.1
Pt = e<sup>h1</sup> / (e<sup>h1</sup> + e<sup>h0</sup>)

= 1 / (1 + e<sup>h0</sup> / e<sup>h1</sup>)

= 1 / (1 + e<sup>h0 - h1</sup>)

= 1 / (1 + e<sup>-θ<sup>T</sup>x</sup>)

#### 13.5.3
g = θ<sup>T</sup>x

f = e<sup>-g</sup>; df/dg = -f

π = 1 / (f - 1): By definition of Bernoulli-logistic

dπ / dθ = (dπ / df) * (df / dg) * (dg / dθ)

= (-1) * (f - 1)<sup>-2</sup> * (-f) * x

= (f / (f - 1)) * (1 / (f - 1)) * x

= (f / (f - 1)) * (f / (f - 1) - 1) * x

= π * (π - 1) * x

d(lnπ) / dθ = (1 / π) * π * (π - 1) * x

= (π - 1) * x
