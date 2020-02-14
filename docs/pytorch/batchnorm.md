#BatchNormalization
##Why
### Normalization
[Batch Normalization](https://www.youtube.com/watch?v=BZh1ltr5Rkg)

Normalization is a widely used pre-processing trick in machine learning. 
The unbalanced feature scopes make learning rate hard to choose -- the same learning rate 
may explore some features while vanish other features at the same time.

After normalization, the same learning rate imposes same effect on all features, 
which makes the LR easier to choose. And then make use of regularization tools to select preferred features  
### Internal Covariate Shift
The topic is brought up by paper: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
](https://arxiv.org/abs/1502.03167)

_We define Internal Covariate Shift as the change in th distribution of network activations
due to the change in network parameters during training._

An intuitive explanation is that the training of parameters of layer H<sub>n+1</sub> is updated based on distribution of output of H<sub>n</sub>. 
But the sequence of back-propagation if from H<sub>n+1</sub> to H<sub>n</sub>. 
So the ongoing update of different layers may conflict with each other or cancel update effect of each other.

![covirate](./images/covariate.jpg) 

Above figure showed a simple case. 
* In batch t training, H<sub>n</sub> decided to move the output distribution to the right;
H<sub>n+1</sub> also decided to move the distribution to the right.
* The result is that, in batch (t + 1) training, H<sub>n</sub> output fitted the target, while H<sub>n+1</sub> moved too much to miss the target.
* If the learning rate is small enough, H<sub>n+1</sub> may move the output back to match the target, 
while it also pushed output of H<sub>n</sub> away from target. So the parameters of the whole network may be updated in a zigzagged pattern.
* If the learning rate is large enough, the update would be oscillating and fail to converge.

When batch normalization has been introduced, the output the each layer would be in the fixed center with limited scope, 
the network just update the shape to fit into the target. 

Some other papers argued that it may not the root cause of improved performance: [Understanding Batch Normalization](https://arxiv.org/abs/1806.02375)
### Other Observations
* De-correlated features 
* Introduced some noises as _mean(batch)_ is not _E(input)_
* [A Gentle Introduction to Batch Normalization for Deep Neural Networks](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)
* Smooth [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604)
## Algorithm
Besides input standardization, batch-norm also introduced a linear transformation to recover 
representation ability of input for non-linear function in following layer. 
The parameters γ and β are independent of input values and learnable.  
### Forward
The _*Algorithm 1*_ from paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).

The following sequence figure is from [Batch Norm Paper Reading](https://www.youtube.com/watch?v=OioFONrSETc)

![BatchNormForward](./images/batchnormbackprop.jpg)
 
### Backward
Follow the sequence shown in above figure as (reverse) topology sort.
1. ∂l/∂x_hat<sub>i</sub> = ∂l/∂y<sub>i</sub> * ∂y<sub>i</sub>/∂x_hat<sub>i</sub> = ∂l/∂y<sub>i</sub> * γ
2. ∂l/∂γ = Σ(∂l/∂y<sub>i</sub> * ∂y<sub>i</sub>/∂γ) = Σ(∂l/∂y<sub>i</sub> * x_hat<sub>i</sub>)
3. ∂l/∂β = Σ(∂l/∂y<sub>i</sub> * ∂y<sub>i</sub>/∂β) = Σ(∂l/∂y<sub>i</sub>)
4. ∂l/∂σ<sup>2</sup> = Σ(∂l/∂x_hat<sub>i</sub> * ∂x_hat<sub>i</sub>/∂σ<sup>2</sup>) = Σ(∂l/∂x_hat<sub>i</sub> * (-1/2) * (x<sub>i</sub> - μ) * (σ<sup>2</sup> + ǫ)<sup>(-3/2)</sup>)
5. ∂σ<sup>2</sup>/∂μ = (-2/m) * Σ(x<sub>i</sub> - μ)
6. ∂x_hat<sub>i</sub>/∂μ = (-1) * (σ2 + ε)<sup>(-1/2)</sup>
7. ∂x_hat<sub>i</sub>/∂x<sub>i</sub> = (σ2 + ǫ)<sup>(-1/2)</sup>
8. ∂μ/∂x<sub>i</sub> = 1/m
9. ∂σ<sup>2</sup>/∂x<sub>i</sub> = (2/m) * (x<sub>i</sub> - μ)

Then
* ∂l/∂μ = ∂l/∂σ<sup>2</sup> * ∂σ<sup>2</sup>/∂μ + Σ(∂l/∂x_hat<sub>i</sub> * ∂x_hat<sub>i</sub>/∂μ)
* ∂l/∂x<sub>i</sub> = ∂l/∂x_hat<sub>i</sub> * ∂x_hat<sub>i</sub>/∂x<sub>i</sub> + ∂l/∂μ * ∂μ/∂x<sub>i</sub> + ∂l/∂σ<sup>2</sup> * ∂σ<sup>2</sup>/∂x<sub>i</sub> 
### In Inference
In inference, the μ and σ<sup>2</sup> are not estimation of batch input, but the E of all previous input instead. 
* μ = E[μ] = mean(Σ(μ<sub>batch</sub>))
* σ<sup>2</sup> = Var[x] = (m/(m - 1)) * mean(Σσ<sup>2</sup><sub>batch</sub>)

The batch-norm output is 
y = γ * x_hat + β = γ * ((x - μ)/(σ<sup>2</sup> + ε)<sup>(-1/2)</sup>) + β
## Benefits
* Larger learning rate
* Less sensible to parameter initiation
* Less epoch required

[Notes of paper](https://gist.github.com/shagunsodhani/4441216a298df0fe6ab0)
## Other References
[Covariate Shift](https://www.youtube.com/results?search_query=covariate+shift)

[Batch Norm(Chinese)](https://www.youtube.com/watch?v=BZh1ltr5Rkg)

[Batch Normalization(Andrew Ang)](https://www.youtube.com/watch?v=nUUqwaxLnWs)