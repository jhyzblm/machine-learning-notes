# Minibatch Stochastic Gradient Descent
As stated in Dive into Deep Learning:

>"The most naive application of gradient descent consists of taking the derivative of the loss function, which is an average of the losses computed on every single example in the dataset. In practice, this can be extremely slow: we must pass over the entire dataset before making a single update, even if the update steps might be very powerful (Liu and Nocedal, 1989). Even worse, if there is a lot of redundancy in the training data, the benefit of a full update is limited.“

>”The other extreme is to consider only a single example at a time and to take update steps based on one observation at a time. The resulting algorithm, stochastic gradient descent (SGD) can be an effective strategy (Bottou, 2010), even for large datasets. Unfortunately, SGD has drawbacks, both computational and statistical. One problem arises from the fact that processors are a lot faster multiplying and adding numbers than they are at moving data from main memory to processor cache. It is up to an order of magnitude more efficient to perform a matrix–vector multiplication than a corresponding number of vector–vector operations. This means that it can take a lot longer to process one sample at a time compared to a full batch. A second problem is that some of the layers, such as batch normalization (to be described in Section 8.5), only work well when we have access to more than one observation at a time.“

# Example: SGD Updates with Two Samples

We want to see how **Stochastic Gradient Descent (SGD)** updates the parameter θ when we have two samples with different targets.

---

## Setup
- Sample 1 prefers **θ = 1**
- Sample 2 prefers **θ = 3**
- Loss function:  
  $$\ell(\theta) = (\theta - y)^2$$
- Gradient:  
  $$\nabla_\theta \ell(\theta) = 2(\theta - y)$$
- Learning rate: **η = 0.1**
- Initial value: **θ₀ = 0**

---

## Update Steps

### Step 1 (Sample 1, target = 1)
- Current θ = 0  
- Gradient = $2(0 − 1) = -2$  
- Update:  
  $$\theta \leftarrow 0 - 0.1 \times (-2) = 0.2$$

➡ θ moves toward 1.

---

### Step 2 (Sample 2, target = 3)
- Current θ = 0.2  
- Gradient = $2(0.2 − 3) = -5.6$  
- Update:  
  $$\theta \leftarrow 0.2 - 0.1 \times (-5.6) = 0.76$$

➡ θ is pulled toward 3.

---

### Step 3 (Sample 1, target = 1)
- Current θ = 0.76  
- Gradient = $2(0.76 − 1) = -0.48$  
- Update:  
  $$\theta \leftarrow 0.76 - 0.1 \times (-0.48) = 0.808$$

➡ θ moves back toward 1.

---

### Step 4 (Sample 2, target = 3)
- Current θ = 0.808  
- Gradient = $2(0.808 − 3) = -4.384$  
- Update:  
  $$\theta \leftarrow 0.808 - 0.1 \times (-4.384) = 1.246$$

➡ θ is pulled toward 3 again.

---

## Conclusion
- **Sample 1** pulls θ toward 1.  
- **Sample 2** pulls θ toward 3.  
- SGD updates oscillate back and forth between the two directions.  
- Over time, θ converges near **θ = 2**, which is the overall optimum (the average of 1 and 3).

>"The solution to both problems is to pick an intermediate strategy: rather than taking a full batch or only a single sample at a time, we take a minibatch of observations (Li et al., 2014). The specific choice of the size of the said minibatch depends on many factors, such as the amount of memory, the number of accelerators, the choice of layers, and the total dataset size. Despite all that, a number between 32 and 256, preferably a multiple of a large power of , is a good start. This leads us to minibatch stochastic gradient descent."




Create Linear Neural Networks
