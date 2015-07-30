# pyGibbsLDA
Python Implementation of Collapsed Gibbs Sampling for Latent Dirichlet Allocation (LDA)

## Module Usage
```python
>>> import pyGibbsLDA
>>> root = /home2/
>>> sampler = pyGibbsLDA.Sampler(root+"TwitterData.dat", 100)
>>> likelihood = sampler.run(500, 300, 2)
```