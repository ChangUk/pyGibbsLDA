# pyGibbsLDA
Python Implementation of Collapsed Gibbs Sampling for Latent Dirichlet Allocation (LDA)

## Develop environment
* Language: Python3
* Prerequisite libraries: [Scipy](http://scipy.org), [Numpy](http://numpy.org), [matplotlib](http://matplotlib.org)

## Input data format
DocumentID \t WordID \t Count \n

## Module usage example
```python
>>> import pyGibbsLDA
>>> root = /home2/
>>> sampler = pyGibbsLDA.Sampler(root+"TwitterData.dat", 100)
>>> likelihood = sampler.run(500, 300, 2)
```

## Reference
* [http://parkcu.com/blog/gibbs-sampling-for-topic-models](http://parkcu.com/blog/gibbs-sampling-for-topic-models)
