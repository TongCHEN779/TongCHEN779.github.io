---
title: 'A Comparison of Probabilistic and Statistic View of Deep Learning Models'
date: 2024-03-20
permalink: /posts/proba/
tags:
  - Probabilistic
  - Statistic
  - Bayesian
  - VAE
  - Diffusion
---

Background
---
We use lower case $x$ to denote the observed sample, and upper case $X$ to denote the associated unknown variable. We denote by $X, Y, Z$ the observation, latent, and lable, respectively. In most cases, only the variable $X$ can be fully observed, the label $Y$ might be partially observed, and the latent $Z$ is unobservable. 

Now we have three probabilistic worlds: the (unobserved) true world $p$, the observed world $\hat{p}$, and the approximated world $q$. The true world is never observable, and the approximated world is derived from our prior and posterior knowledge. Therefore, the goal is to let our knowledge as close to the ground true as possible, i.e., minimize the probability discrepancy between $\hat{p}$ and $q$. For the following discussion, we will no longer distinguish $p$ and $\hat{p}$ and denote both by $p$ for simplicity.

Evidence Lower Bound (ELBO)
---
We have an observed variable $X$ and one latent variable $Z$, the joint distribution $p(X, Z)$ can be parameterized by 

$$
p(X, Z) = p (Z) p (X|Z) = \theta (Z) \theta (X|Z)
$$

or 

$$
p(X, Z) = p (X) p (Z|X)\approx p (X) \phi (Z|X)
$$

In the cases where computing the evidence $\log \phi (x)$ is intractable (due to mixture of distributions), we would rather use the marginal 

$$
\log \int \phi (x, z) \text{d} z = \int \phi (x|z) \phi (z) \text{d} z
$$

If the integral over $z$ is tractable, then EM algorithm can be applied. Otherwise, we need to optimize its lower bound, called evidence lower bound. For any observed sample $x$, the log likelihood is lower bounded by

$$
\log p (x) \ge \mathbb{E}_{Z \sim \phi (\cdot|x)} \bigg[\log \frac{p (x, Z)}{\phi (Z|x)}\bigg] = ELBO (x)
$$

Actually, $\log p (x) = ELBO (x) + \mathcal{D}_{KL} (\phi (\cdot|x) \| p(\cdot|x))$, which means we must pay a price if we use an approximation of the posterior $p(\cdot|x)$. Futhermore, ELBO can be written as the difference between a reconstruction error and a prior matching error:

$$
ELBO (x) = \underbrace{\mathbb{E}_{Z \sim \phi (\cdot|x)} [\log \theta (x|Z)]}_{\text{reconstruction error}} - \underbrace{\mathcal{D}_{KL} (\phi (\cdot|x) \| \theta (\cdot))}_{\text{prior matching error}}
$$

On the other hand,

$$
ELBO (x) = \underbrace{\log p(x)}_{\text{likelihood}} - \underbrace{\mathcal{D}_{KL} (\phi (\cdot|x) \| p (\cdot|x))}_{\text{posterior matching error}}
$$

We see that, by maximizing ELBO, we simutaneously minimize the KL-divergence between pair $(\phi(\cdot|x), \theta(\cdot))$ and $(\phi (\cdot|x), p(\cdot|x))$. The former one ensures that $\phi$ is indeed learning some distribution, and the latter one ensures that $\phi$ is getting close to the real distribution.

Density Estimation
---

Classification/Regression
---

Variational Autoencoder (VAE)
---

Markovian Hierarchical VAE (MHVAE)
---

Variational Diffusion Model (VDM)
---