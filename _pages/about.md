---
permalink: /
title: "Welcome to Tong CHEN's homepage"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Tong Chen is a postdoc researcher with [Raghavendra Selvan](https://raghavian.github.io/) at University of Copenhagen. His current research is mainly about efficient and sustainable machine learning, under the [SustainML](https://sustainml.eu/) project. 

Previously, Tong obtained his PhD's degree at LAAS-CNRS, [POP](https://www.laas.fr/public/fr/pop/) team, in Toulouse, France, under supervision of [Edouard Pauwels](https://www.irit.fr/~Edouard.Pauwels/), [Victor Magron](https://homepages.laas.fr/vmagron/), and [Jean-Bernard Lasserre](https://homepages.laas.fr/lasserre/drupal/home/) in December 2022. He received a master's degree in September 2019 in Statistics and Machine Learning from Université Paris Saclay in France, and a bachelor's degree in June 2017 in Mathematics from [Wuhan University](http://maths.whu.edu.cn/Englishversion/index.htm) in China.

# Research Interests

## 1. Efficient Machine Learning

Large model and dataset play important roles in mordern deep learning. We, on the other hand, are trying to investigate how to make DL model/dataset smaller while preserving the target downstream performance. In theory, it's essential to show whether (and how) one is able to achieve similar performance with smaller models or less data. In practice, we should also be aware that generalization (which is almost the unique consideration in most cases) might be misleading, and a wider range of factors (such as robustness, privacy, fairness, etc.) should be considered.

- Dataset condensation: This includes model-based and model-free methods, optimising in data space and feature space, and DC for generic purpose, e.g., accuracy, robustness, privacy, fairness, etc.

- Data-driven model compression: Similar with dataset condensation, we are interested in model-free (data-driven) model compression methods.

## 2. Polynomial Optimization for Machine Learning (POP4ML)

I have developed an SDP relaxation framework based on Lasserre's moment-SOS hierarchy and its sparse variant, called [Sublevel Hierarchy](https://link.springer.com/article/10.1007/s10589-021-00325-z/),  which can be used for various ML tasks including robustness verification of deep neural networks (see [[Chen et al., 2020](https://proceedings.neurips.cc/paper/2020/file/dea9ddb25cbf2352cf4dec30222a02a5-Paper.pdf)], [[Chen et al., 2021](https://proceedings.neurips.cc/paper/2021/file/e3b21256183cf7c2c7a66be163579d37-Paper.pdf)]). Other potential applications include:

- Robust classification: Robust classifiers can be formulated by polynomial optimization. However, it's not scalable to large-scale ML problems and is sometimes not numerically stable. It would be a nice alternative of adversarial training once those issues are mitigated.

- Implementation of higher-order moment-SOS problems in ML framework: Moment problems have been implemented a lot in Julia/Matlab, but not in Python. This prevents researchers from studying ML problems using moment-SOS tools efficiently. Successful applications include JAX implementation of first-order SDP relaxation, but extension is needed.

## 3. Machine Learning for Polynomial Optimization (ML4POP)

Polynomials are GOOD (in theory), BAD (in practice), and UGLY (from theory to practice). My hope is to make them also GOOD in pactice, and beautiful from theory to practice. In order to approach that, one possible way is to automate the acceleration techniques (such as sparsity, low-rank decomposition) of optimization instead of static methods.

- Learning dynamic SOS proof using RL method: Krivine-Stengle and Putinar's positivstellensatz (P-satz) are common tools to certify positivity of a polynomial. However, existing static hierarchies suffer from numerical intractability. Using RL method such as DQN to dynamically search the optimal hierarchies is a promising and challenging open problem for SOS proof.

- Low-rank decomposition of PSD matrices: In many first-order algorithms for solving SDPs, we frequently pre-decompose a large PSD matrix to the product of two smaller matrices, which is called the rank. However, the rank is a hyperparamter and is usually tuned empirically starting from 1. The idea is to estimate the rank of a matrix using ML models to improve the tuning efficiency.
