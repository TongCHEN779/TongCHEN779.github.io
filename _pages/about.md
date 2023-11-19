---
permalink: /
title: "Welcome to Tong CHEN's homepage"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Tong Chen is a postdoc researcher with [Raghavendra Selvan](https://raghavian.github.io/) at [Machine Learning Section](https://di.ku.dk/english/research/machine-learning/), University of Copenhagen. His current research is mainly about efficient and sustainable machine learning, under the [SustainML](https://sustainml.eu/) project. 

Previously, Tong obtained his PhD's degree at LAAS-CNRS, [POP](https://www.laas.fr/public/fr/pop/) team, in Toulouse, France, under supervision of [Jean-Bernard Lasserre](https://homepages.laas.fr/lasserre/drupal/home/), [Victor Magron](https://homepages.laas.fr/vmagron/) and [Edouard Pauwels](https://www.irit.fr/~Edouard.Pauwels/) in December 2022. His PhD thesis is about "[Robustness Verification of Neural Networks using Polynomial Optimization](http://thesesups.ups-tlse.fr/5493/)", here are the [slides](http://tongchen779.github.io/files/PhD_defense.pdf) for the defense. He received a master's degree in September 2019 in Statistics and Machine Learning from Université Paris Saclay in France, and a bachelor's degree in June 2017 in Mathematics from [Wuhan University](http://maths.whu.edu.cn/Englishversion/index.htm) in China. Here is his [CV](http://tongchen779.github.io/files/CV_in_english.pdf).

# Research Interests

## 1. Efficient machine learning with provable guarantee and robustness.

Opposite to the big trend in deep learning, we are trying to investigate how to compress/condense ML model/data both in theory and in practice. In theory, we are investigating how to propose compression method with provable performance guarantee, or how to verify the performance of a compressed model/data with provable guarantee. We are also drawing the community's attention from test accuracy to other metrics, e.g., compression should not only aim for generalization, but also robustness, privacy, fairness, etc. 

In practice, we are trying to do training-free compression. This includes evaluate the generalization property of a model structure without training, or compare the distribution of datasets without training. We are also trying to understand the difference between features learned by a large model and a compressed one, and how to extract useful features and remove useless features.

## 2. Polynomial optimisation for and with machine learning

We have developed a second-order SDP relaxation method based on Lasserre's moment-SOS hierarchy, called [Sublevel Hierarchy](https://link.springer.com/article/10.1007/s10589-021-00325-z/),  which can be used for various ML tasks including robustness verification of deep neural networks (see [[Chen et al., 2020](https://proceedings.neurips.cc/paper/2020/file/dea9ddb25cbf2352cf4dec30222a02a5-Paper.pdf)], [[Chen et al., 2021](https://proceedings.neurips.cc/paper/2021/file/e3b21256183cf7c2c7a66be163579d37-Paper.pdf)]). He is interested in exploring the sparse structure, together with designing efficient first-/second-order optimization algorithms for large-scale problems arising from ML. For example, the fast first-order SDP solver for robustness verification of DNNs by [[Dathathri et al., 2020](https://proceedings.neurips.cc/paper/2020/file/397d6b4c83c91021fe928a8c4220386b-Paper.pdf)].

Another promising question is how to use ML model to learn the sparsity structures in moment-SOS relaxation. The only related work is called "dynamic polynomial proof" by [[Fawzi et al., 2019](https://arxiv.org/abs/1906.01681)], which use RL algorithm to find sparse Krivine-Stengle P-satz representations. It's natural to investigate how to extend this idea to SOS proof.