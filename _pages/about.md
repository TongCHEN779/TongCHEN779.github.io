---
permalink: /
title: "Welcome to Tong CHEN's homepage"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Tong Chen is a postdoc researcher at [Machine Learning Section](https://di.ku.dk/english/research/machine-learning/), University of Copenhagen. His research is mainly about efficient and sustainable machine learning, under the [SustainML](https://sustainml.eu/) project led by [Raghavendra Selvan](https://raghavian.github.io/). 

Previously, Tong obtained his PhD's degree at LAAS-CNRS, [POP](https://www.laas.fr/public/fr/pop/) team, in Toulouse, France, under supervision of [Jean-Bernard Lasserre](https://homepages.laas.fr/lasserre/drupal/home/), [Victor Magron](https://homepages.laas.fr/vmagron/) and [Edouard Pauwels](https://www.irit.fr/~Edouard.Pauwels/) in December 2022. His PhD thesis is about "[Robustness Verification of Neural Networks using Polynomial Optimization](http://thesesups.ups-tlse.fr/5493/)", here are the [slides](http://tongchen779.github.io/files/PhD_defense.pdf) for the defense. He received a master's degree in September 2019 in Statistics and Machine Learning from Université Paris Saclay in France, and a bachelor's degree in June 2017 in Mathematics from [Wuhan University](http://maths.whu.edu.cn/Englishversion/index.htm) in China. Here is his [CV](http://tongchen779.github.io/files/CV_in_english.pdf).

Tong's research interest mainly lies on optimization methods for and with machine learning. 

From optimization aspect, he has developed a second-order SDP relaxation method based on Lasserre's moment-SOS hierarchy, called [Sublevel Hierarchy](https://link.springer.com/article/10.1007/s10589-021-00325-z/),  which can be used for various ML tasks including robustness verification of deep neural networks. He is interested in exploring the sparse structure, together with designing efficient first-/second-order optimization algorithms for large-scale problems arising from ML. For example, the fast first-order SDP solver for robustness verification of DNNs by [Dathathri et al 2020](https://proceedings.neurips.cc/paper/2020/file/397d6b4c83c91021fe928a8c4220386b-Paper.pdf).

From machine learning aspect, he is concerned about the robustness and safety of intelligent systems. In particular, robustness verification of neural networks is a tough problem both theoretically and practically. There always exists a trade-off between efficiency and accuracy for verification algorithms. I have been working for a long time to design efficient polynomial optimization-based algorithms to verify robustness of neural networks, see [Chen et al 2020](https://proceedings.neurips.cc/paper/2020/file/dea9ddb25cbf2352cf4dec30222a02a5-Paper.pdf), [Chen et al 2021](https://proceedings.neurips.cc/paper/2021/file/e3b21256183cf7c2c7a66be163579d37-Paper.pdf). However, polynomial-based approaches can be only applied to small-scaled networks for the moment. The potential and future of polynomial optimization in machine learning is still unknown.

Research Interests
---
* Efficient and sustainable ML, e.g., model compression, quantum computing.
* Polynomial optimization algorithms with/without exploring sparsity structures and their applications.
* Semidefinite programming (SDP) and first-order algorithms for SDP.
* Relaxation techniques for non-convex optimization, e.g., moment relaxation, sum-of-square method.
* First-/second-order optimization algorithms in machine learning.
* Safety and reliability of neural networks, e.g., robustness verification, robust training.
* Mathematical foundation of deep learning.

Ongoing Projects
---
* Fist-Order SDP solver for Higher-Order Moment-SOS Relaxation with Ball Constraints.
* Exactness and Tightness of SDP Relaxation of Neural Network Verification Problem.
* Understanding the Paradox of Robustness Verification and Training.
* (Benchmark for) Polynomial Architecture Search (PAS).
* Data Condensation Combined with Neural Architecture Search.
