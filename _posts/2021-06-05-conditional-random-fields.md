---
title: 'Conditional Random Fields'
date: 2021-06-05
show_published_date: false
show_title: true
permalink: /posts/conditional_random_fields
tags:
  - machine learning
---

Conditional random fields are a class of discriminative models that uses the dependencies/overlapping information between the neighbors of data points in a sequence. These models are best used in prediction scenarios where the relationship between neighbors affects the current prediction.

The main idea behind CRF models is that we have feature functions with outputs that equal $0$ or $1$ depending on the constraints on the input given to the feature function. More specifically, we have a set of feature functions that look at a pair of neighboring labels $l_i$ and $l_{i-1}$, the whole input sequence $X$, and the current position in the sequence $i$ and output a real value (usually $0$ or $1$). Each feature function is of the form:

$$f_1(l,l_{i-1},X,i) = \begin{cases} 1 \text { if } l_i = A \text { and } l_{i-1} = B \\ 0 \text { otherwise}\end{cases}$$

The choice of the conditions $A$ and $B$ depends on the specific task and function. Note that one can use any or all of the feature function input in the conditional, not just $l_i$ and $l_{i-1}$.

We take a sum of the weighted outputs of all $M$ features across all $N$ points in the input sequence as the score for the label $l$ on the input sequence $X$. More formally, we have

$$z(l|s)=\sum_{j}^M\sum_i^N\lambda_jf_j(l_i,l_{i-1},X,i)$$

$\lambda$ is the scalar weight associated with each feature representing how much the feature should contribute to the score and can is learned.

We can then transform the scores into a normalized probability distribution by exponentiating and dividing by the sum of the exponents. We have:

$$p(l|s)=\frac {\exp(z(l|s))} {\sum_k \exp(z(l_k|s))}$$

This looks an awful lot like logistic regression and in a sense it is. Conditional random fields is a sequential version of logistic regression.

If you're familiar with Hidden Markov Model (HMM), one conceptual difference between HMM and CRF is that CRF defines conditional probabilities while HMM defines joint probabilities. Also, HMMs are generative while CRFs are discriminative.
