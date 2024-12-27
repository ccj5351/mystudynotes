---
layout: post
title:  "Matrix Rank"
date:   2024-12-26 11:18:12 -0700
categories: math, machine learning
---

## Linear independence

In the theory of vector spaces, a set of vectors is said to be `linearly independent` if there exists no nontrivial linear combination of the vectors that equals the zero vector. If such a linear combination exists, then the vectors are said to be linearly dependent. These concepts are central to the definition of dimension.

A vector space can be of finite dimension or infinite dimension depending on the maximum number of linearly independent vectors. The definition of linear dependence and the ability to determine whether a subset of vectors in a vector space is linearly dependent are central to determining the dimension of a vector space.

### Definition

**Linearly dependent**

A sequence of vectors $\\{ v_i \\}_{i=1}^{k}$ from a vector space $V$ is said to be linearly dependent, if there exist scalars $a_1, a_2, \dots, a_k$ not all zero, such that:

$$a_{1}\mathbf {v} _{1}+a_{2}\mathbf {v} _{2}+\cdots +a_{k}\mathbf {v} _{k}=\mathbf {0} $$

where $\mathbf {0}$ denotes the zero vector.

This implies that at least one of the scalars is nonzero, say $a_{1}\neq 0$, and the above equation is able to be written as

$${ \mathbf {v} _{1}={\frac {-a_{2}}{a_{1}}}\mathbf {v} _{2}+\cdots +{\frac {-a_{k}}{a_{1}}}\mathbf {v} _{k},} \text{ if } k>1. $$

Or $\mathbf {v} _{1}=\mathbf {0}$ if $k=1$.

Thus, a set of vectors is linearly dependent if and only if one of them is zero or a linear combination of the others.


**Linearly Independent** 

A sequence of vectors $\mathbf {v}_{1},\mathbf {v}_{2},\dots,\mathbf {v}_{n}$ is said to be linearly independent if it is not linearly dependent, that is, if the equation

$$ a_{1}\mathbf {v} _{1}+a_{2}\mathbf {v} _{2}+\cdots +a_{n}\mathbf {v} _{n}=\mathbf {0}$$

can only be satisfied by ${\displaystyle a_{i}=0}$ for $i=1,\dots, n$. 

This implies that no vector in the sequence can be represented as a linear combination of the remaining vectors in the sequence. In other words, a sequence of vectors is linearly independent if the only representation of $\mathbf {0}$ as a linear combination of its vectors is the trivial representation in which all the scalars $a_{i}$ are zero. Even more concisely, a sequence of vectors is linearly independent if and only if $\mathbf{0}$ can be represented as a linear combination of its vectors in a unique way.

If a sequence of vectors contains the same vector twice, it is necessarily dependent. The linear dependency of a sequence of vectors does not depend of the order of the terms in the sequence. This allows defining linear independence for a finite set of vectors: A finite set of vectors is linearly independent if the sequence obtained by ordering them is linearly independent. In other words, one has the following result that is often useful.

A sequence of vectors is linearly independent if and only if it does not contain the same vector twice and the set of its vectors is linearly independent.


## Rank of matrix

In linear algebra, the rank of a matrix $A$ is the dimension of the vector space generated (or spanned) by its columns. This corresponds to the maximal number of linearly independent columns of $A$. This, in turn, is identical to the dimension of the vector space spanned by its rows. Rank is thus a measure of the `"nondegenerateness"` of the system of linear equations and linear transformation encoded by $A$. There are multiple equivalent definitions of rank. A matrix's rank is one of its most fundamental characteristics.

The rank is commonly denoted by rank(_A_) or rk(_A_); sometimes the parentheses are not written, as in rank  _A_.


## Matrix Rank Definition
In this section, we give some definitions of the rank of a matrix. 

The `column rank` of A is the dimension of the column space of A, while the row rank of A is the dimension of the row space of A.

A fundamental result in linear algebra is that the column rank and the row rank are always equal. This number (i.e., the number of linearly independent rows or columns) is simply called the rank of A.

A matrix is said to have full rank if its rank equals the largest possible for a matrix of the same dimensions, which is the lesser of the number of rows and columns. I.e., given a matrix $A_{m \times n}$
$$\text {rank}A \leq min(m, n)$$

- If $\text {rank}A \leq min(m, n)$, then $A$ is called `full rank` matrix.
- A matrix is said to be rank-deficient if it does not have full rank. The rank deficiency of a matrix is the difference between the lesser of the number of rows and columns, and the rank.

The rank of a linear map or operator  $\Phi$ is defined as the dimension of its image:

$$ \operatorname {rank} (\Phi ):=\dim(\operatorname {img} (\Phi ))$$
where $\dim$  is the dimension of a vector space, and $\operatorname {img}$ is the image of a map.

> Recall:   
> In mathematics, for a function $f:X\to Y$, the image of an input value $x$ is the single output value produced by $f$ when passed $x$.   
> The preimage of an output value $y$ is the set of input values that produce $y$.  
> More generally, evaluating $f$ at each element of a given subset $A$ of its domain $X$ produces a set, called the "image of $A$ under (or through)  $f$".   
> Similarly, the inverse image (or preimage) of a given subset 
$B$ of the codomain $Y$ is the set of all elements of $X$ that map to a member of $B$.  

---

## Examples

The matrix 

$$ A={\begin{bmatrix}1&0&1\\0&1&1\\0&1&1\end{bmatrix}} $$ 

has rank 2: the first two columns are linearly independent, so the rank is at least 2, but since the third is a linear combination of the first two (the first column plus the second), the three columns are linearly dependent so the rank must be less than 3.

The matrix 

$$ A={\begin{bmatrix}1&1&0&2\\-1&-1&0&-2\end{bmatrix}} $$

has rank 1: there are nonzero columns, so the rank is positive, but any pair of columns is linearly dependent. 

Similarly, the transpose

$$A^{\mathrm {T} }={\begin{bmatrix}1&-1\\1&-1\\0&0\\2&-2\end{bmatrix}}$$
 
of $A$ has rank 1. Indeed, since the column vectors of $A$ are the row vectors of the transpose of $A$, the statement that the column rank of a matrix equals its row rank is equivalent to the statement that the rank of a matrix is equal to the rank of its transpose, i.e., $\operatorname {rank} (A) = \operatorname {rank} (A^T)$.
