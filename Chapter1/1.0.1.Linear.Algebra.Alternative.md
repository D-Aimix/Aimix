# Scalars, Vectors, and Matrices
Data represent real-world entities and events as numbers, text, sound, image, and video. We apply various techniques to manipulate these data to explore and understand the world around us. Linear algebra is an efficient way to represent the data, which are collections of numbers.  There are three general levels of abstractions for this collection of numbers.

## Scalars

**Hamid:** What is the temperature in Houston Texas?

**Alexa**: 99 degree F

Alexa's answer is a single number. In other words, it has only one component - magnitude. Nothing else is needed to specify the number other than the magnitude. Direction is meaningless in this context. A single number that can entirely be specified by its magnitude is called a **Scalar**. Scalar is said to have a rank 0 because there is no confusion on how you state the number. It is the ONLY number. We can also describe scalars as having no axis. The numpy *ndim* attribute shows. The space of all continuous real-valued scalars is $\mathbb{R}$.  Continous here means that you have infinite numbers between any two pairs of numbers. On the other hand, space means containing all possible scalars where addition or scalar multiplications give another scalar in the space.

```python
import numpy as np
import pandas as pd
T = np.array(99)
print(f"Temperature in NY is {T} and it has a rank {T.ndim}")

# output
T is 99 and it has a dimention 0
```


## Vectors 
The next abstraction level is vectors, which are a collection of scalars. 
The elements of a  vector are referred to as components, entries, and coefficients. 

### Vector representations
1. $\vec{v} = \left( v_x, v_y \right)$ where $v_x$ and $v_y$ are the components of the vectors or entries in the vector. 
2. $\vec{v} = v_x\hat{i} + v_y\hat{j}$ where $\hat{i}$ and $\hat{j}$ are vectors of length 1, the **unit vectors**.  $\hat{i}$ starts from the origin, 0, and stops at 1 on the x-axis. $\hat{j}$ does the same on the y-axis.
3. $||\vec{v}||\angle\theta$ Where $||\vec{v}||$ is the length of the vector and $\theta$ is the angle that the vector tor make with the x-axis in a clockwise direction. 

The default vector orientation is vertical - column vector.  Note that a vector is entirely specified by:

1. its magbitude
2. its direction

```python
# Vertical vector - column vector
np.random.seed(5)
vec = np.random.randint(1,10, (2,))
print(vec)

# output
[4 7]
```
There are a few interesting facts about **vec:** 

### Dimension of a Vector
the number of entries in $\vec{vec}$ is the _dimension_ of the subspace in which the vector sits.  This is the same for any vectors in the same subspace. In this case, $\vec{vec}$ is a vector sitting in a 2-dimensional subspace, a plane. An n-dimensional subspace contains vectors with n number of entries along their axis. 

```python
len(vec)

#output
The dimension of the column vector vec is 2.
```

### Rank of a Vector
A vector has a rank of 1. That is, it has only one axis. In other words, you need only one index to retrieve a particular entry from $\vec{vec}$.

```python
print(f"The rank of or the number of axis of vec is {vec.ndim}.")

# Retrieve number 7 from vec
print(f"The number sitting at index 1 is {vec[1]}.")

# Output
The rank of or the number of axis of vec is 1.
The number sitting at index 1 is 7.
```
Observe that $vec_1$ is the same as vec[1] in python.

### Span of a vector
We need to introduce linear combinations before we talk about the span of a vector. Linear combination is the sum of the product of a variable and a scalar for each variables in a set. One example of such linear combinations is the vector representation in terms of the unit vectors.

$\vec{v} = v_x\hat{i} + v_y\hat{j}$ where $\hat{i}$ and $\hat{j}$ are vectors of length 1, the **unit vectors**.  $\hat{i}$ starts from the origin, 0, and stops at 1 on the x-axis. $\hat{j}$ does the same on the y-axis.

These two unit vectors form the basis for the two-dimensional space, a plane. The basis here means any vectors in the 2-dimensional space can be obtained from a linear combination of the two unit vectors. We say the two unit vectors are independent and span the 2-dimensional space because any vectors in the space can be expressed in the form of the two  unit vectors.

```python
i = np.array([1,0]) # unit vector along the x axis
j = np.array([0,1]) # unit vector along the y axis

veccombo = (4*i) + (7*j)
print(veccombo)

#output
[4 7]

```
$\vec{vec}$ and $\vec{veccombo}$ are the same vectors expressed in different ways. 


# Linear decomposition {Draft}
- Solution to a system of equation can be considered a linear decomposition problem
- define linear decomposition
- Illustrate with simple examples
- solve system of linear equations  using linear decomposition


# Norm of a Vector {Draft}
The magnitude or norm of a vector is denoted as $||\vec{v}||$. Dotting a vector with itself gives us the square-norm of the vector. We can obtain the norm by taking the square root of the result as shown in the example below:

$$
\begin{equation*}
\vec{v} = 
\begin{bmatrix}
4 \\
3
\end{bmatrix}
\end{equation*}
$$

$$
\begin{equation*}
\vec{v}^T.\vec{v} = 
\begin{bmatrix}
4 && 3
\end{bmatrix}
\
.
\begin{bmatrix}
4 \\
3 
\end{bmatrix}
\end{equation*}
$$

$$
= \left (4 * 4 \right ) + \left( 3 * 3 \right ) 
= 16 + 9 = 25 \Rightarrow \text{the square norm of} \ \vec{v}
$$
$$
\sqrt{25} = 5 \Rightarrow \text{The norm of} \ \vec{v}
$$

```python
# Dotting a vector with itself
v = np.array([3,4])

sqr_norm = np.dot(v,v) # square norm -> sqr_norm = 25
np.sqrt(sqr_norm) # norm -> norm = 5
```

## The general form of the Norm of a Vector
$$
L_p \ \text{norm} = \left ( \sum_i|v_i|^p \right )^{\frac{1}{p}}
$$