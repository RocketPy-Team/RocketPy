.. _sensitivity-theory:

Sensitivity Analysis - Theory
=============================

Introduction
------------

Sensitivity analysis is a set of techniques used to quantify a system's variability due 
to different sources of uncertainty. For rocketry simulators, this amounts to the study 
of the deviation between the observed and the nominal flight.

From all sources of variation, there are four of major importance:

1. **Rocket Physics model**: consists of the physics models used in rocketry. It encompasses
which rocketry elements we can incorporate such as different types of motors, aerodynamic
surfaces, and other rocket components along with the mathematical equations used to describe them.

2. **Numerical approximations**: consists of how well we can solve the physics equations.
Analytic solutions are seldomly available, and therefore we must resort on numerical
approximations.

3. **Weather forecast**: consists of how well the environment is predicted. Accurate predictions 
are crucial for rocketry simulation as many components are influenced by it.

4. **Measurement uncertainty**: consists of measurement errors. Every instrument has a limited
precision, which causes us to simulate flights with parameters values that are not the true
values but should be somewhat close.

Accurate predictions requires analyzing carefully each source of variation. 
The first two sources of variation are naturally handled by the simulator
itself in the implementation of rocketry components and computational methods.
Weather forecasting might be directed implemented in the software or
aided by the use of another specialized simulator. Sensitivity analysis tackles the 
last source of uncertainty by quantifying how the variability in rocket parameters 
causes variability in the variables of interest.

This document provides the mathematical justification for a used of tool that aids the 
practitioner in deciding which parameters he should more accurately measure.

As a motivating example, imagine a rocket designer who wishes to accurately estimate the 
apogee, i.e. the maximal altitude reached by his rocket. His rocket has many parameters, 
most of which are measured with limited precision. This limited precision in the input 
parameters results in variability in the apogee's estimate. 

How can he be more certain can that the rocket will reach a certain altitude? One
approach is to reduce the uncertainty due to parameter measurement, which boils down to 
answering the following question: **which parameters would reduce the variability of 
the apogee the most if they were measured with greater precision?**

Mathematical Modeling
---------------------

Defining the system
~~~~~~~~~~~~~~~~~~~

Let $x\\,\\in\\,\\mathbb{R}^p$ be the vector of input parameters, 
$t\\,\\in\\, \\mathbb{R}_{+}$ be the time variable, and 
$f: \\mathbb{R}_{+}\\times\\mathbb{R}^p \\longrightarrow \\, \\mathbb{R}^d$ be a 
deterministic function that simulates the phenomena of interest. We assume that $f$ is 
an intractable function, meaning that we do not have analytical equations for it.

Studying the system or phenomena translates of studying the function $f$ itself.
For rocketry simulators, the parameters $x$ usually consist of rocket, motor, and 
environment properties relevant for the simulation, and $f(t, x)$ is the trajectory 
point in the $3$ dimensional space at time $t$. The regular use of a simulator consists 
in specifying $x^*$ as the vector of input parameters and studying how $f(t, x^*)$ 
evolves in time. The input parameter $x^*$ is called the nominal parameter value and 
$f(t, x^*)$ is the nominal trajectory.

For a more robust analysis, the user can recognize that his nominal parameters $x^*$ 
might incorrectly reflect the true parameters $x$. The true value $x$ can never be 
known in practice, but we expect that $x^*$ is a good approximation of those values. 
Hence, instead of just analyzing $f(t, x^*)$, he can analyze the nominal trajectory 
and its sensitivity to variations around $x^*$, providing more appropriate conclusions 
to the ideal simulated trajectory $f(t, x)$. Note that $f(t, x)$ will still deviate 
from the real trajectory due to the other three sources of uncertainty, but it will 
be more accurate than $f(t, x^*)$.

Despite the simulator function $f$ being complicated and intractable, we can compute 
its values for any input $x$ and for time values $t$ in a discrete-time grid 
$\\mathcal{T}$. The sensitivity of $f$ with respect to $x$ can be modeled through a 
Monte Carlo approach. But first, we need to define what are target variables for a 
trajectory.

Target variables
~~~~~~~~~~~~~~~~

A target variable $y = y(x)$ is a quantity obtained from the trajectory 
:math:`f(t, x)`  at one specific time instant. For instance, when studying rocket 
trajectories that return to land, the point in which the trajectory attains its maximum 
altitude value is called the apogee. The time until apogee is reached, $t_a$, depends 
on the input parameters, hence $t_a = t_a(x)$. The apogee can then be defined 
as :math:`y(x) = f(t_a(x), x)`.
    
Another example would be the impact point. If we let $y \\, \\in \\, \\mathbb{R}^2$ 
denote the coordinates of the impact point on earth's surface, then the time until 
impact, $t_i$, is also a function of $x$ so that $t_i = t_i(x)$. The impact point would 
then be defined as $y(x) = f(t_i(x), x)$. We could even consider the time until impact, 
$t_i(x)$, as the target variable itself.

Precise prediction of target variables is, sometimes, more important than precise 
prediction of the whole trajectory itself. For instance, having an accurate prediction 
of the landing point is important both for rocket recovery as well as safety regulations 
in competitions. Accurately predicting the apogee is important for rocket competitions 
and payload delivery in rockets.
    
The important takeaway is that target variables are a snapshot of the trajectories so 
its analysis is somewhat simpler. This simplicity comes in handy since it allows us to 
better model the uncertainty due to input parameter error.


Sensitivity analysis using regression
-------------------------------------

Statistical Modeling
~~~~~~~~~~~~~~~~~~~~

From now on, we assume we are modeling a target variable 
$y = y(x) = g(x) \\, \\in \\, \\mathbb{R}^d$. We will assume that $g \\, \\in \\, C^2$.
The first-order Taylor series expansion of $g(x)$ around $x^*$ is given by

$$g(x) = g(x^*) + J_{g}(x^*)(x-x^*) + o(||x-x^*||)\\quad,$$

where $J_{g}$ is the Jacobian of $g$ with respect to $x$.
Recall that the Jacobian expresses the first-order variation of $g$ with respect to 
variations of $x$ around $x^*$, making it a key concept in sensitivity analysis.
    
Since $f$ is intractable, so is $g$ and $J_{g}(x^*)$.
We now show how to estimate the Jacobian using Monte Carlo and regression.

Assume that, for each parameter of interest in $x^* = (x_j^*)_{j=1}^p$, we have a prior 
standard deviation $\\sigma_j$ representing the variability of the true value $x_j$ 
around the nominal value $x_j^*$. The standard deviations can be obtained, for instance, 
from the precision of the instrument used to measure that parameter, e.g. the precision 
of the balance used to measure the rocket total mass. Hence, we consider that the true 
value is a random variable $X$ around $x^*$ and with the specified uncertainty. That is, 
a Gaussian distribution centered in $x^*$ and that the different components of the 
input $X_j$ are independent and each has variance $\\sigma_j^2$, or, more compactly, 
$X \\sim \\mathcal{N}(x^*, D)$ with $D = (diag(\\sigma_j^2))_{j=1}^p$.   

Substituting $x$ by the random variable $X$ and replacing the Taylor expansion error 
terms by a random error 
$\\epsilon \\sim \\mathcal{N}_d(\\mathbf{0}_d, \\Sigma_{d\\times d})$ independent of 
$X$, the conditional distribution of the first-order approximation of 
$\\tilde{Y}$ given $X$ is

$$\\tilde{Y} = g(x^*) + J_{g}(x^*)(X-x^*) + \\epsilon \\sim \\mathcal{N}(g(x^*) + J_{g}(x^*)(X-x^*), \\Sigma_{\\epsilon} ) \\quad.$$

When we replaced the approximation error $o(||x-x^*||)$ by a random error $\\epsilon$, 
the variance of $\\epsilon$ is the conditional variance-covariance matrix of 
$\\tilde{Y}$ given $X$. The $j$-th diagonal term of $\\Sigma_{\\epsilon}$ is the 
variance of $\\tilde{Y}_j$, while the element $(\\Sigma_{\\epsilon})_{jk}$ represent 
the covariance between $\\tilde{Y}_j$ and $\\tilde{Y}_k$.

Assume that we draw Monte Carlo samples 
$X^{(i)} \\overset{i.i.d.}{\\sim}\\mathcal{N}(x^*, D)$ and 
compute the values $Y^{(i)} = g(X^{(i)})$ for all $i\\,\\in\\,[n]$. Then

$$g(X^{(i)}) - g(x^*) \\overset{i.i.d.}\\sim \\mathcal{N}(J_{g}(x^*)(X^{(i)}-x^*), \\Sigma_{\\epsilon}) \\quad.$$

The nominal parameters $x^*$ and nominal target variable $y^* = g(x^*)$ are known.
The Jacobian $J_g(x^*)$ and $\\Sigma_{\\epsilon}$ can be estimated using a multivariate 
linear regression of $X$ on $Y = g(X)$.

**Case $d = 1$** The regression approach is best understood considering the simplest 
case when $d = 1$. Indeed, we have the usual case of multiple linear regression.
The Jacobian is simply the gradient $J_{g}(x^*) = \\nabla g(x^*)$.
Write $\\nabla g(x^*) = \\beta = (\\beta_1, \\ldots, \\beta_p)$, where the coefficient 
$\\beta_j$ is exactly the linear approximation coefficient of $g(x)$ around $x^*$ for 
the $j$-th input parameter.

Denoting target variable vector as $\\mathbf{Y} = \\mathbf{Y}_{n\\times 1}$, 
$\\mathbf{Y^*} = \\mathbf{Y^*}_{n\\times 1} = \\begin{bmatrix} y^*, \\ldots, y^* \\end{bmatrix}^T$ 
the nominal target variable repeated in a vector, the input parameter matrix as 
$\\mathbf{X} = \\mathbf{X}_{n\\times p}$, the regression coefficient vector by 
$\\beta = \\beta_{p\\times 1}$ and the error vector by 
$\\mathbf{\\varepsilon} = \\mathbf{\\varepsilon}_{n\\times 1}$, the regression model 
can be written as

$$
\\mathbf{Y} - \\mathbf{Y^*} = (\\mathbf{X} - \\mathbf{X^*})\\beta + \\varepsilon \\sim \\mathcal{N}_n(\\mathbf{X} - \\mathbf{X^*})\\beta, \\sigma^2 I_{n\\times n})\\quad,
$$

where $\\mathbf{X^*} = \\begin{bmatrix} x^* \\\\ \\vdots \\\\ x^* \\end{bmatrix}$, 
a matrix repeating the nominal parameters at each row.

A good example where this would be the case is when performing sensitivity analysis 
for the apogee only.

**Case $d > 1$** This is case requires the use of multivariate multiple linear regression.
The Jacobian is an $n \\times d$ matrix so that the regression coefficients are also a 
matrix $\\mathbf{B} = (\\mathbf{B}_1, \\ldots, \\mathbf{B}_d)$. The term $\\mathbf{B}_i$ 
is the $i$-th column of $\\mathbf{B}$ and $\\mathbf{B}_{ij}$ is the regression 
coefficient of the $j$-th parameter for the $i$-th variable.

If the variance-covariance matrix $\\Sigma_{\\epsilon}$ is diagonal, then we can just 
fit $d$ separate multiple linear regressions as explained above. If not, then there is 
a correlation between the target variables and we should also estimate it along with 
the variances.

Denoting target variable matrix as $\\mathbf{Y} = \\mathbf{Y}_{n\\times d}$, $\\mathbf{Y^*} = \\mathbf{Y^*}_{n\\times d} = \\begin{bmatrix} y^* \\\\ \\vdots \\\\ y^* \\end{bmatrix}$ the nominal target variable repeated in a matrix, the input parameter matrix as $\\mathbf{X} = \\mathbf{X}_{n\\times p}$, the regression coefficient vector by $\\mathbf{B} = \\mathbf{B}_{p\\times d}$ and the error matrix by $\\mathbf{E} = \\mathbf{E}_{n\\times d}$, the regression model can be written as

$$\\mathbf{Y} - \\mathbf{Y^*}  = (\\mathbf{X} - \\mathbf{X^*})\\mathbf{B} + \\mathbf{E} \\sim \\mathcal{N}_{n\\times d}(\\mathbf{X} - \\mathbf{X^*})\\mathbf{B}, I_{n\\times n} \\otimes \\Sigma_{\\epsilon})\\quad.$$


A good example where this would be the case is when performing sensitivity analysis for 
the impact point. Here, we would have $d = 2$ and there is a correlation between the two 
target variables.

Sensitivity coefficients
~~~~~~~~~~~~~~~~~~~~~~~~

Remember that our goal is to obtain which parameters are important and which are not.
To that end, we need to define the sensitivity coefficient.
The coefficient of a parameter should take into account both 
how much the target variable changes its values depending on that parameter and 
the prior uncertainty in that parameter.

Hence, the sensitivity coefficient should be a metric that answers the following 
question: **how much would the variability of the target variable decrease if we knew 
the true value of the parameter with certainty?**

For the mathematical formulation, we will consider $d = 1$ since it is easily interpretable.
The same calculations can be extended when $d > 1$.

The regression model provides the conditional variance $Var(Y|X = x) = \\sigma_\\epsilon^2$.
However, this conditional variance is just the variability due to first-order Taylor series expansion.
Our true interest resides on $Var(Y)$ and how it depends on $\\beta$.
Assuming $\\epsilon$ is uncorrelated to $X - x^*$, we have

$$Var(Y) = \\sigma_{\\epsilon}^2 + J_{f}(x^*) D [J_{g}(x^*)]^T= \\sigma_{\\epsilon}^2 + \\beta D \\beta^T\\quad.$$

Hence,

$$Var(Y) =\\sigma_{\\epsilon}^2 +  \\sum_{j=1}^p \\sigma_j^2 \\beta_j^2\\quad.$$

We define the sensitivity coefficient of the $j$-th parameter by its relative 
contribution to the total variance in percentage

$$S(j) = 100 \\times \\frac{\\beta_j^2\\sigma_j^2}{\\sigma_{\\epsilon}^2 + \\sum_{k=1}^p \\sigma_k^2 \\beta_k^2} \\quad.$$

The estimator is then

$$\\hat{S}(j) = 100 \\times \\frac{\\hat{\\beta}_j^2\\sigma_j^2}{\\hat{\\sigma}_{\\epsilon}^2 + \\sum_{k=1}^p \\sigma_k^2 \\hat{\\beta}_k^2}  \\quad.$$

Note that $\\beta_j$ and $\\sigma_\\epsilon$ are replaced by their estimators computed 
in the linear regression, but $\\sigma_j$ does not need to be estimated since its is 
known beforehand.

The coefficient represents by what factor would the total variance $Var(Y)$ reduce if we 
knew the true value of that parameter. For instance, if $S(j) = 20\\%$ and we could 
remeasure the $j$-th parameter with certainty so that $\\sigma_j^2 = 0$, then 
$Var(Y)$ would reduce by $20\\%$. 

It is important to observe that the **sensitivity coefficient is a local measure**.
We are performing a local sensitivity analysis in the sense that we are studying how $f$
depends on $x$ around $x^*$. A better notation for it would be $S(j, x^*)$ 
representing the importance of the $j$-th parameter around the nominal parameter $x^*$. 
We prefer to omit the reference to $x^*$ but emphasize that, if $x^*$ is changed, then 
we need to perform the sensitivity analysis again.

Evaluating the model
~~~~~~~~~~~~~~~~~~~~

The results of sensitivity analysis should not be taken at face value.
Along the way to obtain equations for the sensitivity coefficient, we made assumptions.
The most critical assumption is, of course, using a first-order Taylor series expansion.
Even though the simulator function $f$ is certainly non-linear and complicated, a linear 
approximation is justified as long as we are performing the sensitivity analysis around 
a neighborhood of $x^*$.

If the parameters standard deviations $\\sigma_j$ are too large, then the linear 
approximation error might be too large and invalidate the analysis.
We can compute the linear approximation error (LAE) in the same scale of the parameter 
importance by

$$LAE = 100 \\times \\frac{\\sigma_{\\epsilon}^2}{\\sigma_{\\epsilon}^2 + \\sum_{k=1}^p \\sigma_k^2 \\beta_k^2}$$

The estimator for the $LAE$ is then

$$\\widehat{LAE} = 100 \\times \\frac{\\hat{\\sigma}_{\\epsilon}^2}{\\hat{\\sigma}_{\\epsilon}^2 + \\sum_{k=1}^p \\sigma_k^2 \\hat{\\beta}_k^2}$$

If the $\\widehat{LAE}$ is more relevant than all parameters in the model, then we might 
opt for a non-linear model approximation, possibly a quadratic regression including 
interaction terms. Currently, our approach only covers the linear case.

.. seealso::

    For a practical example of sensitivity analysis with code, see :ref:`sensitivity-practical`