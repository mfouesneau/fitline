Fitting a straight line with non-symmetric (but Gaussian) uncertainties
=======================================================================

This repository is an exercise I had with **Iskren Georgiev** when he was
looking at linear correlations between the stellar mass of galaxies and the
properties of their central nuclear star cluster.

The approach here is neither frequentist nor Bayesian but more like in between.
we explored how to quickly get fits through his data even when the uncertainties
were non-symmetric or upper limits.

Example usage
-------------

.. code::
   :language: python

    python template.py <file> -N 200 -n 10
    python template.py <file> -N 200 -n 1 --xnorm 6 --ynorm 0.5 --xfloor 10 --yfloor 10

Definition of the problem
-------------------------

If we define the data set D with uncertainties E of :math:`{d_k, e_k}`
independent observations and a model M predicting what the data should look
like. It comes through Bayes rule:

.. math::

    p(M | D, E) = 1 / Z  prod_k P(d_k| M, e_k) P(M)

:math:`Z` is the evidence (normalization factor), :math:`P(d_k| M, e_k)` the
likelihood the the :math:`k`-th data point, and :math:`P(M)` the prior on the
model (or its parameters).


Non-symmetric uncertainties
---------------------------

This becomes trivial when uncertaintes are non-correlated by observing that it
corresponds to a skewed normal distribution. The latter can be very simply
defined by the closed form of a split normal distribution.

The split normal distribution arises from merging two opposite halves of two
probability density functions (PDFs) of normal distributions in their common
mode.

The PDF of the split normal distribution is given by:

.. math::

    f(x;\mu,\sigma_1,\sigma_2)= A \exp (- \frac {(x-\mu)^2}{2 \sigma_1^2})    if  x < \mu
    f(x;\mu,\sigma_1,\sigma_2)=   A \exp (- \frac {(x-\mu)^2}{2 \sigma_2^2})   otherwise

where

.. math::

    A = \sqrt{2/\pi} (\sigma_1+\sigma_2)^{-1}.


Finite Data Sample
------------------

The finite data can be either discarded or accounted for by bootstrapping the
data. In other words, do multiple fits (with the above) to get one value per
resampling and combine them to get a PDF by integrating the different bootstrap
realizations.


Fitting a straight line to data
-------------------------------

We define our model as follow:

.. math::

    y_{model}(x | \alpha, \beta) = \alpha \, x + \beta

I assume here that we have perfect "fake" data to work with, :math:`(x_{data}, y_{data})`, 
so that the MAP is actually trivial to get from the covariance of the sample:

.. math::

    C = Cov(x_{data}, y_{data})

    \alpha = \frac{C[1,0]}{C[0,0] ^ 2} = \frac{\rho_{x,y}\sigma_x\sigma_y}{\sigma_x ^ 2}

    \beta = \overline{y_{data}} - \alpha \, \overline{x_{data}}


Hat trick: sampling from the data
---------------------------------

The approximation that we propose to do is to replace the data point :math:`d_k` by a
sampling of :math:`d_k` following the distribution given by :math:`e_k`. Here we suppose
normally distributed uncertainties.

it comes that either you can do

.. math::

    p(D | M) p(M) = prod_k \sum_j (P(d_{k,j}| M) \, P(M)),

when you use all the samples once, or

.. math::

    p(D | M) p(M) = \sum_j \prod_k (P(d_{k,j}| M) \, P(M)),

in the more proper way. The latter is marginalization of the posterior
distribution over noise, the former is faking by virtually expanding the data.

It is clear that both version will not lead to the same result, in particular
the final PDF of :math:`M`.

In this version of the script I consider the former trick implementation as it
also give me the possibility to do both with the same code. Simply increase the
bootstrapping number of realizations and put the sampling to 1 random sample.


.. Note::

    I actually ran both modes and get the same answer, which I don't fully
    understand.



Folding in the intrinsic dispersion
-----------------------------------

So far, everything we have done has implicitly assumed that there truly is a
narrow linear relationship between $x$ and $y$ (or there would be if
they were both measured with negligible uncertainties).

Since there are very few problems in science, especially in astrophysics, in
which a relationship between two observables is expected to be either narrow or
linear, we have to revise at least the narrow assumption.

The reasons for intrinsic scatter are not always obvious.  Commonly the
quantities being measured are produced or affected by a large number of
additional, unmeasured or unmeasurable quantities. The relation between
$x$ and $y$ is rarely exact, even in when observed by theoretically
perfect, noise-free observations.

Proper treatment of these problems gets into the complicated area of estimating
density given finite and noisy samples. However, we will limit ourselves to a
linear relation with some intrinsic scatter. As such, we will instead revise our
model that generates the data and infer the parameters of that model.

We introduce an intrinsic Gaussian variance $\sigma^2$, orthogonal to the line. In
this case, the parameters of the relationship become $(\alpha, \beta, \sigma^2)$

In this case, each data point :math:`(x_i, y_i)` can be treated as being drawn
from a projected distribution function that is a convolution of the projected
uncertainty Gaussian.

Adapting from linear algebra projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From a realization drawn from our data, we search for the only model that fits
to this perfect dataset. Hence the intrinsic dispersion perpendicular to the
linear model is also deterministically known.

Projection is a standard linear algebra technique.

A slope :math:`\alpha` can be described by a unit vector :math:`v` orthogonal to
the line representing the model:

.. math::

    v = \frac{1}{\sqrt{ 1 + m ^2}} . (-m, 1)^T

The orthogonal displacement :math:`\Delta_i` of each data point :math:`x_i,y_i`
from the line is given by:

.. math::

        \Delta_i = v^T . (d_i - \overline{d_i}),


The variance is then given by the usual unbiased estimator:

.. math::

        \sigma ^ 2 = \frac{1}{N-1}\sum_i \| \Delta_i \| ^ 2 = \frac{1}{N-1}\sum_i \| v^T . (d_i - \overline{d_i}) \| ^ 2
