Fitting a straight line with non-symmetric uncertainties
========================================================

This directory contains a C version of the python code. This version was mostly
a personal exercise and does not offer all the options (nor the figure) that the
python version offers. However, it is pretty fast!

Just run make to generate the executable code

.. code:: bash

        make

Options are given with the usual `-h` or `--help`

.. code:: bash

        Usage:
          ./fit [OPTION...] Command line usage

          -h, --help  Display help message

         Data options:

          -i, --input arg       Input data file
          -n, --nsamples arg    Number of samples from data to generate the
                                probability distribution
          -b, --bootstrap       set to bootstrap the data (not only their likelihood)
              --errorfloor arg  threshold under which errors are not reliable
              --logxnorm arg    x-data normalization value
              --logynorm arg    y-data normalization value
              --xfloor arg      floor of x-value uncertainty (in %)
              --yfloor arg      floor of y-value uncertainty (in %)

         Mock Data options:

              --mock        Set to run mock data sampling
              --mock_N arg  Number of data points in mock sample
              --mock_a arg  slope of the mock data
              --mock_b arg  intercept of the mock data
              --mock_d arg  intrinsic dispersion of the mock data
