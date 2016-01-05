/**
 *  Some Math and statistical tools
 *  @author: morgan fouesneau
 */


#include "mfstats.hh"
#include <ctime>


/**
 * Returns the mean of an array while filtering for nan values
 *
 *  @param arr      data vector
 *  @param T        type of data
 *  @return mean    mean value
 */
template<typename T> T nan_mean(std::vector<T>& arr){
    size_t N = arr.size();
    T mean = 0.;
    size_t count = 0;
    for (size_t i=0; i<N; ++i){
        if (!std::isnan(arr[i])){
            mean += arr[i];
            count++;
        }
    }
    mean = mean / (T) count;

    return mean;
}

/**
 *  Returns the covariance of an array while filtering for nan values
 *
 *  @param x        array
 *  @param y        array
 *  @return cov    covariance value
 */
template<typename T> T nan_cov(std::vector<T>& x, std::vector<T>& y){
    size_t N = x.size();
    T mx = nan_mean<T>(x);
    T my = nan_mean<T>(y);

    T cov = 0.;
    size_t count = 0;
    for (size_t i=0; i<N; ++i){
        if (!std::isnan(x[i]) && !std::isnan(y[i])){
            cov += (x[i] - mx) * (y[i] - my);
            count++;
        }
    }
    cov = cov / (T) (count - 1);

    return cov;
}

/**
 *  Returns the ariance of an array while filtering for nan values
 *
 *  @param x        array
 *  @param y        array
 *  @return var    variance value
 */
template<typename T> T nan_var(std::vector<T>& x){
    return nan_cov<T>(x, x);
}


/**
 *  Returns the dispersion of an array while filtering for nan values
 *
 *  @param x        array
 *  @param y        array
 *  @return std     standard deviation value
 */
template<typename T> T nan_std(std::vector<T>& x){
    return std::sqrt(nan_var<T>(x));
}


/**
 *  return the number of not nan value in arr
 *
 *  @param arr      array
 *  @return count   number of elements
 */
template<class T> size_t count_notnan (std::vector<T> &arr) {
    size_t N = arr.size();
    size_t count = 0;
    for(size_t i=0; i<N; ++i){
            if (!std::isnan(arr[i])){
                    count++;
            }
    }
    return count;
}
/**
 *  return the number of not nan value in pairs of x and y
 *
 *  @param N        size of the array
 *  @param x        array
 *  @param y        array
 *  @return count   number of elements
 */
template<class T> size_t count_notnan (std::vector<T>& x, std::vector<T>& y) {
    size_t N = x.size();
    size_t count = 0;
    for(size_t i=0; i<N; ++i){
            if (!std::isnan(x[i]) && !std::isnan(y[i])){
                    count++;
            }
    }
    return count;
}

/**
 * Return evenly spaced values within a given interval.
 *
 * @param step      Spacing between values.  
 * @param T         the type of the output array.  
 * @return v        vector of values
 */
template <typename T> std::vector<T> arange(size_t N, T step){
    std::vector<T> r(N);
    for (size_t i = 0; i < N; ++i) {
        r[i] = (T) i * step;
    }
    return r;
}

/**
 * Returns the indices that would sort an array.
 *
 * Perform an indirect sort along the first axis using std::sort
 *
 * @param v     Array to sort against
 * @return idx  vector of sorted indexes
 */
template <typename T> std::vector<size_t> argsort(const std::vector<T>& v) {

    // initialize original index locations
    std::vector<size_t> idx = arange<size_t>(v.size());

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

/**
 * Find the percentile of a list of values.  
 *
 * @param data        a list of values. 
 * @param percent     a float value from 0 to 100.
 * @param sorted      set if the data vector is sorted
 * @return val        the percentile of the values
 */
double percentile(std::vector<double> data, double percent, bool sorted){

    double k = (data.size() - 1) * percent * 0.01;
    size_t f = std::floor(k);
    size_t c = std::ceil(k);

    if (sorted){
        if (f == c){
            return data[(size_t) k];
        } else {
            return data[f] * (c - k) + data[c] * (k - f); 
        }
    } else {
        std::vector<size_t> idx = argsort(data);

        if (f == c){
            return data[idx[(size_t) k]];
        } else {
            return data[idx[f]] * (c - k) + data[idx[c]] * (k - f); 
        }
    }
}

/**
 * Find the percentile of a list of values.  
 *
 * @param data        a list of values. 
 * @param percentiles a float value from 0 to 100.
 * @param sorted      set if the data vector is sorted
 * @return val        the percentile of the values
 */
std::vector<double> percentile(std::vector<double> data, std::vector<double> percentiles){

    std::vector<double> output(percentiles.size());

    std::vector<size_t> idx = argsort(data);

    for (size_t i = 0; i < percentiles.size(); ++i) {
        double percent = percentiles[i];
        double k = (data.size() - 1) * percent * 0.01;
        size_t f = std::floor(k);
        size_t c = std::ceil(k);

        if (f == c){
            output[i] = data[idx[(size_t) k]];
        } else {
            output[i] = data[idx[f]] * (c - k) + data[idx[c]] * (k - f); 
        }
        
    }
    return output;
}

/**
Compute weighted percentiles.

If the weights are equal, this is the same as normal percentiles.  Elements of
the data and wt arrays correspond to each other and must have equal length.

@param data  data points (ndata,)
@param percentiles sequence of percentile values between 0 and 100 (nperc,)
@param weights weights of each data point (ndata, ) All the weights must be
               non-negative and the sum must be greater than zero.

@return p the weighted percentiles of the data.

percentile
----------
A percentile is the value of a variable below which a certain percent of
observations fall.
The term percentile and the related term percentile rank are often used in
the reporting of scores from *normal-referenced tests*, 16th and 84th
percentiles corresponding to the 1-sigma interval of a Normal distribution.
Note that there are very common percentiles values:
* 0th   = minimum value
* 50th  = median value
* 100th = maximum value

Weighted percentile
-------------------
A weighted percentile where the percentage in the total weight is counted
instead of the total number. *There is no standard function* for a weighted
percentile.

Implementation
--------------
The method implemented here extends the commom percentile estimation method
(linear interpolation beteeen closest ranks) approach in a natural way.  Suppose
we have positive weights, \f$W= [W_i]\f$, associated, respectively, with our
\f$N\f$ sorted sample values, \f$D=[d_i]\f$. Let \f$S_n = \sum_{i=0..n} {w_i}\f$
the the n-th partial sum of the weights. Then the n-th percentile value is
given by the interpolation between its closest values \f$v_k, v_{k+1}\f$:
\f$v = v_k + (p - p_k) / (p_{k+1} - p_k) * (v_{k+1} - v_k)\f$
where
\f$p_n = 100/S_n * (S_n - w_n/2)\f$
 */
std::vector<double> percentile(std::vector<double> data,
		std::vector<double> percentiles,
		std::vector<double> weights){
    
    size_t ndata = data.size();
    size_t nperc = percentiles.size();

    // check arguments
    if (weights.size() != ndata){
        std::invalid_argument("Shape mismatch: weights must be of same length as data");
    }

    for (size_t i = 0; i < nperc; ++i) {
        if ((percentiles[i] < 0) || (percentiles[i] > 100)){
            std::invalid_argument("Percentile values must be within [0,100]");
        }
    }

    // sort data by values
    std::vector<size_t> idx = argsort(data);
    
    // copy data into a sorted array sd
    //      weights into sorted weights sw
    //  and make the cumulative distribution of weights aw
    
    std::vector<double> sd(ndata);
    std::vector<double> sw(ndata); 
    for (size_t i = 0; i < ndata; ++i) {
        sd[i] = data[idx[i]];
        sw[i] = weights[idx[i]];
    }
    // build cumulative distribution
    std::vector<double> aw(ndata); // cumulative sum
    aw[0] = sw[0];
    for (size_t i = 1; i < ndata; ++i) {
        aw[i] = aw[i - 1] + sw[i];
    }

    if (aw[ndata - 1] <= 0){
        std::runtime_error("Non-positive weighted sum");
    }

    std::vector<double> w(ndata); // effective weights
    for (size_t i = 1; i < ndata; ++i) {
        w[i] = (aw[i] - 0.5 * sw[i]) / aw[ndata - 1];
    }
    //
    // find index of each percentile value and interpolate
    std::vector<double> output(nperc);
    for (size_t p = 0; p < nperc; ++p) {
        double pval = percentiles[p] * 0.01;
        // find index
        size_t ind = 0;
        while ((w[ind] < pval) && (ind < ndata)){
            ind ++;
        }
        // interpolate
        if (ind == 0) {
            output[p] = sd[0];
        } else if (ind == ndata) {
            output[p] = sd[ndata - 1];
        } else {
            double f1 = (w[ind] - pval) / (w[ind] - w[ind - 1]);
            double f2 = (pval - w[ind - 1]) / (w[ind] - w[ind - 1]);
            if (std::abs(f1 + f2 - 1.0) > 1e-6){
                std::runtime_error("Error during interpolation");
            }
            output[p] = sd[ind - 1] * f1 + sd[ind] * f2;
        }
    }
    return output;
}

void test_percentiles()
{
    // define random generator
    std::random_device rd;
    std::mt19937 generator(rd());

    std::normal_distribution<double> randn(0, 1);
    size_t N = 100000;

    std::vector<double> values(N);
    std::vector<double> weights(N);
    for (size_t i = 0; i < N; ++i) {
        values[i] = randn(generator);
        weights[i] = 1.;
    }

    std::cout << "Gaussian Stats " 
              << std::setw(12) << nan_mean(values) 
              << " +/- " 
              << std::setw(12) << nan_std(values)
              << " | not nan " << count_notnan(values)
              << std::endl;

    // set values to compute
    std::vector<double> perc = {0.13, 15.87, 50, 84.13, 99.87};
    std::cout << std::setw(10) << " ";
    for (size_t i = 0; i < perc.size(); ++i) {
        std::cout << std::setw(12) << perc[i];
    }
    std::cout << std::setw(12) << "time" << std::endl;
    
    // unweighted values
    std::clock_t start = std::clock();
    std::vector<double> p(perc.size());
    for (size_t i = 0; i < perc.size(); ++i) {
        p[i] = percentile(values, perc[i]);
    }
    std::clock_t end = std::clock();
    std::cout << std::setw(10) << "unweighted";
    for (size_t i = 0; i < perc.size(); ++i) {
        std::cout << std::setw(12) << p[i];
    }
    std::cout << std::setw(12) << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    
    // unweighted values
    start = std::clock();
    std::vector<double> p1 = percentile(values, perc);
    end = std::clock();
    std::cout << std::setw(10) << "unweighted";
    for (size_t i = 0; i < perc.size(); ++i) {
        std::cout << std::setw(12) << p[i];
    }
    std::cout << std::setw(12) << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    // weighted version
    start = std::clock();
    std::vector<double> pw = percentile(values, perc, weights);
    end = std::clock();

    std::cout << std::setw(10) << "weighted";
    for (size_t i = 0; i < perc.size(); ++i) {
        std::cout << std::setw(12) << pw[i];
    }
    std::cout << std::setw(12) << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

}

// int main(int argc, char *argv[])
// {
//     test_percentiles();    
//     return 0;
// }
// vim: expandtab:ts=4:softtabstop=4:shiftwidth=4
