/**
 * @file mfstats.hh
 * @brief Some statistics
 *
 * @author Morgan Fouesneau, 
 *   Organization:  
 *
 */

#ifndef __MFSTATS_H__
#define __MFSTATS_H__

#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

template<typename T> T nan_mean(std::vector<T>& arr);

template<typename T> T nan_cov(std::vector<T>& x, std::vector<T>& y);

template<typename T> T nan_var(std::vector<T>& x);

template<typename T> T nan_std(std::vector<T>& x);

template<typename T> size_t count_notnan(std::vector<T>& x, std::vector<T>& y);

template<typename T> size_t count_notnan(std::vector<T>& arr);

template <typename T> std::vector<T> arange(size_t N, T step=(T) 1);

template <typename T> std::vector<size_t> argsort(const std::vector<T>& v);

std::vector<double> percentile(std::vector<double> data, 
		                       std::vector<double> percentiles,
		                       std::vector<double> weights);

std::vector<double> percentile(std::vector<double> data, 
                               std::vector<double> percentiles);

double percentile(std::vector<double> data, double percent, bool sorted=false);
	
#endif
// vim: expandtab:ts=4:softtabstop=4:shiftwidth=4
