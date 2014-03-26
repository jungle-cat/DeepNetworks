/*
 * base.h
 *
 *  Created on: 2014-3-25
 *      Author: Feng
 */

#ifndef __BASE_H__
#define __BASE_H__

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <opencv/cv.h>

namespace dl {

using namespace cv;
using namespace std;




/*!
 * @param dim. shuffle the matrix by rows (1) or by columns(0) or by elements (-1)
 */
template <typename Tp>
void shuffle( const Mat_<Tp>& data, Mat_<Tp>& out, int dim )
{
	if (dim < 0) {
		out = data.clone();
		randShuffle(out);
	}

	if (dim == 0) {
		vector<int> indices(data.cols);
		iota(indices.begin(), indices.end(), 0);
		random_shuffle(indices.begin(), indices.end());

		out.create(data.size());
		// construct new output matrix
		for (std::size_t i = 0; i < indices.size(); ++i) {
			int idx = indices[i];
			data.col(idx).copyTo(out.col(i));
		}
	}

	if (dim > 0 ) {
		vector<int> indices(data.rows);
		iota(indices.begin(), indices.end(), 0);
		random_shuffle(indices.begin(), indices.end());

		out.create(data.size());
		for (std::size_t i = 0; i < indices.size(); ++i) {
			int idx = indices[i];
			data.row(idx).copyTo(out.row(i));
		}
	}
}

/*!
 * @param dim. shuffle the matrix by rows (1) or by columns(0) or by elements (-1)
 */
template <typename Tp>
Mat_<Tp> shuffle( const Mat_<Tp>& data, int dim )
{
	Mat_<Tp> out;
	shuffle(data, out, dim);
	return out;
}

template <typename Tp>
void sigm( const Mat_<Tp>& data, Mat_<Tp>& out )
{
	Mat_<Tp> s;
	exp(-data, s);

	s = 1 / (1 + s);
	if (out.empty())
		out = s;
	else
		s.copyTo(out);
}


template <typename Tp>
Mat_<Tp> sigm( const Mat_<Tp>& data )
{
	Mat_<Tp> ret;
	sigm(data, ret);
	return ret;
}



}

#endif /* __BASE_H__ */
