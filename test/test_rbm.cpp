/*
 * test_rbm.cpp
 *
 *  Created on: 2014年3月26日
 *      Author: Feng
 */

#ifdef __TEST_RMB__

#include "../modules/dbn/rbm.h"


int main()
{
	dl::Mat_<float> data(1000, 100);
	dl::randu(data, dl::Scalar::all(0), dl::Scalar::all(100));
	return 1;
}

#endif
