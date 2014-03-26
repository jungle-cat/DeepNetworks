/*
 * main.cpp
 *
 *  Created on: 2014年3月25日
 *      Author: Feng
 */

#include "modules/dbn/rbm.h"
#include "modules/core/utils.h"

#include <ctime>




int main()
{
	std::clock_t t = std::clock();
	dl::Mat_<double> data;
	auto t2 = std::clock();
	std::cerr << static_cast<double>(t2-t) / CLOCKS_PER_SEC << "\t" << static_cast<double>(t2-t) / CLOCKS_PER_SEC << std::endl;

	dl::load_data("D:/a.txt", data);

//	randu(data, dl::Scalar::all(0), dl::Scalar::all(1));
	auto t3 = std::clock();
	std::cerr << static_cast<double>(t3-t) / CLOCKS_PER_SEC << "\t" << static_cast<double>(t3-t2) / CLOCKS_PER_SEC << std::endl;




	dl::RBM<double> rbm(784,100);
	rbm.train(data, 20, 100, 1, 0);

	return 1;
}
