/*
 * utils.h
 *
 *  Created on: 2014年3月26日
 *      Author: Feng
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <fstream>
#include <sstream>
#include <opencv/cv.h>

namespace dl {

using namespace std;
using namespace cv;

template <typename Tp>
void load_data( const string& name, Mat_<Tp>& data )
{
	fstream file(name, ios_base::in);
	if (! file.is_open())
		throw "Error: failed to load file\n";

	string line;
	int count = 0;
	while (getline(file, line)) {
		vector<Tp> d;
		stringstream ss(line);
		Tp c;
		while ( ss >> c){
			d.push_back(c);
		}

		Mat_<Tp> t = static_cast<Mat>(d).t();
		if (count == 0) {
			data.create(1, d.size());
			t.copyTo(data);
		}
		else
			data.push_back(t);
		count ++;
	}
}

}

#endif /* __UTILS_H__ */
