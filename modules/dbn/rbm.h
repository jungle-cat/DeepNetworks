/*
 * rbm.h
 *
 *  Created on: 2014-3-25
 *      Author: Feng
 */

#ifndef __RBM_H__
#define __RBM_H__

#include "../core/base.h"

namespace dl {


template <typename Tp = double>
class RBM
{
	public:
		RBM( const Mat_<Tp>& x, std::size_t nvisi, std::size_t nhidd,
			 const Mat_<Tp>& w = Mat_<Tp>(), const Mat_<Tp>& biasv = Mat_<Tp>(),
			 const Mat_<Tp>& biash = Mat_<Tp>() )
		{
			num_vis = nvisi;
			num_hid = nhidd;

			if (w.empty()) {
				auto low = Scalar::all(-4 * sqrt(6.0 / static_cast<double>(num_hid + num_vis)));
				auto high = Scalar::all(4 * sqrt(6.0 / static_cast<double>(num_hid + num_vis)));

				weights.create(num_hid, num_vis);
				randu(weights, low, high);
			}
			else
				w.copyTo(weights);

			if (biasv.empty())
				bias_vis = Mat_<Tp>::zeros(1, num_vis);
			else
				biasv.copyTo(bias_vis);

			if (biash.empty())
				bias_hid = Mat_<Tp>::zeros(1, num_hid);
			else
				biash.copyTo(bias_hid);
		}

		double train( const Mat_<Tp>& x, int nepochs, int batch_size, double learningrate = 0.1, double momentum = 0 )
		{
			const int num = x.rows;
			const int nbatches = num / batch_size;

			double err = 0;
			for ( std::size_t i = 0; i < nepochs; ++i) {
				auto td = shuffle(x, 1);
				err = 0;

				for ( auto j = 0; j < nbatches; ++j) {
					auto batch = td.rowRange(j*batch_size, (j+1)*batch_size);

					auto w_m = Mat_<Tp>::zeros(weights.size());
					auto v_m = Mat_<Tp>::zeros(bias_vis.size());
					auto h_m = Mat_<Tp>::zeros(bias_hid.size());
					double err_epoch = train_epoch(batch, w_m, v_m, h_m, learningrate, momentum);

					err += err_epoch;
				} // end of each batch training
			} // end of each epoch training

			return err / static_cast<double>(nbatches);
		}

	private:
		void propagate( const Mat_<Tp>& units, const Mat_<Tp>& w, const Mat_<Tp>& b, Mat_<Tp>& out )
		{
			if ( units.cols != w.cols || w.cols != b.cols )
				throw "Error."; //TODO Implement specific error exception.

			Mat_<Tp> t;
			if (units.rows != 1) {
				repeat(b, units.rows, 1, t);
			}
			else t = b;

			Mat_<Tp> ret = units * w + t;
			if (out.empty())
				out = ret;
			else
				ret.copyTo(out);
		}

		Mat_<Tp> propagate( const Mat_<Tp>& units, const Mat_<Tp>& w, const Mat_<Tp>& b )
		{
			Mat_<Tp> ret;
			propagate(units, w, b, ret);
			return ret;
		}

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

		Mat_<Tp> sigm( const Mat_<Tp>& data )
		{
			Mat_<Tp> ret;
			sigm(data, ret);
			return ret;
		}

		void sigmrand( const Mat_<Tp>& data, Mat_<Tp>& out )
		{
			auto size = data.size();
			Mat_<Tp> sigm;
			exp(-data, sigm);

			Mat_<Tp> sample(size);
			randu(sample, Scalar::all(0), Scalar::all(1));

			Mat ret;
			threshold(1 / (1 + sigm) > sample, ret, 127, 1, THRESH_BINARY);
			ret.convertTo(out, out.type());
		}

		Mat_<Tp> sigmrand( const Mat_<Tp>& data )
		{
			Mat_<Tp> ret;
			sigmrand(data, ret);
			return ret;
		}

		double train_epoch( const Mat_<Tp>& samples, Mat_<Tp>& weight_m, Mat_<Tp>& biasvisi_m, Mat_<Tp>& biashidd_m, double learningrate, double momentum )
		{
			// for all hidden units compute the probability of 1 and perform gibbs sample
			auto v1 = samples;
			auto h1 = sigmrand( propagate(v1, weights.t(), bias_vis) );
			// for all visible units compute the probability of 1 and perform gibbs sample
			auto v2 = sigmrand( propagate(h1, weights, bias_hid) );
			auto h2 = static_cast<Mat_<Tp> >( sigm(propagate(v2, weights.t(), bias_vis)) );

			auto dc = h1.t() * bias_vis - h2.t() * bias_hid;
			reduce(v1 - v2, v2, 0, CV_REDUCE_SUM);
			reduce(h1 - h2, h2, 0, CV_REDUCE_SUM);

			weight_m = momentum * weight_m + learningrate * dc;
			bias_vis = momentum * bias_vis + learningrate * h2;
			bias_hid = momentum * bias_hid + learningrate * v2;

			double err = norm(v1 - v2, NORM_L2) / static_cast<double>(samples.rows);
			return err*err;
		}

	public:
		Mat_<Tp> weights;  //!< weights between visible and hidden units
		Mat_<Tp> bias_vis; //!< visible bias
		Mat_<Tp> bias_hid; //!< hidden bias

		std::size_t num_vis; //!< number of visible units
		std::size_t num_hid; //!< number of hidden units

};



}
#endif /* __RBM_H__ */
