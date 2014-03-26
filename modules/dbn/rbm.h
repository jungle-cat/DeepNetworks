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
		/*!
		 * @param Construct a Restricted Boltzmann Machine.
		 * @param nvisi Total number of visible units.
		 * @param nhidd Total number of hidden units.
		 * @param w The Weights between visible units and hidden units.
		 * @param biasv Bias of visible units.
		 * @param biash Bias of hidden units.
		 */
		RBM( std::size_t nvisi, std::size_t nhidd, const Mat_<Tp>& w = Mat_<Tp>(),
			 const Mat_<Tp>& biasv = Mat_<Tp>(), const Mat_<Tp>& biash = Mat_<Tp>() )
		{
			// TODO check the parameter validity.
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

		/*!
		 * @brief Train a RBM given specific parameters.
		 * @param x The training data with each sample per row.
		 * @param nepochs
		 * @param batch_size
		 * @param learningrate
		 * @param momentum
		 * @return
		 */
		double train( const Mat_<Tp>& x, int nepochs, int batch_size, double learningrate = 0.1, double momentum = 0 )
		{
			// TODO check the parameter validity.
			const int num = x.rows;
			const int nbatches = num / batch_size;

			double err = 0;
			Mat_<Tp> w_m = Mat_<Tp>::zeros(weights.size());
			Mat_<Tp> v_m = Mat_<Tp>::zeros(bias_vis.size());
			Mat_<Tp> h_m = Mat_<Tp>::zeros(bias_hid.size());

			for ( std::size_t i = 0; i < nepochs; ++i) {
				auto td = shuffle(x, 1);
				err = 0;

				for ( auto j = 0; j < nbatches; ++j) {
					Mat_<Tp> batch = td.rowRange(j*batch_size, (j+1)*batch_size);

					double err_epoch = train_epoch(batch, w_m, v_m, h_m, learningrate, momentum);

					err += err_epoch;
				} // end of each batch training
				std::cerr << "epoch " << i << " : " << err / static_cast<double>(nbatches) << std::endl;
			} // end of each epoch training

			return err / static_cast<double>(nbatches);
		}

	private:
		/*!
		 * @brief
		 * @param units
		 * @param w
		 * @param b
		 * @param out
		 */
		void propagate( const Mat_<Tp>& units, const Mat_<Tp>& w, const Mat_<Tp>& b, Mat_<Tp>& out )
		{
			if ( units.cols != w.rows || w.cols != b.cols )
				throw invalid_argument("Error: propagate"); //TODO Implement specific error exception.

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


		void sigmrand( const Mat_<Tp>& data, Mat_<Tp>& out )
		{
			auto size = data.size();
			Mat_<Tp> sigm;
			exp(-data, sigm);

			Mat_<Tp> sample(size);
			randu(sample, Scalar::all(0), Scalar::all(1));

			Mat ret = 1 / (1 + sigm) > sample;
			ret.convertTo(out, out.type(), 1/255.0);
		}

		Mat_<Tp> sigmrand( const Mat_<Tp>& data )
		{
			Mat_<Tp> ret;
			sigmrand(data, ret);
			return ret;
		}

		double train_epoch( const Mat_<Tp>& samples, Mat_<Tp>& weight_m, Mat_<Tp>& biasvis_m, Mat_<Tp>& biashid_m, double learningrate, double momentum )
		{

			// for all hidden units compute the probability of 1 and perform gibbs sample
			auto v1 = samples;
			auto h1 = sigmrand( propagate(v1, weights.t(), bias_hid) );

			// for all visible units compute the probability of 1 and perform gibbs sample
			auto v2 = sigmrand( propagate(h1, weights, bias_vis) );
			auto h2 = static_cast<Mat_<Tp> >( sigm(propagate(v2, weights.t(), bias_hid)) );



			//TODO trick and exception using reduce with integer type.
			// while it's ok using float/double type matrix.
			Mat_<Tp> dc = h1.t() * v1 - h2.t() * v2, dv, dh;
			reduce(v1 - v2, dv, 0, CV_REDUCE_SUM);
			reduce(h1 - h2, dh, 0, CV_REDUCE_SUM);

			learningrate /= static_cast<double>(samples.rows);
			addWeighted(weight_m, momentum, dc, learningrate, 0, weight_m);
			addWeighted(biasvis_m, momentum, dv, learningrate, 0, biasvis_m);
			addWeighted(biashid_m, momentum, dh, learningrate, 0, biashid_m);
//			weight_m = momentum * weight_m + learningrate * dc;
//			biasvis_m = momentum * biasvis_m + learningrate * dv;
//			biashid_m = momentum * biashid_m + learningrate * dh;

			weights += weight_m;
			bias_vis += biasvis_m;
			bias_hid += biashid_m;

			double err = pow(norm(v1 - v2, NORM_L2), 2) / static_cast<double>(samples.rows);
			return err;
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
