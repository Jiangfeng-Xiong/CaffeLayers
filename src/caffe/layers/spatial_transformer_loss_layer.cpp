/*
 * SpatialTransformerLosss.cpp
 *
 *  Created on: 8 Oct, 2016
 *      Author: lab-xiong.jiangfeng
 */
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_transformer_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void STLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);

	string prefix = "\t\tST Loss Layer:: LayerSetUp: \t";

	std::cout<<prefix<<"Getting output_h_ and output_w_"<<std::endl;

	output_h_ = this->layer_param_.st_loss_param().output_h();
	output_w_ = this->layer_param_.st_loss_param().output_w();

	std::cout<<prefix<<"output_h_ = "<<output_h_<<", output_w_ = "<<output_w_<<std::endl;

}

template <typename Dtype>
void STLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	vector<int> tot_loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(tot_loss_shape);

	CHECK_EQ(bottom[0]->count(1), 6) << "Inputs theta must have dimension of 6.";

	num_ = bottom[0]->shape(0);

	vector<int> loss_shape(3);
	loss_shape[0] = num_;
	loss_shape[1] = output_h_;
	loss_shape[2] = output_w_;
	loss_.Reshape(loss_shape);

	vector<int> dtheta_tmp_shape(2);
	dtheta_tmp_shape[0] = num_ * 6;
	dtheta_tmp_shape[1] = output_h_ * output_w_;
	dtheta_tmp_.Reshape(dtheta_tmp_shape);

	vector<int> all_ones_vec_shape(1);
	all_ones_vec_shape[0] = output_h_ * output_w_;
	all_ones_vec_.Reshape(all_ones_vec_shape);


	vector<int> U2V_shape(4);
	U2V_shape[0] = num_;
	U2V_shape[1] = 2;
	U2V_shape[2] = output_h_;
	U2V_shape[3] = output_w_;
	source_coordinates_.Reshape(U2V_shape);


	vector<int> V_shape(4);
	V_shape[0] = num_;//share coordinates in (channel and num for target coordinate(-1 ~ +1)
	V_shape[1] = 3;
	V_shape[2] = output_h_;
	V_shape[3] = output_w_;
	target_coordinates_.Reshape(V_shape);


	Dtype* target_corrdinates_data = target_coordinates_.mutable_cpu_data();
	for (int n=0;n < num_;++n){
		for(int h = 0; h < output_h_;++h){
			for(int w = 0; w < output_w_;++w){
				target_corrdinates_data[target_coordinates_.offset(n,0,h,w)] = (Dtype) w/(Dtype)(output_w_-1)*2.0 -1.0; //x

				target_corrdinates_data[target_coordinates_.offset(n,1,h,w)] = (Dtype) h/(Dtype)(output_h_-1)*2.0 -1.0; //y

				target_corrdinates_data[target_coordinates_.offset(n,2,h,w)] = (Dtype) 1.0;//homogeneous coordinate

			}
		}
	}

}

template <typename Dtype>
void STLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//CHECK(false) << "Error: not implemented.";
	// implementd by jfxiong

	Dtype* theta_data = bottom[0]->mutable_cpu_data();
	const Dtype* target_coordinate_data = target_coordinates_.cpu_data();
	Dtype* source_coordinate_data  = source_coordinates_.mutable_cpu_data();

	for(int n = 0; n <num_;++n){

		//compute the source coordinate corresponding to target coordinate using theta
		//C = alpha*A*B + beta* C  A:M(2)*K(3)  B:K(3)*N(h*w)  C:M(2)*N(h*w)
		//A: theta	   B: target_coordinate_data    C: source_coodinate_data
		caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,2,output_h_*output_w_,3,Dtype(1.0),
				theta_data + n * 6 ,target_coordinate_data + n*3*output_h_*output_w_,Dtype(0.0),source_coordinate_data + n*2*output_h_*output_w_);
	}
	int N = num_ * 2 * output_h_ * output_w_;
	caffe_abs<Dtype>(N,source_coordinates_.cpu_data(),source_coordinate_data);
	caffe_add_scalar<Dtype>(N,Dtype(-1.0),source_coordinate_data);

	Dtype total_loss=0;
	for(int n=0;n<N;n++){
		source_coordinate_data[n] = source_coordinate_data[n] >0 ? source_coordinate_data[n]:0;
		total_loss  = total_loss + source_coordinate_data[n];
	}
	top[0]->mutable_cpu_data()[0] = total_loss/N;
}

template <typename Dtype>
void STLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	/* not implemented */
//CHECK(false) << "Error: not implemented.";
	//implemented by jfxiong
	const Dtype* source_coordinate_data  = source_coordinates_.cpu_data();
	const Dtype* target_coordinates_data = target_coordinates_.cpu_data();

	Dtype* dtheta_temp = dtheta_tmp_.mutable_cpu_data();
	Dtype* dtheta = bottom[0]->mutable_cpu_diff();
	Dtype* all_ones_vec = all_ones_vec_.mutable_cpu_diff();

	caffe_set<Dtype>(all_ones_vec_.count(),(Dtype)1,all_ones_vec);

	for(int n = 0 ;n < num_;n++){
		for(int h = 0;h < output_h_ ; h++){
			for(int w =0; w< output_w_; w++){
				Dtype output_x = source_coordinate_data[source_coordinates_.offset(n,0,h,w)];// -1 - +1
				Dtype output_y = source_coordinate_data[source_coordinates_.offset(n,1,h,w)];

				Dtype input_x = target_coordinates_data[target_coordinates_.offset(n,0,h,w)];
				Dtype input_y = target_coordinates_data[target_coordinates_.offset(n,1,h,w)];

				// d{L}/d{output_x}
				Dtype dL_xo = (Dtype)0;
				if(output_x < -1 ){
					dL_xo = -1;
				}
				else if(output_x > 1){
					dL_xo = 1;
				}

				//d{L}/d{output_y}
				Dtype dL_yo = (Dtype)0;
				if(output_y < -1){
					dL_yo = -1;
				}
				else if(output_y > 1){
					dL_yo = 1;
				}

				dtheta_temp[(6*n)*output_h_*output_w_ + h* output_w_ + w] = dL_xo * input_x;
				dtheta_temp[(6*n+1)*output_h_*output_w_ + h* output_w_ + w] = dL_xo * input_y;
				dtheta_temp[(6*n+2)*output_h_*output_w_ + h* output_w_ + w] = dL_xo ;
				dtheta_temp[(6*n+3)*output_h_*output_w_ + h* output_w_ + w] = dL_yo * input_x;
				dtheta_temp[(6*n+4)*output_h_*output_w_ + h* output_w_ + w] = dL_yo * input_y;
				dtheta_temp[(6*n+5)*output_h_*output_w_ + h* output_w_ + w] = dL_yo;
			}
		}
	}
	caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,num_*6,1,output_h_*output_w_,(Dtype)1.,dtheta_temp,all_ones_vec,(Dtype)0.,dtheta);
	caffe_scal<Dtype>(bottom[0]->count(),top[0]->cpu_diff()[0]/(num_*output_h_*output_w_),dtheta);
}

#ifdef CPU_ONLY
STUB_GPU(STLossLayer);
#endif

INSTANTIATE_CLASS(STLossLayer);
REGISTER_LAYER_CLASS(STLoss);

}  // namespace caffe



