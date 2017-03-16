#include<vector>
#include<cmath>
#include<algorithm>

#include"caffe/layer.hpp"
#include"caffe/layers/spatial_transformer_layer.hpp"
#include"caffe/util/math_functions.hpp"

#include<iostream>

namespace caffe{

	template<typename Dtype>
	void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top){

		//initialize parameter from probuf
		output_h_ = bottom[0]->shape(2);
		if(this->layer_param_.st_param().output_h()){
			output_h_ = this->layer_param_.st_param().output_h();
		}

		output_w_ = bottom[0]->shape(3);
		if(this->layer_param_.st_param().output_w()){
			output_w_ = this->layer_param_.st_param().output_w();
		}

		if(is_pre_defined_theta[0]=this->layer_param_.st_param().has_theta_1_1()){
			pre_defined_theta[0] = this->layer_param_.st_param().theta_1_1();
			std::cout<<"using predefined theta_1_1: "<<pre_defined_theta[0]<<std::endl;
		}
		if(is_pre_defined_theta[1]=this->layer_param_.st_param().has_theta_1_2()){
			pre_defined_theta[1] = this->layer_param_.st_param().theta_1_2();
			std::cout<<"using predefined theta_1_2: "<<pre_defined_theta[1]<<std::endl;
		}
		if(is_pre_defined_theta[2]=this->layer_param_.st_param().has_theta_1_3()){
			pre_defined_theta[2] = this->layer_param_.st_param().theta_1_3();
			std::cout<<"using predefined theta_1_3: "<<pre_defined_theta[2]<<std::endl;
		}
		if(is_pre_defined_theta[3]=this->layer_param_.st_param().has_theta_2_1()){
			pre_defined_theta[3] = this->layer_param_.st_param().theta_2_1();
			std::cout<<"using predefined theta_2_1: "<<pre_defined_theta[3]<<std::endl;
		}
		if(is_pre_defined_theta[4]=this->layer_param_.st_param().has_theta_2_2()){
			pre_defined_theta[4] = this->layer_param_.st_param().theta_2_2();
			std::cout<<"using predefined theta_2_2: "<<pre_defined_theta[4]<<std::endl;
		}
		if(is_pre_defined_theta[5]=this->layer_param_.st_param().has_theta_2_3()){
			pre_defined_theta[5] = this->layer_param_.st_param().theta_2_3();
			std::cout<<"using predefined theta_2_3: "<<pre_defined_theta[5]<<std::endl;
		}

	}

	template<typename Dtype>
	void SpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> & bottom,
			const vector<Blob<Dtype>*>& top){


			num_ = bottom[0]->shape(0);
			channel_ = bottom[0]->shape(1);

			input_h_ = bottom[0]->shape(2);
			input_w_ = bottom[0]->shape(3);

			//reshape V
			vector<int> top_shape(4);
			top_shape[0] = num_;
			top_shape[1] = channel_;
			top_shape[2] = output_h_;
			top_shape[3] = output_w_;

			top[0]->Reshape(top_shape);


			map_size_ = output_h_ * output_w_;

			//target coordinates
			vector<int> V_shape(4);//U(source)--->V(target)
			V_shape[0] = 1;//share coordinates in (channel and num for target coordinate(-1 ~ +1)
			V_shape[1] = 3;
			V_shape[2] = output_h_;
			V_shape[3] = output_w_;
			target_coordinates_.Reshape(V_shape);



			CHECK_EQ(bottom[1]->shape(1),6);//affine transform six parameters

			//Initialize target coordinates

			Dtype* target_corrdinates_data = target_coordinates_.mutable_cpu_data();

			// Normalize coordinates to -1 ~ 1
			for(int h = 0; h < output_h_;++h){
				for(int w = 0; w < output_w_;++w){
					target_corrdinates_data[target_coordinates_.offset(0,0,h,w)] = (Dtype) w/(Dtype)(output_w_-1)*2.0 -1.0; //x

					target_corrdinates_data[target_coordinates_.offset(0,1,h,w)] = (Dtype) h/(Dtype)(output_h_-1)*2.0 -1.0; //y

					target_corrdinates_data[target_coordinates_.offset(0,2,h,w)] = (Dtype) 1.0;//homogeneous coordinate

				}
			}

			//source coordinates corresponding to target coordinates,so shape like output V
			vector<int> U2V_shape(4);
			U2V_shape[0] = num_;
			U2V_shape[1] = 2;
			U2V_shape[2] = output_h_;
			U2V_shape[3] = output_w_;
			source_coordinates_.Reshape(U2V_shape);


			//bilinear sampling
			vector<int> sample_shape(4);
			sample_shape[0] = num_;
			sample_shape[1] = output_h_;
			sample_shape[2] = output_w_;
			sample_shape[3] = 4;//sample 4 point to compute weighted average value
			source_sample_range_.Reshape(sample_shape);


			//for gpu
			vector<int> source_grad_shape(5);
			source_grad_shape[0] = channel_;
			source_grad_shape[1] = num_;
			source_grad_shape[2] = 2;
			source_grad_shape[3] = output_h_;
			source_grad_shape[4] = output_w_;
			source_grad_cache_.Reshape(source_grad_shape);

			vector<int> all_ones_shape(1);
			all_ones_shape[0] = channel_;
			source_grad_op_.Reshape(all_ones_shape);
			caffe_set<Dtype>(channel_,1,source_grad_op_.mutable_cpu_data());
	}

	template<typename Dtype>
	void SpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
			const vector<Blob<Dtype>*>& top){

		//get top and bottom data
		//Dtype* U = bottom[0]->cpu_data();
		Dtype* V = top[0]->mutable_cpu_data();
		Dtype* theta_data = bottom[1]->mutable_cpu_data();
		const Dtype* target_coordinate_data = target_coordinates_.cpu_data();

		Dtype* source_coordinate_data  = source_coordinates_.mutable_cpu_data();
		int* source_sample_range_data  = source_sample_range_.mutable_cpu_data();

		caffe_set<Dtype>(top[0]->count(),0,V);//memset

		for(int n = 0; n <num_;++n){

			//set predefine theta
			for(int j=0 ;j<6;j++)
			{
				if(is_pre_defined_theta[j])
				{
					theta_data[n*6 + j] = pre_defined_theta[j];
				}
			}

			//compute the source coordinate corresponding to target coordinate using theta
			//C = alpha*A*B + beta* C  A:M(2)*K(3)  B:K(3)*N(h*w)  C:M(2)*N(h*w)
			//A: theta	   B: target_coordinate_data    C: source_coodinate_data
			caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,2,map_size_,3,Dtype(1.0),
					theta_data + n * 6 ,target_coordinate_data,Dtype(0.0),source_coordinate_data + n*2*map_size_);


			//map source coordinate from(-1,1) to (0,input_h) and (0,input_w)
			caffe_add_scalar<Dtype>(2*map_size_,Dtype(1.0),source_coordinate_data + n*2*map_size_);//(0-2)
			caffe_scal<Dtype>(map_size_,0.5*(input_w_-1),source_coordinate_data + n*2*map_size_);
			caffe_scal<Dtype>(map_size_,0.5* (input_h_-1),source_coordinate_data  + n*2*map_size_ + map_size_);


			//compute U
			for(int h = 0; h < output_h_; h++){
				for(int w = 0; w < output_w_ ; w++){

					Dtype x = source_coordinate_data[source_coordinates_.offset(n,0,h,w)];
					Dtype y = source_coordinate_data[source_coordinates_.offset(n,1,h,w)];

					//at most four different point (when x,y is not integer)
					int x1 = (floor(x) > 0) ? floor(x) : 0;
					int x2 = (ceil(x) < (input_w_ -1 )) ? ceil(x):(input_w_ -1 );
					int y1 = (floor(y) > 0) ? floor(y) : 0;
					int y2 = (ceil(y) < (input_h_ -1 )) ? ceil(y):(input_h_ -1 );

					//save range data  for back-propagation
					source_sample_range_data[source_sample_range_.offset(n,h,w,0)] = x1;
					source_sample_range_data[source_sample_range_.offset(n,h,w,1)] = x2;
					source_sample_range_data[source_sample_range_.offset(n,h,w,2)] = y1;
					source_sample_range_data[source_sample_range_.offset(n,h,w,3)] = y2;

					//get V  given  U and source coordinate using bilinear sampling
					for(int yi = y1; yi <= y2; ++yi){
						for(int xi = x1; xi <= x2 ; ++xi){
							for(int c =0; c < channel_;++c){
								V[top[0]->offset(n,c,h,w)] += bottom[0]->data_at(n,c,yi,xi)*(1-fabs(x-xi))*(1-fabs(y-yi));
							}
						}
					}
				}
			}
		}
	}

	template<typename Dtype>
	void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom){

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* data_diff = bottom[0]->mutable_cpu_diff();
		Dtype* theta_diff = bottom[1]->mutable_cpu_diff();
		Dtype* source_coordinates_diff = source_coordinates_.mutable_cpu_diff();

		const Dtype* target_coordinates_data = target_coordinates_.cpu_data();
		const Dtype* source_coordinates_data = source_coordinates_.cpu_data();
		const int * source_sample_range_data = source_sample_range_.cpu_data();

		caffe_set<Dtype>(bottom[0]->count(),0,data_diff);
		caffe_set<Dtype>(source_coordinates_.count(),0,source_coordinates_diff);

		for(int n = 0;n < num_;n++){

			for (int h = 0; h < output_h_; ++h){
				for(int w = 0; w < output_w_; ++w){
					Dtype x = source_coordinates_data[source_coordinates_.offset(n,0,h,w)];
					Dtype y = source_coordinates_data[source_coordinates_.offset(n,1,h,w)];

					int x_min = source_sample_range_data[source_sample_range_.offset(n,h,w,0)];
					int x_max = source_sample_range_data[source_sample_range_.offset(n,h,w,1)];
					int y_min = source_sample_range_data[source_sample_range_.offset(n,h,w,2)];
					int y_max = source_sample_range_data[source_sample_range_.offset(n,h,w,3)];

					for(int yi=y_min;yi <= y_max; yi++){
						for(int xi=x_min; xi <=x_max; xi++){
							int sign_x = caffe_sign<Dtype>(xi - x);
							int sign_y = caffe_sign<Dtype>(yi- y);
							for(int c = 0; c < channel_;++c){
								//dL/dxs  and dL/dys
								source_coordinates_diff[source_coordinates_.offset(n,0,h,w)] += top_diff[top[0]->offset(n,c,h,w)]*
										bottom[0]->data_at(n,c,yi,xi)*(1 - fabs(yi - y)) * sign_x * output_w_/2.0;
								source_coordinates_diff[source_coordinates_.offset(n,1,h,w)] += top_diff[top[0]->offset(n,c,h,w)]*
										bottom[0]->data_at(n,c,yi,xi)*(1 - fabs(xi - x)) * sign_y * output_h_/2.0;

								//dL/dU
								data_diff[bottom[0]->offset(n,c,yi,xi)] += top_diff[top[0]->offset(n,c,h,w)]*
								        (1 - fabs(xi - x)) * (1 - fabs(yi - y));

							}
						}
					}
				}
			}

			//dL/d(theta)
			caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans,2,3,map_size_,Dtype(1.0),
					source_coordinates_diff + n*2*map_size_,
					target_coordinates_data,//shared
					(Dtype)0.0,theta_diff + 6*n);


			//tricky way for not changing fixed theta
			for(int j=0 ;j<6;j++){
				if(is_pre_defined_theta[j]){
					theta_diff[n*6 + j] = 0;
				}
			}

		}
	}
#ifdef CPU_ONLY
	STUB_GPU(SpatialTransformerLayer);
#endif

INSTANTIATE_CLASS(SpatialTransformerLayer);
REGISTER_LAYER_CLASS(SpatialTransformer);

}// namespace caffe
