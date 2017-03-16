#ifndef CAFFE_ST_LAYER_HPP_
#define CAFFE_ST_LAYER_HPP_

#include<vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe{

template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype>{

public:
	explicit SpatialTransformerLayer(const LayerParameter& param)
	:Layer<Dtype>(param){}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);


	virtual inline const char* type() const{return "SpatialTransformer";}
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 1; }




protected:
	bool is_backward_; //in the case, the input feature map is the input image
					   //thus we don't need to backward
	int num_;
	int channel_;
	int input_h_;
	int input_w_;
	int output_h_;
	int output_w_;
	int map_size_;



	Blob<Dtype> target_coordinates_;//coordinates in V(1,3,output_h,output_w)
	Blob<Dtype> source_coordinates_;//Corresponding coordinates in U(num_,2,output_h,output_w)
	Blob<int> source_sample_range_;// for bilinear sampling (only four points,x1,x2 y1,y2) (num,output_h,output_w,4)


	bool is_pre_defined_theta[6];
	Dtype pre_defined_theta[6];




	//for gpu
	Blob<Dtype>source_grad_cache_; //five dimension blob
	Blob<Dtype> source_grad_op_;// sum over channel

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_dowm, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_dowm, const vector<Blob<Dtype>*>& bottom);

};


}// namespace caffe



#endif
