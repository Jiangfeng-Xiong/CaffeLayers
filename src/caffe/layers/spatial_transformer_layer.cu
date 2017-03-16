#include <algorithm>
#include <vector>
#include "cuda.h"
#include "caffe/layer.hpp"
#include"caffe/layers/spatial_transformer_layer.hpp"

namespace caffe {
    // compute each Vi

    template <typename Dtype>
    __global__ void ComputeSource(const int total, const int num,
    							  const int output_h, const int output_w,const int input_h, const int input_w,
            					  const Dtype* target_data, const Dtype* theta, Dtype* source_data, int* source_range_data) {
        // total = num * output_h * output_w

        CUDA_KERNEL_LOOP(index, total) {
            int div = output_h * output_w;
            int n = index / div;
            int n_rem = index % div;
            div /= output_h;
            int h = n_rem / div;
            int w = n_rem % div;

            Dtype x_target = target_data[h * output_w + w];
            Dtype y_target = target_data[h * output_w + w + output_w * output_h];

            int offset_theta = 6 * n;
            Dtype x = x_target * theta[offset_theta] + y_target * theta[offset_theta + 1] + theta[offset_theta + 2];
            Dtype y = x_target * theta[offset_theta + 3] + y_target * theta[offset_theta + 4] + theta[offset_theta + 5];

            x = (x + (Dtype) 1.) / (Dtype) 2. * (input_w - 1);
            y = (y + (Dtype) 1.) / (Dtype) 2. * (input_h - 1);

            int offset_source = n * output_h * output_w * 2 + h * output_w + w;
            source_data[offset_source] = x;
            source_data[offset_source + output_h * output_w] = y;

            int w_min = (floor(x) > 0) ? floor(x) : 0;
            int w_max = (ceil(x) < input_w - 1) ? ceil(x) : (input_w - 1);
            int h_min = (floor(y) > 0) ? floor(y) : 0;
            int h_max = (ceil(y) < input_h - 1) ? ceil(y) : (input_h - 1);
            int offset_range = (n * output_h * output_w + h * output_w + w) * 4;
            source_range_data[offset_range] = w_min;
            source_range_data[offset_range + 1] = w_max;
            source_range_data[offset_range + 2] = h_min;
            source_range_data[offset_range + 3] = h_max;
        }
    }

    template <typename Dtype>
    __global__ void AffineForward(const int count, const int channels,
    							  const int height, const int width,const int input_h,const int input_w,
            const Dtype* in, const Dtype* source_data, const int* source_range_data, Dtype* out) {

        CUDA_KERNEL_LOOP(index, count) {
            int div = channels * height * width;
            int n = index / div;
            int n_rem = index % div;
            div /= channels;
            int c = n_rem / div;
            int c_rem = n_rem % div;
            div /= height;
            int h = c_rem / div;
            int w = c_rem % div;

            int offset_source = n * 2 * height * width + h * width + w;
            Dtype x = source_data[offset_source];
            Dtype y = source_data[offset_source + height * width];
            int offset_range = (n * height * width + h * width + w) * 4;
            int w_min = source_range_data[offset_range];
            int w_max = source_range_data[offset_range + 1];
            int h_min = source_range_data[offset_range + 2];
            int h_max = source_range_data[offset_range + 3];

            int offset_input = n * channels * input_h * input_w + c * input_h * input_w;
            int offset_output = n * channels * height * width + c * height * width;
            Dtype tmp = 0;
            for (int hh = h_min; hh <= h_max; ++hh) {
                for (int ww = w_min; ww <= w_max; ++ww) {
                    tmp += in[offset_input + hh * input_w + ww]*(1 - fabs(x - ww)) * (1 - fabs(y - hh));//in bottom
                }
            }
            out[offset_output + h * width + w] = tmp;
        }
    }

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        Dtype* theta_data = bottom[1]->mutable_gpu_data();
        const Dtype* target_data = target_coordinates_.gpu_data();
        Dtype* source_data = source_coordinates_.mutable_gpu_data();
        int* range_data = source_sample_range_.mutable_gpu_data();
        
        
        int count = top[0]->count();
        caffe_gpu_set<Dtype>(count, 0, top_data);
        
        for(int n=0;n<num_;n++){
        	for(int j=0; j<6;j++){
        		if(is_pre_defined_theta[j]){
        			caffe_gpu_set<Dtype>(1, pre_defined_theta[j], theta_data+n*6+j);
					//theta_data[n*6 + j] = pre_defined_theta[j];
				}
        	}
        }
        
        ComputeSource<Dtype> << <CAFFE_GET_BLOCKS(num_ * output_h_ * output_w_),
                CAFFE_CUDA_NUM_THREADS >> >(num_ * output_h_ * output_w_, num_, output_h_, output_w_,input_h_, input_w_,
                target_data, theta_data, source_data, range_data);
                
                
        AffineForward<Dtype> << <CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS >> >(count, channel_, output_h_, output_w_,input_h_, input_w_,
                bottom_data, source_data, range_data, top_data);
        CUDA_POST_KERNEL_CHECK;
    }

    __device__ inline void atomic_add(float * address, float val) {
        atomicAdd(address, val);
    }

    __device__ inline void atomic_add(double * address, double val) {
        unsigned long long int* address_as_ull =
                (unsigned long long int*) address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
    }
   
  
    // compute (1) d{V_i} / d{x_i}, then (2) d{V_i} / d{theta}
    // compute sum_{i} d{V_i} / d{U_nm}

    template <typename Dtype>
    __global__ void AffineBackward(const int count, const int num, const int channels,
    		const int height, const int width,
    		const int input_h, const int input_w,
            const Dtype* data, const Dtype* source_data, int* source_range_data, const Dtype* top_diff,
            Dtype* data_diff, Dtype* source_grad_cache) {
        // count = num * channel * height * width

        CUDA_KERNEL_LOOP(index, count) {
            int div = channels * height * width;
            int n = index / div;
            int n_rem = index % div;
            div /= channels;


            int c = n_rem / div;
            int c_rem = n_rem % div;
            div /= height;


            int h = c_rem / div;
            int w = c_rem % div;

            int offset_source = n * 2 * height * width + h * width + w;
            Dtype x = source_data[offset_source];
            Dtype y = source_data[offset_source + height * width];

            int offset_range = (n * height * width + h * width + w) * 4;
            int w_min = source_range_data[offset_range];
            int w_max = source_range_data[offset_range + 1];
            int h_min = source_range_data[offset_range + 2];
            int h_max = source_range_data[offset_range + 3];
            int source_diff_x = c * num * 2 * height * width + n * 2 * height * width + h * width + w;
            int source_diff_y = source_diff_x + height * width;
            Dtype tmp_source_x = 0;
            Dtype tmp_source_y = 0;
            Dtype buffer = top_diff[n * channels * height * width + c * height * width + h * width + w];

            for (int hh = h_min; hh <= h_max; ++hh) {
                for (int ww = w_min; ww <= w_max; ++ww) {
                    int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
                    int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
                    Dtype buffer2 = buffer * data[n * channels * input_h * input_w + c * input_h * input_w + hh * input_w + ww]; //bottom len-1???
                    Dtype tmp_hh = 1 - fabs(y - hh);
                    Dtype tmp_ww = 1 - fabs(x - ww);
                    tmp_source_x += buffer2 * tmp_hh * sign_x;
                    tmp_source_y += buffer2 * tmp_ww * sign_y;
                    Dtype inc = buffer * tmp_hh * tmp_ww;
                    int offset = n * channels * input_h * input_w + c * input_h * input_w + hh * input_w + ww;//bottom diff
                    atomic_add(data_diff + offset, inc);
                }
            }
            source_grad_cache[source_diff_x] = tmp_source_x * (width - 1) / (Dtype) 2.;
            source_grad_cache[source_diff_y] = tmp_source_y * (height - 1) / (Dtype) 2.;
        }
    }

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();


        Dtype* data_diff = bottom[0]->mutable_gpu_diff();
        Dtype* theta_diff = bottom[1]->mutable_gpu_diff();
        Dtype* source_grad_cache = source_grad_cache_.mutable_gpu_data();

        const Dtype* target_data = target_coordinates_.gpu_data();
        const Dtype* source_data = source_coordinates_.gpu_data();
        Dtype* source_diff = source_coordinates_.mutable_gpu_diff();
        int* source_range_data = source_sample_range_.mutable_gpu_data();
        caffe_gpu_set<Dtype>(bottom[0]->count(), 0, data_diff);
        int count = top[0]->count();
        // compute gradient with respect to theta
        AffineBackward<Dtype> << <CAFFE_GET_BLOCKS(count),
                CAFFE_CUDA_NUM_THREADS >> >(count, num_, channel_, output_h_, output_w_,input_h_,input_w_,
                bottom_data, source_data, source_range_data, top_diff,
                data_diff, source_grad_cache);

        // merge gradient for theta 
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, num_ * 2 * map_size_, channel_,
                Dtype(1), source_grad_op_.gpu_data(), source_grad_cache, Dtype(0), source_diff);

        for (int index = 0; index < num_; ++index) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2, 3, map_size_,
                    Dtype(1), source_diff + index * 2 * map_size_, target_data, Dtype(0), theta_diff + index * 6);
                    
                    
            //tricky way for not changing fixed theta
			for(int j=0 ;j<6;j++){
				if(is_pre_defined_theta[j]){
						//theta_diff[index*6 + j] = 0;
						caffe_gpu_set<Dtype>(1,0, theta_diff+index*6+j);
				}
			}
        }
                
        CUDA_POST_KERNEL_CHECK;
    }
    INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);
} // namespace caffe
