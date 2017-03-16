#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

template <typename TypeParam>
class NormalizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  void normlize(const Dtype* src,std::vector<Dtype>* dst,int dim){
    double sum = 0;
    for(int i=0;i<dim;i++){
      sum+=src[i]*src[i];
    }
    Dtype vector_norm = std::sqrt(sum);
    for(int i=0;i<dim;i++){
      dst->push_back(src[i]/vector_norm);
    }
}

  NormalizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>())
  {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NormalizeLayerTest() { delete blob_bottom_; delete blob_top_; }


  void TestForward(Dtype filler_std)
  {
    FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    LayerParameter layer_param;
    NormalizeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = 1e-5;
	  std::vector<Dtype> test_vector;//2*3*4*5
    int dim = this->blob_bottom_->count(1);
    for(int i=0;i<this->blob_bottom_->shape()[0];i++){
    normlize(bottom_data+i*dim,&test_vector,dim);
    }  

  for(int i=0;i<this->blob_bottom_->shape()[0];i++){
    for(int j=0;j<dim;j++){
      Dtype expected_value = test_vector[i*dim+j];
      Dtype precision = std::max(
        Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
      EXPECT_NEAR(expected_value, top_data[i*dim+j], precision);
    }
  }
}
   void TestBackward(Dtype filler_std)
  {
    FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    LayerParameter layer_param;
    NormalizeLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
    checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};
TYPED_TEST_CASE(NormalizeLayerTest, TestDtypesAndDevices);
TYPED_TEST(NormalizeLayerTest, TestL2Norm) {
  this->TestForward(1.0);
}
TYPED_TEST(NormalizeLayerTest, TestL2NormGradient) {
  this->TestBackward(1.0);
}
}
