/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"



namespace tensorflow {

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

/**
  expend ..
static ::tensorflow::register_op::OpDefBuilderReceiver register_op0 __attribute__((unused)) = ::tensorflow::register_op::OpDefBuilderWrapper<true>("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
*/


REGISTER_OP("ZeroOut1")
    .Input("to_zero: int32")  
    .Attr("preserve_index: int") 
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
/*
expend ...
static ::tensorflow::register_op::OpDefBuilderReceiver register_op1 __attribute__((unused)) = ::tensorflow::register_op::OpDefBuilderWrapper<true>("ZeroOut1")
    .Input("to_zero: int32")
    .Attr("preserve_index: int")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
*/

REGISTER_OP("ZeroOut2")
		.Attr("T: realnumbertype")
		.Input("to_zero: T")
		.Output("zeroed: T")
		.Doc(R"doc(
	Zeros out all but the first value of a Tensor.
	zeroed: A Tensor whose first value is identical to `to_zero`, and 0
	  otherwise.
	)doc");

/*
expend ...
static ::tensorflow::register_op::OpDefBuilderReceiver register_op2 __attribute__((unused)) = ::tensorflow::register_op::OpDefBuilderWrapper<true>("ZeroOut2")
  .Attr("T: realnumbertype")
  .Input("to_zero: T")
  .Output("zeroed: T")
  .Doc(R"doc(
        Zeros out all but the first value of a Tensor.
        zeroed: A Tensor whose first value is identical to `to_zero`, and 0
          otherwise.
        )doc");
*/
template <typename T>
class ZeroOutOp2 : public OpKernel {
	 public:
	  explicit ZeroOutOp2(OpKernelConstruction* context) : OpKernel(context) {}
	
	  void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<T>();
	
		// Create an output tensor
		Tensor* output = nullptr;
		OP_REQUIRES_OK(context,
					   context->allocate_output(0, input_tensor.shape(), &output));
		auto output_flat = output->template flat<T>();
	
		// Set all the elements of the output tensor to 0
		const int N = input.size();
		for (int i = 0; i < N; i++) {
		  output_flat(i) = T(0);
		}
	
		// Preserve the first input value
		if (N > 0) output_flat(0) = input(0);
	  }
	};
/*
expend ....
template <typename T>
class ZeroOutOp2 : public OpKernel {
  public:
   explicit ZeroOutOp2(OpKernelConstruction* context) : OpKernel(context) {}

   void Compute(OpKernelContext* context) override {

  const Tensor& input_tensor = context->input(0);
  auto input = input_tensor.flat<T>();


  Tensor* output = nullptr;
  do { ::tensorflow::Status _s(context->allocate_output(0, input_tensor.shape(), &output)); if (!(__builtin_expect(!!(_s.ok()), 1))) { (context)->CtxFailureWithWarning(
 "word2vec_ops.cc"
# 64 "word2vec_ops.cc"
  ,
 65
# 64 "word2vec_ops.cc"
  , _s); return; } } while (0)
                                                                   ;
  auto output_flat = output->template flat<T>();


  const int N = input.size();
  for (int i = 0; i < N; i++) {
    output_flat(i) = T(0);
  }


  if (N > 0) output_flat(0) = input(0);
   }
 };

*/	

#define REGISTER_KERNEL(type)                                       \
	  REGISTER_KERNEL_BUILDER(											\
		  Name("ZeroOut2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
		  ZeroOutOp2<type>)
	
	REGISTER_KERNEL(float);
	REGISTER_KERNEL(double);
	REGISTER_KERNEL(int32);
	
#undef REGISTER_KERNEL

/*
  expend ...

 constexpr bool should_register_3__flag = true; static ::tensorflow::kernel_factory::OpKernelRegistrar registrar__body__3__object( should_register_3__flag ? ::tensorflow::register_kernel::Name("ZeroOut2").Device(DEVICE_CPU).TypeConstraint<float>("T").Build() : nullptr, "ZeroOutOp2<float>", [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* { return new ZeroOutOp2<float>(context); });;


 constexpr bool should_register_4__flag = true; static ::tensorflow::kernel_factory::OpKernelRegistrar registrar__body__4__object( should_register_4__flag ? ::Tensorflow::register_kernel::Name("ZeroOut2").Device(DEVICE_CPU).TypeConstraint<double>("T").Build() : nullptr, "ZeroOutOp2<double>", [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* { return new ZeroOutOp2<double>(context); });;

 constexpr bool should_register_5__flag = true; static ::tensorflow::kernel_factory::OpKernelRegistrar registrar__body__5__object( should_register_5__flag ? ::orflow::register_kernelensorflow::register_kernel::Name("ZeroOut2").Device(DEVICE_CPU).TypeConstraint<int32>("T").Build() : nullptr, "ZeroOutOp2<int32>", [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* { return new ZeroOutOp2<int32>(context); });;

*/

class ZeroOutOp1 : public OpKernel {
 public:
  explicit ZeroOutOp1(OpKernelConstruction* context) : OpKernel(context) {

  // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("preserve_index", &preserve_index_));

				    // Check that preserve_index is positive
    OP_REQUIRES(context, preserve_index_ >= 0,
                errors::InvalidArgument("Need preserve_index >= 0, got ",
                                        preserve_index_));

 }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;

	// Get the index of the value to preserve    

				    // Check that preserve_index is positive
  OP_REQUIRES(context, preserve_index_ >= 0,
                errors::InvalidArgument("Need preserve_index >= 0, got ",
                                        preserve_index_));
	// We're using saved attr to validate potentially dynamic input
    // So we check that preserve_index is in range
    OP_REQUIRES(context, preserve_index_ < input.dimension(0),
                errors::InvalidArgument("preserve_index out of range"));
										 
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);

	 // Preserve the requested input value
  //  output_flat(preserve_index_) = input(preserve_index_);

	
  }
  private:
     int preserve_index_;
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut1").Device(DEVICE_CPU), ZeroOutOp1);

/*
expend ...
constexpr bool should_register_6__flag = true; static ::tensorflow::kernel_factory::OpKernelRegistrar registrar__body__6__object( should_register_6__flag ? ::tensorflow::register_kernel::Name("ZeroOut1").Device(DEVICE_CPU).Build() : nullptr, "ZeroOutOp1", [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* { return new ZeroOutOp1(context); });;

*/

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {

  // Get the index of the value to preserve   
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;

	// We're using saved attr to validate potentially dynamic input
    // So we check that preserve_index is in range  
										 
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);

	 // Preserve the requested input value
   // output_flat(preserve_index_) = input(preserve_index_);

	
  }
//  private:
//     int preserve_index_;
};

/*
 expend ...
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {


  }

  void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();
    Tensor* output_tensor = __null;
    do { ::tensorflow::Status _s(context->allocate_output(0, input_tensor.shape(), &output_tensor)); if (!(__builtin_expect(!!(_s.ok()), 1))) { (context)->CtxFailureWithWarning(
 "word2vec_ops.cc"
# 170 "word2vec_ops.cc"
    ,
 171
# 170 "word2vec_ops.cc"
    , _s); return; } } while (0)
                                                                     ;
    auto output_flat = output_tensor->flat<int32>();

    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }
  }
};

*/

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);

/*
expend ...
constexpr bool should_register_7__flag = true; static ::tensorflow::kernel_factory::OpKernelRegistrar registrar__body__7__object( should_register_7__flag ? ::tensorflow::register_kernel::Name("ZeroOut").Device(DEVICE_CPU).Build() : nullptr, "ZeroOutOp", [](::tensorflow::OpKernelConstruction* context) -> ::tensorflow::OpKernel* { return new ZeroOutOp(context); });;

*/

REGISTER_OP("SkipgramWord2vec")
    .Output("vocab_word: string")
    .Output("vocab_freq: int32")
    .Output("words_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_words_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("window_size: int = 5")
    .Attr("min_count: int = 5")
    .Attr("subsample: float = 1e-3")
    .Doc(R"doc(
Parses a text file and creates a batch of examples.
vocab_word: A vector of words in the corpus.
vocab_freq: Frequencies of words. Sorted in the non-ascending order.
words_per_epoch: Number of words per epoch in the data file.
current_epoch: The current epoch number.
total_words_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
filename: The corpus's text file name.
batch_size: The size of produced batch.
window_size: The number of words to predict to the left and right of the target.
min_count: The minimum number of word occurrences for it to be included in the
    vocabulary.
subsample: Threshold for word occurrence. Words that appear with higher
    frequency will be randomly down-sampled. Set to 0 to disable.
)doc");

REGISTER_OP("NegTrainWord2vec")
    .Input("w_in: Ref(float)")
    .Input("w_out: Ref(float)")
    .Input("examples: int32")
    .Input("labels: int32")
    .Input("lr: float")
    .SetIsStateful()
    .Attr("vocab_count: list(int)")
    .Attr("num_negative_samples: int")
    .Doc(R"doc(
Training via negative sampling.
w_in: input word embedding.
w_out: output word embedding.
examples: A vector of word ids.
labels: A vector of word ids.
vocab_count: Count of words in the vocabulary.
num_negative_samples: Number of negative samples per example.
)doc");

}  // end namespace tensorflow
