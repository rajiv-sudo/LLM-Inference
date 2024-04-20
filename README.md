## LLM Inference
In this section, we will explore how you can download a Large Language Model (LLM) from Huggingface and use that model to do inference on different hardware platforms.

Model Used: **mistralai/Mistral-7B-v0.1**

We will do our setup and testing in AWS EC2 instances. Here are the scenarios we will test. Please click the link for each scenario to nagivate to the details of setup, code and running the test.

***Here are the differenct scenarios***
## [01 - GPU EC2 instance in AWS P3.8xlarge with model's native precision](https://github.com/rajiv-sudo/LLM-Inference/tree/main/LLM_Mistral_7B_Inference_EC2_GPU)

In this tutorial, we will:

1. Create a GPU instance in AWS – p3.8xl instance that contains four NVIDIA Tesla V100 GPUs

2. Install compatible version of CUDA toolkit and ensure that the toolkit is installed and visible and ready to go

3. Install compatible version of Pytorch that is compatible with the CUDA toolkit version

4. Run inference on the GPU instance using the Mistral 7B model and the provided Jupyter notebook

## [02 - GPU EC2 instance in AWS P3.2xlarge with quantized model weights](https://github.com/rajiv-sudo/LLM-Inference/tree/main/LLM_Mistral_7B_Quantized_Inference_EC2_GPU)
In the previous tutorial, we ran the inference on P3.8xlarge GPU based EC2 instance. This instance is $12+ per hour based on on-demand pricing. This machine has four NVIDIA Tesla V100 GPUs each with 16 GB of GPU memory. So, in total this machine had 64 GB of GPU memory. 

For the 7B parameter Mistral model with native precision (BF16), we need approx 16GB of GPU memory to load the model. The next smaller EC2 instance config in P3 family is the P3.2xlarge machine. This machine has one V100 GPU with 16 GB of memory. So, if we try to run the model in P3.2xlarge EC2 instance, then it errors out saying ```**out of memory**```

The P3.2xlarge machine is $3+ per hour based on on-demand pricing. This is almost one-fourth the price for the P3.8xlarge instance

So if we want to get acceptable performance from the Mistral 7B model with sparse inferencing, but use a smaller GPU machine, not as expensive as the P3.8xlarge machine, our option is to quantize the model in a lower precision and be able to run it on a smaller GOU machine like P3.2xlarge.

In this tutorial, we will:

1. Create a GPU instance in AWS – p3.2xl instance that contains one NVIDIA Tesla V100 GPU

2. Install compatible version of CUDA toolkit and ensure that the toolkit is installed and visible and ready to go

3. Install compatible version of Pytorch that is compatible with the CUDA toolkit version

4. Using the provided Jupyter notebook, we will first quantize the weights of the Mistral 7B model to INT8. This will reduce the model memeory requirement to almost half of what is needed to load the model in the native BF16 precision.

5. Run inference on the GPU instance using the Mistral 7B ```quantized to INT8``` model and the provided Jupyter notebook

## CPU EC2 instance in AWS M7i.8xlarge with quantized model weights
