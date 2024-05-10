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

## [02 - GPU EC2 instance in AWS P3.2xlarge with model's native precision]([https://github.com/rajiv-sudo/LLM-Inference/tree/main/LLM_Mistral_7B_Quantized_Inference_EC2_GPU](https://github.com/rajiv-sudo/LLM-Inference/blob/main/Mistral-7B-GPU-EC2-P3.2xl)
In the previous tutorial, we ran the inference on P3.8xlarge GPU based EC2 instance. This instance is $12+ per hour based on on-demand pricing. This machine has four NVIDIA Tesla V100 GPUs each with 16 GB of GPU memory. So, in total this machine had 64 GB of GPU memory. 

The next smaller EC2 instance config in P3 family is the P3.2xlarge machine. This machine has one V100 GPU with 16 GB of memory. We will run the inference in this ```P3.2xlarge EC2``` instance.

The P3.2xlarge machine is $3+ per hour based on on-demand pricing. This is almost one-fourth the price for the P3.8xlarge instance

We will see how the LLM inferencing performance is for the Mistral 7B model, on this smaller GPU machine, not as expensive as the P3.8xlarge machine.

In this tutorial, we will:

1. Create a GPU instance in AWS – p3.2xl instance that contains one NVIDIA Tesla V100 GPU

2. Install compatible version of CUDA toolkit and ensure that the toolkit is installed and visible and ready to go

3. Install compatible version of Pytorch that is compatible with the CUDA toolkit version

4. Using the provided Jupyter notebook, we will run inference on the GPU instance using the Mistral 7B model


## CPU EC2 instance in AWS M7i.8xlarge with quantized model weights
