## LLM Inference
In this section, we will explore how you can download a Large Language Model (LLM) from Huggingface and use that model to do inference on different hardware platforms.

Model Used: **mistralai/Mistral-7B-v0.1**

We will do our setup and testing in AWS EC2 instances. Here are the scenarios we will test. Please click the link for each scenario to nagivate to the details of setup, code and running the test.

***Here are the differenct scenarios***
## [01 - GPU EC2 instance in AWS P3.8xlarge with model's native precision](https://github.com/rajiv-sudo/LLM-Inference/tree/main/LLM_Mistral_7B_Inference_EC2_GPU)

In this tutorial, we will:

Create a GPU instance in AWS – p3.8xl instance that contains four NVIDIA Tesla V100 GPUs

Install compatible version of CUDA toolkit and ensure that the toolkit is installed and visible and ready to go

Install compatible version of Pytorch that is compatible with the CUDA toolkit version

Run inference on the GPU instance using the Mistral 7B model and the provided Jupyter notebook

## [02 - GPU EC2 instance in AWS P3.2xlarge with quantized model weights](https://github.com/rajiv-sudo/LLM-Inference/tree/main/LLM_Mistral_7B_Quantized_Inference_EC2_GPU)

## CPU EC2 instance in AWS M7i.8xlarge with quantized model weights
