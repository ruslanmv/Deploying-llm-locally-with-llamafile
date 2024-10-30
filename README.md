# Deploying Your LLM Locally with llamafile: A Step-by-Step Guide

This guide provides a detailed tutorial on transforming your custom LLaMA model, **llama3**, into a llamafile, enabling it to run locally as a standalone executable. We'll cover the steps for converting and executing your model on a CPU and GPU setup, emphasizing CPU usage.

## Model Overview

- **Model Name**: llama3
- **Developer**: ruslanmv
- **License**: Apache-2.0
- **Base Model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Key Features**:
  - **Medical Focus**: Fine-tuned for health inquiries.
  - **Comprehensive Knowledge Base**: Trained on medical datasets.
  - **Text Generation**: Delivers informative responses.

This model is compatible with the Hugging Face Transformers library and PyTorch, allowing for local execution without server dependencies.

## Step 1: Environment Setup

Install the required libraries and dependencies:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch and necessary dependencies
pip install torch==2.2.1 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121

# Install BitsAndBytes and Accelerate for quantization and model loading
pip install bitsandbytes accelerate
```

These dependencies ensure compatibility with large language models and facilitate efficient execution on both CPU and GPU.

## Step 2: Loading Your Model in Python

Let’s load **llama3** in Python to ensure it is functioning before converting it into a llamafile. Here’s a basic example of loading the model and generating responses:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Model and tokenizer setup
model_name = "ruslanmv/ai-medical-model-32bit"
device_map = 'auto'

# Configure BitsAndBytes for 4-bit quantization to reduce memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Define a function to interact with the model
def askme(question):
    prompt = f"<|start_header_id|>system<|end_header_id|> You are a Medical AI chatbot assistant. <|eot_id|><|start_header_id|>User: <|end_header_id|>This is the question: {question}<|eot_id|>"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(answer)

# Test with a sample question
askme("What was the main cause of the inflammatory CD4+ T cells?")
```

This step verifies the model is functioning correctly and serves as a baseline before conversion.

## Step 3: Converting to a llamafile

### 3.1 Install llamafile

If you haven’t yet installed **llamafile**, download it from [Mozilla Builders’ GitHub repository](https://github.com/Mozilla-Ocho/llamafile). Follow the instructions on the page to set up the executable environment.

### 3.2 Quantizing Your Model

To prepare the model for the llamafile format, we must ensure it is in a quantized format compatible with llama.cpp. Save your model in **GGUF** format, which is supported by llamafile.

Run the following commands in your terminal:

```bash
# Save model in GGUF format
python -m transformers-cli quantize --model ruslanmv/ai-medical-model-32bit --bits 4 --output_format GGUF --output ./llama3.gguf
```

Now, **llama3.gguf** is ready for conversion into a llamafile.

### 3.3 Creating the llamafile

Use llamafile’s CLI tools to create an executable file containing the model weights and configurations. In your terminal, navigate to the model’s directory and run:

```bash
llamafile -j8 -m llama3.gguf --temp 0.7 -p '[INST]You are a Medical AI. How can I help you?[/INST]'
```

This command compiles the model into an executable llamafile, enabling it to run without dependencies on other libraries.

## Step 4: Running the llamafile

To execute the model locally, you’ll need to make it executable and then run it as follows:

```bash
chmod +x llama3.llamafile
./llama3.llamafile
```

For CPU usage, the model will operate by default on CPU, while specifying `--gpu` will allow GPU usage if available.

### GPU Configuration (Optional)

For GPU acceleration on supported systems, you can enable GPU usage:

```bash
./llama3.llamafile --gpu nvidia
```

Or for systems with AMD GPUs:

```bash
./llama3.llamafile --gpu amd
```

If multiple GPUs are available, llamafile will automatically distribute processing across them, or you can specify which GPU to use by setting the `HIP_VISIBLE_DEVICES` environment variable.

## Step 5: Accessing the Model Locally

Upon running the llamafile, a local server will be initiated. You can access it via:

```bash
http://localhost:8080
```

This interface allows you to interact with the model directly. For programmatic access, llamafile also includes an OpenAI-compatible API endpoint at `http://127.0.0.1:8080/`.

## Sample Usage in a Script

To automate queries, here’s a sample script in Python that accesses the local llamafile API:

```python
import requests

def query_llama(question):
    url = "http://127.0.0.1:8080/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": f"User: {question}\nAssistant:",
        "max_tokens": 150
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["text"]

question = "What are the main causes of inflammatory CD4+ T cells?"
print(query_llama(question))
```

## Conclusion

By following these steps, you've converted your custom LLaMA model into a llamafile, optimized it for efficient local execution, and ensured compatibility with both CPU and GPU environments. This approach enables you to run powerful LLMs like **llama3** on various platforms, making it easier to deploy without dependency overhead.


This blog post provides a robust introduction to deploying models with llamafile, demonstrating how accessible high-performance AI can be with the right tools. For more advanced customization, refer to llamafile’s [official documentation](https://github.com/Mozilla-Ocho/llamafile).