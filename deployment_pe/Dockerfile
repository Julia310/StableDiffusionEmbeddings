# Start from the Hugging Face PyTorch GPU image
FROM huggingface/transformers-pytorch-gpu

# Set the working directory
WORKDIR /workspace

# Copy the requirements.txt file from the "deployment" folder to the container
COPY ./deployment_pe/requirements.txt /workspace

# Install any additional dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the "ldm" package folder to the container
COPY ./ldm /workspace/ldm

# Copy the prompt_engineering_ui.py script from the "scripts" folder to the container
COPY ./scripts/prompt_engineering_ui.py /workspace

# Default command to run the script
CMD ["python3", "prompt_engineering_ui.py"]
