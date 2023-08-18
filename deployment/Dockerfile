# Start from the Hugging Face PyTorch GPU image
FROM huggingface/transformers-pytorch-gpu

# Set the working directory
WORKDIR /workspace

# Copy the requirements.txt file from the "deployment" folder to the container
COPY ./deployment/requirements.txt /workspace

# Install any additional dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the packages to the container
COPY . /workspace/

# Default command to run the script
CMD ["python3", "scripts/prompt_embedding_space_traversal.py"]
