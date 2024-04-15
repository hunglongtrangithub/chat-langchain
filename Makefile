# Define variables for common parameters
IMAGE_NAME=chat-backend
CONTAINER_NAME=chat-backend
PORT=8080
USE_GPU=1
.PHONY: start format build run stop remove rebuild

# Start the local server using uvicorn
start:
	uvicorn main:app --reload --port $(PORT)

# Format the code using black and isort
format:
	black .
	isort .

# Build the Docker image
build:
	docker build --no-cache --progress=plain -t $(IMAGE_NAME) . 

# Run the Docker container
run:
	docker run --name $(CONTAINER_NAME) \
		$(if $(findstring $(USE_GPU), 1),--gpus all,) \
		--entrypoint "/bin/bash" \
		-p $(PORT):$(PORT) \
		--env-file .env \
		-v $(shell pwd)/voices:/app/voices \
		-v $(shell pwd)/faiss_index:/app/faiss_index \
		-v $(shell pwd)/llm_llama:/app/llm_llama \
		$(IMAGE_NAME) \
		-c "./expect.exp && uvicorn main:app --host 0.0.0.0 --port 8080 --reload"

# Interactive shell into the Docker CONTAINER_NAME
shell:
	docker run -it $(CONTAINER_NAME) /bin/bash

# Stop the Docker container
stop:
	docker stop $(CONTAINER_NAME)

# Remove the Docker container
remove:
	docker rm $(CONTAINER_NAME)

# Rebuild and restart the container
rebuild: 
	stop remove build run

