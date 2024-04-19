# Define variables for common parameters
IMAGE_NAME=chat-frontend
CONTAINER_NAME=chat-frontend
PORT=3000
LOG?=
.PHONY: build run stop remove rebuild

# Build the Docker image
build:
	docker build $(if $(findstring $(LOG), 1),--no-cache --progress=plain,) -t $(IMAGE_NAME) .


# Run the Docker container
run:
	docker run --name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		--entrypoint "/bin/sh" \
		--mount type=bind,source=$(shell pwd),target=/app \
		$(IMAGE_NAME) \
		-c "yarn dev"


# Interactive shell into the Docker CONTAINER_NAME
shell:
	docker run -it --name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		--entrypoint "/bin/sh" \
		--mount type=bind,source=$(shell pwd),target=/app \
		$(IMAGE_NAME) \
		-c "/bin/sh"

# Stop the Docker container
stop:
	docker stop $(CONTAINER_NAME)

# Start the Docker container
start:
	docker start $(CONTAINER_NAME)

# Remove the Docker container
remove:
	docker rm $(CONTAINER_NAME)

# Rebuild and restart the container
rebuild: 
	stop remove build run

