# Define environment variables
IMAGE_NAME=gassen77/ci-mlops
CONTAINER_NAME=mlops-container

# Set Python version and Docker version
PYTHON_VERSION=3.8
DOCKER_VERSION=latest

# Target: lint the code using flake8
lint:
        @echo "Linting code with flake8..."
        flake8 . | tee linting_report.txt

# Target: format the code using black
format:
        @echo "Formatting code with black..."
        black . || true

# Target: install dependencies for Python
install-deps:
        @echo "Installing Python dependencies..."
        pip install -r requirements.txt

# Target: build the Docker image
docker-build:
        @echo "Building Docker image..."
        docker build -t $(IMAGE_NAME):$(DOCKER_VERSION) .

# Target: run the model training inside a Docker container
docker-run:
        @echo "Running model training in Docker container..."
        docker run --rm -v $(PWD):/app $(IMAGE_NAME):$(DOCKER_VERSION) python train.py

# Target: clean Docker images and containers
docker-clean:
        @echo "Cleaning up Docker containers and images..."
        docker rm -f $(CONTAINER_NAME) || true
        docker rmi $(IMAGE_NAME):$(DOCKER_VERSION) || true

# Target: full pipeline (lint, format, build, run)
ci-cd: lint format docker-build docker-run

# Default target
all: ci-cd
