# Build the Docker image using the Dockerfile
podman build -t assignment4:v1 -f docker/Dockerfile .

#docker volume create ml-models6

#docker run -v ml-models6:/digit/models assignment4:v1
podman run -v $(pwd)/models:/digit/models assignment4:v1
