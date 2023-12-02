podman build -t dependencyimage -f docker/DependencyDockerfile .
podman build -t finalimage -f docker/FinalDockerfile .
