import os

import docker


# TODO: NEEDS BULLETPROOF WINDOWS IMPLEMENTATION
def get_default_client() -> docker.DockerClient:
    if 'DOCKER_HOST' not in os.environ:
        return docker.DockerClient(base_url='unix://var/run/docker.sock')
    return docker.from_env()
