#!/usr/bin/env python3

import sys
import subprocess
import os
import argparse

def create_docker_container(container_name, hostname, version="latest"):
    """
    Create a Docker container with specified name and hostname
    """

    mount_dir = "OpeNTF"
    image_name = f"kmthang/opennmt:{version}"

    current_dir = os.getcwd()
    command = [
        "docker", "run",
        "-it",              # Interactive mode with TTY
        "-d",               # Detached mode
        "--name", container_name,
        "--hostname", hostname,
        "--gpus", "all",    # Enable all GPUs
        "-v", f"{current_dir}:/{mount_dir}",  # Mount current directory
        image_name
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully created container: {container_name} with hostname: {hostname}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating container {container_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error creating container {container_name}: {e}")
        return False
    
    return True

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Create Docker containers for OpenNMT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Create containers with latest image:
    python3 create_containers.py c1 c2 c3

  Create containers with specific version:
    python3 create_containers.py --v 1.0 c1 c2 c3

Note:
  - Each container will be created with GPU support enabled
  - Current directory will be mounted to /OpeNTF in each container
  - Container names will be used as their hostnames
''')
    parser.add_argument('--v', default='latest', help='Image version (default: latest)')
    parser.add_argument('containers', nargs='+', help='One or more container names to create')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create containers
    for container_name in args.containers:
        success = create_docker_container(container_name, container_name, args.v)
        if not success:
            print(f"Failed to create container {container_name}")
            continue

if __name__ == "__main__":
    main()