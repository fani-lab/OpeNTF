#!/usr/bin/env python3

import sys
import subprocess
import os

def create_docker_container(container_name, hostname):
    """
    Create a Docker container with specified name and hostname
    """

    mount_dir = "OpeNTF"
    image_name = "kmthang/opennmt:3.0.4-torch1.10.1"

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
    # Check if we have the correct number of arguments
    if len(sys.argv) < 2:
        print("Usage: python3 create_containers.py container1 container2 container3 ...")
        sys.exit(1)
    
    # Get container names from command line arguments
    container_names = sys.argv[1:]
    
    # Create containers
    for container_name in container_names:
        success = create_docker_container(container_name, container_name)
        if not success:
            print(f"Failed to create container {container_name}")
            continue

if __name__ == "__main__":
    main()