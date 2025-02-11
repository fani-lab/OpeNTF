#!/usr/bin/env python3

import sys
import subprocess
import os
import argparse

def create_docker_container(container_name, hostname, version="latest", is_windows=False):
    """
    Create a Docker container with specified name and hostname
    """
    mount_dir = "OpeNTF"
    image_name = f"kmthang/opennmt:{version}"
    current_dir = os.path.dirname(os.getcwd())

    # Check if image exists
    check_image_cmd = ["docker", "image", "inspect", image_name]
    try:
        result = subprocess.run(check_image_cmd, capture_output=True, check=False)
        if result.returncode != 0:
            print(f"Docker image {image_name} not found. Pulling image first...")
            try:
                subprocess.run(["docker", "pull", image_name], check=True)
                print(f"Successfully pulled image {image_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error pulling Docker image: {e}")
                return False
    except Exception as e:
        print(f"Error checking for Docker image: {e}")
        return False

    # Base command for both Windows and Linux
    command = [
        "docker", "run",
        "-it",              # Interactive mode with TTY
        "-d",               # Detached mode
        "--name", container_name,
        "--hostname", hostname,
    ]

    # Add GPU and volume mounting based on OS
    if is_windows:
        command.extend([
            "--gpus", "all",    # Enable all GPUs
            "-v", f"{current_dir}:C:/{mount_dir}",  # Mount using Windows path
        ])
    else:
        # Linux version with user mapping
        user_id = subprocess.check_output(['id', '-u']).decode('utf-8').strip()
        group_id = subprocess.check_output(['id', '-g']).decode('utf-8').strip()
        
        command.extend([
            "--user", f"{user_id}:{group_id}",  # Map container user to host user
            "--gpus", "all",    # Enable all GPUs
            "-v", f"{current_dir}:/{mount_dir}",  # Mount using Linux path
        ])

    command.append(image_name)
    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully created container: {container_name} with hostname: {hostname}")
        print(f"Container created with user mapping: {user_id}:{group_id}" if not is_windows else "Container created for Windows")
    except subprocess.CalledProcessError as e:
        print(f"Error creating container {container_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error creating container {container_name}: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Create Docker containers for OpenNMT with proper user permissions and GPU support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Create containers with latest image (Linux):
    python3 create_containers.py c1 c2 c3
    # Creates containers with host user permissions mapped automatically

  Create containers for Windows:
    python3 create_containers.py -w c1 c2 c3
    # or
    python3 create_containers.py --windows c1 c2 c3

  Create containers with specific version:
    python3 create_containers.py -v 1.0 c1 c2 c3
    # or
    python3 create_containers.py --version 1.0 c1 c2 c3

Note:
  - Each container will be created with GPU support enabled
  - For Linux:
    * Current directory will be mounted to /OpeNTF
    * Host user permissions will be mapped automatically to prevent permission issues
    * Uses host user and group IDs for container operations
  - For Windows:
    * Current directory will be mounted to C:/OpeNTF
    * Uses default Docker user permissions
  - Container names will be used as their hostnames
  - All containers are created in detached mode (-d) with interactive TTY (-it)
''')
    parser.add_argument('-v', '--version', dest='v', default='latest', 
                       help='Image version (default: latest)')
    parser.add_argument('-w', '--windows', action='store_true', 
                       help='Create containers for Windows environment')
    parser.add_argument('containers', nargs='+', 
                       help='One or more container names to create')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create containers
    for container_name in args.containers:
        success = create_docker_container(container_name, container_name, args.v, args.windows)
        if not success:
            print(f"Failed to create container {container_name}")
            continue

if __name__ == "__main__":
    main()