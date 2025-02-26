#!/usr/bin/env python3

import sys
import subprocess
import os
import argparse

def create_docker_container(container_name, hostname, version="latest", is_windows=False, gpu_mode="all"):
    """
    Create a Docker container with specified name and hostname
    
    gpu_mode options:
    - "all": Standard --gpus all flag
    - "none": No GPU access
    - "device": Use device mounting instead of --gpus flag
    - "runtime": Use nvidia runtime instead of --gpus flag
    - "host-libs": Mount host NVIDIA libraries
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
    # full command above
    # docker run -it -d --name c1 --hostname c1 kmthang/opennmt:latest

    # Add user mapping for Linux
    if not is_windows:
        user_id = subprocess.check_output(['id', '-u']).decode('utf-8').strip()
        group_id = subprocess.check_output(['id', '-g']).decode('utf-8').strip()
        command.extend(["--user", f"{user_id}:{group_id}"])  # Map container user to host user

    # Add GPU configuration based on mode
    if gpu_mode == "all":
        command.extend(["--gpus", "all"])
    elif gpu_mode == "runtime":
        command.extend(["--runtime", "nvidia"])
    elif gpu_mode == "device":
        # Try to get GPU device information
        try:
            # Get a list of NVIDIA device files
            nvidia_devices = []
            for device in ['/dev/nvidia0', '/dev/nvidia1', '/dev/nvidia2', '/dev/nvidia3', 
                          '/dev/nvidiactl', '/dev/nvidia-modeset', '/dev/nvidia-uvm', 
                          '/dev/nvidia-uvm-tools']:
                if os.path.exists(device):
                    nvidia_devices.append(device)
                    command.extend(["--device", device])
            
            print(f"Added GPU devices: {nvidia_devices}")
        except Exception as e:
            print(f"Warning: Could not add GPU devices: {e}")
    elif gpu_mode == "host-libs":
        # Mount host NVIDIA libraries
        try:
            # Add device mounts
            nvidia_devices = []
            for device in ['/dev/nvidia0', '/dev/nvidia1', '/dev/nvidia2', '/dev/nvidia3', 
                          '/dev/nvidiactl', '/dev/nvidia-modeset', '/dev/nvidia-uvm', 
                          '/dev/nvidia-uvm-tools']:
                if os.path.exists(device):
                    nvidia_devices.append(device)
                    command.extend(["--device", device])
            
            # Mount NVIDIA driver libraries
            nvidia_lib_paths = [
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/usr/lib/x86_64-linux-gnu/libcuda.so",
                "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
            ]
            
            for lib_path in nvidia_lib_paths:
                if os.path.exists(lib_path):
                    command.extend(["-v", f"{lib_path}:{lib_path}"])
            
            print(f"Added GPU devices: {nvidia_devices}")
            print(f"Mounted NVIDIA libraries from host")
        except Exception as e:
            print(f"Warning: Could not set up host GPU libraries: {e}")
    elif gpu_mode == "none":
        print("Creating container without GPU access")
    else:
        print(f"Unknown GPU mode: {gpu_mode}, defaulting to --gpus all")
        command.extend(["--gpus", "all"])

    # Add volume mounting based on OS
    if is_windows:
        command.extend(["-v", f"{current_dir}:C:/{mount_dir}"])  # Mount using Windows path
    else:
        command.extend(["-v", f"{current_dir}:/{mount_dir}"])  # Mount using Linux path

    command.append(image_name)
    
    try:
        print(f"Running command: {' '.join(command)}")
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
    
  Create containers with different GPU modes:
    python3 create_containers.py --gpu-mode none c1 c2 c3  # No GPU access
    python3 create_containers.py --gpu-mode runtime c1 c2 c3  # Use nvidia runtime
    python3 create_containers.py --gpu-mode device c1 c2 c3  # Use device mounting
    python3 create_containers.py --gpu-mode host-libs c1 c2 c3  # Mount host NVIDIA libraries

Note:
  - Each container will be created with GPU support enabled by default
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
    parser.add_argument('--gpu-mode', choices=['all', 'none', 'runtime', 'device', 'host-libs'], default='all',
                       help='GPU access mode (default: all)')
    parser.add_argument('containers', nargs='+', 
                       help='One or more container names to create')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create containers
    for container_name in args.containers:
        success = create_docker_container(container_name, container_name, args.v, args.windows, args.gpu_mode)
        if not success:
            print(f"Failed to create container {container_name}")
            continue

if __name__ == "__main__":
    main()
    
    print("\n" + "="*80)
    print("IMPORTANT NOTES ABOUT GPU COMPATIBILITY:")
    print("="*80)
    print("If you're using NVIDIA H100 GPUs or other newer GPUs, you may need to:")
    print("1. Use the '--gpu-mode host-libs' option which has been shown to work")
    print("2. Update PyTorch inside the container to support newer GPU architectures")
    print("\nTo update PyTorch inside a running container, you can try:")
    print("  docker exec -u 0 <container_name> pip3 install --upgrade torch torchvision torchaudio")
    print("\nAlternatively, you can create a custom Docker image with updated PyTorch:")
    print("  1. Create a Dockerfile.updated with:")
    print("     FROM kmthang/opennmt:latest")
    print("     RUN pip3 install --upgrade torch torchvision torchaudio")
    print("  2. Build the image: docker build -t kmthang/opennmt:updated -f Dockerfile.updated .")
    print("  3. Use the updated image: python create_containers.py -v updated --gpu-mode host-libs <container_name>")
    print("="*80)