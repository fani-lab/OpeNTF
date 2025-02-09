#!/usr/bin/env python3

import subprocess
import sys
import os

user = 'root'

def get_processes_in_container(container_name):
    """Get Python3 and OpenNMT processes in the specified container."""
    try:
        # Filter out multiprocessing spawn processes and only show main python3 and onmt_ processes
        cmd = f"docker exec {container_name} ps aux | grep -E 'python3|onmt_' | grep -v 'multiprocessing' | grep -v 'resource_tracker'"
        # Use grep -v to exclude the grep command itself from results
        cmd += " | grep -v grep"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode not in [0, 1]:  # grep returns 1 if no matches found
            print(f"Error checking processes in container {container_name}")
            return None
            
        return result.stdout.strip()
    
    except subprocess.SubprocessError as e:
        print(f"Error executing command for container {container_name}: {str(e)}")
        return None

def get_matching_containers(pattern):
    """Get list of container names matching the given pattern."""
    try:
        grep_pattern = pattern.replace('*', '')
        cmd = f"docker ps --format '{{{{.Names}}}}' | grep '{grep_pattern}'"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode not in [0, 1]:
            print(f"Error getting container list for pattern: {pattern}")
            return []
            
        containers = [c.strip() for c in result.stdout.split('\n') if c.strip()]
        return containers
        
    except subprocess.SubprocessError as e:
        print(f"Error listing containers: {str(e)}")
        return []

def check_container_exists(container_name):
    """Check if the specified container exists and is running."""
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=True
        )
        containers = result.stdout.strip().split('\n')
        return container_name in containers
    except subprocess.CalledProcessError:
        return False

def show_container_processes(container_name):
    """Display all processes running in the container."""
    try:
        print(f"\nProcesses running in container '{container_name}':")
        subprocess.run(
            ['docker', 'top', container_name],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error getting container processes: {str(e)}")

def run_shell_script_in_container(container_name, script_name):
    """Run the specified shell script from the container's run_scripts folder."""
    if not script_name.endswith('.sh'):
        print("Error: The provided file must be a shell script (.sh)")
        sys.exit(1)

    script_path = f"/OpeNTF/run_scripts/{script_name}"

    try:
        # Check if script exists in container
        check_script = subprocess.run(
            ['docker', 'exec', '-u', user, container_name, 'test', '-f', script_path],
            capture_output=True,
            check=False
        )
        
        if check_script.returncode != 0:
            print(f"Error: Script '{script_name}' not found in /OpeNTF/run_scripts folder")
            sys.exit(1)

        subprocess.run(
            ['docker', 'exec', '-u', user, container_name, 'chmod', '+x', script_path],
            check=True
        )
        
        subprocess.run(
            ['docker', 'exec', '-u', user, '-w', '/OpeNTF/run_scripts', container_name, 'bash', '-c', f'./{script_name}'],
            check=True
        )
        
        print(f"Script started. Check run_logs directory for {script_name.replace('.sh', '')}.log")
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {str(e)}")
        sys.exit(1)

def kill_process_in_container(container_name, script_name):
    """Kill a specific script process running in the container."""
    script_path = f"/OpeNTF/run_scripts/{script_name}"
    try:
        # Show processes before stopping
        print("\nBefore stopping the script:")
        processes = get_processes_in_container(container_name)
        if processes:
            print(processes)
        else:
            print("No Python/OpenNMT processes found")

        # Kill the process
        subprocess.run(
            ['docker', 'exec', '-u', user, container_name, 'pkill', '-f', script_path],
            check=True
        )
        print(f"\nAttempted to stop process running {script_name}")
        
        # Show remaining processes
        print("\nRemaining processes:")
        processes = get_processes_in_container(container_name)
        if processes:
            print(processes)
        else:
            print("No Python/OpenNMT processes found")
            
    except subprocess.CalledProcessError as e:
        print(f"Error stopping the process: {str(e)}")

def kill_all_user_processes(container_name):
    """Kill all Python and OpenNMT processes started by the specified user in the container."""
    try:
        # Show processes before stopping
        print("\nBefore stopping all processes:")
        processes = get_processes_in_container(container_name)
        if processes:
            print(processes)
        else:
            print("No Python/OpenNMT processes found")

        subprocess.run(
            ['docker', 'exec', '-u', user, container_name, 'pkill', '-f', 'python3'],
            check=False
        )
        
        subprocess.run(
            ['docker', 'exec', '-u', user, container_name, 'pkill', '-f', 'onmt'],
            check=False
        )
        
        print(f"\nAttempted to stop all Python and OpenNMT processes for user {user}")
        
        # Show remaining processes
        print("\nRemaining processes:")
        processes = get_processes_in_container(container_name)
        if processes:
            print(processes)
        else:
            print("No Python/OpenNMT processes found")
        
    except subprocess.CalledProcessError as e:
        print(f"Error stopping processes: {str(e)}")

def remove_path_using_docker(path):
    """Remove a file or directory using a temporary Docker container with root access."""
    try:
        # Create a temporary container with root access and mount the path to be deleted
        abs_path = os.path.abspath(path)
        parent_dir = os.path.dirname(abs_path)
        target_name = os.path.basename(abs_path)
        
        cmd = [
            'docker', 'run', '--rm',  # Remove container after execution
            '-v', f'{parent_dir}:/workspace',  # Mount parent directory
            'ubuntu:latest',  # Use ubuntu image for basic operations
            'rm', '-rf', f'/workspace/{target_name}'  # Remove the target
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Successfully removed: {path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error removing path: {str(e)}")
        sys.exit(1)

def print_help():
    """Print available commands and their usage."""
    print("\nUsage: python3 in_docker.py <command> [options]")
    print("\nAvailable commands:")

    print("\n  ps: List Python3/OpenNMT processes in containers")
    print("     Usage: python3 in_docker.py ps <container_name> [container2 ...]")
    print("            python3 in_docker.py ps <pattern>  (e.g., kap_*)")
    print("     Example: python3 in_docker.py ps my_container")
    print("     Example: python3 in_docker.py ps my_container kap_*")

    print("\n  run: Run a shell script inside a container")
    print("     Usage: python3 in_docker.py run <container_name> <script_name.sh>")
    print("     Example: python3 in_docker.py run my_container train.sh")
    print("     Note: Scripts should be placed in the /OpeNTF/run_scripts directory inside the container.")

    print("\n  stop: Stop a specific script process in a container")
    print("     Usage: python3 in_docker.py stop <container_name> <script_name.sh>")
    print("     Example: python3 in_docker.py stop my_container train.sh")

    print("\n  stopall: Stop all Python/OpenNMT processes in a container")
    print("      Usage: python3 in_docker.py stopall <container_name>")
    print("      Example: python3 in_docker.py stopall my_container")

    print("\n  rm: Remove a file or directory (useful for docker-created files)")
    print("     Usage: python3 in_docker.py rm <path_to_folder_or_file>")
    print("     Example: python3 in_docker.py rm /path/to/folder")
    print("     Note: Uses a temporary Docker container with root access to remove the path")

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0)

    command = sys.argv[1]

    if command == "ps":
        if len(sys.argv) < 3:
            print("Error: Please provide container name(s) or pattern")
            sys.exit(1)
        
        patterns = sys.argv[2:]
        containers = []
        
        for pattern in patterns:
            if '*' in pattern:
                containers.extend(get_matching_containers(pattern))
            else:
                containers.append(pattern)
        
        if not containers:
            print("No matching containers found")
            sys.exit(1)
        
        for container in containers:
            print(f"\n=== Processes in container: {container} ===")
            processes = get_processes_in_container(container)
            if processes:
                print(processes)
            else:
                print("No Python/OpenNMT processes found or container not running")

    elif command == "run":
        if len(sys.argv) != 4:
            print("Error: Please provide container name and script name")
            sys.exit(1)
        container_name = sys.argv[2]
        script_name = sys.argv[3]
        if not check_container_exists(container_name):
            print(f"Error: Container '{container_name}' does not exist.")
            sys.exit(1)
        run_shell_script_in_container(container_name, script_name)

    elif command == "stop":
        if len(sys.argv) != 4:
            print("Error: Please provide container name and script name")
            sys.exit(1)
        container_name = sys.argv[2]
        script_name = sys.argv[3]
        if not check_container_exists(container_name):
            print(f"Error: Container '{container_name}' does not exist.")
            sys.exit(1)
        kill_process_in_container(container_name, script_name)

    elif command == "stopall":
        if len(sys.argv) != 3:
            print("Error: Please provide container name")
            sys.exit(1)
        container_name = sys.argv[2]
        if not check_container_exists(container_name):
            print(f"Error: Container '{container_name}' does not exist.")
            sys.exit(1)
        kill_all_user_processes(container_name)

    elif command == "rm":
        if len(sys.argv) != 3:
            print("Error: Please provide path to remove")
            sys.exit(1)
        path = sys.argv[2]
        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist.")
            sys.exit(1)
        remove_path_using_docker(path)

    elif command in ("-h", "--help"):
        print_help()
        sys.exit(0)

    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 