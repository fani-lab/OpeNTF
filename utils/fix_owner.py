#!/usr/bin/env python

import os
import sys
import subprocess

def fix_ownership_using_docker(path):
    """Change ownership of a path to current host user using a temporary Docker container."""
    try:
        # Get current user and group IDs
        user_id = subprocess.check_output(['id', '-u']).decode('utf-8').strip()
        group_id = subprocess.check_output(['id', '-g']).decode('utf-8').strip()

        # Create absolute path and get parent directory
        abs_path = os.path.abspath(path)
        parent_dir = os.path.dirname(abs_path)
        target_name = os.path.basename(abs_path)
        
        cmd = [
            'docker', 'run', '--rm',  # Remove container after execution
            '-v', f'{parent_dir}:/workspace',  # Mount parent directory
            'ubuntu:latest',  # Use ubuntu image for basic operations
            'chown', '-R', f'{user_id}:{group_id}', f'/workspace/{target_name}'  # Change ownership
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Successfully changed ownership of '{path}' to user {user_id}:{group_id}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error changing ownership: {str(e)}")

def print_help():
    print("\nFix Ownership Tool")
    print("=================")
    print("A utility to change ownership of files/directories to the current host user.")
    print("\nUsage:")
    print("  python fix_owner.py <path_to_folder_or_file>")
    print("\nDescription:")
    print("  Changes ownership of a file/directory to the current host user.")
    print("  Uses a temporary Docker container with root access to change ownership to current host user.")
    print("  Useful for fixing permission issues with Docker-created files.")
    print("\nExample:")
    print("  python fix_owner.py /path/to/folder")

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0)
        
    if len(sys.argv) != 2:
        print("Error: Please provide a path to change ownership")
        print_help()
        sys.exit(1)
        
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        sys.exit(1)
        
    fix_ownership_using_docker(path)

if __name__ == "__main__":
    main() 