#!/usr/bin/env python3

import subprocess
import time

def run_command_in_container(container_name, script_name):
    """Run a script inside a specified container"""
    command = [
        "docker", "exec",
        container_name,
        "/bin/bash", "-c",
        f"cd /OpeNTF/run_scripts && chmod +x {script_name} && ./{script_name}"
    ]
    
    try:
        print(f"Starting {script_name} in container {container_name}")
        subprocess.run(command, check=True)
        print(f"Completed {script_name} in container {container_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name} in container {container_name}: {e}")
        return False
    return True

def main():
    # Define container assignments for ConvS2S experiments
    convs2s_experiments = [
        ("kap_c1", "1.4.1-imdb_convs2s_final.sh"),
        ("kap_c2", "1.4.1-gith_convs2s_final.sh"),
        ("kap_c3", "1.4.1-dblp_convs2s_final.sh"),
        ("kap_c4", "1.4.1-uspt_convs2s_final.sh")
    ]

    # Define container assignments for RNN experiments
    rnn_experiments = [
        ("kap_r1", "1.4.1-imdb_rnn_final.sh"),
        ("kap_r2", "1.4.1-gith_rnn_final.sh"),
        ("kap_r3", "1.4.1-dblp_rnn_final.sh"),
        ("kap_r4", "1.4.1-uspt_rnn_final.sh")
    ]

    # Run ConvS2S experiments
    print("Starting ConvS2S experiments...")
    for container, script in convs2s_experiments:
        run_command_in_container(container, script)

    # Run RNN experiments
    print("\nStarting RNN experiments...")
    for container, script in rnn_experiments:
        run_command_in_container(container, script)

if __name__ == "__main__":
    main() 