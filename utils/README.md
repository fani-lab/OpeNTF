# OpeNTF Utilities

This folder contains various utility scripts that assist with data processing, analysis, and Docker container management for the OpeNTF project. Each script is designed for a specific purpose. Below is a brief description of what each script does and why you might need to use it.

---

## GPU & System Checks

### checkgpu.py
- **Purpose:** Verifies if CUDA is available and displays GPU information.
- **Usage:** Run the script to print CUDA availability, device name, and current device index.

### checkmax.py
- **Purpose:** Scans log files for numerical patterns based on a given prefix and threshold.
- **Usage Example:**
  ```bash
  python3 checkmax.py m 2011 /path/to/logfile.log
  ```
- **When to Use:** Useful for checking if specific metrics or timestamps in log files exceed a defined value.

---

## Data Conversion & Analysis

### convert_metrics.py
- **Purpose:** Converts evaluation metrics from CSV files by selecting specific metrics rows, scaling values (e.g., to percentage), and outputting a formatted result.
- **Usage Example:**
  ```bash
  python3 convert_metrics.py path/to/metrics.csv
  ```
- **When to Use:** Helps in transforming raw metric data into a concise format for reporting or further analysis.

### datashape.py
- **Purpose:** Displays the shapes and sample contents of the data structures (like indexes and teams vectors) to verify preprocessing.
- **Usage Example:**
  ```bash
  python3 datashape.py dblp dblp.v12.json.filtered.mt75.ts3 --rows 5
  ```
- **When to Use:** Typically run after data preprocessing to inspect that the data's structure is as expected.

### team_stats.py
- **Purpose:** Analyzes team data to generate detailed statistics such as team sizes, expert participation, duplicate teams, and threshold-based summaries.
- **Usage Example:**
  ```bash
  python3 team_stats.py gith gith.data.csv.filtered.mt75.ts3 --minimum-teams 50 --team-size 4
  ```
- **When to Use:** Ideal for understanding the distribution of skills and experts across teams and assessing overall data quality.

### fix_data.py
- **Purpose:** Cleans and filters teams data by removing teams or experts that do not meet specified criteria, handles duplicate teams, and updates corresponding indexes.
- **Usage Example:**
  ```bash
  python3 fix_data.py gith gith.data.csv.filtered.mt75.ts3 --minimum-teams 75 --team-size 3
  ```
- **When to Use:** Run this script to ensure the dataset meets quality standards before further analysis or training.

---

## Docker Container Management

### create_containers.py
- **Purpose:** Creates Docker containers for setting up OpenNMT training environments with GPU support.
- **Usage Example:**
  ```bash
  python3 create_containers.py --v 1.0 container1 container2
  ```
- **When to Use:** Facilitates the quick provision of containerized environments needed for OpenNMT with customizable container names and image versions.

### in_docker.py
- **Purpose:** Provides a suite of Docker container management commands, including:
  - Listing processes inside containers.
  - Running shell scripts within containers.
  - Stopping specific or all processes.
  - Removing files or directories via a temporary container.
- **Usage Examples:**
  ```bash
  # List processes in a container
  python3 in_docker.py ps container_name

  # Run a shell script inside a container
  python3 in_docker.py run container_name script.sh

  # Stop a specific script process
  python3 in_docker.py stop container_name script.sh

  # Stop all Python/OpenNMT processes
  python3 in_docker.py stopall container_name

  # Remove a problematic file or folder
  python3 in_docker.py rm /path/to/remove
  ```
- **When to Use:** Use this script for managing iterative training processes, troubleshooting container issues, or cleaning up resources.

---

Each utility is designed to streamline different parts of the workflow associated with the OpeNTF project. For further details or usage instructions, run the respective script with the `-h` flag to see its help message.

Happy processing!
