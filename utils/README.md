# OpeNTF-NMT Utilities

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
- **Features:**
  - Automatic user permission mapping for Linux environments
  - Windows compatibility mode
  - GPU support enabled by default
- **Usage Examples:**
  ```bash
  # Create containers in Linux (with user permission mapping)
  python3 create_containers.py c1 c2 c3

  # Create containers for Windows environment
  python3 create_containers.py -w c1 c2 c3
  # or
  python3 create_containers.py --windows c1 c2 c3

  # Create containers with specific version
  python3 create_containers.py -v 1.0 c1 c2 c3
  # or
  python3 create_containers.py --version 1.0 c1 c2 c3
  ```
- **When to Use:** When setting up new training environments or scaling existing ones.

### in_docker.py
- **Purpose:** Provides a comprehensive suite of Docker container management commands.
- **Available Commands:**
  - **ps**: List Python/OpenNMT processes in containers
    ```bash
    python3 in_docker.py ps container_name
    python3 in_docker.py ps "container_prefix_*"  # Pattern matching
    ```
  - **run**: Execute shell scripts within containers
    ```bash
    python3 in_docker.py run container_name script.sh
    ```
  - **stop**: Stop specific script processes
    ```bash
    python3 in_docker.py stop container_name script.sh
    ```
  - **stopall**: Stop all Python/OpenNMT processes
    ```bash
    python3 in_docker.py stopall container_name
    ```
  - **rm**: Remove files/directories using a temporary container
    ```bash
    python3 in_docker.py rm /path/to/remove
    ```
  - **mv**: Rename files/directories using a temporary container
    ```bash
    python3 in_docker.py mv /old/path /new/path
    ```
  - **fixowner**: Change ownership of files/directories to current host user
    ```bash
    python3 in_docker.py fixowner /path/to/fix
    ```
- **When to Use:**
  - Managing container processes
  - Running training scripts
  - Handling permission issues with Docker-created files
  - Cleaning up resources
  - Fixing ownership of files created by Docker containers

---

Each utility is designed to streamline different parts of the workflow associated with the OpeNTF project. For further details or usage instructions, run the respective script with the `-h` flag to see its help message.

Happy processing!
