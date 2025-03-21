#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a toy dataset from a DBLP jsonl file.

This script reads a DBLP jsonl file and creates a smaller version with the specified
number of teams, preserving the data format and structure.

Example usage:
    python create_raw_toy_dblp.py -i data/raw/dblp/dblp_v12.jsonl -ts 500
"""

import os
import sys
import json
import argparse
import random
from tqdm import tqdm
from pathlib import Path
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.tprint import tprint

# Default thread usage as percentage of total available threads
DEFAULT_THREAD_PERCENTAGE = 0.5

# Try to import the parse_gpus_string utility function
try:
    from utils.parse_gpus_string import parse_gpus_string
except ImportError:
    # Fallback implementation if the import fails
    def parse_gpus_string(gpus_str):
        """Parse a comma-separated string of GPU indices."""
        if not gpus_str:
            return None

        # Handle special strings
        if gpus_str.lower() == "all":
            return "all"
        if gpus_str.lower() == "first":
            return "first"

        # Parse comma-separated indices
        try:
            if "," in gpus_str:
                return [int(idx.strip()) for idx in gpus_str.split(",")]
            else:
                return [int(gpus_str.strip())]
        except ValueError:
            tprint(f"Invalid GPU specification: {gpus_str}. Using first available GPU.")
            return "first"


# Function to count lines in a chunk of a file
def count_lines_in_chunk(file_path, start_pos, end_pos):
    """Count lines in a chunk of a file."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(start_pos)
        # If not at the beginning of the file, discard the first partial line
        if start_pos > 0:
            f.readline()

        current_pos = f.tell()
        while current_pos < end_pos:
            f.readline()
            count += 1
            current_pos = f.tell()

            # Break if we've reached the end of the file
            if current_pos >= os.path.getsize(file_path):
                break

    return count


# Function to process lines in a chunk of a file
def process_chunk(args):
    """Process a chunk of a file and write selected lines to output file."""
    file_path, start_pos, end_pos, selected_indices, output_file, chunk_id = args

    # Create a temporary output file for this chunk
    temp_output = f"{output_file}.part{chunk_id}"

    count = 0
    found_entries = 0
    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(start_pos)
        # If not at the beginning of the file, discard the first partial line
        if start_pos > 0:
            f.readline()

        current_pos = f.tell()
        lines_written = 0

        with open(temp_output, "w", encoding="utf-8") as out:
            while current_pos < end_pos:
                line = f.readline()

                # Check if this line index is in our selected indices
                if count in selected_indices:
                    out.write(line)
                    lines_written += 1
                    found_entries += 1

                count += 1
                current_pos = f.tell()

                # Break if we've reached the end of the file
                if current_pos >= os.path.getsize(file_path):
                    break

    # If no entries were written, return None instead of an empty file
    if lines_written == 0:
        try:
            os.unlink(temp_output)
            return None, 0
        except:
            pass

    return temp_output, lines_written


def create_toy_dblp(
    input_file, team_size, output_file=None, seed=42, gpus=None, threads=None
):
    """
    Create a toy dataset from a DBLP jsonl file.

    Args:
        input_file (str): Path to the input DBLP jsonl file
        team_size (int): Number of teams (publications) to include in the toy dataset
        output_file (str, optional): Path to the output file. If None, generates a default name
        seed (int, optional): Random seed for reproducibility
        gpus (str/list, optional): GPU indices to use. Can be None, a list of indices, or a string like "0,1,2"
        threads (int, optional): Number of CPU threads to use. If None, uses 50% of available threads

    Returns:
        str: Path to the created output file
    """
    start_time = time.time()

    # Set random seed for reproducibility
    random.seed(seed)

    # Determine total available CPU threads
    total_threads = multiprocessing.cpu_count()

    # Determine number of threads to use
    if threads is None:
        # Use default percentage of available threads
        threads = max(1, int(total_threads * DEFAULT_THREAD_PERCENTAGE))
    else:
        # Make sure threads is not more than available
        threads = min(threads, total_threads)

    tprint(f"Using {threads} CPU threads out of {total_threads} available")

    # Generate default output file name if not provided
    if output_file is None:
        base_name = os.path.basename(input_file)
        name_without_ext, ext = os.path.splitext(base_name)
        if ext.lower() != ".jsonl":
            ext = ".jsonl"
        output_file = os.path.join(
            os.path.dirname(input_file), f"{name_without_ext}_toy_ts{team_size}{ext}"
        )

    tprint(f"Creating toy dataset with {team_size} teams")
    tprint(f"Input file: {input_file}")
    tprint(f"Output will be saved to: {output_file}")

    # Check if GPU acceleration is requested and available
    GPU_AVAILABLE = False
    gpu_indices = None

    if gpus is not None:
        try:
            import cupy as cp

            # Parse GPU indices if given as string
            if isinstance(gpus, str):
                gpu_indices = parse_gpus_string(gpus)
            else:
                gpu_indices = gpus

            # Check if GPU indices are valid
            if gpu_indices == "all":
                # Use all available GPUs
                device_count = cp.cuda.runtime.getDeviceCount()
                gpu_indices = list(range(device_count))
                tprint(f"Using all {device_count} available GPUs")
            elif gpu_indices == "first":
                # Use first GPU
                gpu_indices = [0]
                tprint(f"Using the first GPU (index 0)")
            elif isinstance(gpu_indices, list):
                # Use specified GPU indices
                device_count = cp.cuda.runtime.getDeviceCount()
                valid_indices = [idx for idx in gpu_indices if idx < device_count]
                if not valid_indices:
                    tprint("No valid GPU indices provided. Using CPU.")
                    GPU_AVAILABLE = False
                else:
                    gpu_indices = valid_indices
                    tprint(f"Using GPUs with indices: {gpu_indices}")
                    GPU_AVAILABLE = True

            if GPU_AVAILABLE:
                # Set the visible devices
                for i, idx in enumerate(gpu_indices):
                    if i == 0:  # Use the first GPU for computations
                        cp.cuda.Device(idx).use()
                    # Print device information
                    device_props = cp.cuda.runtime.getDeviceProperties(idx)
                    tprint(
                        f"GPU {idx}: {device_props['name'].decode('utf-8')}, "
                        f"Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB"
                    )

        except ImportError:
            tprint(
                "GPU acceleration requested but CuPy not available. Using CPU instead."
            )
            GPU_AVAILABLE = False
        except Exception as e:
            tprint(f"Error initializing GPUs: {e}. Using CPU instead.")
            GPU_AVAILABLE = False

    # Get file size
    file_size = os.path.getsize(input_file)

    # First, count the total number of entries in the file
    tprint("Counting total entries in the input file...")
    count_start = time.time()

    if (
        file_size > 100_000_000 and threads > 1
    ):  # Only use parallel counting for files > 100MB
        # Divide file into chunks for parallel processing
        chunk_size = file_size // threads
        chunk_positions = [
            (i * chunk_size, min((i + 1) * chunk_size, file_size))
            for i in range(threads)
        ]

        # Count lines in each chunk in parallel
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(count_lines_in_chunk, input_file, start, end)
                for start, end in chunk_positions
            ]

            # Get results with progress bar
            total_entries = 0
            with tqdm(total=threads, desc="Counting entries") as pbar:
                for future in futures:
                    total_entries += future.result()
                    pbar.update(1)
    else:
        # Count lines sequentially for smaller files
        total_entries = 0
        with open(input_file, "r", encoding="utf-8") as f:
            for _ in tqdm(f, desc="Counting entries"):
                total_entries += 1

    count_time = time.time() - count_start
    tprint(f"Found {total_entries} entries in {count_time:.2f} seconds")

    # If requested team size is greater than or equal to total entries, just copy the file
    if team_size >= total_entries:
        tprint(
            f"Requested team size ({team_size}) is greater than or equal to total entries ({total_entries})"
        )
        tprint("Creating a copy of the original file...")
        copy_start = time.time()
        with open(input_file, "r", encoding="utf-8") as f_in, open(
            output_file, "w", encoding="utf-8"
        ) as f_out:
            for line in tqdm(f_in, total=total_entries, desc="Copying file"):
                f_out.write(line)
        copy_time = time.time() - copy_start
        tprint(
            f"Created toy dataset with {total_entries} teams in {copy_time:.2f} seconds"
        )
        return output_file

    # Select random indices to include in the toy dataset
    select_start = time.time()
    if GPU_AVAILABLE:  # Use GPU for selection if available
        try:
            # Use GPU for selection
            tprint("Using GPU to select random indices...")
            import cupy as cp

            # Create array of all indices
            indices = cp.arange(total_entries)
            # Shuffle the indices
            cp.random.seed(seed)
            cp.random.shuffle(indices)
            # Take the first team_size indices
            selected_indices_gpu = indices[:team_size]
            # Convert to set for faster lookup
            selected_indices = set(selected_indices_gpu.get().tolist())

            tprint("Successfully used GPU for index selection")
        except Exception as e:
            tprint(f"Error using GPU for index selection: {e}. Falling back to CPU.")
            # Use CPU for selection
            selected_indices = set(random.sample(range(total_entries), team_size))
    else:
        # Use CPU for selection
        tprint("Using CPU for index selection")
        selected_indices = set(random.sample(range(total_entries), team_size))

    select_time = time.time() - select_start
    tprint(f"Selected {team_size} indices in {select_time:.2f} seconds")

    # Create the toy dataset
    write_start = time.time()

    # For very small team sizes or small files, use sequential processing to avoid parallelization overhead
    if team_size < 100 or file_size < 100_000_000 or threads <= 1:
        # Process sequentially
        tprint(f"Using sequential processing for {team_size} teams...")
        with open(input_file, "r", encoding="utf-8") as f_in, open(
            output_file, "w", encoding="utf-8"
        ) as f_out:
            for i, line in tqdm(
                enumerate(f_in), total=total_entries, desc="Processing file"
            ):
                if i in selected_indices:
                    f_out.write(line)
    else:
        # Use parallel processing for larger files and team sizes
        tprint(
            f"Using parallel processing with {threads} threads to create toy dataset..."
        )

        # Divide file into chunks for parallel processing
        chunk_size = file_size // threads
        chunk_positions = [
            (i * chunk_size, min((i + 1) * chunk_size, file_size))
            for i in range(threads)
        ]

        # Process chunks in parallel without distributing indices
        temp_files = []
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = []
            
            for i, (start, end) in enumerate(chunk_positions):
                # Process each chunk with the full set of selected indices
                task = (input_file, start, end, selected_indices, output_file, i)
                futures.append(executor.submit(process_chunk, task))

            # Get results with progress bar
            with tqdm(total=threads, desc="Processing chunks") as pbar:
                for future in futures:
                    temp_file, lines_written = future.result()
                    if temp_file and lines_written > 0:
                        temp_files.append(temp_file)
                    pbar.update(1)

        # Combine temporary files into final output
        tprint("Combining chunk results into final output file...")
        with open(output_file, "w", encoding="utf-8") as out:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    with open(temp_file, "r", encoding="utf-8") as f:
                        out.write(f.read())
                    # Delete the temporary file
                    os.unlink(temp_file)

    write_time = time.time() - write_start
    tprint(f"Wrote entries in {write_time:.2f} seconds")

    total_time = time.time() - start_time
    tprint(f"Created toy dataset with {team_size} teams in {total_time:.2f} seconds")
    tprint(f"Output saved to: {output_file}")

    return output_file


def main():
    """Main entry point for the script."""
    # Define our custom help text
    help_text = """Create a toy dataset from a DBLP jsonl file

Required:
   -i INPUT, --input INPUT
	Path to the input DBLP jsonl file

   -ts TEAM_SIZE, --team-size TEAM_SIZE
	Number of teams (publications) to include in the toy dataset


Optionals:
   -o OUTPUT, --output OUTPUT
	Path to the output file (default: input_file_toy_ts{team_size}.jsonl)

   -s SEED, --seed SEED
	Random seed for reproducibility (default: 42)

   -gpus GPUS
	GPU indices to use, comma separated (e.g., "0,1,2") (default: None)

   -t THREADS, --threads THREADS
	Number of CPU threads to use (default: 50% of available CPU threads)
"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a toy dataset from a DBLP jsonl file",
        add_help=False,  # Disable default help
    )

    # Override the help option to print our custom help text
    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    # Required arguments group
    required = parser.add_argument_group("Required")
    required.add_argument(
        "-i",
        "--input",
        dest="input",
        help=argparse.SUPPRESS,
        required=True,
        metavar="INPUT",
    )

    required.add_argument(
        "-ts",
        "--team-size",
        dest="team_size",
        type=int,
        help=argparse.SUPPRESS,
        required=True,
        metavar="TEAM_SIZE",
    )

    # Optional arguments group
    optionals = parser.add_argument_group("Optionals")
    optionals.add_argument(
        "-o", "--output", dest="output", help=argparse.SUPPRESS, metavar="OUTPUT"
    )

    optionals.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help=argparse.SUPPRESS,
        metavar="SEED",
    )

    optionals.add_argument(
        "-gpus", dest="gpus", default=None, help=argparse.SUPPRESS, metavar="GPUS"
    )

    optionals.add_argument(
        "-t",
        "--threads",
        dest="threads",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
        metavar="THREADS",
    )

    # Override the print_help method to print our custom help text
    def custom_print_help(file=None):
        print(help_text, file=file)

    parser.print_help = custom_print_help

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input):
        tprint(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)

    if args.team_size <= 0:
        tprint(f"Error: Team size must be positive, got {args.team_size}")
        sys.exit(1)

    if args.threads is not None and args.threads <= 0:
        tprint(f"Error: Thread count must be positive, got {args.threads}")
        sys.exit(1)

    # Create the toy dataset
    output_file = create_toy_dblp(
        args.input, args.team_size, args.output, args.seed, args.gpus, args.threads
    )

    tprint(f"Done! Toy dataset saved to: {output_file}")


if __name__ == "__main__":
    main()
