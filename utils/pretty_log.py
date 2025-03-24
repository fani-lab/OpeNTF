import sys
import shutil
import pytz
from datetime import datetime
from utils.tprint import tprint

# Dictionary to track the last label used for in-place updates
_last_progress_labels = {}


def pretty_log(
    label, memory_gb=None, is_header=False, is_subheader=False, progress=None
):
    """
    Format logs with pretty headers and right-aligned memory usage and progress bar.

    Args:
        label: Message to display
        memory_gb: Memory usage in GB (optional)
        is_header: Whether this is a main section header
        is_subheader: Whether this is a subsection header
        progress: Dictionary with progress info (optional):
            - current: Current progress value
            - total: Total for 100% completion
            - suffix: Suffix to display (like "MB/s")
            - bar_length: Length of the progress bar in chars
    """
    global _last_progress_labels

    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Format based on header type
    if is_header:
        header_line = "#" * terminal_width
        print(f"\n{header_line}")
        tprint(f"{label.upper()}")
        print(f"{header_line}\n")
        _last_progress_labels = {}  # Reset progress tracking after headers
    elif is_subheader:
        header_line = "-" * terminal_width
        print(f"\n{header_line}")
        tprint(f"{label}")
        print(f"{header_line}\n")
        _last_progress_labels = {}  # Reset progress tracking after headers
    else:
        # Regular log with optional memory usage and progress bar
        message = tprint(label)

        # Calculate remaining space
        remaining_space = terminal_width - len(message)

        # If we have progress info, create a progress bar
        progress_bar = ""
        if progress and "current" in progress and "total" in progress:
            current = progress["current"]
            total = progress["total"]

            # Default bar length - shorter than before (half the length)
            bar_length = progress.get(
                "bar_length", min(15, int(remaining_space * 0.25))
            )

            # Calculate percentage
            if total > 0:
                percent = min(1.0, current / total)  # Cap at 100%
            else:
                percent = 0

            # Determine bar characters
            filled_length = int(bar_length * percent)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)

            # Format with percentage on the left side
            suffix = progress.get("suffix", "")
            progress_bar = f"{percent*100:.1f}% [{bar}] {suffix}"

            # Update remaining space
            remaining_space = remaining_space - len(progress_bar) - 2  # 2 for spacing

        # Format memory info
        memory_info = ""
        if memory_gb is not None:
            memory_info = f"Mem: {memory_gb:.2f} GB"

        # Calculate spacing to right-align progress and memory info
        if progress_bar and memory_info:
            # Both progress and memory info
            spacing = remaining_space - len(memory_info) - 2  # 2 for spacing
            output = f"{message}{' ' * max(2, spacing)}{progress_bar}  {memory_info}"
        elif progress_bar:
            # Only progress bar
            spacing = remaining_space - 2  # 2 for spacing
            output = f"{message}{' ' * max(2, spacing)}{progress_bar}"
        elif memory_info:
            # Only memory info
            spacing = remaining_space - 2  # 2 for spacing
            output = f"{message}{' ' * max(2, spacing)}{memory_info}"
        else:
            # Just the message
            output = message

        # Check if we need to update in-place
        if progress and label in _last_progress_labels:
            # Move cursor to beginning of line and clear line
            sys.stdout.write("\r" + " " * terminal_width + "\r")
            sys.stdout.write(output)
            sys.stdout.flush()
        else:
            # For memory-only logs (no progress), add a newline before the message
            if memory_gb is not None and not progress:
                sys.stdout.write("\n")

            # New line for new messages or non-progress messages
            print(output)

            # Ensure we have a newline after memory-only logs (no progress)
            if memory_gb is not None and not progress:
                sys.stdout.write("\n")

            sys.stdout.flush()

        # Remember this label for future updates
        if progress:
            _last_progress_labels[label] = True
        else:
            # Remove from tracking if no progress
            _last_progress_labels.pop(label, None)


def log_memory_usage(label, is_header=False, is_subheader=False, progress=None):
    """
    Log memory usage with pretty formatting and optional progress bar.

    This is a convenience wrapper that automatically calculates memory usage.

    Args:
        label: Message to display
        is_header: Whether this is a main section header
        is_subheader: Whether this is a subsection header
        progress: Dictionary with progress info (optional)
    """
    import psutil  # Import here to avoid making it a required dependency

    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)

    # Use the pretty_log function
    pretty_log(label, memory_gb, is_header, is_subheader, progress)


def main():
    """
    Main function to demonstrate pretty_log usage and show help menu
    """
    import argparse
    import time

    # Create example outputs
    example_text = """
Example Outputs:

1. Header:
   ##################################################
   PROCESSING DATA
   ##################################################

2. Subheader:
   --------------------------------------------------
   Loading Files
   --------------------------------------------------

3. Progress bar:
   Processing files    75.0% [███████████----] files

4. Memory usage:
   Training model                        Mem: 2.45 GB

5. Combined progress and memory:
   Training model    50.0% [███████--------] epoch  Mem: 2.45 GB
"""

    parser = argparse.ArgumentParser(
        description="Pretty logging utility with progress bars and memory tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_text,
    )

    # Required arguments group
    required = parser.add_argument_group("Required")
    required.add_argument("-m", "--message", type=str, help="Message to display")

    # Optional arguments group
    optional = parser.add_argument_group("Optional")
    optional.add_argument(
        "-p", "--progress", action="store_true", help="Show progress bar example"
    )
    optional.add_argument(
        "-mem", "--memory", action="store_true", help="Show memory tracking example"
    )
    optional.add_argument(
        "-d", "--demo", action="store_true", help="Run full demo with all features"
    )

    args = parser.parse_args()

    # If no arguments, show demo
    if len(sys.argv) == 1:
        print("\nPretty Log Demo:\n")

        # Header example
        pretty_log("Processing Data", is_header=True)

        # Subheader example
        pretty_log("Loading Files", is_subheader=True)

        # Regular log with memory
        pretty_log("Reading configuration files", memory_gb=1.24)

        # Progress example
        for i in range(5):
            pretty_log(
                "Processing files",
                progress={
                    "current": i,
                    "total": 4,
                    "suffix": "files",
                    "bar_length": 15,
                },
            )
            time.sleep(0.5)

        # Memory and progress combined
        for i in range(5):
            pretty_log(
                "Training model",
                memory_gb=2.45,
                progress={
                    "current": i,
                    "total": 4,
                    "suffix": "epoch",
                    "bar_length": 15,
                },
            )
            time.sleep(0.5)

        print("\nUsage Examples:\n")
        print("1. Basic message:")
        print("   python pretty_log.py -m 'Processing data'\n")

        print("2. Show progress bar:")
        print("   python pretty_log.py -m 'Processing files' -p\n")

        print("3. Show memory usage:")
        print("   python pretty_log.py -m 'Training model' -mem\n")

        print("4. Full demo with all features:")
        print("   python pretty_log.py -d\n")

        return

    # Handle actual command execution
    if args.message:
        if args.demo:
            # Run the full demo with the user's message
            pretty_log(args.message, is_header=True)
            for i in range(5):
                pretty_log(
                    "Processing",
                    memory_gb=1.24,
                    progress={
                        "current": i,
                        "total": 4,
                        "suffix": "steps",
                        "bar_length": 15,
                    },
                )
                time.sleep(0.5)
        elif args.progress:
            # Show progress example
            for i in range(5):
                pretty_log(
                    args.message,
                    progress={
                        "current": i,
                        "total": 4,
                        "suffix": "steps",
                        "bar_length": 15,
                    },
                )
                time.sleep(0.5)
        elif args.memory:
            # Show memory example
            pretty_log(args.message, memory_gb=1.24)
        else:
            # Basic message
            pretty_log(args.message)


if __name__ == "__main__":
    main()
