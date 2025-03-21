#!/usr/bin/env python3

import os
import sys
import json
from tqdm import tqdm


def convert_dblp_json_to_jsonl(input_file, output_file=None):
    """
    Convert DBLP JSON file to JSON Lines format.

    This script is specifically designed for the DBLP dataset format:
    [
    {"id":1091,...},
    ,{"id":1388,...}
    ,{"id":1674,...}
    ]

    It handles the special case where lines start with a comma.

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output file (default: input_file with '.jsonl' extension)
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".jsonl"

    print(f"Converting DBLP JSON file: {input_file}")
    print(f"Output will be saved to: {output_file}")

    # Count lines for progress tracking
    print("Counting lines in file...")
    line_count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for _ in f:
            line_count += 1

    print(f"Found {line_count} lines in file")

    # Process the file line by line
    objects_count = 0

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        # Skip the opening '[' line
        first_line = infile.readline().strip()
        if first_line != "[":
            print(
                f"Warning: First line is not '[', but '{first_line}'. Continuing anyway."
            )

        # Set up progress bar
        with tqdm(total=line_count, unit="line", desc="Processing") as pbar:
            pbar.update(1)  # Update for the first line

            for line in infile:
                pbar.update(1)

                # Skip empty lines
                line = line.strip()
                if not line:
                    continue

                # Skip the closing ']'
                if line == "]":
                    continue

                # Remove trailing comma if present
                if line.endswith(","):
                    line = line[:-1]

                # Remove leading comma if present
                if line.startswith(","):
                    line = line[1:]

                # Skip if line is still empty after removing commas
                if not line:
                    continue

                # Parse and write the JSON object
                try:
                    # Parse to validate
                    parsed_obj = json.loads(line)
                    # Write to output file
                    outfile.write(json.dumps(parsed_obj) + "\n")
                    objects_count += 1

                    # Update progress description periodically
                    if objects_count % 10000 == 0:
                        pbar.set_description(f"Processed {objects_count:,} objects")
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {str(e)}")
                    print(f"Line text: {line[:100]}...")

    print(f"Conversion complete. {objects_count:,} objects written to {output_file}")
    return True


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(
            "Usage: python convert_dblp_json_to_jsonl.py <input_json_file> [output_file]"
        )
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_dblp_json_to_jsonl(input_file, output_file)


if __name__ == "__main__":
    main()
