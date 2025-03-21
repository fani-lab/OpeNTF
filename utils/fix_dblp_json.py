#!/usr/bin/env python3

import os
import sys
import json
from tqdm import tqdm
import re


def fix_dblp_json(input_file, output_file=None):
    """
    Convert DBLP JSON array format to JSON Lines format.

    This script is specifically designed for the DBLP dataset, which has a format like:
    [
    {"id":1091,...},
    ,{"id":1388,...}
    ,{"id":1674,...}
    ]

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output file (default: input_file with '.jsonl' extension)
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".jsonl"

    print(f"Converting DBLP JSON file: {input_file}")
    print(f"Output will be saved to: {output_file}")

    # Get file size for progress tracking
    file_size = os.path.getsize(input_file)

    # Compile regex pattern to match JSON objects
    # This pattern matches either:
    # 1. A complete JSON object starting with { and ending with }
    # 2. A complete JSON object starting with ,{ (with comma) and ending with }
    object_pattern = re.compile(
        r"(?:^|\A|\s*,\s*)(\{.*?\})(?=\s*,|\s*\]|\s*$|\Z)", re.DOTALL
    )

    objects_count = 0
    bytes_processed = 0

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        # Skip the opening [
        first_line = infile.readline()
        bytes_processed += len(first_line)
        if "[" not in first_line:
            print("Error: First line doesn't contain '['")
            return False

        # Set up progress bar
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Processing"
        ) as pbar:
            pbar.update(bytes_processed)

            # Process the file in chunks of 1MB
            buffer = ""
            chunk_size = 1024 * 1024  # 1MB

            while True:
                chunk = infile.read(chunk_size)
                if not chunk:
                    break

                bytes_processed += len(chunk)
                pbar.update(len(chunk))

                # Add chunk to buffer
                buffer += chunk

                # Process any complete objects in the buffer
                while True:
                    # Find a complete JSON object in the buffer
                    match = object_pattern.search(buffer)
                    if not match:
                        break

                    # Extract the object and remove any leading comma
                    json_obj = match.group(1)
                    if json_obj.startswith(","):
                        json_obj = json_obj[1:]

                    # Validate and write the object
                    try:
                        # Parse to validate
                        parsed_obj = json.loads(json_obj)
                        # Write to output file
                        outfile.write(json.dumps(parsed_obj) + "\n")
                        objects_count += 1

                        # Update progress description
                        if objects_count % 10000 == 0:
                            pbar.set_description(f"Processed {objects_count:,} objects")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing object: {str(e)}")
                        print(f"Object text: {json_obj[:100]}...")

                    # Remove processed object from buffer
                    buffer = buffer[match.end() :]

            # Process any remaining data in buffer
            if buffer.strip():
                # Handle remaining partial object
                if "{" in buffer and "}" in buffer:
                    try:
                        remaining = buffer[buffer.find("{") : buffer.rfind("}") + 1]
                        parsed_obj = json.loads(remaining)
                        outfile.write(json.dumps(parsed_obj) + "\n")
                        objects_count += 1
                    except json.JSONDecodeError:
                        print("Error parsing final object, skipping")

    print(f"Conversion complete. {objects_count:,} objects written to {output_file}")
    return True


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_dblp_json.py <input_json_file> [output_file]")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    fix_dblp_json(input_file, output_file)


if __name__ == "__main__":
    main()
