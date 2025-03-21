#!/usr/bin/env python3
import os
import sys
import json


def check_json_file(filepath, num_lines=10):
    """
    Check the format of a JSON file and display the first few lines.

    Args:
        filepath: Path to the JSON file
        num_lines: Number of lines to display (default: 10)
    """
    print(f"Checking JSON file: {filepath}")

    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return False

    filesize = os.path.getsize(filepath)
    print(f"File size: {filesize / (1024*1024):.2f} MB")

    try:
        # Read the first few lines
        lines = []
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < num_lines:
                    lines.append(line.strip())
                else:
                    break

        print(f"\nFirst {len(lines)} lines of the file:")
        for i, line in enumerate(lines):
            print(f"{i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")

        # Check for JSON array format
        with open(filepath, "r", encoding="utf-8") as f:
            first_char = f.read(1).strip()

        is_array = first_char == "["
        print(f"\nFile appears to be a JSON {'array' if is_array else 'lines'} format.")

        # Try to parse the first line as JSON
        if len(lines) > 0:
            try:
                # For JSON Lines format
                parsed = json.loads(lines[0])
                print(f"Successfully parsed first line as JSON.")
                print(f"Keys in first object: {list(parsed.keys())}")

                # Check for DBLP-specific keys
                expected_keys = ["id", "title", "year", "authors", "venue", "fos"]
                missing_keys = [key for key in expected_keys if key not in parsed]

                if missing_keys:
                    print(f"Warning: Missing expected DBLP keys: {missing_keys}")
                else:
                    print(f"All expected DBLP keys are present.")

                return True
            except json.JSONDecodeError as e:
                print(f"Error parsing first line as JSON: {str(e)}")

                # Check if it's a JSON array
                if is_array:
                    print("Attempting to parse as a JSON array...")
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            # Read a small chunk to check the format
                            chunk = f.read(1000)
                            # Add closing bracket if it's not in the chunk
                            if "]" not in chunk:
                                chunk += "]"
                            json.loads(chunk)
                            print("Successfully parsed initial chunk as JSON array.")
                            return True
                    except json.JSONDecodeError as e2:
                        print(f"Error parsing as JSON array: {str(e2)}")

                return False

        return False
    except Exception as e:
        print(f"Error checking file: {str(e)}")
        return False


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python check_json_format.py <json_file_path> [num_lines]")
        return

    filepath = sys.argv[1]
    num_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    check_json_file(filepath, num_lines)


if __name__ == "__main__":
    main()
