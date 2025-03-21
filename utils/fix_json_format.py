#!/usr/bin/env python3
import os
import sys
import json
import traceback
from tqdm import tqdm


def fix_json_array(input_file, output_file=None, chunk_size=10000):
    """
    Fix a malformed JSON array file by converting it to JSON Lines format.

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output file (default: input_file + '.fixed')
        chunk_size: Size of chunks to process at once (default: 10000)
    """
    if output_file is None:
        output_file = input_file + ".fixed"

    print(f"Fixing JSON array file: {input_file}")
    print(f"Output will be saved to: {output_file}")

    # Get file size for progress bar
    file_size = os.path.getsize(input_file)

    # Process the file in chunks
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        # Check first character to see if it's an array
        first_char = infile.read(1)
        if first_char != "[":
            print("Error: File doesn't start with '['")
            return False

        # Reset file pointer
        infile.seek(0)

        # Initialize status variables
        in_array = False
        in_object = False
        in_string = False
        escape_next = False
        object_text = ""
        brace_count = 0
        line_count = 0
        objects_written = 0

        # Setup progress bar
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Processing"
        ) as pbar:
            while True:
                chunk = infile.read(chunk_size)
                if not chunk:
                    break

                pbar.update(len(chunk))

                for char in chunk:
                    # Process character
                    if escape_next:
                        object_text += char
                        escape_next = False
                        continue

                    if char == "\\":
                        object_text += char
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string

                    if not in_string:
                        if char == "[":
                            if not in_array:
                                in_array = True
                                continue
                        elif char == "]":
                            if in_array and not in_object:
                                in_array = False
                                continue
                        elif char == "{":
                            if not in_object:
                                in_object = True
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                in_object = False
                                object_text += char
                                # Complete object found, write it
                                try:
                                    # Parse to validate
                                    json_obj = json.loads(object_text)
                                    # Write as a line
                                    outfile.write(json.dumps(json_obj) + "\n")
                                    objects_written += 1

                                    # Show progress
                                    if objects_written % 100000 == 0:
                                        pbar.set_description(
                                            f"Processed {objects_written} objects"
                                        )
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing object: {str(e)}")
                                    print(f"Object text: {object_text[:100]}...")

                                object_text = ""
                                continue
                        elif char == "," and not in_object:
                            # Comma between objects, skip
                            continue

                    if in_object:
                        object_text += char

        print(
            f"Processing complete. {objects_written} objects written to {output_file}"
        )
        return True


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_json_format.py <input_json_file> [output_file]")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        fix_json_array(input_file, output_file)
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
