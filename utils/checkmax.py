import sys
import re
import os

def scan_file(prefix, threshold, file_path):
    # Convert threshold to integer
    threshold = int(threshold)
    
    # Check if file has .log extension
    if not file_path.endswith('.log'):
        raise ValueError("File must have .log extension")
    
    found_matches = False
    # Read from file using absolute path
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            # Find all matches of pattern prefix + number
            # \d+ matches one or more digits
            matches = re.finditer(rf'{prefix}(\d+)', line)
            
            # Check each match
            for match in matches:
                number = int(match.group(1))  # Get the number part
                if number > threshold:
                    found_matches = True
                    # Print line number, the full match, and the line content
                    print(f"Line {line_num}: Found {match.group(0)} in: {line.strip()}")
    
    if not found_matches:
        print(f"No numbers found after '{prefix}' that exceed {threshold}")

def main():
    # Check if correct number of arguments provided
    if len(sys.argv) != 4:
        print("Usage: python3 checkmax.py prefix threshold path/to/file.log")
        print("Example: python3 checkmax.py m 2011 /home/user/logs/logfile.log")
        sys.exit(1)
    
    prefix = sys.argv[1]
    threshold = sys.argv[2]
    file_path = os.path.abspath(sys.argv[3])  # Convert to absolute path
    
    try:
        scan_file(prefix, threshold, file_path)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)

if __name__ == "__main__":
    main() 