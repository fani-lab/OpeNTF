import sys
import re

def scan_file(prefix, threshold, filename):
    # Convert threshold to integer
    threshold = int(threshold)
    
    found_matches = False
    # Read from file instead of stdin
    with open(filename, 'r') as file:
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
        print("Usage: python3 checkmax.py prefix threshold filename")
        print("Example: python3 checkmax.py m 2011 logfile.txt")
        sys.exit(1)
    
    prefix = sys.argv[1]
    threshold = sys.argv[2]
    filename = sys.argv[3]
    
    try:
        scan_file(prefix, threshold, filename)
    except ValueError:
        print("Error: Threshold must be a number")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)

if __name__ == "__main__":
    main() 