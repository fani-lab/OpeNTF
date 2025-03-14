#!/usr/bin/env python3
"""
GPU-accelerated preprocessing for dataset sparse vector generation
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import needed modules
from cmn_v3.team import Team
from cmn_v3.dblp import Publication
from utils.tprint import tprint

# Dictionary mapping domain names to their respective classes
DOMAIN_CLASSES = {
    "dblp": Publication,
    # Add other domains here as they are implemented
    # "github": Repo,
    # "imdb": Movie,
    # "patent": Patent,
}

def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='GPU-accelerated preprocessing for dataset sparse vector generation')
    
    # Required arguments
    required = parser.add_argument_group('Required')
    required.add_argument(
        "-i", "--input",
        dest="input",
        help="Input file path",
        required=True
    )
    
    required.add_argument(
        "-o", "--output",
        dest="output",
        help="Output directory path",
        required=True
    )
    
    required.add_argument(
        "-d", "--domain",
        dest="domain",
        help="Domain name (dblp, github, imdb, patent, etc.)",
        required=True
    )
    
    # Optional arguments
    optionals = parser.add_argument_group('Optionals')
    optionals.add_argument(
        "-y", "--year",
        dest="year",
        help="Filter by year",
        type=int
    )
    
    optionals.add_argument(
        "-b", "--batch-size",
        dest="batch_size",
        help="Batch size for processing",
        default=5_000_000,
        type=int
    )
    
    optionals.add_argument(
        "-t", "--threads",
        dest="threads",
        help="Number of threads for CPU fallback",
        default=128,
        type=int
    )
    
    optionals.add_argument(
        "-g", "--gpus",
        dest="gpus",
        help="CUDA Visible GPUs (comma-separated list, e.g., '0' or '0,1')",
        default=None
    )
    
    args = parser.parse_args()
    
    # Get the domain class
    domain = args.domain.lower()
    if domain not in DOMAIN_CLASSES:
        print(f"Error: Domain '{domain}' not supported. Available domains: {', '.join(DOMAIN_CLASSES.keys())}")
        sys.exit(1)
        
    domain_class = DOMAIN_CLASSES[domain]
    
    # Start timing
    start_time = time.time()
    
    tprint(f"Starting GPU-accelerated preprocessing for {domain} domain")
    
    # Run preprocessing with the appropriate domain class
    domain_class.generate_sparse_vectors_v3(
        datapath=args.input,
        output_dir=args.output,
        gpus=args.gpus
    )
    
    # Calculate and print processing time
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    tprint(f"Preprocessing complete for {domain} domain")
    tprint(f"Total processing time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (HH:MM:SS)")

if __name__ == '__main__':
    main() 