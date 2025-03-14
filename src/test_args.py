#!/usr/bin/env python3
import argparse
import sys

# Copy of the modified addargs function to avoid importing from main
def addargs(parser):
    """Parse and Set Arguments."""

    # Define our custom help text
    help_text = """OpenNTF: Open Neural Team Formation

Required:
   -i INPUT, --input INPUT
\tLocation of dataset

   -d DOMAIN, --domain DOMAIN
\tDomain of the dataset. Options: dblp, gith, imdb, uspt


Optionals:
   -m MODEL, --model MODEL
\tModel to perform the task, or the type of the experiments to run, e.g., random, heuristic, expert, etc. If not provided, process will stop after data loading.

   -train TRAIN, --train TRAIN
\tWhether to train the model

   -filter FILTER, --filter FILTER
\tWhether to filter data: zero: no filtering, one: filter zero degree nodes, two: filter one degree nodes

   -future FUTURE, --future FUTURE
\tForecast future teams: zero: no need to forecast future teams, one: predict future teams

   -fair FAIR, --fair FAIR
\tApply fairness to model

   -o OUTPUT, --output OUTPUT
\tOutput file or folder

   -gpus GPUS, --gpus GPUS
\tCUDA Visible GPUs

   -t THREADS, --threads THREADS
\tNumber of threads to use for parallel processing (0 for auto, defaults to 75% of available CPU cores)

   -b BATCH_SIZE, --batch-size BATCH_SIZE
\tBatch size for processing large datasets (default: IMDB: 10000, DBLP: 10000, GITH: 1000, USPT: 5000)
"""

    # Override the help option to print our custom help text
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    
    # Required arguments group
    required = parser.add_argument_group('Required')
    required.add_argument(
        "-i", "--input",  # Updated to match help menu
        dest="data",      # Keep the original destination
        help=argparse.SUPPRESS,
        required=True,
        metavar="INPUT"
    )

    required.add_argument(
        "-d", "--domain",  # Updated to match help menu
        dest="domain",
        help=argparse.SUPPRESS,
        required=True,
        metavar="DOMAIN"
    )

    # Optional arguments group
    optionals = parser.add_argument_group('Optionals')
    optionals.add_argument(
        "-m", "--model",  # Updated to match help menu
        dest="model",
        help=argparse.SUPPRESS,
        required=False,
        default=None,
        metavar="MODEL"
    )

    optionals.add_argument(
        "-train", "--train",
        dest="train",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="TRAIN"
    )

    optionals.add_argument(
        "-filter", "--filter",
        dest="filter",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="FILTER"
    )

    optionals.add_argument(
        "-future", "--future",
        dest="future",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="FUTURE"
    )

    optionals.add_argument(
        "-fair", "--fair",
        dest="fair",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="FAIR"
    )

    optionals.add_argument(
        "-o", "--output",  # Updated to match help menu
        dest="output",
        help=argparse.SUPPRESS,
        default=None,
        type=str,
        metavar="OUTPUT"
    )

    optionals.add_argument(
        "-gpus", "--gpus",
        dest="gpus",
        help=argparse.SUPPRESS,
        default=None,
        metavar="GPUS"
    )
    
    optionals.add_argument(
        "-t", "--threads",
        dest="threads",
        help=argparse.SUPPRESS,
        default=0,
        type=int,
        metavar="THREADS"
    )
    
    optionals.add_argument(
        "-b", "--batch-size",
        dest="batch_size",
        help=argparse.SUPPRESS,
        default=0,  # 0 means use domain-specific defaults
        type=int,
        metavar="BATCH_SIZE"
    )
    
    # Override the print_help method to print our custom help text
    def custom_print_help(file=None):
        print(help_text, file=file)
    
    parser.print_help = custom_print_help
    
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser = addargs(parser)
    
    # Test with -h flag
    if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit(0)
        
    # Otherwise parse args
    args = parser.parse_args()
    print("Arguments:", args) 