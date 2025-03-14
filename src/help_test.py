import argparse
import textwrap

class CustomHelpFormatter(argparse.HelpFormatter):
    """
    Custom help formatter to match the required format:
    - 3 spaces for argument names
    - 1 tab for descriptions
    """
    def __init__(self, prog):
        super().__init__(prog, max_help_position=24, width=80)
    
    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            # Use 3 spaces before argument names
            parts.append('   ')
            parts.append(', '.join(action.option_strings))
            if action.nargs != 0:
                default = self._get_default_metavar_for_optional(action)
                metavar, = self._metavar_formatter(action, default)(1)
                parts.append(' ' + metavar)
            return ''.join(parts)
    
    def _get_help_string(self, action):
        help_text = action.help
        if help_text is None:
            return None
        # Add a tab character before the help text
        if not help_text.startswith('\t'):
            help_text = '\t' + help_text
        return help_text
    
    def _format_action(self, action):
        parts = super()._format_action(action)
        # Add a blank line after each argument (except the last one in a group)
        parts += '\n'
        return parts

def addargs(parser):
    """Parse and Set Arguments."""
    # Use our custom formatter that handles the specific formatting requirements
    parser.formatter_class = CustomHelpFormatter
    parser.description = "OpenNTF: Open Neural Team Formation"

    required = parser.add_argument_group('Required')
    required.add_argument(
        "-data", "--data",
        dest="data",
        help="Location of dataset",
        required=True,
        metavar="DATA"
    )

    required.add_argument(
        "-domain", "--domain",
        dest="domain",
        help="Domain of the dataset. Options: dblp, gith, imdb, uspt",
        required=True,
        metavar="DOMAIN"
    )

    optionals = parser.add_argument_group('Optionals')
    optionals.add_argument(
        "-model", "--model",
        dest="model",
        help="Model to perform the task, or the type of the experiments to run, e.g., random, heuristic, expert, etc. If not provided, process will stop after data loading.",
        required=False,
        default=None,
        metavar="MODEL"
    )

    optionals.add_argument(
        "-train", "--train",
        dest="train",
        help="Whether to train the model",
        default=0,
        type=int,
        metavar="TRAIN"
    )

    optionals.add_argument(
        "-filter", "--filter",
        dest="filter",
        help="Whether to filter data: zero: no filtering, one: filter zero degree nodes, two: filter one degree nodes",
        default=0,
        type=int,
        metavar="FILTER"
    )

    optionals.add_argument(
        "-future", "--future",
        dest="future",
        help="Forecast future teams: zero: no need to forecast future teams, one: predict future teams",
        default=0,
        type=int,
        metavar="FUTURE"
    )

    optionals.add_argument(
        "-fair", "--fair",
        dest="fair",
        help="Apply fairness to model",
        default=0,
        type=int,
        metavar="FAIR"
    )

    optionals.add_argument(
        "-output", "--output",
        dest="output",
        help="Output file or folder",
        default=None,
        type=str,
        metavar="OUTPUT"
    )

    optionals.add_argument(
        "-gpus", "--gpus",
        dest="gpus",
        help="CUDA Visible GPUs",
        default=None,
        metavar="GPUS"
    )
    
    optionals.add_argument(
        "-t", "--threads",
        dest="threads",
        help="Number of threads to use for parallel processing (0 for auto, defaults to 75% of available CPU cores)",
        default=0,
        type=int,
        metavar="THREADS"
    )
    
    optionals.add_argument(
        "-b", "--batch-size",
        dest="batch_size",
        help="Batch size for processing large datasets (default: IMDB: 10000, DBLP: 10000, GITH: 1000, USPT: 5000)",
        default=0,  # 0 means use domain-specific defaults
        type=int,
        metavar="BATCH_SIZE"
    )
    
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser = addargs(parser)
    parser.print_help() 