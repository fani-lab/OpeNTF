'''

This is a helper file for test runs on different sets of parameters.
The objective is to change the values in param.py based on the command line
arguments passed to this module each time before running OpeNTF

The reason to create this file is to automate multiple OpeNTF runs
which require different sets of values in param.py. Running this file
with proper arguments automated by bash script can easily serve the purpose
"Why not just add the arguments in the main.py? - Because too many arguments which
are not changed that much without tests, will clutter the main.py"

1. Set the command line arguments through bash script
2. Add the command for running this file in bash script (This will change the values in param.py for the particular run)
3. Add the command for running OpeNTF with updated params
4. Repeatedly add 1-3 for new set of values

'''

import argparse
from param import settings
import pprint


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Modify settings in param.py")
    parser.add_argument("-baseline", help="The model to use") # mandatory argument
    parser.add_argument("--l", type=int, nargs='+', help="Nodes in the layer(s)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--b", type=int, help="Batch size")
    parser.add_argument("--e", type=int, help="Epoch")
    parser.add_argument("--nns", type=int, help="Number of negative samples")
    parser.add_argument("--ns", help="NS")
    parser.add_argument("--weight", type=float, help="Weight")
    parser.add_argument("--loss", help="Loss")
    parser.add_argument("--cmd", help="Pipeline")

    args = parser.parse_args()

    bsl = args.baseline
    if args.l is not None: settings['model']['baseline'][bsl]['l'] = args.l
    if args.lr is not None: settings['model']['baseline'][bsl]['lr'] = args.lr
    if args.b is not None: settings['model']['baseline'][bsl]['b'] = args.b
    if args.e is not None: settings['model']['baseline'][bsl]['e'] = args.e
    if args.nns is not None: settings['model']['baseline'][bsl]['nns'] = args.nns
    if args.ns is not None: settings['model']['baseline'][bsl]['ns'] = args.ns
    if args.weight is not None: settings['model']['baseline'][bsl]['weight'] = args.weight
    if args.loss is not None: settings['model']['baseline'][bsl]['loss'] = args.loss

    if args.cmd is not None: settings['model']['cmd'] = args.cmd

    param_file_path = 'param.py'

    # Writing the modified dictionary back to the file
    with open('param.py', 'w') as file:
        file.write('import random\n')
        file.write('import torch\n')
        file.write('import numpy as np\n')
        file.write('\n')
        file.write('random.seed(0)\n')
        file.write('torch.manual_seed(0)\n')
        file.write('torch.cuda.manual_seed_all(0)\n')
        file.write('\n')
        file.write('np.random.seed(0)\n')
        file.write('\n')
        file.write('settings = ')
        pprint.pprint(settings, stream=file)

if __name__ == "__main__":
    main()