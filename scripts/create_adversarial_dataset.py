import os
import argparse
import logging
import random

from pprint import pprint
from collections import defaultdict


def load_tables(filename):
    """
    Load the mappings from the 64 first lines of the training file.

    Args:
        filename (str): training tsv filename

    Returns:
        tables (dict of dicts): maps table names to tables
    """

    # Read atomic mappings from the training file, the first 64 lines
    atomic_mappings = [line.split() for line in open(filename).readlines()[:64]]

    # Turn the mappings into a dictionary format, with a dict per table
    tables = defaultdict(dict)
    for mapping in atomic_mappings:
        table_number = mapping[1]
        key = mapping[0]
        value = mapping[4]
        tables[table_number][key] = value

    return tables


def load_heldout(filename):
    """
    Load the heldout dataset and split it by lines and tabs.

    Args:
        filename (str): heldout tsv filename

    Returns:
        list of lines, lines split by tabs
    """
    return [line.split("\t") for line in open(filename).readlines()]


def swap(attention, level):
    """
    Swap pairs of attentions, where the number of swappings is equal to level.

    Args:
        attention (list): attention indices
        level (int): number of desired swappings

    Returns:
        list with attention swapped
    """
    if level >= len(attention):
        logging.warning("Number of swappings >= the attention length.")

    for i in range(level):
        j, k = tuple(random.sample(range(0, len(attention)), 2))
        attention[j], attention[k] = attention[k], attention[j]
    return attention


def add_attacks(heldout, swap_input, level):
    """
    Change heldout data using adversarial attacks.

    Args:
        heldout (list): heldout lines split by tabs
        swap_input (bool): whether to swap the input index
        level (int): number of desired swappings

    Returns:
        heldout data adapted using attacks
    """
    for i, line in enumerate(heldout):
        # Attention is in the third position
        attention = line[-1].split()

        # If the input should also be swapped, include, else, exclude
        if not swap_input:
            swapped = swap(attention[1:-1], level)
            attention = attention[:1] + swapped + attention[-1:]
        else:
            swapped = swap(attention[:-1], level)
            attention = swapped + attention[-1:]

        # Adapt the heldout line with the new attention
        attention = " ".join(attention)
        line[-1] = attention
        heldout[i] = line
    return heldout


def update_output(heldout, tables):
    """
    Update the (intermediate) output steps based on the new attention.

    Args:
        heldout (list): heldout lines split by tabs
        tables (dict of dicts): lookup table mappings

    Returns:
        heldout with the (intermediate) outputs adapted
    """
    for i, line in enumerate(heldout):
        # Extract all info of the line
        input_sequence = line[0].split()
        output_sequence = line[1].split()
        attention = line[2].split()
        word = output_sequence[0]
        new_output = [word]

        # Generate new output based on the order in the attention
        for j in range(1, len(input_sequence) - 1):
            table = tables[input_sequence[int(attention[j])]]
            word = table[word]
            new_output.append(word)
        line[1] = " ".join(new_output)
        heldout[i] = line
    return heldout


parser = argparse.ArgumentParser()
parser.add_argument('--train', help='training data to extract atomic tables from.', required=True)
parser.add_argument('--heldout', help='heldout data to be adapted by attacks', required=True)
parser.add_argument('--output_dir', default="", help='path to data directory.')
parser.add_argument('--log-level', default='info', help='logging level.')
parser.add_argument('--swap_input', action='store_true', help='whether input is included in swapping')
parser.add_argument('--level', default=1, help='number of swappings', type=int)

opt = parser.parse_args()
log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=log_format, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

tables = load_tables(opt.train)
heldout = load_heldout(opt.heldout)
adversarial_heldout = add_attacks(heldout, opt.swap_input, opt.level)
heldout_with_attacks = "\n".join(["\t".join(line)
                                  for line in adversarial_heldout])
filename = "{}_attacks.tsv".format(opt.heldout.split("/")[0].split('.')[0])
with open(os.path.join(opt.output_dir, filename), 'w') as f:
    f.write(heldout_with_attacks)

# We cannot update the output if the input is swapped
if not opt.swap_input:
    adversarial_heldout = update_output(adversarial_heldout, tables)
    heldout_with_attacks_outputs = "\n".join(["\t".join(line)
                                              for line in adversarial_heldout])
    filename = "{}_attacks_outputs.tsv".format(opt.heldout.split("/")[-1].split('.')[0])
    print(filename)
    print(opt.output_dir)
    with open(os.path.join(opt.output_dir, filename), 'w') as f:
        f.write(heldout_with_attacks_outputs)
