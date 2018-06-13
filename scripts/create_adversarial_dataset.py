import os
import argparse
import logging
import random

from pprint import pprint
from collections import defaultdict
import Levenshtein


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

    while True:

        old_attention = attention
        old_attention_string = ''.join(old_attention)

        for i in range(level):
            j, k = tuple(random.sample(range(0, len(old_attention)), 2))
            old_attention[j], old_attention[k] = old_attention[k], old_attention[j]

        new_attention_string = ''.join(old_attention)

        if Levenshtein.distance(old_attention_string, new_attention_string) == (level + 1):
            attention = old_attention
            break

    return attention


def add_attacks(heldout, swap_input, level, ignore_output_eos):
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
            tables_attention = attention[1:-1]
            swapped = swap(tables_attention, level)
            swapped = attention[:1] + swapped
        else:
            tables_attention = attention[:-1]
            swapped = swap(tables_attention, level)

        if not ignore_output_eos:
            attention = swapped + attention[-1:]
        else:
            attention = swapped

        # Adapt the heldout line with the new attention
        attention = " ".join(attention)
        line[-1] = attention
        heldout[i] = line
    return heldout


def remove_doubles(heldout):
    new_heldout = []
    for i, line in enumerate(heldout):
        input_sequence = line[0].split()[1:-1]
        if len(set(input_sequence)) != 1:
            new_heldout.append(line)
    return new_heldout


def update_output(heldout, tables, ignore_output_eos):
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
parser.add_argument('--ignore_output_eos', action='store_true', help='whether to ignore EOS token')
parser.add_argument('--level', default=1, help='number of swappings', type=int)

opt = parser.parse_args()
log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=log_format, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

tables = load_tables(opt.train)
heldout = remove_doubles(load_heldout(opt.heldout))

# If the ignore EOS flag is set, also save the regular dataset without EOS
if opt.ignore_output_eos:
    heldout_without_eos = []
    for line in heldout:
        no_eos = " ".join(line[-1].split()[:-1])
        heldout_without_eos.append([line[0], line[1], no_eos])
    heldout_without_eos = "\n".join(["\t".join(line)
                                      for line in heldout_without_eos])
    filename = "{}_no_eos.tsv".format(opt.heldout.split("/")[-1].split('.')[0])
    with open(os.path.join(opt.output_dir, filename), 'w') as f:
        f.write(heldout_without_eos)

# Add the attacks
adversarial_heldout = add_attacks(heldout, opt.swap_input, opt.level, opt.ignore_output_eos)
heldout_with_attacks = "\n".join(["\t".join(line)
                                  for line in adversarial_heldout])
filename = "{}_attacks.tsv".format(opt.heldout.split("/")[-1].split('.')[0])
with open(os.path.join(opt.output_dir, filename), 'w') as f:
    f.write(heldout_with_attacks)

# We cannot update the output if the input is swapped
if not opt.swap_input:
    adversarial_heldout = update_output(adversarial_heldout, tables, opt.ignore_output_eos)
    heldout_with_attacks_outputs = "\n".join(["\t".join(line)
                                              for line in adversarial_heldout])
    filename = "{}_attacks_outputs.tsv".format(opt.heldout.split("/")[-1].split('.')[0])
    with open(os.path.join(opt.output_dir, filename), 'w') as f:
        f.write(heldout_with_attacks_outputs)
