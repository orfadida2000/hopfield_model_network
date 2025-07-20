import argparse
from typing import Tuple, Dict

import hopfield_model
from hopfield_model.utilities import read_alphabet_xlsx
from hopfield_model.network import HopfieldNetwork
from hopfield_model.typing import Vector

# My name
MY_NAME = "Or"

# Path to the Excel file containing letter patterns
FILEPATH = "exercise/alphabet.xlsx"


def parse_and_set_run_modes():
	"""
	Parse command line arguments to set the run modes for the Hopfield network assignments.
	"""
	
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Run Hopfield network assignments.")
	parser.add_argument('--debug', action='store_true',
						help="Enable debug mode for full network state plotting.")
	parser.add_argument('--save', action='store_true',
						help="Enable saving plots to disk.")
	args = parser.parse_args()
	
	# Set the global debug mode dynamically
	hopfield_model.DEBUG_MODE = args.debug
	if hopfield_model.DEBUG_MODE:
		print("Debug mode is enabled. Detailed output will be printed.")
	
	# Set the global saving mode dynamically
	hopfield_model.SAVING_MODE = args.save
	if hopfield_model.SAVING_MODE:
		print("Saving mode is enabled. Plots will be saved to disk.")


def general_init() -> Tuple[Dict[str, Vector], HopfieldNetwork, str]:
	"""
	General initialization function to set up the Hopfield network and load patterns from an Excel file.
	
	Returns:
		tuple: A tuple containing:
			- patterns (Dict[str, Vector]): Dictionary of letter patterns loaded from the Excel file.
			- network (HopfieldNetwork): Instance of the Hopfield network initialized with the size of the patterns.
			- letter (str): The first letter of my name, used as uppercase.
	"""
	# Parse command line arguments to set run modes
	parse_and_set_run_modes()
	
	# Read patterns from Excel file
	try:
		patterns = read_alphabet_xlsx(FILEPATH)
	except Exception as e:
		raise RuntimeError(f"Failed to read patterns from the Excel file '{FILEPATH}': {e}")
	print("Letters loaded successfully from Excel file.")
	print()
	
	# Create a Hopfield network with the size of the patterns
	n = len(next(iter(patterns.values())))
	network = HopfieldNetwork(n=n)
	
	# The first letter of my name to use as uppercase.
	letter = MY_NAME[0].upper()  # Use the first letter of my name, e.g., 'O'
	
	return patterns, network, letter
