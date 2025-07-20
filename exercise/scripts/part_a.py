from typing import Dict
from hopfield_model.network import HopfieldNetwork
from hopfield_model.typing import Vector, Ordering
from hopfield_model.utilities import letters_to_patterns
from exercise.scripts.initialization import general_init


def run_part_a(network: HopfieldNetwork, patterns: Dict[str, Vector], letter: str,
			   order: Ordering = Ordering.SEQUENTIAL) -> None:
	"""
	Run Part A: Train the network on a single letter pattern and check if it is a fixed point.

	Parameters:
		network (HopfieldNetwork): Instance of the Hopfield network.
		patterns (Dict[str, Vector]): Dictionary of letter patterns.
		letter (str): Single letter to train on, e.g., 'A'.
		order (Ordering): Order of neuron updates (sequential or random), default is sequential.
	"""
	assert isinstance(network, HopfieldNetwork), "Network must be an instance of HopfieldNetwork."
	assert isinstance(letter, str) and len(letter) == 1, "Letter must be a single character string."
	
	patterns_list = letters_to_patterns([letter], patterns)
	network.train(patterns_list, is_initial=True)
	
	# Check if this state is steady
	is_fixed_point = network.check_if_fixed_point(patterns_list[0], order)
	if is_fixed_point:
		print(f"The pattern of the letter '{letter}' is a fixed point.")
	else:
		print(f"The pattern of the letter '{letter}' is NOT a fixed point.")


if __name__ == "__main__":
	# Run only part Α
	
	# General initialization: Load patterns and create the Hopfield network
	patterns_dict, new_network, first_letter = general_init()
	
	# Part A: Train on first letter of my name and check if it is a fixed point
	print("Starting Part A: Training the network on a single letter (the first letter of my name):")
	run_part_a(new_network, patterns_dict, first_letter, Ordering.SEQUENTIAL)
	print("Part A completed successfully.\n")
