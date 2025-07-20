import matplotlib

# Ensure the backend is set to 'TkAgg' for interactive plotting
if matplotlib.get_backend() != 'TkAgg':
	matplotlib.use('TkAgg')

from typing import Dict, Tuple, Optional, List
import hopfield_model
from hopfield_model.network import HopfieldNetwork
from hopfield_model.typing import Vector, Ordering
from hopfield_model.utilities import (letters_to_patterns, pretty_print_matrix, perturbate_pattern,
									  check_overall_convergence)
from exercise.scripts.initialization import general_init


def run_part_b(network: HopfieldNetwork, patterns: Dict[str, Vector], letter: str,
			   noise_level: float = 0.2,
			   max_cycles: int = 100, order: Ordering = Ordering.SEQUENTIAL,
			   with_saving: bool = False,
			   with_interactive: bool = True,
			   shape: Tuple[int, int] = (10, 9),
			   file_base_name: Optional[str] = None, directory_path_lst: Optional[List[str]] = None,
			   extension: Optional[str] = None,
			   dpi: Optional[int] = None,
			   is_initial_training: bool = False) -> None:
	"""
	Run Part B: Perturb the initial state and observe the network's convergence.

	Parameters:
		network (HopfieldNetwork): Instance of the Hopfield network (already trained).
		patterns (Dict[str, Vector]): Dictionary of letter patterns.
		letter (str): The letter the network should be trained on, will initialize the network with perturbation of this letter's pattern and observe convergence to it.
		noise_level (float): Fraction of bits to flip in the initial state (0 to 1).
		max_cycles (int): Max number of cycles to run.
		order (Ordering): Order of neuron updates (sequential or random), default is sequential.
		with_saving (bool): If True, enables saving plots to disk.
		with_interactive (bool): If True, enables interactive plotting.
		shape (Optional[Tuple[int, int]]): Shape of the network state for visualization, default is (10, 9).
		file_base_name (Optional[str]): Base name for saving files, if applicable.
		directory_path_lst (Optional[List[str]]): Relative or absolute path to the directory where files will be saved (given as a list of strings), if applicable.
		extension (Optional[str]): File extension for saving plots, if applicable.
		dpi (Optional[int]): DPI for saving plots, if applicable.
		is_initial_training (bool): If True, the network will be trained on the letter pattern from scratch, otherwise it will continue without training abd just perturb the pattern.
	"""
	assert isinstance(network, HopfieldNetwork), "Network must be an instance of HopfieldNetwork."
	assert isinstance(letter, str) and len(
			letter) == 1, "letter must be a single character string."
	assert with_saving or with_interactive, "At least one of with_saving or with_interactive must be True."
	assert isinstance(noise_level,
					  float) and 0 <= noise_level <= 1, "Noise level must be between 0 and 1."
	
	patterns_list = letters_to_patterns([letter], patterns)
	if is_initial_training:
		print(f"Training the network on the pattern of the letter '{letter}'...")
		network.train(patterns_list, is_initial=True)
	
	target_state = patterns_list[0]
	print(f"Target state that will undergo perturbation:")
	pretty_print_matrix(target_state.reshape(shape))
	noisy_state = perturbate_pattern(target_state, noise_level)
	print(f"Noisy state after perturbation:")
	pretty_print_matrix(noisy_state.reshape(shape))
	
	network.set_state(noisy_state)
	print("Initial state set. Now running asynchronous updates...")
	
	converged, final_state = network.run_until_convergence(max_cycles=max_cycles, order=order,
														   with_visualization=True,
														   with_saving=with_saving,
														   with_interactive=with_interactive,
														   shape=shape,
														   file_base_name=file_base_name,
														   directory_path_lst=directory_path_lst,
														   extension=extension,
														   dpi=dpi)
	overall_convergence = check_overall_convergence(converged, final_state, target_state)
	if converged:
		print("Network converged to a fixed point.")
		if overall_convergence:
			print("The network converged to the intended target state.")
		else:
			print("The network converged, but not to the intended target state.")
			print(f"The final state was:")
			pretty_print_matrix(final_state.reshape(shape))
	else:
		print(f"Network did not converge within {max_cycles} cycles.")


if __name__ == "__main__":
	# Run only part B
	
	# General initialization: Load patterns and create the Hopfield network
	patterns_dict, new_network, first_letter = general_init()
	
	# Part B: Perturb the pattern and run convergence
	print("Starting Part B: Perturbing the target state and observing convergence:")
	run_part_b(new_network, patterns_dict, first_letter, noise_level=0.2, max_cycles=100,
			   order=Ordering.SEQUENTIAL,
			   with_saving=hopfield_model.SAVING_MODE,
			   with_interactive=True,
			   shape=(10, 9),
			   file_base_name=f"convergence_to_letter_{first_letter}",
			   directory_path_lst=["exercise", "plots", "network_convergence_plots",
								   f"letter_{first_letter}"],
			   extension="png",
			   dpi=300,
			   is_initial_training=True)
	print("Part B completed successfully.\n")
