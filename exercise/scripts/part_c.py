import matplotlib

# Ensure the backend is set to 'TkAgg' for interactive plotting
if matplotlib.get_backend() != 'TkAgg':
	matplotlib.use('TkAgg')

from typing import List, Tuple, Dict, Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

import hopfield_model
from hopfield_model.network import HopfieldNetwork
from hopfield_model.typing import Vector, Ordering
from hopfield_model.utilities import perturbate_pattern, check_overall_convergence, letters_to_patterns
from hopfield_model.visualization.plotting import create_probabilities_trajectory_figure, save_figure
from exercise.scripts.initialization import general_init

# Noise levels to test in Part C
NOISE_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.375, 0.4, 0.4125,
			  0.425, 0.4375, 0.45, 0.4625, 0.475, 0.48, 0.485, 0.49, 0.495, 0.5]


def calc_prob_of_conv_vs_noise(network: HopfieldNetwork, target_state: Vector, noise_level_list: List[float],
							   max_cycles: int, order: Ordering, trials_per_noise: int) -> Tuple[
	NDArray[np.float64], NDArray[np.float64]]:
	"""
	Calculate the probability of convergence for different noise levels.

	Parameters:
		network (HopfieldNetwork): Instance of the Hopfield network (already trained).
		target_state (Vector): The target state to check convergence against, e.g., the pattern of a letter.
		noise_level_list (List[float]): List of noise levels to test as fractions of bits to flip (0 to 1).
		max_cycles (int): Max number of cycles to run for each trial.
		order (Ordering): Order of neuron updates (sequential or random).
		trials_per_noise (int): Number of trials for each noise level, to estimate the probability of convergence.

	Returns:
		Tuple[NDArray[np.float64], NDArray[np.float64]]: Two 1D arrays, one for noise level percentages and one for corresponding probabilities of convergence.
	"""
	assert isinstance(network, HopfieldNetwork), "Network must be an instance of HopfieldNetwork."
	network.assert_network_valid_pattern(target_state)
	assert isinstance(noise_level_list, List) and len(
			noise_level_list) > 0, "Noise levels must be a non-empty list."
	assert isinstance(trials_per_noise,
					  int) and trials_per_noise > 0, "trials_per_noise must be a positive integer."
	
	probabilities = []
	noise_level_percentages = []
	for noise_level in noise_level_list:
		assert isinstance(noise_level,
						  float) and 0 <= noise_level <= 1, "Noise level must be between 0 and 1."
		print(f"Testing noise level: {noise_level:.4%}")
		probability = calc_prob_of_conv_for_noise_level(network, target_state, noise_level, max_cycles, order,
														trials_per_noise)
		probabilities.append(probability)
		noise_level_percentages.append(noise_level * 100)
		print(f"Probability of convergence at noise level {noise_level:.4%}: {probability:.4f}")
	
	return np.asarray(noise_level_percentages, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)


def calc_prob_of_conv_for_noise_level(network: HopfieldNetwork, target_state: Vector, noise_level: float,
									  max_cycles: int, order: Ordering, trials_per_noise: int) -> float:
	"""
	Calculate the probability of convergence for a specific noise level.

	Parameters:
		network (HopfieldNetwork): Instance of the Hopfield network (already trained).
		target_state (Vector): The target state to check convergence against, e.g., the pattern of a letter.
		noise_level (float): Fraction of bits to flip in the target state (0 to 1).
		max_cycles (int): Max number of cycles to run for each trial.
		order (Ordering): Order of neuron updates (sequential or random).
		trials_per_noise (int): Number of trials for this noise level to estimate the probability of convergence.

	Returns:
		float: Probability of convergence for the given noise level.
	"""
	success_count = 0
	for _ in range(trials_per_noise):
		success_count += int(check_convergence_trial(network, target_state, noise_level, max_cycles, order))
	probability = success_count / trials_per_noise
	return probability


def check_convergence_trial(network: HopfieldNetwork, target_state: Vector, noise_level: float,
							max_cycles: int, order: Ordering) -> bool:
	"""
	Check if the network converges to the target state after perturbation.

	Parameters:
		network (HopfieldNetwork): Instance of the Hopfield network (already trained).
		target_state (Vector): The target state to check convergence against, e.g., the pattern of a letter.
		noise_level (float): Fraction of bits to flip in the target state (0 to 1).
		max_cycles (int): Max number of cycles to run for convergence.
		order (Ordering): Order of neuron updates (sequential or random).

	Returns:
		bool: True if the network converged to the target state, False otherwise.
	"""
	noisy_state = perturbate_pattern(target_state, noise_level)
	network.set_state(noisy_state)
	converged, final_state = network.run_until_convergence(max_cycles=max_cycles, order=order,
														   with_visualization=False)
	return check_overall_convergence(converged, final_state, target_state)


def run_part_c(network: HopfieldNetwork, patterns: Dict[str, Vector], trained_letter: str,
			   noise_level_list: List[float],
			   max_cycles: int, order: Ordering, trials_per_noise: int, with_saving: bool = False,
			   with_interactive: bool = True, filename: Optional[str] = None,
			   directory_path_lst: Optional[List[str]] = None,
			   extension: Optional[str] = None, dpi: Optional[int] = None,
			   is_initial_training: bool = False) -> None:
	"""
	Run Part C: Calculate the probability of convergence for different noise levels and plot the results.
	
	Parameters:
		network (HopfieldNetwork): Instance of the Hopfield network (already trained).
		patterns (Dict[str, Vector]): Dictionary of letter patterns.
		trained_letter (str): The letter the network is already trained on, the network should be trained on the next letter in the alphabet.
		noise_level_list (List[float]): List of noise levels to test as fractions of bits to flip (0 to 1).
		max_cycles (int): Max number of cycles to run for each trial.
		order (Ordering): Order of neuron updates (sequential or random).
		trials_per_noise (int): Number of trials for each noise level, to estimate the probability of convergence.
		with_saving (bool): If True, enables saving plots to disk.
		with_interactive (bool): If True, enables interactive plotting.
		filename (Optional[str]): Base name for saving files, if applicable.
		directory_path_lst (Optional[List[str]]): Relative or absolute path to the directory where files will be saved (given as a list of strings), if applicable.
		extension (Optional[str]): File extension for saving plots, if applicable.
		dpi (Optional[int]): DPI for saving plots, if applicable.
		is_initial_training (bool): If True, the network will be trained on both letters from scratch, otherwise it will continue training only on the next letter.
	"""
	assert isinstance(network, HopfieldNetwork), "Network must be an instance of HopfieldNetwork."
	assert isinstance(trained_letter, str) and len(
			trained_letter) == 1, "Trained letter must be a single character string."
	assert with_saving or with_interactive, "At least one of with_saving or with_interactive must be True."
	
	next_letter = chr(
			ord("A") + ((ord(trained_letter) - ord("A") + 1) % 26))  # Get the next letter in the alphabet
	patterns_list = letters_to_patterns(letters_list=[trained_letter, next_letter], patterns=patterns)
	target_state, train_state = patterns_list[0], patterns_list[1]
	if is_initial_training:
		network.train(patterns_list, is_initial=True)
	else:
		network.train([train_state], is_initial=False)
	noise_level_percentages, probabilities = calc_prob_of_conv_vs_noise(network, target_state,
																		noise_level_list, max_cycles, order,
																		trials_per_noise)
	sort_indices = np.argsort(noise_level_percentages)
	fig = create_probabilities_trajectory_figure(noise_level_percentages[sort_indices],
												 probabilities[sort_indices], trained_letter)
	if with_saving:
		save_figure(fig, filename, directory_path_lst, extension, dpi)
	if with_interactive:
		plt.show()
	plt.close(fig)  # Close the figure to free up resources


if __name__ == "__main__":
	# Run only part C
	
	# General initialization: Load patterns and create the Hopfield network
	patterns_dict, new_network, first_letter = general_init()
	
	# Part C: Calculate probability of convergence for different noise levels
	noise_levels_list = NOISE_LIST  # Noise levels to test in Part C
	print("Starting Part C: Calculating probability of convergence for different noise levels:")
	run_part_c(new_network, patterns_dict, first_letter,
			   noise_levels_list,
			   max_cycles=100, order=Ordering.SEQUENTIAL, trials_per_noise=2500,
			   with_saving=hopfield_model.SAVING_MODE,
			   with_interactive=True,
			   filename=f"probability_of_convergence_letter_{first_letter}",
			   directory_path_lst=["exercise", "plots", "probability_vs_noise_plots",
								   f"letter_{first_letter}"],
			   extension="png",
			   dpi=300,
			   is_initial_training=True)
	print("Part C completed successfully.\n")
