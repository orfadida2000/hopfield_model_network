import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Dict, Tuple, Union, List
import os
from hopfield_model.typing import Vector
import re
from matplotlib import pyplot as plt


# ───────────────────────────────────────────────────────────────────────
# For the Vector, Matrix type aliases, look at hopfield_model/typing.py
# ───────────────────────────────────────────────────────────────────────


def pretty_print_matrix(matrix: Union[NDArray[np.integer], NDArray[np.floating]]) -> None:
	"""
	Pretty-prints a 2D numpy array with aligned columns and dynamic box borders.

	Parameters:
		matrix (Union[NDArray[np.integer], NDArray[np.floating]]): 2D numpy array to print.
	"""
	assert isinstance(matrix,
					  np.ndarray) and matrix.ndim == 2 and matrix.size > 0, "Input must be a non-empty 2D numpy array (matrix)."
	
	'''
	format_spec ::= [options][width][grouping]["." precision][type]
	options     ::= [[fill]align][sign]["z"]["#"]["0"]
	fill        ::= <any character>
	align       ::= "<" | ">" | "=" | "^"
	sign        ::= "+" | "-" | " "
	width       ::= digit+
	grouping    ::= "," | "_"
	precision   ::= digit+
	type        ::= "b" | "c" | "d" | "e" | "E" | "f" | "F" | "g"
					| "G" | "n" | "o" | "s" | "x" | "X" | "%"
	'''
	
	basic_float_format = lambda num: f"{num:-z,.3f}"
	basic_decimal_format = lambda num: f"{num:-,d}"
	
	float_format = lambda num, width: f"{num:>-z{width},.3f}"
	decimal_format = lambda num, width: f"{num:>-{width},d}"
	
	rows, cols = matrix.shape
	
	# Detect float or int, format accordingly
	if np.issubdtype(matrix.dtype, np.floating):
		basic_formatted = [[basic_float_format(matrix[i, j]) for j in range(cols)] for i in range(rows)]
	elif np.issubdtype(matrix.dtype, np.integer):
		basic_formatted = [[basic_decimal_format(matrix[i, j]) for j in range(cols)] for i in range(rows)]
	else:
		raise ValueError("Matrix must be of integer or floating point type.")
	
	basic_formatted_length = np.asarray(
			[[len(basic_formatted[i][j]) for j in range(cols)] for i in range(rows)])
	max_len_per_col = np.max(basic_formatted_length, axis=0)
	
	if np.issubdtype(matrix.dtype, np.floating):
		formatted = [[float_format(matrix[i, j], max_len_per_col[j]) for j in range(cols)] for i in
					 range(rows)]
	else:
		formatted = [[decimal_format(matrix[i, j], max_len_per_col[j]) for j in range(cols)] for i in
					 range(rows)]
	
	# Prepare content lines
	content_lines = ["│ " + " ".join(row) + " │" for row in formatted]
	
	# Compute content width (all lines should have the same width)
	content_width = len(content_lines[0]) - 2  # Subtract 2 for the (""┌"", "┐", "└", "┘) characters
	
	# Prepare top and bottom borders
	top_border = f"┌" + ("─" * content_width) + "┐"
	bottom_border = "└" + ("─" * content_width) + "┘"
	
	# Final print
	print(top_border)
	for line in content_lines:
		print(line)
	print(bottom_border)


def read_alphabet_xlsx(filepath: str) -> Dict[str, Vector]:
	"""
	Reads Excel file with binary column vectors (0/1), converts to {-1, +1}.

	Parameters:
		filepath (str): Path to .xlsx file.

	Returns:
		Dict[str, Vector]: Dictionary of vectors keyed as 'A', 'B', etc.
	"""
	df = pd.read_excel(filepath, header=None)
	num_cols = df.shape[1]
	num_rows = df.shape[0]
	
	assert num_cols > 0, "Excel file has no columns."
	assert num_rows > 0, "Excel file has no rows."
	
	patterns: Dict[str, Vector] = {}
	
	for i, col in enumerate(df.columns):
		vec = df[col].to_numpy()
		
		assert vec.shape[0] == num_rows, f"Column {i} has inconsistent length."
		assert np.all(np.isin(vec, [0, 1])), "Column values must be binary (0 or 1)."
		
		vec = np.where(vec == 0, -1, 1)
		key = chr(ord('A') + i)
		patterns[key] = vec
	
	return patterns


def perturbate_pattern(pattern: Vector, noise_level: float) -> Vector:
	"""
	Applies random noise to a pattern, flipping bits with a given probability.

	Parameters:
		pattern (Vector): Original pattern.
		noise_level (float): Fraction of bits to flip (0 to 1).

	Returns:
		Vector: Noisy pattern.
	"""
	assert_valid_pattern(pattern, len(pattern))
	assert isinstance(noise_level, float) and 0 <= noise_level <= 1, "Noise level must be between 0 and 1."
	
	rng = np.random.default_rng()
	noisy_pattern = pattern.copy()
	noise_mask = np.ones_like(noisy_pattern)
	num_flips = int(len(pattern) * noise_level)
	indices_to_flip = rng.choice(len(pattern), size=num_flips, replace=False)
	noise_mask[indices_to_flip] = -1  # Flip the sign of selected indices
	noisy_pattern *= noise_mask
	
	return noisy_pattern


def get_unique_plot_filename(base_name: str, extension: str = "png") -> str:
	"""
	Generates a unique filename by appending a number if the base name already exists.
	
	Parameters:
		base_name (str): Base name for the file (relative or absolute path without extension).
		extension (str): File extension (default is "png").
		
	Returns:
		str: Unique filename with the specified base name and extension.
	"""
	
	filename = f"{base_name}.{extension}"
	if not os.path.exists(filename):
		return filename
	i = 1
	while True:
		filename = f"{base_name}({i}).{extension}"
		if not os.path.exists(filename):
			return filename
		i += 1


def is_valid_pattern(pattern: Vector, n: int) -> Tuple[bool, str]:
	"""
	Checks that a pattern is a valid 1D vector of length n with values in {-1, +1} and returns True if valid, False otherwise.

	Parameters:
		pattern (Vector): 1D vector to validate.
		n (int): Expected length of the pattern.
	
	Returns:
		Tuple[bool, str]: Tuple containing a boolean indicating validity and an error message if invalid.
	"""
	if not isinstance(pattern, np.ndarray):
		return False, "Pattern must be a numpy array."
	if pattern.ndim != 1:
		return False, "Pattern must be 1D."
	if len(pattern) != n:
		return False, f"Pattern length {len(pattern)} must match n={n}."
	if not np.all(np.isin(pattern, [-1, 1])):
		return False, "Pattern must contain only -1 and +1 values."
	return True, ""


def assert_valid_pattern(pattern: Vector, n: int) -> None:
	"""
	Asserts that a pattern is a valid 1D vector of length n with values in {-1, +1}.

	Parameters:
		pattern (Vector): 1D vector to validate.
		n (int): Expected length of the pattern.
	"""
	valid, error_message = is_valid_pattern(pattern, n)
	if not valid:
		raise ValueError(error_message)


def is_valid_fs_name(name: str) -> bool:
	"""
	Checks if name is a valid filesystem name (not empty, alphanumeric, underscores, hyphens, and spaces).
	
	Parameters:
		name (str): Name to validate.
	
	Returns:
		bool: True if the name is valid, False otherwise.
	"""
	if not isinstance(name, str):
		return False
	BASIC_SAFE_PATTERN = re.compile(r'^[A-Za-z0-9_\- ]+$')
	if not bool(BASIC_SAFE_PATTERN.match(name)):
		return False
	return name == name.strip()


def is_valid_plt_extension(extension: str) -> bool:
	"""
	Checks if the given file extension is supported by Matplotlib for saving figures.
	
	Parameters:
		extension (str): File extension to check (e.g., 'png', 'pdf').
		
	Returns:
		bool: True if the extension is valid, False otherwise.
	"""
	valid_plt_extensions = plt.gcf().canvas.get_supported_filetypes().keys()
	if not isinstance(extension, str):
		return False
	if extension.lower() in valid_plt_extensions:
		return True
	return False


def letters_to_patterns(letters_list: List[str], patterns: Dict[str, Vector]) -> List[Vector]:
	"""
	Convert a list of letters to their corresponding patterns.

	Parameters:
		letters_list (List[str]): List of single-character letters.
		patterns (Dict[str, Vector]): Dictionary mapping letters to their patterns.

	Returns:
		List[Vector]: List of patterns corresponding to the letters.
	"""
	assert isinstance(patterns, Dict), "Patterns must be a dictionary."
	assert isinstance(letters_list, List) and len(letters_list) > 0, "Letters must be a non-empty list."
	assert all(isinstance(letter, str) and len(letter) == 1 for letter in
			   letters_list), "Each letter must be a single character string."
	try:
		patterns_list = [patterns[letter] for letter in letters_list]
	except KeyError as e:
		raise KeyError(f"Letter '{e.args[0]}' not found in the patterns dictionary that was provided.")
	return patterns_list


def check_overall_convergence(converged, final_state, target_state) -> bool:
	"""
	Check if the network converged to the target state.

	Parameters:
		converged (bool): Whether the network converged within the max cycles, tha value that was returned by the run_until_convergence method.
		final_state (Vector): The final state of the network after updates.
		target_state (Vector): The target state to check against.
	"""
	if converged and np.array_equal(final_state, target_state):
		return True
	return False
