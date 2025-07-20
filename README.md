# Hopfield Model Project

This project implements a Hopfield network model in Python, designed for educational purposes - learning and simulating
associative memory tasks.  
It includes a core implementation of the Hopfield network, methods for training and recalling patterns, and tools for
analyzing the stability of the network.  
The project is structured to facilitate modular testing and execution of different parts of
the assignment.

---

## 📚 Overview

- **Core Implementation**: A Hopfield network class with methods for training, recalling patterns, and analyzing
  stability.
- **Pattern Management**: Load and manage patterns from Excel files, including support for custom patterns.
- **Visualization**: Interactive plotting of network states, convergence, and stability analysis.
- **Assignment Structure**: Organized scripts for different parts of the assignment, allowing for modular testing and
  execution.
- **Type Annotations**: Comprehensive type annotations for better code clarity and maintainability.
- **Documentation**: Detailed docstrings and a structured README for easy navigation and understanding.
- **Testing**: Basic test cases to ensure functionality, with a focus on pattern loading and network behavior.
- **Interactive Plotting**: Use of `matplotlib` for visualizing network states and convergence, with options for static
  and dynamic plots.
- **Modular Design**: Clear separation of concerns with a main package for the Hopfield network and a subpackage for
  visualization.
- **Assignment Resources**: Includes an Excel file with letter patterns and scripts for each part of the assignment.
- **Easy Setup**: Simple setup instructions with a virtual environment and dependency management
  via **[requirements.txt](requirements.txt)**.
- **CLI Arguments**: Support for command-line arguments to enable different modes (e.g., debug mode for detailed
  plotting).

---

## 📁 Project Structure

```
hopfield_model_project/         # Project root directory
│
├── hopfield_model/             # Main package
│   ├── __init__.py
│   ├── network.py              # Core Hopfield network implementation
│   ├── typing.py               # Type aliases
│   ├── utilities.py            # Helper functions (e.g., loading patterns)
│   └── visualization/          # Plotting subpackage
│       ├── __init__.py
│       ├── plot_controller.py  # Handles interactive plotting flow
│       ├── plotting.py         # Plotting functions (static & dynamic)
│       └── typing.py           # Plot-related type aliases
│
├── exercise/                   # Assignment-specific resources
│   ├── alphabet.xlsx           # Letter patterns
│   └── scripts/                # Assignment parts
│       ├── initialization.py   # Initialization script
│       ├── main.py
│       ├── part_a.py
│       ├── part_b.py
│       ├── part_c.py
│       └── part_d.py
│
├── .gitignore
├── LICENSE                     # MIT License
├── README.md   
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Project metadata
```

---

## 🛠 Setup

### 1. Python Version

This project requires *Python 3.12 or higher* due to usage of modern string formatting options (e.g., the z alignment
modifier).  
You can check your Python version with:

```bash
python --version
```

### 2. Create Virtual Environment

#### Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Tkinter Setup (Required for Interactive Plotting)

Tkinter is required for *interactive plotting* (via matplotlib.use('TkAgg')).

#### Linux (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install python3-tk
```

#### Windows:

Tkinter is *bundled* with most Python installations.  
If you encounter import errors, reinstall Python from
the [official Python website](https://www.python.org/downloads/windows/) and ensure *"tcl/tk and IDLE"* is checked
during installation.

#### macOS:

Tkinter is *usually pre-installed*. If missing, you can install it via:

```bash
brew install python-tk
```

⚠ *Important:*  
If you skip this step, running interactive visualizations will fail.


---

## ▶ Running the Project

All parts of the assignment are located inside **[exercise/scripts/](exercise/scripts)**:

- **[main.py](exercise/scripts/main.py)**
- **[part_a.py](exercise/scripts/part_a.py)**
- **[part_b.py](exercise/scripts/part_b.py)**
- **[part_c.py](exercise/scripts/part_c.py)**
- **[part_d.py](exercise/scripts/part_d.py)**

### You can run the unified script:

```bash
python3 -m exercise.scripts.main
```

### You can also run each part individually:

```bash
python3 -m exercise.scripts.part_a
```

```bash
python3 -m exercise.scripts.part_b
```

```bash
python3 -m exercise.scripts.part_c
```

```bash
python3 -m exercise.scripts.part_d
```

### 📝 Important Notes

- Current working directory should be the top-level **[hopfield_model_project/](.)** directory.
- No relative imports are used; absolute imports handle project navigation cleanly.
- The main script **[main.py](exercise/scripts/main.py)** orchestrates the execution of all parts, allowing for modular
  testing and execution.
- The different parts are designed to be run independently or as a whole.

### ⚙ Modes

#### Enable modes by passing CLI arguments:

- `--debug` — Extra plotting: when plotting the network convergence, it will plot the state of the network at each step,
  and not just when an update occurs, this mode is relevant only for **[part_b.py](exercise/scripts/part_b.py)**.
- `--save` — Plot Saving: saves the displayed plots to disk.

**Examples:**

```bash
python -m exercise.scripts.part_b --debug
```

```bash
python -m exercise.scripts.part_d --save
```

```bash
python -m exercise.scripts.main --debug --save
```

---

## 📌 Key Components

- **[hopfield_model/network.py](hopfield_model/network.py)**: The core class implementing the Hopfield model, with
  methods for training and recalling patterns.
- **[hopfield_model/utilities.py](hopfield_model/utilities.py)**: Functions for loading patterns from Excel files and
  managing network states.
- **[hopfield_model/visualization/plotting.py](hopfield_model/visualization/plotting.py)**: Functions for plotting
  network states and convergence, including interactive plotting capabilities.
- **[exercise/scripts/main.py](exercise/scripts/main.py)**: The main script that orchestrates the execution of the
  assignment parts, allowing for modular testing and execution.
- **[exercise/alphabet.xlsx](exercise/alphabet.xlsx)**: An Excel file containing letter patterns used in the
  assignment.

## 📄 License

MIT License.
See **[LICENSE](LICENSE)** for details.

## 👤 Author

- **Name:** Or Fadida
- **Email:** [orfadida@mail.tau.ac.il](mailto:orfadida@mail.tau.ac.il)
- **GitHub:** [orfadida2000](https://github.com/orfadida2000)

## 🔑 Keywords

Hopfield, neural networks, associative memory, stability analysis, capacity testing
