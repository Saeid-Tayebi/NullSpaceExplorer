# NullSpaceExplorer

## Overview

**NullSpaceExplorer** is a Python-based tool for exploring the null space in modeling systems using PLS (Partial Least Squares) regression. It enables users to identify sets of input variables (`X`) that yield the same predicted output (`Y_pre`). The tool supports three approaches to null space calculation:

1. **NS_All**: Finds `X` inputs where the predicted outputs (`Y_pre`) for all output variables match the target output (`Y_target`).
2. **NS_Single (Score to Y)**: Finds `X` inputs where the predicted output matches the target output for a specific column of `Y` by calculating scores.
3. **NS_Single (X to Y)**: Directly calculates `X` inputs using the `X` data space where `Y_pre` remains unchanged for a specific column of `Y`.

## Key Features

- **Null Space Calculation**: Supports global null space (NS_All) and column-specific calculations (NS_Single).
- **Modeling Framework Validation**: Ensures the generated null space is within the modeling framework by checking Hotelling’s \( T^2 \) and SPE limits.
- **Customizable Parameters**: Specify the number of points (`number_points`) for the null space exploration.
- **Visualization**: Offers tools for visualizing null space results.
- **Testing**: Includes a `tests` folder with test cases to validate the functionality of the code.

## Important Notes

- Null space for **NS_All** exists only if the number of PLS components exceeds the number of `Y` variables.
- For **NS_Single**, null space can be calculated for individual `Y` variables, even if the global null space does not exist.
- If there is no intersection in the null space results across all `Y` variables, the system has no null space.
- The difference between **NS_Single (Score to Y)** and **NS_Single (X to Y)**:
  - **NS_Single (Score to Y)**: Calculates scores for which `Y_pre` is unchanged and projects them back into the `X` space.
  - **NS_Single (X to Y)**: Directly identifies `X` data for which `Y_pre` remains unchanged.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Saeid-Tayebi/NullSpaceExplorer.git
   ```
2. Navigate to the project directory:
   ```bash
   cd NullSpaceExplorer
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To get started, check out the `example_usage.py` file in the root directory. This file demonstrates how to import and use the `nspls` library.

### Example Usage

```python
# Import the necessary modules
import numpy as np
from nspls.lib.pls import PlsClass as pls
from nspls import null_space as nspls

# Calibration Dataset Parameters
Num_observation = 30
Ninput = 4
Noutput = 2
Num_testing = 1
n_component = Noutput+1             # Number of PLS components (=Number of X Variables)

# Calibration Dataset
X = np.random.rand(Num_observation, Ninput)
Beta = np.random.rand(Ninput, Noutput) * 2 - 1  # np.array([3,2,1])
Y = (X @ Beta)

# Targeted Output (For which Null space is to be explored)
X_target = np.random.rand(Num_testing, Ninput)
Y_target = (X_target @ Beta)

# Model Development
MyPlsModel = pls().fit(X, Y, n_component=3)


# NS All : Y prediction for all NS_X equals Y_targeted
NS_t, NS_X, NS_Y = nspls.null_space_all_col(
    X, Y, Y_des=Y_target, n_component=n_component, number_points=100, model_inversion=1)
MyPlsModel.visual_plot(X_test=NS_X)

# NS single : which_col=1 Y prediction for all NS_X equals which_col=1 of Y_targeted
NS_t, NS_X, NS_Y = nspls.null_space_single_col_score_to_Y(
    X, Y, Y_des=Y_target, which_col=1, n_component=n_component, number_points=100, model_inversion=1)
MyPlsModel.visual_plot(X_test=NS_X)

# NS XtoY : the same as NS single yet NS_X has been calculated directly using the X space
NS_t, NS_X, NS_Y = nspls.null_space_single_col_X_to_Y(
    X, Y, Y_des=Y_target, which_col=1, n_component=n_component, number_points=100, model_inversion=1)
MyPlsModel.visual_plot(X_test=NS_X)

```

For more detailed examples, refer to the `example_usage.py` file.

## Testing

The `tests` folder contains test cases to validate the functionality of the code. To run the tests, use the following command:

```bash
python -m pytest tests/
```

## Releases

The latest release of **NullSpaceExplorer** can be found on the [Releases page](https://github.com/Saeid-Tayebi/NullSpaceExplorer/releases). Download the release package to get started with the tool.

## Folder Structure

```
NullSpaceExplorer/
├── nspls/               # Main library code
│   ├── __init__.py      # Make the folder a Python package
│   ├── PlsClass.py      # PLS class implementation
│   ├── NSFcn.py         # Null space calculation functions
│   └── ...              # Other code files
├── tests/               # Test cases for validating the code
│   ├── test_ns_all.py   # Tests for NS_All functionality
│   ├── test_ns_single.py # Tests for NS_Single functionality
│   └── ...              # Other test files
├── example_usage.py     # Demo/tutorial file
├── README.md            # Project overview and instructions
├── LICENSE              # License file
├── requirements.txt     # List of dependencies
└── .gitignore           # Files to ignore in Git
```

## License

This project is licensed under the [MIT License](LICENSE). Please ensure to cite relevant publications used in the development of this tool.

## Contributions

Contributions are welcome! Feel free to submit a pull request or report issues via GitHub.

---
