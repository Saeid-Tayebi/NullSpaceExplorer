# NullSpaceExplorer

## Overview
**NullSpaceExplorer** is a Python-based tool for exploring the null space in modeling systems using PLS (Partial Least Squares) regression. It enables users to identify sets of input variables (`X`) that yield the same predicted output (`Y_pre`). The tool supports three approaches to null space calculation:
1. **NS_All**: Finds `X` inputs where the predicted outputs (`Y_pre`) for all output variables match the target output (`Y_target`).
2. **NS_Single**: Finds `X` inputs where the predicted output matches the target output for a specific column of `Y`.
3. **NS_XtoY**: Directly calculates `X` inputs using the `X` data space where `Y_pre` remains unchanged.

## Key Features
- **Null Space Calculation**: Supports both global null space (NS_All) and column-specific calculations (NS_Single, NS_XtoY).
- **Modeling Framework Validation**: Ensures the generated null space is within the modeling framework by checking Hotelling’s \( T^2 \) and SPE limits.
- **Customizable Parameters**: Specify the number of points (`Num_point`) for the null space exploration.
- **Visualization**: Offers tools for visualizing null space results.

## Important Notes
- Null space for **NS_All** exists only if the number of PLS components exceeds the number of `Y` variables.
- For **NS_Single** and **NS_XtoY**, null space can be calculated for individual `Y` variables, even if the global null space does not exist.
- If there is no intersection in the null space results across all `Y` variables, the system has no null space.
- The difference between **NS_Single** and **NS_XtoY**:
  - **NS_Single** calculates scores for which `Y_pre` is unchanged and projects them back into the `X` space.
  - **NS_XtoY** directly identifies `X` data for which `Y_pre` remains unchanged.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Saeid-Tayebi/NullSpaceExplorer.git
   ```

## Usage
Here’s an example of how to use the tool for null space calculations:

```python
import numpy as np
from MyPlsClass import MyPls as pls
import NSFcn

# Calibration Dataset Parameters
Num_observation = 30
Ninput = 4
Noutput = 2
Num_testing = 1
Num_com = 3  # Number of PLS components (=Number of X Variables)

# Calibration Dataset
X = np.random.rand(Num_observation, Ninput)
Beta = np.random.rand(Ninput, Noutput) * 2 - 1
Y = (X @ Beta)

# Targeted Output (For which Null space is to be explored)
X_target = np.random.rand(Num_testing, Ninput)
Y_target = (X_target @ Beta)

# Null Space determination
MyPlsModel = pls()
MyPlsModel.train(X, Y, Num_com=Num_com)

# NS All
NS_t, NS_X, NS_Y = NSFcn.NS_all(plsModel=MyPlsModel, Y_des=Y_target, MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)

# NS Single
NS_t, NS_X, NS_Y = NSFcn.NS_single(plsModel=MyPlsModel, which_col=1, Num_point=1000, Y_des=Y_target, MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)

# NS XtoY
NS_t, NS_X, NS_Y = NSFcn.NS_XtoY(plsModel=MyPlsModel, which_col=2, Num_point=1000, Y_des=Y_target, MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)
```

## Parameters
- **`Y_des`**: Target `Y` values for which the null space is calculated.
- **`Num_point`**: Number of points to include in the null space exploration. The actual number may be fewer due to modeling constraints.
- **`which_col`**: Specifies the column of `Y` for column-specific null space calculations (NS_Single and NS_XtoY).

## License
This project is licensed under the [MIT License](LICENSE). Please ensure to cite relevant publications used in the development of this tool.

## Contributions
Contributions are welcome! Feel free to submit a pull request or report issues via GitHub.
