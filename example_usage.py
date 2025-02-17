
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
