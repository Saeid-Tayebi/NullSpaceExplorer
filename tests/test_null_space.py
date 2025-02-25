from nspls.lib.pls import PlsClass as pls, plseval
from nspls import null_space as ns
import numpy as np


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

mypls = pls().fit(X, Y, n_component=n_component)


def test_null_space() -> None:

    # all
    NS_t, NS_X, NS_Y = ns.null_space_all_col(
        X, Y, Y_des=Y_target, n_component=n_component, number_points=100, model_inversion=1)
    Y_pre = mypls.predict(NS_X)
    assert np.allclose(Y_pre.shape, NS_Y.shape)
    assert np.allclose(Y_pre, NS_Y)
    assert np.allclose(Y_pre, Y_target)
    assert np.allclose(NS_Y, Y_target)

    # single t to y
    for i in range(Y_target.shape[1]):
        NS_t, NS_X, NS_Y = ns.null_space_single_col_score_to_Y(
            X, Y, Y_des=Y_target, which_col=i, n_component=n_component, number_points=100, model_inversion=1)
        Y_pre = mypls.predict(NS_X)
        assert np.allclose(Y_pre.shape, NS_Y.shape)
        assert np.allclose(Y_pre[:, i], NS_Y[:, i])
        assert np.allclose(Y_pre[:, i], Y_target[:, i])
        assert np.allclose(NS_Y[:, i], Y_target[:, i])

    # single X to y
    for i in range(Y_target.shape[1]):
        NS_t, NS_X, NS_Y = ns.null_space_single_col_X_to_Y(
            X, Y, Y_des=Y_target, which_col=i, n_component=n_component, number_points=100, model_inversion=1)
        Y_pre = mypls.predict(NS_X)
        assert np.allclose(Y_pre.shape, NS_Y.shape)
        assert np.allclose(Y_pre[:, i], NS_Y[:, i])
        assert np.allclose(Y_pre[:, i], Y_target[:, i])
        assert np.allclose(NS_Y[:, i], Y_target[:, i])
