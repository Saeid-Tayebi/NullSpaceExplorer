import numpy as np
from .lib.pls import PlsClass as pls


def null_space_all_col(X, Y, Y_des, n_component, number_points: int = 100, model_inversion: int = 1):

    # train pls model
    plsmodel = pls().fit(X, Y, n_component=n_component)
    # calculate null space
    return plsmodel.null_space_all(Y_des=Y_des, Num_point=number_points, MI_method=model_inversion)


def null_space_single_col_score_to_Y(X, Y, Y_des, which_col, n_component, number_points: int = 100, model_inversion: int = 1):
    ''' Calculate NS for each column separately once at a time based on Garcia paper '''
    # train pls model
    plsmodel = pls().fit(X, Y, n_component=n_component)
    return plsmodel.null_space_single_col_t_to_Y(Y_des=Y_des, which_col=which_col, Num_point=number_points, MI_method=model_inversion)


def null_space_single_col_X_to_Y(X, Y, Y_des, which_col, n_component, number_points: int = 100, model_inversion: int = 1):
    ''' Calculate NS for each column separately once at a time based on Garcia paper '''
    # train pls model
    plsmodel = pls().fit(X, Y, n_component=n_component)
    return plsmodel.null_space_single_col_X_to_Y(Y_des=Y_des, which_col=which_col, Num_point=number_points, MI_method=model_inversion)
