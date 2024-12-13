#%%
import numpy as np
from MyPlsClass import MyPls as pls
import NSFcn

# Calibration Dataset Parameters
Num_observation=30
Ninput=4
Noutput=2
Num_testing=1
Num_com = 3             # Number of PLS components (=Number of X Variables)

# Calibration Dataset
X =np.random.rand(Num_observation,Ninput)
Beta=np.random.rand(Ninput,Noutput) * 2 -1 #np.array([3,2,1])
Y=(X @ Beta)

# Targeted Output (For which Null space is to be explored)
X_target=np.random.rand(Num_testing,Ninput)
Y_target=(X_target @ Beta)

# Null Space determination
MyPlsModel=pls()
MyPlsModel.train(X,Y,Num_com=Num_com)

# NS All : Y prediction for all NS_X equals Y_targeted
NS_t,NS_X,NS_Y=NSFcn.NS_all(plsModel=MyPlsModel,Y_des=Y_target,MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)

# NS single : which_col=1 Y prediction for all NS_X equals which_col=1 of Y_targeted
NS_t,NS_X,NS_Y=NSFcn.NS_single(plsModel=MyPlsModel,which_col=1,Num_point=1000,Y_des=Y_target,MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)

# NS XtoY : the same as NS single yet NS_X has been calculated directly using the X space
NS_t,NS_X,NS_Y=NSFcn.NS_XtoY(plsModel=MyPlsModel,which_col=2,Num_point=1000,Y_des=Y_target,MI_method=1)
MyPlsModel.visual_plot(X_test=NS_X)
# %%
