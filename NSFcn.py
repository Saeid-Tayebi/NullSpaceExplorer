import numpy as np
from scipy.linalg import sqrtm


def NS_all(plsModel, Num_point=100, Y_des=None, MI_method=1):
    ''' Calculate NS for all columns at the same time using SVD '''
    if plsModel.Null_Space!=2:
        print('Null Space Does Not Exist')
        return None
    
    X_ds,__=plsModel.MI(Y_des,MI_method)   # direct x answer
    __,t_ds,__,__,__=plsModel.evaluation(X_ds)

    T=plsModel.T
    P=plsModel.P
    Q=plsModel.Q
    S=sqrtm(T.T @ T)
    

    left_singular, v, Vt = np.linalg.svd(S@Q.T, full_matrices=True)
    
    i=len(v)
    G=left_singular[:,i:]
    NS_dim=G.shape[1]

    Ub= np.min(plsModel.ellipse_radius)
    Lb=-Ub

    gamma= (Ub-Lb)*np.random.rand(10*Num_point,NS_dim) + Lb

    # refine the acceptable gamma
    u_ds = t_ds @ np.linalg.inv(S)
    x_preScaled= (u_ds+gamma@G.T)@S@P.T
    x_pre,__=plsModel.unscaler(X_new=x_preScaled,Y_new=None)
    Ypre,t_score,HotelingT2,__,__=plsModel.evaluation(x_pre)
    isvalid=HotelingT2<plsModel.T2_lim[-1]
    
    NS_t=t_score[isvalid,:]
    NS_X=x_pre[isvalid,:]
    NS_Y=Ypre[isvalid,:]

    return NS_t,NS_X,NS_Y


def NS_single(plsModel, which_col,Num_point=100, Y_des=None, MI_method=1):
    ''' Calculate NS for each column separately once at a time based on Garcia paper '''
    def meshgrid_multi_dimension(Lb,Ub,NumRow,NumCol):
        outData=np.zeros([NumRow,NumCol])
        randomData=np.linspace(Lb,Ub,10*NumRow)
        for i in range(NumCol):
            outData[:,i]=np.random.choice(randomData, size=NumRow, replace=True)
        return outData
    
    X_ds,__=plsModel.MI(Y_des,MI_method)   # direct x answer
    __,t_ds,__,__,__=plsModel.evaluation(X_ds)
    P=plsModel.P
    Q=plsModel.Q
    NumNs=Q.shape[1]
    Ub= np.min(plsModel.ellipse_radius)
    Lb=-Ub
    NumRow=Num_point
    NumCol=NumNs-1

    if which_col>Q.shape[1]:
        print('Null Space Does Not Exist')
        return None

    delta_t=meshgrid_multi_dimension(Lb,Ub,NumRow,NumCol)
    which_col=which_col-1
    c=np.zeros([1,NumCol])
    for j in range(NumCol):
        c[0,j]= -Q[which_col,j]/Q[which_col,-1]
        
    z=np.sum(c*delta_t,axis=1)
    delta_t = np.column_stack((delta_t, z))
    ptNS=t_ds+delta_t
    x_preScaled= ptNS @ P.T
    x_pre,__=plsModel.unscaler(X_new=x_preScaled,Y_new=None)
    Ypre,t_score,HotelingT2,__,__=plsModel.evaluation(x_pre)
    isvalid=HotelingT2<plsModel.T2_lim[-1]
    NS_t=t_score[isvalid,:]
    NS_X=x_pre[isvalid,:]
    NS_Y=Ypre[isvalid,:]

    return NS_t,NS_X,NS_Y
    
def NS_XtoY(plsModel, which_col,Num_point=100, Y_des=None, MI_method=1):
    ''' Calculate NS for each column of Y seperately directly using the X data  '''
    def meshgrid_multi_dimension(Lb,Ub,NumRow,NumCol):
        outData=np.zeros([NumRow,NumCol])
        for i in range(NumCol):
            randomData=np.linspace(Lb[i],Ub[i],10*NumRow)
            outData[:,i]=np.random.choice(randomData, size=NumRow, replace=True)
        return outData
    
    X_ds,__=plsModel.MI(Y_des,MI_method)   # direct x answer
    X_ds_scaled,__=plsModel.scaler(X_new=X_ds)
    __,t_ds,__,__,__=plsModel.evaluation(X_ds)

    B=plsModel.B_pls
    NumNs=B.shape[0]
    Ub= np.min(plsModel.Xtrain_scaled,axis=0)
    Lb=-Ub
    NumRow=Num_point
    NumCol=NumNs-1

    if which_col>B.shape[1]:
        print('Null Space Does Not Exist')
        return None

    delta_x=meshgrid_multi_dimension(Lb,Ub,NumRow,NumCol)
    which_col=which_col-1
    c=np.zeros([1,NumCol])
    for j in range(NumCol):
        c[0,j]= -B[j,which_col]/B[-1,which_col]
        
    z=np.sum(c*delta_x,axis=1)
    delta_x = np.column_stack((delta_x, z))
    pxNS=X_ds_scaled+delta_x
    x_preScaled=pxNS
    x_pre,__=plsModel.unscaler(X_new=x_preScaled,Y_new=None)
    Ypre,t_score,HotelingT2,__,__=plsModel.evaluation(x_pre)
    isvalid=HotelingT2<plsModel.T2_lim[-1]
    NS_t=t_score[isvalid,:]
    NS_X=x_pre[isvalid,:]
    NS_Y=Ypre[isvalid,:]

    return NS_t,NS_X,NS_Y