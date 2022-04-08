# -*- coding: utf-8 -*-
"""
This is the numerical code of the box model used in Lorenz et al. submitted to GRL
It includes functions that are used to prepare the input as well as the numerical integration itself.

authors: Marvin Lorenz (marvin.lorenz@io-warnemuende.de)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def eta_lin(amp,slope,T,T_max,N_max):
    """
    this function creates a linear surge
    on time steps of the box model based on the input parameters
    
    amp        : surge height
    slope      : rate, how fast the surge should grow
    T_max      : length of the box model simulation in seconds
    N_max      : number of time steps 
    """
    #create box model time values
    t = np.linspace(0,T_max,N_max+1)
    #create the linear surge
    eta= slope*t
    i = np.argwhere(eta>=amp)[0][0]
    eta[i:] = 2*amp-slope*t[i:]
    eta[eta<0]=0
    return(eta)   


def eta_sin(amp,T,T_max,N_max):
    """
    this function creates a sine shaped surge
    on time steps of the box model based on the input parameters
    
    amp        : surge height
    T          : period of the sine
    T_max      : length of the box model simulation in seconds
    N_max      : number of time steps 
    """
    t = np.linspace(0,T_max,N_max+1)
    eta=amp*np.sin(2*np.pi*t/(2*T))
    eta[eta<0]=0
    return(eta)

def eta_gauss(amp,width,T_max,N_max):
    """
    this function creates a Gaussian shaped surge
    on time steps of the box model based on the input parameters.
    The Gaussian is centered at T_max/2
    
    amp        : surge height
    width      : shae parameter of the Gaussian
    T_max      : length of the box model simulation in seconds
    N_max      : number of time steps 
    """
    t = np.linspace(0,T_max,N_max+1)
    eta = amp*np.exp(-np.power((t - T_max/2.0)/width, 2.)/2)
    return(eta)

def linear_func(x,alpha):
    return alpha*x

def U_np1(U_nm1,eta_n,eta_0,W,H,g,dx,dt,z_0):
    """
    function to compute the transport U at new time step n based on 
    variables at time step n - 1
    
    U_nm1 : transport at the old time step
    eta_n : eta at the new time step n
    eta_0 : eta at the time step n from the prescribed surge forcing
    W     : width of the inlet
    H     : depth of the inlet
    g     : gravitational constant 9.81 m s^-2
    dx    : distance between the two elevation points to compute the gradient
    dt    : time step
    z_0   : roughness length for the friction term
    """
    #total water depth in the inlet
    D_0=H+(eta_0+eta_n)/2.0
    #compute R from eq. (3)
    R=(0.4/np.log((D_0/2+z_0)/z_0))**2
    #have a limiter for R
    R = np.fmax(0.0025,R)
    #compute new U with eq. (4)
    RU = np.abs(U_nm1)*R/D_0**2
    U = ( U_nm1 - dt/dx*g*D_0*(eta_0-eta_n) ) / ( 1. + dt*RU )
    #save friction separately
    friction=U*RU
    return(U,-dt*friction)

def eta_np1_via_U(eta_nm1,U_n,dt,A,W):
    """
    compute new eta based on old transport U with eq. (5)
    
    eta_nm1 : old eta in the lagoon at time_step n -1
    U_n     : transport at timestep n - 1
    dt      : time step
    A       : area of the lagoon
    W       : width of the inlet
    """
    eta_new=eta_nm1 - W/A*U_n*dt
    return(eta_new)

#def model(A,H,W,z_0,dt,T,T_max,MSL,eta_function,amp,slope,eta_data=None,Regression=False,dx=None): #old
def model(A,H,W,z_0,dt,T_max,eta_0,SLR=0.0,Regression=False,dx=None):    
    """
    this function integrates the simulation based on the prescribed surge and input parameters
    
    A     : area of the lagoon in m^2
    H     : depth of the inlet in m 
    W     : width of the inlet in m
    z_0   : roughness length in m
    dt    : time step in s
    T_max : maximum length of the simulation in s
    eta_0 : prescribed surge on time step of the model
    SLR   : additional sea level rise in m
    Regression : True/False, do a linear regression to compute the slope alpha
    dx    : distane between the eta in the lagoon and eta_0 to compute gradients, in m; if None: dx=sqrt(A)
    """
    g  = 9.81 #m s^-2
    if dx is None:
        dx = np.sqrt(A) #m length for the gradient
    N_max=int(T_max/dt) #number of time steps
    ###
    
    eta_0=np.copy(eta_0)
    
    ### initialize the array, where the computed eta should be stored
    # Note that SLR will create a second dimension 
    if type(SLR) is list or type(SLR) is np.ndarray:
        U_result = np.zeros((N_max+2,len(SLR)))
        friction_result = np.zeros((N_max+2,len(SLR)))
        eta_result = np.full((N_max+2,len(SLR)),eta_0[0])#plus 2 to keep index 1 as the inital and index 0 is needed for eta(n-1)
    elif type(SLR) is float:
        U_result = np.zeros((N_max+2,))
        friction_result = np.zeros((N_max+2,))
        eta_result = np.full((N_max+2,),eta_0[0])#plus 2 to keep index 1 as the inital and index 0 is needed for eta(n-1)
    else:
        import sys
        sys.exit('SLR not valid input')
        
    # Integration of the simulation
    # note that all SLR scenarios will be integrated at the same time, therefor only one loop over time
    for n in np.arange(N_max):
        #compute new transport U
        U_result[n],friction_result[n] = U_np1(U_result[n-1],eta_result[n-1],eta_0[n],W,H+np.array(SLR),g,dx,dt,z_0)
        #compute new elevation eta
        eta_result[n]=eta_np1_via_U(eta_result[n-1],U_result[n],dt,A,W)


    #compute growths rate (not used in the publication)
    rate_eta_0 = np.max(eta_0)/(np.argwhere(eta_0==np.max(eta_0))[0][0]*dt)
    rate_eta = np.max(eta_result)/(np.argwhere(eta_result==np.max(eta_result,axis=0))[:,0]*dt)

    #compute eta, relative to the prescribed surge
    relative_eta=np.max(eta_result,axis=0)/np.max(eta_0) 
    
    if Regression:
        #linear regression, interception is forced to equal zero since linear_func = alpha*x
        SLR=np.array(SLR)
        y = np.max(eta_result,axis=0)-np.max(eta_result,axis=0)[0]+SLR
        popt, pcov = curve_fit(linear_func, SLR, y)
        a = popt[0]
        b = 0
        errors = np.sqrt(np.diag(pcov))
        #return output
        return(a,b,relative_eta,eta_result,rate_eta_0,rate_eta)

    else:
        #return output
        return(relative_eta,eta_result,rate_eta_0,rate_eta)










