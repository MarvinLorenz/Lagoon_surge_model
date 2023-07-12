"""
This is the numerical code of the box model based on Stigebrandt (1980) and Hill (1994) used in Lorenz et al. submitted to GRL
It includes functions that are used to prepare the input as well as the numerical integration itself.

authors: Marvin Lorenz (marvin.lorenz@io-warnemuende.de)
"""

import numpy as np

####
#Dimensional model functions
####

def function(A,H,W,L,k,eta_0,eta,g,tau,func):
    D = func(H,eta,eta_0)
    result = -np.sqrt( np.abs(g*W**2/(A**2*k*L)*D**3*(eta-eta_0)+ 
                              (W**2*tau*D**2/(1000.0*k*A**2))
                             )
                     ) * np.sign(g*W**2/(A**2*k*L)*D**3*(eta-eta_0)+ (W**2*tau*D**2/(1000.0*k*A**2)))
    return result

def eta_np1_euler(A,H,W,L,k,dt,eta_0,eta_old,g,tau,Q_r,func):
    #eq (3) of Hill (1994) and eq. (3) of Stigebrandt (1980)
    #print(A,H,W,L,R,dt,eta_0,eta_old,g, SLR)
    #eta_new = eta_old + dt*(g*W**2*D**3/(A**2*R*L))**0.5*np.sqrt(np.abs(eta_0-eta_old))*np.sign(eta_0-eta_old)
    #print(dt*(g*W**2*D**3/(A**2*R*L))**0.5,np.sqrt(np.abs(eta_0-eta_old))*np.sign(eta_0-eta_old))
    eta_new = eta_old + dt*np.sum(function(A,H,W,L,k,eta_0,eta_old,g,tau,func)) + dt*Q_r/A
    #print(eta_new)
    return eta_new

def eta_np1_RK(A,H,W,L,k,dt,eta_0,eta_0p1,eta,g,tau,Q_r,func):
    #eq (4) of Hill (1994) and eq. (5) of Stigebrandt (1980)
    #Runge Kutta 4th Order
    #interpolate forcing eta_0 to half time step into the future
    eta_0p05 = 0.5*(eta_0+eta_0p1)
    #Runge Kutta scheme
    k1 = function(A,H,W,L,k,eta_0,eta,g,tau,func)
    k2 = function(A,H,W,L,k,eta_0p05,eta+0.5*dt*k1,g,tau,func)
    k3 = function(A,H,W,L,k,eta_0p05,eta+0.5*dt*k2,g,tau,func)
    k4 = function(A,H,W,L,k,eta_0p1,eta+dt*k3,g,tau,func)
    eta_new = eta+1.0/6.0*dt*np.sum(k1+2*k2+2*k3+k4) + dt*Q_r/A
    return eta_new


def depth_ar(H,eta,eta_0,SLR=0.0):
    #limit eta_0 to -H
    #print(H,eta,eta_0)
    eta_0 = np.max([eta_0,-H],axis=0)
    D = H+SLR+0.5*(eta_0+eta)
    D[D<0] = 0.0
    #print(D)
    return D
    
def depth_float(H,eta,eta_0,SLR=0.0):
    eta_0 = max(eta_0,-H)
    D = max(H+SLR+0.5*(eta_0+eta),0.0)
    return D

####
# Dimensional version of the equation
####

def model(A,H,W,L,k,dt,T_max,eta_0,SLR=0.0,integration='Euler',k_method='const',z_0=0.001,tau=0.0,Q_r=0.0): 
    """
    this function integrates the simulation based on the prescribed surge and input parameters
    Note: if multiple inlets are used the variable should be a numpy array
    A     : area of the lagoon in m^2
    H     : depth of the inlet in m; this can be multiple inlets 
    W     : width of the inlet in m; this can be multiple inlets
    L     : length of the inlet in m; this can be multiple inlets
    k     : friction parameter no dimension; this can be multiple inlets
    dt    : time step in s
    T_max : maximum length of the simulation in s
    eta_0 : prescribed surge on time step of the model; this can be multiple time series, one for each inlet
    SLR   : additional sea level rise in m
    integration: which numerical scheme is used for integration, either 'Euler' or 'RK' (Runge Kutta 4th order)
    friction_method : either constant or dynmaic
        constant: provided k is used
    z_0   : friction length, only used if k_method == 'dynamic' with law of the wall
        k will be overwritten
    tau   : wind stress in Pa; can be time series; this can be multiple inlets (negative stress pushes water into the lagoon)
    Q_r   : other fluxes into the lagoon, like river discharge or precipitation in m3/s
    """
    g  = 9.81 #m s^-2
    N_max=int(T_max/dt) #number of time steps
    ###
    
    #initialize the lagoon sea level with the first sea level point of the forcing
    eta_0=np.copy(eta_0)
    #for Runge Kutta append eta_0 by its last value
    #check also for number of inlets and chose the depth function accordingly
    if len(eta_0.shape)==1:
        #only one inlet
        #print('single eta_0')
        fill = eta_0[0]
        func = depth_float 
        eta_0=np.append(eta_0,eta_0[-1])
    elif len(eta_0.shape)==2:
        #multiple inlets
        #print('multiple eta_0')
        fill = eta_0[0,0]
        func = depth_ar
        eta_0=np.append(eta_0,np.array([eta_0[-1]]),axis=0)
    else:
        fill=0.0
    
    #consider wind stress
    if type(tau) is float or type(tau) is int or type(tau) is np.float or type(tau) is np.float32 or type(tau) is np.float64 or type(tau) is np.float128:
        tau = np.full(eta_0.shape,tau)
    elif type(tau) is list or type(tau) is np.ndarray:
        if len(tau) < len(eta_0[:-1]):
            print('time series of tau is too short')
            
    #consider river discharge and/or precipitation
    if type(Q_r) is float or type(Q_r) is int or type(Q_r) is np.float or type(Q_r) is np.float32 or type(Q_r) is np.float64 or type(Q_r) is np.float128:
        Q_r = np.full((eta_0.shape[0],),Q_r)
        if len(eta_0.shape)==2:
            fill += Q_r[0]**2*np.mean(k)*np.mean(L)/(g*np.sum(W)**2*np.mean(H)**3)
        else:
            fill += Q_r[0]**2*k*L/(g*W**2*H**3)
    elif type(Q_r) is list or type(Q_r) is np.ndarray:
        if len(Q_r) < len(eta_0[:-1]):
            print('time series of Q_r is too short')

    print(fill)
    eta_result = np.full((N_max+2,),fill)

    # Integration of the simulation
    if integration=='Euler':
        if k_method == 'const':
            for n in np.arange(N_max):
                #compute new elevation eta
                eta_result[n]=eta_np1_euler(A,H+SLR,W,L,k,dt,eta_0[n],eta_result[n-1],g,tau[n],Q_r[n],func)
        elif k_method=='dynamic':
            for n in np.arange(N_max):
                #compute new elevation eta
                D = func(H,eta_result[n-1],eta_0[n],SLR=SLR)
                k = (0.4/np.log((D/2+z_0)/z_0))**2
                eta_result[n]=eta_np1_euler(A,H+SLR,W,L,k,dt,eta_0[n],eta_result[n-1],g,tau[n],Q_r[n],func)
            #print(eta_result[n])
    elif integration=='RK':
        if k_method == 'const':
            for n in np.arange(N_max):
                #compute new elevation eta
                eta_result[n]=eta_np1_RK(A,H+SLR,W,L,k,dt,eta_0[n],eta_0[n+1],eta_result[n-1],g,tau[n],Q_r[n],func)
                #print(eta_result[n])
        elif k_method=='dynamic':
            for n in np.arange(N_max):
                #compute new elevation eta
                
                D = func(H,eta_result[n-1],eta_0[n],SLR=SLR)
                k = (0.4/np.log((D/2+z_0)/z_0))**2
                
                eta_result[n]=eta_np1_RK(A,H+SLR,W,L,k,dt,eta_0[n],eta_0[n+1],eta_result[n-1],g,tau[n],Q_r[n],func)
    
    #return output
    return(eta_result)



####################
#Non dimensional Version of the equations
# Note: no wind stress here
####################


def function_non_dim(zeta,P,eta_0,eta,S):
    result = P*(1.0+zeta*(eta_0+eta)/2.0)**(3./2.) * np.sqrt(np.abs(eta_0 - eta))*np.sign(eta_0 - eta) + S
    return result

def eta_np1_non_dim_RK(zeta,P,dt,eta_0,eta_0p1,eta,S):
    #eq (4) of Hill (1994) and eq. (5) of Stigebrandt (1980)
    #Runge Kutta 4th Order
    #interpolate forcing eta_0 to half time step into the future
    eta_0p05 = 0.5*(eta_0+eta_0p1)
    #Runge Kutta scheme
    k1 = function_non_dim(zeta,P,eta_0,eta,S)
    k2 = function_non_dim(zeta,P,eta_0p05,eta+0.5*dt*k1,S)
    k3 = function_non_dim(zeta,P,eta_0p05,eta+0.5*dt*k2,S)
    k4 = function_non_dim(zeta,P,eta_0p1,eta+dt*k3,S)
    eta_new = eta+1.0/6.0*dt*np.sum((k1+2*k2+2*k3+k4))
    return eta_new

def eta_np1_non_dim_Euler(zeta,P,dt,eta_0,eta,S):
    #eq (4) of Hill (1994) and eq. (5) of Stigebrandt (1980)
    #Euler forward
    eta_new = eta+dt*np.sum(function_non_dim(zeta,P,eta_0,eta,S))
    return eta_new

def model_non_dim(zeta,P,dt,T_max,eta_0,S=0,integration='Euler'): 
    """
    this function integrates the simulation based on the prescribed surge and input parameters
    Note: This non-dimensional model cannot consider wind stress and discharge/precipitation
    zeta  : non-dimensional Amplitude/water depth; this can be multiple inlets
    P     : Choking parameter; this can be multiple inlets
    dt    : time step, scaled with tidal period
    T_max : maximum length of the simulation in number of tidal cycles
    eta_0 : prescribed surge on time step of the model scaled with tidal amplitude
    S     : prescribed non-dim freshwater supply
    """

    N_max=int(T_max/dt) #number of time steps
    ###
    
    eta_0=np.copy(eta_0)
    #print(eta_0.shape)
    #for Runge Kutta append eta_0 plus one
    if len(eta_0.shape)==1:
        #print('single eta_0')
        fill = eta_0[0]
        eta_0=np.append(eta_0,eta_0[-1])
    elif len(eta_0.shape)==2:
        #print('multiple eta_0')
        fill = eta_0[0,0]
        eta_0=np.append(eta_0,np.array([eta_0[-1]]),axis=0)
    else:
        fill=0.0
    #print(eta_0.shape
    
    #consider river discharge and/or precipitation
    if type(S) is float or type(S) is int or type(S) is np.float or type(S) is np.float32 or type(S) is np.float64 or type(S) is np.float128:
        #print('in S')
        S = np.full((eta_0.shape[0],),S)
        fill += S[0]**2/np.sum(P)**2
    elif type(S) is list or type(S) is np.ndarray:
        if len(S) < len(eta_0[:-1]):
            print('time series of Q_r is too short')

    #storage here
    eta_result = np.full((N_max+2,),fill)
    #print(eta_result.shape)
            
    # Integration of the simulation
    if integration == 'Euler':
        for n in np.arange(N_max):
            #compute new elevation eta
            eta_result[n]=eta_np1_non_dim_Euler(zeta,P,dt,eta_0[n],eta_result[n-1],S=S[n])
        #print(eta_result[n])
    elif integration == 'RK':
        for n in np.arange(N_max):
            #compute new elevation eta
            eta_result[n]=eta_np1_non_dim_RK(zeta,P,dt,eta_0[n],eta_0[n+1],eta_result[n-1],S=S[n])

    return(eta_result)

####
# other functions
####

def eta_gauss(amp,width,T_0,T_max,N_max):
    """
    this function creates a Gaussian shaped surge
    on time steps of the box model based on the input parameters.
    The Gaussian is centered at T_0
    
    amp        : surge height
    width      : shape parameter of the Gaussian
    T_max      : length of the box model simulation in seconds
    N_max      : number of time steps 
    """
    t = np.linspace(0,T_max,N_max+1)
    eta = amp*np.exp(-np.power((t - T_0)/width, 2.)/2)
    return(eta)

def Gauss_cross(W_center,W,H_max,H_min,W_gauss):
    """
    this function creates an artificial inlet cross section
    W_center   : array of the central positions of the subinlets
    W          : total width of the inlet / m
    H_max      : maximum depth of the inlet / m
    H_min      : minimum depth of the inlet / m
    -> (H_max - H_min) is the additional depth of the Gaussian thalweg
    W_gauss    : Gaussian width parameter / m
    """
    H_gauss = H_min + (H_max-H_min)*np.exp(-np.power((W_center - W/2.0)/W_gauss, 2.)/2)
    return H_gauss

def NNL(amp_ref,amp_new,SLR):
    """
    Normalized nonlineartiy Index (Bilskie et al. (2014)
    amp_ref : maximum sea level of the simulation without sea level rise
    amp_new : maximum sea level of the simulation with sea level rise
    SLR     : sea level rise 
    """
    return ((amp_new-amp_ref)/SLR - 1)