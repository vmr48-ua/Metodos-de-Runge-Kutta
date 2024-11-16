import numpy as np

def RKII(r0,t,drdt, param ,p= 1/2) -> np.ndarray:
    '''
    Función que implementa el método de Runge-Kuta 2 genérico
    drdt = func(t,r,param)
    r0 = r(t = t[0])
    '''

    import numpy as np

    a1 = 1- p
    a2 = p
    
    c = 1/(2*p)

    dt = t[1]-t[0]
    N = len(t)

    r = np.zeros((N,len(r0)))

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(t[i],r[i],param)
        k2 = dt*drdt(t[i] + c*dt,r[i] + c*k1,param)
        r[i+1] = r[i] + a1*k1 + a2*k2
    
    return r

def RKIV(r0,t,drdt, param) -> np.ndarray:
    '''
    Función que implementa el método de Runge-Kuta 4
    
    Input:
    - r0: np.ndarray, r(t = t[0])
    - t: np.ndarray
    - drdt: function(t,r,param)
    - param: np.ndarray
    
    Output:
    - r: np.ndarray
    '''

    import numpy as np

    dt = t[1]-t[0]
    N = len(t)

    r = np.zeros((N,len(r0)))

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(t[i],r[i],param)
        k2 = dt*drdt(t[i] + dt/2,r[i] + k1/2,param)
        k3 = dt*drdt(t[i] + dt/2,r[i] + k2/2,param)
        k4 = dt*drdt(t[i] + dt,r[i] + k3,param)
        r[i+1] = r[i] + (k1+ 2*k2 + 2*k3 + k4)/6
    
    return r

def RKIV(r0,t,drdt, param) -> np.ndarray:
    '''
    Función que implementa el método de Runge-Kuta 4
    drdt = func(t,r)
    r0 = r(t = t[0])
    '''

    import numpy as np

    dt = t[1]-t[0]
    N = len(t)

    r = np.zeros((N,len(r0)))

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(t[i],r[i],param)
        k2 = dt*drdt(t[i] + dt/2,r[i] + k1/2,param)
        k3 = dt*drdt(t[i] + dt/2,r[i] + k2/2,param)
        k4 = dt*drdt(t[i] + dt,r[i] + k3,param)
        r[i+1] = r[i] + (k1+ 2*k2 + 2*k3 + k4)/6
    
    return r