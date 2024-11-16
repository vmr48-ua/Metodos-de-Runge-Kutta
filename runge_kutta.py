import numpy as np

def RKIIG(r0,t,drdt, param ,p = 1/2) -> np.ndarray:
    '''
    Función que implementa el método de Runge-Kuta 2, genérico
    
    Input:
    - r0: np.ndarray, r(t = t[0])
    - t: np.ndarray
    - drdt: function(t,r,param)
    - param: np.ndarray
    - p = 1/2:
    
    Output:
    - r: np.ndarray
    '''
    
    import numpy as np

    if p == 0:
        raise ValueError('Incorrect value of p, p=0')

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


def RKIIIG(r0,t,drdt, param ,a= 1/2 ,b = 1) -> np.ndarray:
    '''
    Función que implementa el método de Runge-Kuta 3, genérico
    
    Input:
    - r0: np.ndarray, r(t = t[0])
    - t: np.ndarray
    - drdt: function(t,r,param)
    - param: np.ndarray
    - a = 1/2:
    - b = 1 
    
    Output:
    - r: np.ndarray
    '''
    
    import numpy as np

    if a == 0 or a == 2/3 or b == 0 or a == b:
        raise ValueError('Incorrect value of a or b')

    dt = t[1]-t[0]
    N = len(t)

    r = np.zeros((N,len(r0)))

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(t[i],r[i],param)
        k2 = dt*drdt(t[i] + a*dt,r[i] + a*k1,param)
        k3 = dt*drdt(t[i] + b*dt,r[i] + ((b*(b-3*a(1-a)))/(a*(3*a-2)))*k1 + ((-b*(b-a))/(a*(3*a-2)))*k2,param)
        r[i+1] = r[i] + (1-(3*a+3*b -2 )/(6*a*b))*k1 + ((3*b-1)/(6*a*(b-a)))*k2 + ((2-3*a)/(6*b(b-a)))*k3
    
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