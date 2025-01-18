import numpy as np
import time
import numpy.linalg as la

def progress(i, N) -> None:
    '''
    Función que imprime el progreso de un ciclo for
    
    Input:
    - i: int, iteración actual
    - N: int, número total de iteraciones
    '''
    print(f'\rProgreso: {100*i/N:.2f}%', end='')
    return None

def error(psi_analitica, psi_numerica, dx):
    return la.norm(psi_analitica - psi_numerica,axis=1)*dx

def timear(funcion, param):
    """ 
    Devuelve el tiempo que tarda en ejecutarse una función
    """
    t0 = time.perf_counter()
    sol = funcion(*param)
    t1 = time.perf_counter()
    return [sol, (np.abs(t1-t0))]

def RKII_G(r0, t, drdt, param, p = 1/2) -> np.ndarray:
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
    if p == 0:
        raise ValueError('Incorrect value of p, p=0')

    a1 = 1- p
    a2 = p
    
    c = 1/(2*p)

    dt = t[1]-t[0]
    N = len(t)

    r = np.zeros((N, *r0.shape), dtype=np.complex128 if np.iscomplexobj(r0) else np.float64)
    r[0,:] = r0
    
    for i in range(N-1):
        progress(i,N-1)
        k1 = dt*drdt(t[i],r[i],param)
        k2 = dt*drdt(t[i] + c*dt,r[i] + c*k1,param)
        r[i+1] = r[i] + a1*k1 + a2*k2
    return r

def RKIII_G(r0, t, drdt, param, a = 1/2 ,b = 1) -> np.ndarray:
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
    if a == 0 or a == 2/3 or b == 0 or a == b:
        raise ValueError('Incorrect value of a or b')

    dt = t[1]-t[0]
    N = len(t)

    r = np.zeros((N, *r0.shape), dtype=np.complex128 if np.iscomplexobj(r0) else np.float64)

    r[0,:] = r0

    for i in range(N-1):
        progress(i,N-1)
        k1 = dt*drdt(t[i],r[i],param)
        
        # k2 = dt*drdt(t[i] + a*dt,r[i] + a*k1,param)
        # k3 = dt*drdt(t[i] + b*dt,r[i] + ((b*(b-3*a*(1-a)))/(a*(3*a-2)))*k1 + ((-b*(b-a))/(a*(3*a-2)))*k2,param)
        # r[i+1] = r[i] + (1-(3*a+3*b -2 )/(6*a*b))*k1 + ((3*b-1)/(6*a*(b-a)))*k2 + ((2-3*a)/(6*b*(b-a)))*k3
        k2 = dt*drdt(t[i] + 0.5*dt,r[i] + 0.5*k1,param)
        k3 = dt*drdt(t[i] + 1*dt,r[i] + (-1)*k1 + (2)*k2,param)
        r[i+1] = r[i] + (1/6)*k1 + (2/3)*k2 + (1/6)*k3
    return r

def RKIV(r0, t, drdt, param) -> np.ndarray:
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

    # Si estamos en schrodinger es complejo si no es float, si es klein gordon las dimensiones
    # cuadran si no me falta una dimensión por eso el *r0.shape
    r = np.zeros((N, *r0.shape), dtype=np.complex128 if np.iscomplexobj(r0) else np.float64)
    r[0,:] = r0

    for i in range(N-1):
        progress(i,N-1)
        k1 = dt*drdt(t[i],r[i],param)
        k2 = dt*drdt(t[i] + dt/2,r[i] + k1/2,param)
        k3 = dt*drdt(t[i] + dt/2,r[i] + k2/2,param)
        k4 = dt*drdt(t[i] + dt,r[i] + k3,param)
        r[i+1] = r[i] + (k1+ 2*k2 + 2*k3 + k4)/6
    
    return r

def RKVI(r0, t, drdt, param) -> np.ndarray: 
    '''
    Función que implementa el método de Runge-Kuta 6
    
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

    r = np.zeros((N, *r0.shape), dtype=np.complex128 if np.iscomplexobj(r0) else np.float64)

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(t[i],r[i],param)
        k2 = dt*drdt(t[i] + dt/3    ,r[i] + k1/3,param)
        k3 = dt*drdt(t[i] + 2*dt/3  ,r[i] + 2*k2/3,param)
        k4 = dt*drdt(t[i] + dt/3    ,r[i] + k1/12 + k2/3 - k3/12,param)
        k5 = dt*drdt(t[i] + dt/2    ,r[i] - k1/16 + 9*k2/8 - (3*k3)/16 - 3*k4/8,param)
        k6 = dt*drdt(t[i] + dt/2    ,r[i] + 9*k2/8 - 3*k3/8 - 3*k4/4 + k5/2,param)
        k7 = dt*drdt(t[i] + dt      ,r[i] + 9*k1/44 - 9*k2/11 + 63*k3/44 + 18*k4/11 -16*k6/11,param)

        r[i+1] = r[i] + 11*k1/120 + 27*k3/40 + 27*k4/40 - 4*k5/15 - 4*k6/15 + 11*k7/120 
    
    return r