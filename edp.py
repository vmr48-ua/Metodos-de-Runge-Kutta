import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp, sin, cos
from matplotlib.animation import FuncAnimation, FFMpegWriter

from runge_kutta import *

def diffusion(t, u, params) -> np.ndarray:
    """
    Derivada temporal para la ecuación de difusión.
    
    Input:
    - t: tiempo actual (no se usa en este caso)
    - u: np.ndarray, distribución de la cantidad a difundir en el tiempo t
    - params: tuple (D, dx)
    
    Output:
    - du_dt: np.ndarray, derivada de u con respecto al tiempo
    """
    D, dx = params
    d2u_dx2 = np.zeros(len(u))
    d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    d2u_dx2[0] = d2u_dx2[-1] = 0  # Dirichlet

    du_dt = D * d2u_dx2
    return du_dt
def diffusion_total(L, T, Nx, Nt, D, u0_diff, params_diff, Nt_vec, anim=True):
    def animation():
        fig, ax = plt.subplots(figsize=(8, 4))
        line_RKII_G, = ax.plot(x, u_diff_RKII_G[0], label='RKII')
        line_RKIII_G, = ax.plot(x, u_diff_RKIII_G[0], label='RKIII')
        line_RKIV, = ax.plot(x, u_diff_RKIV[0], label='RKIV')
        line_RKVI, = ax.plot(x, u_diff_RKVI[0], label='RKVI')
        line_anal, = ax.plot(x, u_diff_anal[0], label='Analítica')
        time_text = ax.text(0.8*L, 0.8*np.max(u_diff_anal[0]), '', fontsize=12)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title('Animación Difusión')
        ax.legend()

        def update(frame):
            line_RKII_G.set_ydata(u_diff_RKII_G[frame])
            line_RKIII_G.set_ydata(u_diff_RKIII_G[frame])
            line_RKIV.set_ydata(u_diff_RKIV[frame])
            line_RKVI.set_ydata(u_diff_RKVI[frame])
            line_anal.set_ydata(u_diff_anal[frame])
            time_text.set_text(f"t = {np.round(t[frame],2)}s")
            return line_RKII_G, line_RKIII_G, line_RKIV, line_RKVI, line_anal, time_text

        ani = FuncAnimation(fig, update, frames=range(0,Nt,10), blit=True, interval=1)
        writervideo = FFMpegWriter(fps=120) 
        # ani.save('difusion.gif', writer=writervideo, savefig_kwargs=dict(facecolor='#f1f1f1')) 
        
        plt.show()
    
    x = np.linspace(0, L, Nx)
    dx = L/(Nx-1)

    t = np.linspace(0, T, Nt)
    dt = T/(Nt-1)
    
    # if estabilidad_diff >= 0.5:
    #     raise Exception('Condición de estabilidad no satisfecha')
    
    u_diff_RKII_G, t_RKII_G = timear(RKII_G,(u0_diff, t, diffusion, params_diff))
    u_diff_RKIII_G, t_RKIII_G = timear(RKIII_G,(u0_diff, t, diffusion, params_diff))
    u_diff_RKIV, t_RKIV = timear(RKIV,(u0_diff, t, diffusion, params_diff))
    u_diff_RKVI, t_RKVI = timear(RKVI,(u0_diff, t, diffusion, params_diff))

    u_diff_anal = np.zeros((Nt,Nx))
    for i in range(Nt):
        u_diff_anal[i,:] = (4*pi*(t[i]+1)*D)**(-0.5)*(exp(-((x-L/3)**2)/(4*D*(t[i]+1))) + 5*exp(-((x-2*L/3)**2)/(4*D*(t[i]+1))))
        
    def diff_anal(Nt):
        t = np.linspace(0, T, Nt)
        u_diff_anal = np.zeros((Nt,Nx))
        for i in range(Nt):
            u_diff_anal[i,:] = (4*pi*(t[i]+1)*D)**(-0.5)*(exp(-((x-L/3)**2)/(4*D*(t[i]+1))) + 5*exp(-((x-2*L/3)**2)/(4*D*(t[i]+1))))
        return u_diff_anal


    ########################################################################################3
    # EVOLUCIÓN DEL ERROR
    ########################################################################################3
    error2 = error(u_diff_anal, u_diff_RKII_G, dx)[1:]    
    error3 = error(u_diff_anal, u_diff_RKIII_G, dx)[1:]
    error4 = error(u_diff_anal, u_diff_RKIV, dx)[1:]
    error6 = error(u_diff_anal, u_diff_RKVI, dx)[1:]

    error2_max = np.max(error2)
    error3_max = np.max(error3)
    error4_max = np.max(error4)
    error6_max = np.max(error6)

    fig, ax = plt.subplots(facecolor='#f1f1f1',figsize=(8, 4))
    ax.plot(t[1:], error2, label='RKII_G')
    ax.plot(t[1:], error3, label='RKIII_G')
    ax.plot(t[1:], error4, label='RKIV')
    ax.plot(t[1:], error6, label='RKVI')
    ax.set_xlabel('t')
    ax.semilogx(), ax.semilogy()
    ax.set_ylim(0.8*np.min(error6), 1.2*np.max(error2))
    ax.set_ylabel('Error')
    ax.set_title('Difusión: evolución del error')
    ax.legend()
    plt.show()

    ########################################################################################3
    # EVOLUCIÓN TIEMPO EJECUCIÓN
    ########################################################################################3
    fig, ax = plt.subplots(ffacecolor='#f1f1f1',figsize=(8, 4))
    ax.scatter(t_RKII_G, error2_max, label='RKII_G', marker='x')
    ax.scatter(t_RKIII_G, error3_max, label='RKIII_G', marker='x')
    ax.scatter(t_RKIV, error4_max, label='RKIV', marker='x')
    ax.scatter(t_RKVI, error6_max, label='RKVI', marker='x')
    ax.set_xlabel('Tiempo de ejecución')
    ax.set_ylabel('Error máximo')
    ax.semilogx()
    ax.set_title('Difusión: error en función del tiempo de ejecución')
    ax.legend()
    plt.show()

    ########################################################################################3
    # EVOLUCIÓN DT
    ########################################################################################3
    rk2, rk3, rk4, rk6 = [], [], [], []
    for Nt0 in Nt_vec:
        u_diff_anal0 = diff_anal(Nt0)
        rk2.append(np.max(error(u_diff_anal0, RKII_G(u0_diff, np.linspace(0, T, Nt0), diffusion, params_diff), dx)))
        rk3.append(np.max(error(u_diff_anal0, RKIII_G(u0_diff, np.linspace(0, T, Nt0), diffusion, params_diff), dx)))
        rk4.append(np.max(error(u_diff_anal0, RKIV(u0_diff, np.linspace(0, T, Nt0), diffusion, params_diff), dx)))
        rk6.append(np.max(error(u_diff_anal0, RKVI(u0_diff, np.linspace(0, T, Nt0), diffusion, params_diff), dx)))
    fig, ax = plt.subplots(facecolor='#f1f1f1',figsize=(8, 4))
    ax.plot(T/(Nt_vec-1), rk2, label=f'RKII_G')
    ax.plot(T/(Nt_vec-1), rk3, label=f'RKIII_G')
    ax.plot(T/(Nt_vec-1), rk4, label=f'RKIV ')
    ax.plot(T/(Nt_vec-1), rk6, label=f'RKVI')
    ax.semilogx(), ax.semilogy()
    ax.set_xlabel('dt')
    ax.set_ylabel('Error')
    ax.set_title('Difusión: error en función del dt')
    ax.legend()
    plt.show()

    if anim:
        animation()

def schrodinger(t, psi, params) -> np.ndarray:
    """
    Derivada temporal para la ecuación de Schrödinger.
    
    Input:
    - t: tiempo actual (no se usa porque es psi independiente de t)
    - psi: np.ndarray, función de onda en el tiempo t
    - params: tuple (hbar, m, dx, V)
    
    Output:
    - dpsi_dt: np.ndarray, derivada de psi con respecto al tiempo
    """
    hbar, m, dx, V = params
    d2psi_dx2 = np.zeros(len(psi), dtype=np.complex128)
    d2psi_dx2[1:-1] = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / dx**2
    d2psi_dx2[0] = d2psi_dx2[-1] = 0  # Dirichlet

    dpsi_dt = -1j * (hbar / (2 * m)) * d2psi_dx2 + (-1j / hbar) * V * psi
    return dpsi_dt
def schrodinger_total(L, T, Nx, Nt, n, u0_schr, params_schr, Nt_vec, anim=True):
    def animation():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(-1.1*np.abs(np.min(u_schr_RKVI)), 1.1*np.abs(np.max(u_schr_RKVI)))
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\phi(x,t)$")
        ax.set_title("Animación Schrödinger")

        line_phi_RKII_G,  = ax.plot([], [], label="$\phi(x,t)$ - RKII")
        line_phi_RKIII_G,  = ax.plot([], [], label="$\phi(x,t)$ - RKIII")
        line_phi_RKIV,  = ax.plot([], [], label="$\phi(x,t)$ - RKIV")
        line_phi_RKVI,  = ax.plot([], [], label="$\phi(x,t)$ - RKVI")
        line_phi_anal,  = ax.plot([], [], label="$\phi(x,t)$ - analítica")
        time_text = ax.text(0.8*L, 0.8*np.max(u_schr_RKVI), '', fontsize=12)
        ax.legend(loc='upper left')

        def update(frame):
            line_phi_RKII_G.set_data(x, u_schr_RKII_G[frame])
            line_phi_RKIII_G.set_data(x, u_schr_RKIII_G[frame])
            line_phi_RKIV.set_data(x, u_schr_RKIV[frame])
            line_phi_RKVI.set_data(x, u_schr_RKVI[frame])
            line_phi_anal.set_data(x, u_schr_anal[frame])
            time_text.set_text(f"t = {np.round(t[frame],2)}s")
            return line_phi_RKII_G, line_phi_RKIII_G, line_phi_RKIV, line_phi_RKVI, line_phi_anal, time_text

        ani = FuncAnimation(fig, update, frames=range(0,Nt,5), blit=True, interval=1)
        writervideo = FFMpegWriter(fps=120) 
        # ani.save('schrodinger.gif', writer=writervideo, savefig_kwargs=dict(facecolor='#f1f1f1')) 
        plt.show()
    
    x = np.linspace(0, L, Nx)
    dx = L/(Nx-1)

    t = np.linspace(0, T, Nt)
    dt = T/(Nt-1)

    sol_RKII_G, t_RKII_G = timear(RKII_G,(u0_schr, t, schrodinger, params_schr))
    sol_RKIII_G, t_RKIII_G = timear(RKIII_G,(u0_schr, t, schrodinger, params_schr))
    sol_RKIV, t_RKIV = timear(RKIV,(u0_schr, t, schrodinger, params_schr))
    sol_RKVI, t_RKVI = timear(RKVI,(u0_schr, t, schrodinger, params_schr))

    u_schr_RKII_G = np.real(sol_RKII_G[:,:])   # Campo phi
    u_schr_RKIII_G = np.real(sol_RKIII_G[:,:])   # Campo phi
    u_schr_RKIV  = np.real(sol_RKIV[:,:])       # Campo phi
    u_schr_RKVI  = np.real(sol_RKVI[:,:])   # Campo phi

    u_schr_anal = np.zeros((Nt,Nx))
    for i in range(Nt):
        u_schr_anal[i,:] = np.sin(n*np.pi*x/L)*np.cos(-((n**2)*(np.pi**2)*t[i])/(2*L**2))
        
    def schr_anal(Nt):
        t = np.linspace(0, T, Nt)
        u_schr_anal0 = np.zeros((Nt,Nx))
        for i in range(Nt):
            u_schr_anal0[i,:] = np.sin(n*np.pi*x/L)*np.cos(-((n**2)*(np.pi**2)*t[i])/(2*L**2))
        return u_schr_anal0
        
    ########################################################################################3
    # EVOLUCIÓN DEL ERROR
    ########################################################################################3
    error2 = error(u_schr_anal, u_schr_RKII_G, dx)
    error3 = error(u_schr_anal, u_schr_RKIII_G, dx)
    error4 = error(u_schr_anal, u_schr_RKIV, dx)
    error6 = error(u_schr_anal, u_schr_RKVI, dx)
    
    error2_max = np.max(error2)
    error3_max = np.max(error3)
    error4_max = np.max(error4)
    error6_max = np.max(error6)

    fig,ax = plt.subplots(facecolor='#f1f1f1',figsize=(8, 4))
    ax.set_title('Schrödinger: evolución del error')
    ax.plot(t, error2, label='RKII_G')
    ax.plot(t, error3, label='RKIII_G')
    ax.plot(t, error4, label='RKIV')
    ax.plot(t, error6, label='RKVI')
    ax.set_xlabel('t')
    ax.semilogx(), ax.semilogy()
    ax.legend()
    plt.show()

    ########################################################################################3
    # ERROR TIEMPO EJECUCIÓN
    ########################################################################################3
    fig, ax = plt.subplots(facecolor='#f1f1f1',figsize=(8, 4))
    ax.scatter(t_RKII_G, error2_max, label='RKII_G', marker='x')
    ax.scatter(t_RKIII_G, error3_max, label='RKIII_G', marker='x')
    ax.scatter(t_RKIV, error4_max, label='RKIV', marker='x')
    ax.scatter(t_RKVI, error6_max, label='RKVI', marker='x')
    ax.set_xlabel('Tiempo de ejecución')
    ax.set_ylabel('Error máximo')
    ax.semilogx(), ax.semilogy()
    ax.set_title('Schrödinger: error en función del tiempo de ejecución')
    ax.legend()
    plt.show()

    ########################################################################################3
    # ERROR DT
    ########################################################################################3
    rk2, rk3, rk4, rk6 = [], [], [], []
    for Nt0 in Nt_vec:
        u_schr_anal0 = schr_anal(Nt0)
        rk2.append(np.max(error(u_schr_anal0, np.real(RKII_G(u0_schr, np.linspace(0, T, Nt0), schrodinger, params_schr)), dx)))
        rk3.append(np.max(error(u_schr_anal0, np.real(RKIII_G(u0_schr, np.linspace(0, T, Nt0), schrodinger, params_schr)), dx)))
        rk4.append(np.max(error(u_schr_anal0, np.real(RKIV(u0_schr, np.linspace(0, T, Nt0), schrodinger, params_schr)), dx)))
        rk6.append(np.max(error(u_schr_anal0, np.real(RKVI(u0_schr, np.linspace(0, T, Nt0), schrodinger, params_schr)), dx)))
    fig, ax = plt.subplots(facecolor='#f1f1f1',figsize=(8, 4))
    ax.plot(T/(Nt_vec-1), rk2, label=f'RKII_G')
    ax.plot(T/(Nt_vec-1), rk3, label=f'RKIII_G')
    ax.plot(T/(Nt_vec-1), rk4, label=f'RKIV ')
    ax.plot(T/(Nt_vec-1), rk6, label=f'RKVI')
    ax.set_xlabel('dt')
    ax.set_ylabel('Error')
    ax.set_title('Schrödinger: error en función del dt')
    ax.legend()
    plt.show()
    
    if anim:
        animation()

def klein_gordon(t, state, params) -> np.ndarray:
    """
    Derivada temporal para la ecuación de Klein-Gordon.
    
    Input:
    - t: tiempo actual (no se usa en este caso)
    - phi: np.ndarray, campo en el tiempo t
    - params: tuple (c, m, dx)
    
    Output:
    - dphi_dt: np.ndarray, derivada de phi con respecto al tiempo
    """
    c, m, dx = params
    phi, dphi_dt = state
    d2phi_dx2 = np.zeros(len(phi))
    d2phi_dx2[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx**2
    d2phi_dx2[0] = d2phi_dx2[-1] = 0

    dphi_dt = dphi_dt
    d2phi_dt2 = c**2 * d2phi_dx2 - m**2 * phi

    return np.array([dphi_dt, d2phi_dt2])
def klein_gordon_total(L, T, Nx, Nt, u0_kg, params_kg, Nt_vec, anim=True):
    def animation():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(-1.1*np.abs(np.min(du_kg_RKIV)), 1.1*np.abs(np.max(du_kg_RKIV)))
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\phi(x,t)$")
        ax.set_title("Animación Klein-Gordon")

        line_phi_RKII_G,  = ax.plot([], [], label="$\phi(x,t)$ - RKII")
        line_dphi_RKII_G, = ax.plot([], [], label="$\dfrac{d\phi}{dt}(x,t)$ - RKII")
        
        line_phi_RKIII_G,  = ax.plot([], [], label="$\phi(x,t)$ - RKIII")
        line_dphi_RKIII_G, = ax.plot([], [], label="$\dfrac{d\phi}{dt}(x,t)$ - RKIII")

        line_phi_RKIV,  = ax.plot([], [], label="$\phi(x,t)$ - RKIV")
        line_dphi_RKIV, = ax.plot([], [], label="$\dfrac{d\phi}{dt}(x,t)$ - RKIV")
        
        line_phi_RKVI,  = ax.plot([], [], label="$\phi(x,t)$ - RKVI")
        line_dphi_RKVI, = ax.plot([], [], label="$\dfrac{d\phi}{dt}(x,t)$ - RKVI")
        
        time_text = ax.text(0.8*L, 0.77*np.max(u_kg_RKIV), '', fontsize=12)
        ax.legend(loc='lower center', ncol=4)

        def update(frame):
            line_phi_RKII_G.set_data(x, u_kg_RKII_G[frame])
            line_dphi_RKII_G.set_data(x, du_kg_RKII_G[frame])
            
            line_phi_RKIII_G.set_data(x, u_kg_RKIII_G[frame])
            line_dphi_RKIII_G.set_data(x, du_kg_RKIII_G[frame])
            
            line_phi_RKIV.set_data(x, u_kg_RKIV[frame])
            line_dphi_RKIV.set_data(x, du_kg_RKIV[frame])
            
            line_phi_RKVI.set_data(x, u_kg_RKVI[frame])
            line_dphi_RKVI.set_data(x, du_kg_RKVI[frame])
            
            time_text.set_text(f"t = {np.round(t[frame],2)}s")
            return line_phi_RKII_G, line_dphi_RKII_G, line_phi_RKIII_G, line_dphi_RKIII_G, line_phi_RKIV, line_dphi_RKIV, line_phi_RKVI, line_dphi_RKVI, time_text

        ani = FuncAnimation(fig, update, frames=range(0,Nt,25), blit=True, interval=1)
        writervideo = FFMpegWriter(fps=120) 
        # ani.save('klein_gordon.gif', writer=writervideo, savefig_kwargs=dict(facecolor='#f1f1f1')) 
        plt.show()

    x = np.linspace(0, L, Nx)
    dx = L/(Nx-1)

    t = np.linspace(0, T, Nt)
    dt = T/(Nt-1)
    
    sol_RKII_G = RKII_G(u0_kg, t, klein_gordon, params_kg)
    u_kg_RKII_G  = sol_RKII_G[:,0,:] # Campo phi
    du_kg_RKII_G = sol_RKII_G[:,1,:] # Velocidad dphi/dt
    
    sol_RKIII_G = RKIII_G(u0_kg, t, klein_gordon, params_kg)
    u_kg_RKIII_G  = sol_RKIII_G[:,0,:] # Campo phi
    du_kg_RKIII_G = sol_RKIII_G[:,1,:] # Velocidad dphi/dt
    
    sol_RKIV = RKIV(u0_kg, t, klein_gordon, params_kg)
    u_kg_RKIV  = sol_RKIV[:,0,:] # Campo phi
    du_kg_RKIV = sol_RKIV[:,1,:] # Velocidad dphi/dt
    
    sol_RKVI = RKVI(u0_kg, t, klein_gordon, params_kg)
    u_kg_RKVI  = sol_RKVI[:,0,:] # Campo phi
    du_kg_RKVI = sol_RKVI[:,1,:] # Velocidad dphi/dt

    if anim:
        animation()