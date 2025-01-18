import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp, sin, cos
from matplotlib.animation import FuncAnimation

# Locales
from runge_kutta import *
from edp import *
from plot_func import *

def schrodinger_total(L, T, Nx, Nt, n, u0_schr, params_schr, Nt_vec, plot=True):
    def animation():
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(-1.1*np.abs(np.min(u_schr_RKVI)), 1.1*np.abs(np.max(u_schr_RKVI)))
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\phi(x,t)$")
        ax.set_title("Evolución de Schrodinger $\\phi(x, t)$")
        ax1.plot(t, error6, label='error RKVI')
        ax1.set_ylabel('Error')
        ax1.legend()

        line_phi_RKVI,  = ax.plot([], [], label="phi(x,t) - RKVI")
        line_phi_anal,  = ax.plot([], [], label="phi(x,t) - analítica")
        time_text = ax.text(0.8*L, 0.8*np.max(u_schr_RKVI), '', fontsize=12)
        ax.legend()

        def update(frame):
            line_phi_RKVI.set_data(x, u_schr_RKVI[frame])
            line_phi_anal.set_data(x, u_schr_anal[frame])
            time_text.set_text(f"t = {np.round(t[frame],2)}s")
            return line_phi_RKVI, line_phi_anal, time_text

        ani = FuncAnimation(fig, update, frames=range(0,Nt,25), blit=True, interval=1)
        plt.show()
    
    x = np.linspace(0, L, Nx)
    dx = L/(Nx-1)

    t = np.linspace(0, T, Nt)
    dt = T/(Nt-1)

    sol_RKII_G = RKII_G(u0_schr, t, schrodinger, params_schr)
    sol_RKIII_G = RKIII_G(u0_schr, t, schrodinger, params_schr)
    sol_RKIV = RKIV(u0_schr, t, schrodinger, params_schr)
    sol_RKVI = RKVI(u0_schr, t, schrodinger, params_schr)

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
        
    error2 = error(u_schr_anal, u_schr_RKII_G, dx)
    error3 = error(u_schr_anal, u_schr_RKIII_G, dx)
    error4 = error(u_schr_anal, u_schr_RKIV, dx)
    error6 = error(u_schr_anal, u_schr_RKVI, dx)

    fig,ax = plt.subplots()
    ax.plot(t, error2, label='RKII_G')
    ax.plot(t, error3, label='RKIII_G')
    ax.plot(t, error4, label='RKIV')
    ax.plot(t, error6, label='RKVI')
    ax.set_xlabel('t')
    ax.semilogx(), ax.semilogy()
    ax.legend()
    plt.show()

    rk2, rk3, rk4, rk6 = [], [], [], []
    for Nt in Nt_vec:
        u_schr_anal0 = schr_anal(Nt)
        rk2.append(np.max(error(u_schr_anal0, RKII_G(u0_schr, np.linspace(0, T, Nt), schrodinger, params_schr), dx)))
        rk3.append(np.max(error(u_schr_anal0, RKIII_G(u0_schr, np.linspace(0, T, Nt), schrodinger, params_schr), dx)))
        rk4.append(np.max(error(u_schr_anal0, RKIV(u0_schr, np.linspace(0, T, Nt), schrodinger, params_schr), dx)))
        rk6.append(np.max(error(u_schr_anal0, RKVI(u0_schr, np.linspace(0, T, Nt), schrodinger, params_schr), dx)))
    fig, ax = plt.subplots()
    ax.plot(T/(Nt_vec-1), rk2, label=f'RKII_G')
    ax.plot(T/(Nt_vec-1), rk3, label=f'RKIII_G')
    ax.plot(T/(Nt_vec-1), rk4, label=f'RKIV ')
    ax.plot(T/(Nt_vec-1), rk6, label=f'RKVI')
    ax.semilogx(), ax.semilogy()
    ax.set_xlabel('dt')
    ax.set_ylabel('Error')
    ax.set_title('Error de la solución numérica frente a la analítica')
    ax.legend()
    plt.show()
    
    if plot:
        animation()
