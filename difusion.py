import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp, sin, cos
from matplotlib.animation import FuncAnimation

# Locales
from runge_kutta import *
from edp import *
from plot_func import *

def diffusion_total(L, T, Nx, Nt, D, u0_diff, params_diff, Nt_vec, anim=True):
    def animation():
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.tight_layout()
        line_RKII_G, = ax.plot(x, u_diff_RKII_G[0], label='RKIIG')
        line_RKIII_G, = ax.plot(x, u_diff_RKIII_G[0], label='RKIIIG')
        line_RKIV, = ax.plot(x, u_diff_RKIV[0], label='RKIV')
        line_RKVI, = ax.plot(x, u_diff_RKVI[0], label='RKVI')
        line_anal, = ax.plot(x, u_diff_anal[0], label='Analítica')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title('Evolución temporal de la Ecuación de Difusión')
        ax.legend()

        def update(frame):
            line_RKII_G.set_ydata(u_diff_RKII_G[frame])
            line_RKIII_G.set_ydata(u_diff_RKIII_G[frame])
            line_RKIV.set_ydata(u_diff_RKIV[frame])
            line_RKVI.set_ydata(u_diff_RKVI[frame])
            line_anal.set_ydata(u_diff_anal[frame])
            return line_RKII_G, line_RKIII_G, line_RKIV, line_RKVI, line_anal

        ani = FuncAnimation(fig, update, frames=range(0,Nt0,1), blit=True, interval=50)
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

    error2 = error(u_diff_anal, u_diff_RKII_G, dx)[1:]    
    error3 = error(u_diff_anal, u_diff_RKIII_G, dx)[1:]
    error4 = error(u_diff_anal, u_diff_RKIV, dx)[1:]
    error6 = error(u_diff_anal, u_diff_RKVI, dx)[1:]

    error2_max = np.max(error2)
    error3_max = np.max(error3)
    error4_max = np.max(error4)
    error6_max = np.max(error6)

    fig, ax = plt.subplots()
    ax.plot(t[1:], error2, label='RKII_G')
    ax.plot(t[1:], error3, label='RKIII_G')
    ax.plot(t[1:], error4, label='RKIV')
    ax.plot(t[1:], error6, label='RKVI')
    ax.set_xlabel('t')
    ax.semilogx(), ax.semilogy()
    ax.set_ylim(0.8*np.min(error6), 1.2*np.max(error2))
    ax.set_ylabel('Error')
    ax.set_title('Error de la solución numérica frente a la analítica')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(t_RKII_G, error2_max, label='RKII_G', marker='x')
    ax.scatter(t_RKIII_G, error3_max, label='RKIII_G', marker='x')
    ax.scatter(t_RKIV, error4_max, label='RKIV', marker='x')
    ax.scatter(t_RKVI, error6_max, label='RKVI', marker='x')
    ax.set_xlabel('Tiempo de ejecución')
    ax.set_ylabel('Error máximo')
    ax.semilogx(), ax.semilogy()
    ax.set_title('Error de la solución numérica frente a la analítica')
    ax.legend()
    plt.show()

    rk2, rk3, rk4, rk6 = [], [], [], []
    for Nt in Nt_vec:
        u_diff_anal0 = diff_anal(Nt)
        rk2.append(np.max(error(u_diff_anal0, RKII_G(u0_diff, np.linspace(0, T, Nt), diffusion, params_diff), dx)))
        rk3.append(np.max(error(u_diff_anal0, RKIII_G(u0_diff, np.linspace(0, T, Nt), diffusion, params_diff), dx)))
        rk4.append(np.max(error(u_diff_anal0, RKIV(u0_diff, np.linspace(0, T, Nt), diffusion, params_diff), dx)))
        rk6.append(np.max(error(u_diff_anal0, RKVI(u0_diff, np.linspace(0, T, Nt), diffusion, params_diff), dx)))
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

    if anim:
        animation()