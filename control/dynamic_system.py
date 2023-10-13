import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import random as rn

# Sistema masa resorte comun y corriente, con la masa sobre una superficie con friccion 
class MassDamper:
    def __init__(self):
        self.m = 2 # En kg
        self.k = 0.5 # Constante del resorte
        self.c = 2 # Coeficiente de friccion
        self.n = 2000 # Numero de ms por simulacion
        self.s_t = 1000 # Numero de sample times, o muestreos
        self.x0 = 0 # Posicion inicial
        self.y0 = 0 # Velocidad inicial
        self.step_interval = 100 # En ms
        self.last_update_time = -self.step_interval
        self.U = 0 # Entrada del sistema

    def update_force(self, t):
        if t-self.last_update_time >= self.step_interval:
            self.U = rn.uniform(-1, 1)
            self.last_update_time = t

    def system_equation(self, t, y):
        x, v = y # Posicion actual y velocidad del sistema
        dxdt = v
        dvdt = -(self.k/self.m)*x - (self.c/self.m)*v + self.U/self.m
        return (dxdt, dvdt)

    def run_simulation(self):
        y0 = [self.x0, self.y0]
        t_span = [0, self.n] # El vector de tiempo donde vamos a simular
        sol = solve_ivp(self.system_equation, t_span, y0, dense_output=True)
        t = np.linspace(0, self.n, num=self.s_t)
        x, v = sol.sol(t)
        plt.figure(figsize=[12, 6])
        plt.xlabel('Time [ms]')
        plt.ylabel('Value')
        plt.title('Mass damper simulation system')
        plt.grid(True)
        position_line, = plt.plot([], [], label='Position x') # , = significa que se va a estar actualizando
        velocity_line, = plt.plot([], [], label='Velocity x')
        input_line, = plt.plot([], [], label='System input [U]')
        plt.legend()
        x = np.zeros(len(t))
        v = np.zeros(len(t))
        U = np.zeros(len(t))
        for i in range(1, len(t)):
            self.update_force(t[i-1])
            U[i-1] = self.U
            y0 = [x[i-1], v[i-1]]
            t_span = (t[i-1], t[i])
            sol = solve_ivp(self.system_equation, t_span, y0, dense_output=True)
            x[i], v[i] = sol.sol(t[i])
            position_line.set_data(t[:i+1], x[:i+1])
            velocity_line.set_data(t[:i+1], v[:i+1])
            input_line.set_data(t[:i+1], U[:i+1])
            plt.xlim(0, t[i])
            plt.ylim(min(min(x), min(v), min(U)) - 0.5, max(max(x), max(v), max(U)) + 0.5)
            plt.pause(self.n/self.s_t)
        plt.show()

s = MassDamper()
s.run_simulation()