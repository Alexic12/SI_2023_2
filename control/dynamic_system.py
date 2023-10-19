import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import random as rn
import openpyxl
import os
import random

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
        ##error acummulative for PID integral part
        self.error_sum = 0
        #previous error for calculation of change in the error between iterations
        self.error_previous = 0
        self.arreglo = []
        self.workbook = openpyxl.Workbook()
        self.sheet = self.workbook.active
        ##las value for filling sp arr
        self.last_value = 0

    def update_force(self, t, type, force):
        if type == 'random':
            if t-self.last_update_time >= self.step_interval:
                self.U = rn.uniform(-1, 1)
                self.last_update_time = t
        elif type == 'external':
            self.U = force

    def system_equation(self, t, y):
        x, v = y # Posicion actual y velocidad del sistema
        dxdt = v
        dvdt = -(self.k/self.m)*x - (self.c/self.m)*v + self.U/self.m
        return (dxdt, dvdt)

    def fill_sp(self, size, min_val, max_val, interval):
        arr = np.zeros(size)
        last_update_time = 0
        for i in range(size):
            if i - last_update_time >= interval:
                self.last_value = random.uniform(min_val, max_val)
                last_update_time = i
            arr[i] = self.last_value
        return arr

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
        sp_line, = plt.plot([],[],label='System SetPoint (SP)')
        plt.legend()
        x = np.zeros(len(t))
        v = np.zeros(len(t))
        U = np.zeros(len(t))
        sp_a = np.zeros(len(t))
        sp_a = self.fill_sp(len(sp_a), -1, 1, 100)
        indice = np.linspace(1, len(t), len(t))
        for i in range(1, len(t)):
            #self.update_force(t[i-1])
            self.pid_control(x[i-1], sp_a[i-1])
            U[i-1] = self.U
            y0 = [x[i-1], v[i-1]]
            t_span = (t[i-1], t[i])
            sol = solve_ivp(self.system_equation, t_span, y0, dense_output=True)
            x[i], v[i] = sol.sol(t[i])
            position_line.set_data(t[:i+1], x[:i+1])
            velocity_line.set_data(t[:i+1], v[:i+1])
            input_line.set_data(t[:i+1], U[:i+1])
            sp_line.set_data(t[:i+1], sp_a[:i+1])
        plt.xlim(0, t[i])
        plt.ylim(min(min(x), min(v), min(U)) - 0.5, max(max(x), max(v), max(U)) + 0.5)
        #plt.pause(self.n/self.s_t)
        self.arreglo = np.vstack((indice, sp_a, x, U))
        for fila, fila_datos in enumerate(self.arreglo, start=1):
            for columna, valor in enumerate(fila_datos, start=1):
                self.sheet.cell(row=fila, column=columna, value=valor)

        # Guardar el archivo Excel
        directorio_actual = os.path.abspath(os.path.dirname(__file__))
        ruta_excel = os.path.join(directorio_actual, "..", "data", "dynamic_system_dates.xlsx")
        self.workbook.save(ruta_excel)
        #print('Arreglo'+str(self.arreglo.shape))
        plt.show()

    def pid_control(self, x, sp):
        ## Dynamic system output
        kp = 1
        ki = 0.1
        kd = 0.5
        setpoint = sp
        error = setpoint - x
        error_derivative = error - self.error_previous
        self.error_sum += error
        self.U = kp*error +ki*self.error_sum+kd*error_derivative
        self.error_previous = error

s = MassDamper()
s.run_simulation()