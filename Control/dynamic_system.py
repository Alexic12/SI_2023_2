import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random

class MassDamper:

    def __init__(self):

        self.m = 2  # Masa del objeto 
        
        self.k = 0.2  # Constante del resorte
        
        self.c = 2  # Coeficiente de fricción
        
        self.n = 2000  # Número de tiempo por simulación
        
        self.s_t = 1000  # Tiempo de muestreo
        
        self.x0 = 0  # Posición inicial
        
        self.v0 = 0  # Velocidad inicial
        
        self.step_interval = 100  # Tiempo de muestreo en ms
        
        self.last_update_time = -self.step_interval  # Última actualización de tiempo para la simulación
        
        self.U = 0  # System input

        self.error_integral = 0 #error acumulativo para el PID parte integral

        self.error_previous = 0 #error previo 

    def update_force(self, t, tipo, force):

        if tipo == 'random':

            if t - self.last_update_time >= self.step_interval:
                
                self.U = random.uniform(-1, 1)
                
                self.last_update_time = t

            elif tipo == 'external':

                self.U = force

    def system_equations(self, t, y):
        
        x, v = y  # Posición y velocidad actual
        
        dxdt = v
        
        dvdt = -(self.k / self.m) * x - (self.c / self.m) * v + (self.U / self.m)
        
        return [dxdt, dvdt]

    def run_simulation(self):

        # Establecer condiciones iniciales y tiempo de simulación
        y0 = [self.x0, self.v0]
        
        time_span = [0, self.n]

        # Resolver las ecuaciones del sistema
        sol = solve_ivp(self.system_equations, time_span, y0, dense_output=True)

        # Crear puntos de tiempo para la simulación
        t = np.linspace(0, self.n, num=self.s_t)

        # Obtener posición y velocidad para todos los puntos en el tiempo
        x, v = sol.sol(t)

        plt.figure(figsize=(12, 6))
        
        plt.xlabel('Tiempo [ms]')
        
        plt.ylabel('Valor')
        
        plt.title('Simulación de sistema masa-amortiguador')
        
        plt.grid()

        # Inicializar líneas para posición, velocidad y U
        position_line, = plt.plot([], [], label='Posición (x)')
        
        velocity_line, = plt.plot([], [], label='Velocidad (v)')
        
        input_line, = plt.plot([], [], label='Entrada del sistema (u)')

        plt.legend()

        # Inicializar vectores de posición, velocidad y U
        x_data = np.zeros(len(t))
        
        v_data = np.zeros(len(t))
        
        U_data = np.zeros(len(t))

        # Simular el sistema durante el tiempo de simulación
        for i in range(1, len(t)):
            # Actualizar la fuerza si queremos identificar el sistema
            ##self.update_force(t[i - 1])
            
            self.pid_control(x[i-1], 2)


            # Almacenar la entrada del sistema
            U_data[i - 1] = self.U

            # Establecer condiciones iniciales y tiempo de simulación
            y0 = [x_data[i - 1], v_data[i - 1]]
            
            time_span = [t[i - 1], t[i]]

            # Resolver las ecuaciones del sistema para esta iteración
            sol = solve_ivp(self.system_equations, time_span, y0, dense_output=True)

            # Obtener la posición y velocidad para el próximo tiempo de muestreo
            x_data[i], v_data[i] = sol.sol(t[i])

            # Actualizar las líneas del gráfico para mostrar el estado del sistema
            position_line.set_data(t[:i + 1], x_data[:i + 1])
            
            velocity_line.set_data(t[:i + 1], v_data[:i + 1])
            
            input_line.set_data(t[:i + 1], U_data[:i + 1])

            # Mostrar datos hasta el tiempo de muestreo actual
            plt.xlim(0, t[i])
            
            plt.ylim(min(x_data[:i + 1]) - 0.5, max(x_data[:i + 1]) + 0.5)

            # Pausar el gráfico brevemente
            plt.pause(self.n/self.s_t)


        plt.show()


    def pid_control(self, x, sp):

        ##dynamic sustem output
        ## x system position

        Kp = 15

        Ki = 0.05

        Kd = 0.1

        sp = 1

        error = sp - x

        self.error_integral += error

        error_derivativo = error - self.error_previous

        self.U = Kp*error + Ki*self.error_integral + Kd*error_derivativo

        ##We store the previous value of the error

        self.error_previous = error



s = MassDamper()

s.run_simulation()