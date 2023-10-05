import numpy as np
import pandas as pd 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random


class MassDamper:
    
    def __init__(self):
        
        self.m = 2 ##masa del objeto
        
        self.k = 0.5 ##constante del resorte
        
        self.c = 2 ##coeficiente de friccion
        
        self.n = 2000 ##numero de tiempo por simulacion
        
        self.s_t = 1000 ##tiempo de muestreo 
        
        self.x0 = 0 ##posicion inicial
        
        self.v0 = 0 ##velocidad inicial
        
        self.step_interval = 100 ##tiempo de muestreo en ms
        
        self.last_update_time = -self.step_interval ##ultima actualizacion de tiempo para la simulacion
        
        self.U = 0 ##system input
        
        
    def update_force(self, t):
        
        if t-self.last_update_time >= self.step_interval:
            
            self.U = random.uniform(-1,1)
            
            self.last_update_time = t
            
    def system_equiations(self, t, y):
        
        x, v = y ##posicion y velocidad actual 
        
        dxdt = v
        
        dvdt = -((self.k/self.m)*x - (self.c/self.m)*v + (self.U/self.m))
        
        return [dxdt, dvdt]
    
    def run_simulation(self):
        
        ##set initial condition and simulation time
        
        y0 =[self.x0, self.v0]
        
        time_span = [0, self.n]
        
        ##resolver las ecuaciones del sistema
        
        sol = solve_ivp(self.system_equiations, time_span, y0, dense_output = True)
        
        ##create time points for simulation 
        
        t = np.linspace(0, self.n, num = self.s_t)
        
        ##obtener posicion y velocidad para todos los puntos en el tiempo 
        
        x, v = sol.sol(t)
        
        plt.figure(figsize = (12,6))
        
        plt.xlabel('Tiempo [ms]')
        
        plt.ylabel('Value')
        
        plt.title('mass damper simulation system')
        
        plt.grid()
        
        ##initial lines for posicion, velocity and U
        
        position_line, = plt.plot([], [], label = 'position (x)')
        
        velocity_line, = plt.plot([], [], label = 'Velocity (v)')
        
        input_line, = plt.plot([], [], label = 'system input (u)')
        
        plt.legend()
        
        ##initial position and velocity vector
        
        x = np.zeros(len(t))
        
        v = np.zeros(len(t))
        
        U = np.zeros(len(t))
        
        ##lets simulate the system for the simulation time
        
        for i in range(1, len(t)):
            
            ##if we want to identify the system lets update the force
            
            self.update_force(t[i-1])
            
            ##store the system input
            
            U[i-1] = self.U
            
            ##set initial conditions and timespan
            
            y0 = [x[i-1], v[i-1]] 
            
            time_span = [t[i-1], t[i]]  
            
            ##solve the system equations for this iteration
            
            sol = solve_ivp(self.system_equiations, time_span, y0, dense_output = True)
            
            ##get the pisition and velocity for the nex sample time
            
            x[i], v[i] = sol.sol(t[i])
            
            ##lets update the graph lines for showing system status
            
            position_line.set_data(t[i+1], x[i+1])
            
            velocity_line.set_data(t[i+1], v[i+1])
            
            input_line.set_data(t[i+1], U[i+1])
            
            ##lets show the data up until the actual sample time
            
            plt.xlim(0, t[i])
            
            plt.ylim(min(min(x), min(v), min(U)) - 0.5, max(max(x), max(v), max(U)) + 0.5)
            
            ##lets pause the graph
            
            plt.pause(self.n/self.s_t)
            
        plt.show()  
            
            
s = MassDamper()

s.run_simulation()               