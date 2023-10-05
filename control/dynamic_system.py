import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas
import random

class MassDamper:
    def __init__(self) :
        ##System Mass
        self.m = 2#Kg
        ##String constant
        self.k = 0.5
        ##Friccion coefficent
        self.c = 2
        ##Number of ms for simuation
        self.N =2000 #[ms]
        ##Number of samples times
        self.s_t = 1000
        ##Initial position
        self.x0 = 0
        ##initial velocity
        self.v0 = 0
        ##Sample time
        self.step_interval = 100 ##ms
        #Lasta update time for simulation
        self.last_update_time = -self.step_interval
        ##system input
        self.U = 0

    def update_force(self,t):
        if t-self.last_update_time >= self.step_interval:
            self.U = random.uniform(-1,1) ##Fuerza aplicada
            self.last_update_time = t

    def system_equation(self,t,y):
        x ,v = y ##actual position and velocity of system
        dxdt = v
        dvdt = -(self.k/self.m)*x - (self.c/self.m)*v + self.U/self.m
        return [dxdt,dvdt]

    def run_simulation(self):
        ##set initial conditions and simulation time
        y0 = [self.x0,self.v0]
        t_span = [0,self.N]


        ##solve the system equations
        sol = solve_ivp(self.system_equation,t_span,y0,dense_output = True)

        ##create time points for simulation
        t = np.linspace(0,self.N, num= self.s_t)

        ##get the position and velocity data for the times points
        x, v = sol.sol(t)

        ##setup the plot for visualizing the system output in real time
        plt.figure(figsize=(12,6))
        plt.xlabel("Time [ms]")
        plt.ylabel("Value")
        plt.title("Mass Damper Simulation System")
        plt.grid(True)

        ##initialize lines for position , velocity and U
        position_line, = plt.plot([],[],label="Position (X)")
        velocity_line, = plt.plot([],[],label="Velocity (y)")
        input_line, = plt.plot([],[], label="System Input (U)")

        ##lets show graph legeng
        plt.legend()

        ##initialize position and velocity vectpr and U
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        U = np.zeros(len(t))

        ##Lets simulate the system for the simulation time
        for i in range(1,len(t)):
            ##if we want to identify the system last update the force
            self.update_force(t[i-1])


            #Store the system input
            U[i-1] = self.U

            #
            y0 = [x[i-1],v[i-1]]
            t_span = [t[i-1],t[i]]

            ##solve the system equation for this iteration
            sol = solve_ivp(self.system_equation,t_span,y0,dense_output=True)

            ##get the position and velocity for the next sample time
            x[i], v[i] = sol.sol(t[i])

            ##lets update the graph lines for showing system status 
            position_line.set_data(t[:i+1], x[:i+1])
            velocity_line.set_data(t[:i+1], v[:i+1])
            input_line.set_data(t[:i+1], U[:i+1])

            ##lets show the data up until actual samples time 
            plt.xlim(0, t[i])
            plt.ylim(min(min(x),min(v),min(U)) - 0.5,max(max(x),max(v),max(U)) + 0.5)

            ##Lets pause the graph
            plt.pause(self.N/self.s_t)
        plt.show()

S=MassDamper()
S.run_simulation()








