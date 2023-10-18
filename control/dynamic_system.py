import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import random 

class MassDamper :
    def __init__(self) :
        #System mass
        self.m = 2 ##[kg]
        ##String constant
        self.k = 0.5
        ##Friction coefficient
        self.c = 2

        ##Number of seconds for simulation 
        self.N = 2000 #[ms]

        ##number of sample times
        self.s_t = 1000
        ##Initial position
        self.x0 = 0
        ##Initial velocity
        self.v0 = 0
        ## sample time
        self.step_interval = 100 ##[ms]
        ##  last update time for simulation
        self.last_update_time = -self.step_interval

        ##System input
        self.U = 0

        ##error acumulative for PID integral part
        self.error_sum = 0
        self.error_previous = 0

    def update_force (self, t,type,force) :

        if type == 'random':
            if t-self.last_update_time >= self.step_interval :
                self.U = random.uniform (-1,1) ## fuerza aplicada
                self.last_update_time = t
        elif typ == 'external':
            self.U = force



    def system_equations(self, t, y):
        x,v = y ##actual position and velocity of system 
        dxdt = v
        dvdt = -(self.k/self.m)* x -(self.c/self.m) * v + self.U / self.m
        return [dxdt, dvdt]
    
    def run_simulation(self):
        ## set initial conditions and simulation time
        y0 = [self.x0,self.v0]
        t_span = [0,self.N]

        ##solve the system equation
        sol = solve_ivp(self.system_equations, t_span, y0, dense_output=True)

        ##create time points for simulation
        t = np.linspace(0,self.N, num=self.s_t)

        ##get the position and velocity data for the time points 
        x,v = sol.sol(t)
        ##setup the plot for visualizing the system output in real time
        plt.figure(figsize=(12,6))
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.title('Mass damper Simulation system')
        plt.grid(True)


        ##Initialize lines fo position, velocity and U
        position_line, = plt.plot([],[],label='Position in (X)') ##la coma porque es dinamico y para que se vaya actualizando 
        velocity_line, = plt.plot([],[],label='Velocity in (V)')
        input_line, = plt.plot([],[], label = 'System input (U)')


        ##lets show graph legend
        plt.legend()


        ##initialize position and velocity vector
        x = np.zeros(len(t))
        v = np.zeros(len(t))

        U = np.zeros(len(t))

        ## Lets simulate the system for the simulation time
        for i in range (1, len(t)):
            ##if we want to identify the system lets update the force
            ##self.update_force(t[i-1])
            self.pid_control(x[i-1], 2)

            U[i-1] = self.U

            ##
            y0 = [x[i-1], v[i-1]]

            t_span = [t[i-1], t[i]]

            ##Solve the system equations for this iteration
            sol = solve_ivp(self.system_equations, t_span, y0, dense_output=True)

            ##get the position and Velocity for the next sample time
            x[i], v[i] = sol.sol(t[i])

            ## lets update the graph lines for showing system status
            position_line.set_data(t[:i+1], x[:i+1])
            velocity_line.set_data(t[:i+1],v[:i+1])
            input_line.set_data(t[:i+1], U[:i+1])


            plt.xlim(0,t[i])
            plt.ylim(min(min(x),min(v), min(U))- 0.5,max(max(x),max(v), max(U))+ 0.5 ) 

            ##lets pause the graph 
            ##plt.pause(self.N/self.s_t)

        plt.show()

    def pid_control(self, x, sp):
        ##dynamic system output
        ##x system output
        setpoint = sp

        Kp= 1
        Ki = 0.1
        Kd = 0.5

        error = setpoint - x

        ##error acumulative for integral part of PID

        self.error_sum += error

        error_derivative = error - self.error_previous

        self.U = Kp * error + Ki * self.error_sum + Kd * error_derivative

        ##storing the previous value of the error
        
        self.error_previous = error


S = MassDamper()
S.run_simulation()