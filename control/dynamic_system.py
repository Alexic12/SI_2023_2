import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import random
from hubs.models.xgboost import xgb
#from hubs.neural_hub import Neural as N

class MassDamper:
    def __init__(self):
        ##System Mass
        self.m = 2 ##Kg
        ##String Constant
        self.k = 0.5
        ##Friction coefficient
        self.c = 2
        ##Number of ms for simulation
        self.N = 2000
        ##number of sample times
        self.s_t = 1000
        ##initial Position
        self.x0 = 0
        ##initial velocity
        self.v0 = 0
        ##Sample time
        self.step_interval = 100 ## ms
        ##Last update time for simulation
        self.last_update_time = -self.step_interval
        ##System input
        self.U = 0
        ##error acummulative for PID integral part
        self.error_sum = 0
        ##Previous error for calculation of change in the error between iterations
        self.error_previous = 0
        ##las value for filling sp arr
        self.last_value = 0
        ##lets load the neural model
        self.model = xgb.load_model('PID_IDENT_1', 7, 0.02)

    def update_force(self, t, type, force):

        if type == 'random':
            if t-self.last_update_time >= self.step_interval:
                self.U = random.uniform(-1,1)
                self.last_update_time = t   
        elif type == 'external':
            self.U = force

    def fill_sp(self, size, min_val, max_val, interval):
        arr = np.zeros(size)
        last_update_time = 0

        for i in range(size):
            if i - last_update_time >= interval:
                self.last_value = random.uniform(min_val, max_val)
                last_update_time = i

            arr[i] = self.last_value

        return arr




    def system_equations(self, t, y):
        x, v = y ##actual position and velocity of system
        dxdt = v
        dvdt = -(self.k/self.m) * x - (self.c/self.m) * v + self.U/self.m
        return [dxdt, dvdt]
    
    def run_simulation(self):
        ##set initial conditions and simulation time
        y0 = [self.x0, self.v0]
        t_span = [0, self.N]

        ##solve the system equations
        sol = solve_ivp(self.system_equations, t_span, y0, dense_output=True)

        ##Create time points for simulation
        t = np.linspace(0, self.N, num=self.s_t)

        ##index array
        index = np.zeros(len(t))

        ##set SP array
        sp_arr = np.zeros(len(t))

        ##fill the 
        sp_arr = self.fill_sp(len(sp_arr), -1, 1, 100)


        ##get the position and velocity data for the time points
        x, v = sol.sol(t)

        ##setup the plot for visualizing the system output in real time
        plt.figure(figsize=(12,6))
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.title('Mass Damper Simulation System')
        plt.grid(True)

        ##Itialize lines for position, velocity and U
        position_line, = plt.plot([],[],label='Position (X)')
        velocity_line, = plt.plot([],[],label='Velocity (V)')
        input_line, = plt.plot([],[],label='System Input (U)')
        sp_line, = plt.plot([],[],label='System SetPoint (SP)')

        ##lets show graph legend
        plt.legend()

        ##initialize position and velocity vector and U
        x = np.zeros(len(t))
        v = np.zeros(len(t))
        U = np.zeros(len(t))

        ##lets create a vector for neural controller
        control_vector = np.zeros(7)

        ##lets simulate the system fpr the simulation time
        for i in range(1, len(t)):
            ##if we want to identify the system lets update the force

            ##self.update_force(t[i-1])
            ##self.pid_control(x[i-1], sp_arr[i-1])

            control_vector[0] = 2 ##setpoint

            ##for storing the position
            control_vector[1] = control_vector[2]
            control_vector[2] = control_vector[3]
            control_vector[3] = x[i-1]

            ##for storing the control input
            control_vector[4] = control_vector[5]
            control_vector[5] = control_vector[6]
            control_vector[6] = U[i-1]
            
            ##lets perform the control action
            self.U = self.inverse_neuronal_control(control_vector)

            ##fill index
            index[i-1] = i


            ##Store the system input
            U[i-1] = self.U

            ##set initial conditions and timespan
            y0 = [x[i-1], v[i-1]]
            t_span = [t[i-1], t[i]]

            ##solve the system equations for this iteration
            sol = solve_ivp(self.system_equations, t_span, y0, dense_output=True)

            ##get the position and velocity for the next sample time
            x[i], v[i] = sol.sol(t[i])

            ##lets udate the graph lines for showing system status
            position_line.set_data(t[:i+1], x[:i+1])
            velocity_line.set_data(t[:i+1], v[:i+1])
            input_line.set_data(t[:i+1], U[:i+1])
            sp_line.set_data(t[:i+1], sp_arr[:i+1])

            ##lets show te data up until the actual sample time
            plt.xlim(0, t[i])
            plt.ylim(min(min(x), min(v), min(U)) - 0.5, max(max(x), max(v), max(U)) + 0.5)

            ##lets pause the graph 
            #plt.pause(self.N/self.s_t)

        '''
        data = np.vstack((x, U, sp_arr))
        print(data)
        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        excel_filename = 'your_data.xlsx'
        df.to_excel(excel_filename, index=False)

        print(f'Data saved to {excel_filename}')
        '''
        plt.show()


    def pid_control(self, x, sp):
        ## dynamic system 
        ## x system position
        Kp = 1
        Ki = 0.1
        Kd = 0.5

        setpoint = sp

        ##error for proportional part of PID
        error = setpoint - x

        ##error acumulative for integral part of PID
        self.error_sum += error

        ##error derivative for derivative part of PID
        error_derivative = error - self.error_previous

        self.U = Kp*error + Ki*self.error_sum + Kd*error_derivative

        ##we store the previous value of the error
        self.error_previous = error






S = MassDamper()
S.run_simulation()






