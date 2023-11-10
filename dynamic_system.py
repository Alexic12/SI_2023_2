import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import random
from hubs.models.xgboost import xgb
from hubs.data_hub import Data

class Pendulum():
    def __init__(self):
        ##System Mass
        self.m = 0.2 ##Kg
        ##Gravity Constant
        self.g = 9.8 
        ##Longitud pendulo
        self.l = 0.5
        ##Number of ms for simulation
        self.N = 2000
        ##Number of sample Times
        self.s_t = 1000
        ##Initial Position
        self.x0 = 0
        ##Initial velocity
        self.v0 = 0
        ##Sample time
        self.step_interval = 100 ##ms
        ##Last update time for simulation
        self.last_update_time = -self.step_interval
        ##System Input
        self.U = 0
        ##error accumulative for PID integral part
        self.error_sum = 0
        ##Previous error for calculation of change in the error between iterations
        self.error_previous = 0
        ##Setpoint
        self.sp = []
        ##last value for filling sp arr
        self.last_value = 0
        ##Let's load the model
        xg = xgb(10)
        self.inputs = 7
        self.model = xg.load_model(name = 'IDENT_PID_DIR',inputs = self.inputs ,alfa = 0.02) ##Cambiar nombre modelo y tamaño de input (en todos lados donde sea necesario)

    def update_force(self, t, type, force):

        if type == 'random':
           if t-self.last_update_time >= self.step_interval:
                self.U = random.uniform(-4,4)
                self.last_update_time = t  
        elif type =='external':
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
        x, v = y ##current position and velocity of system
        dxdt = v
        #dvdt = -(self.k/self.m) * x - (self.c/self.m) * v + self.U/self.m  #ident directa para este sistema
        dvdt = -(self.g / self.l) * x + self.U / (self.m * self.l)
        return [dxdt, dvdt]
    
    def run_simulation(self):
        ##set initial conditions and simulation time
        y0 = [self.x0 , self.v0]
        t_span = [0, self.N]

        ##Solve the system of equations
        sol = solve_ivp(self.system_equations, t_span, y0, dense_output = True)

        ##Create time points for simulations
        t = np.linspace(0, self.N, num = self.s_t)

        ##index array
        index = np.zeros(len(t))

        ##set SP array
        sp_arr = np.zeros(len(t))

        ##fill the 
        sp_arr = self.fill_sp(len(sp_arr), -1, 1, 100)

        ##create error vector 
        err_arr = np.zeros(len(t))

        ##get the position and velocity data for the time points 
        x, v = sol.sol(t)

        ##setup the plot for visualizing the system output in real time 
        plt.figure(figsize = (12,6)) 
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.title('Pendulum Simulation System')
        plt.grid(True)

        ##Initialize lines for position, velocity and U
        position_line, = plt.plot([],[],label = 'Position (X)')
        velocity_line, = plt.plot([],[],label = 'Velocity (V)')
        input_line, = plt.plot([],[],label = 'System Input (U)')
        sp_line, = plt.plot([],[],label='System SetPoint (SP)')
        err_line, = plt.plot([],[],label='System Error (ERR)')

        ##Let's show graph legend
        plt.legend()

        ##Initialize position and velocity vector
        x = np.zeros(len(t))
        v = np.zeros(len(t))
        U = np.zeros(len(t))

        ##Let's create a vector for neural controller
        control_vector = np.zeros(self.inputs)

        # Inicializar la lista donde se guardarán los datos
        data = []

        ##Let's simulate the system for the simulation time
        for i in range(1, len(t)):
            ##if we want to identify the system let's update the force 

            #self.update_force(t[i-1],'random',0) ##Para una fuerza random, para identificar la planta
            #self.pid_control(x[i-1], sp_arr[i-1]) ##Para identificar un control PID

            ##Let's store the error for that specific time sample
            err_arr[i-1] =sp_arr[i-1] - x[i-1] 

            ####Para control vector se debe trocar SP y X ***Cuando entreno es sp - x pero para predecir es sp - x
            control_vector[0] = control_vector[1]
            control_vector[1] = sp_arr[i-1]
            control_vector[2] = control_vector[3]
            control_vector[3] = err_arr[i-1]
            control_vector[4] = control_vector[5]
            control_vector[5] = x[i-1]
            control_vector[6] = U[i] ## U anterior
            
            # ##for storing the position
            # control_vector[0] = control_vector[1]
            # control_vector[1] = control_vector[2]
            # control_vector[2] = control_vector[3]
            # control_vector[3] = control_vector[4]
            # control_vector[4] = control_vector[5]
            # control_vector[5] = control_vector[6]
            # control_vector[6] = sp_arr[i-1]##Setpoint

            # ##for storing the control point
            # control_vector[7] = control_vector[8]
            # control_vector[8] = control_vector[9]
            # control_vector[9] = control_vector[10]
            # control_vector[10] = control_vector[11]
            # control_vector[11] = control_vector[12]
            # control_vector[11] = control_vector[13]
            # control_vector[13] = err_arr[i-1] #error

            # ##for storing the position
            # control_vector[14] = control_vector[15]
            # control_vector[15] = control_vector[16]
            # control_vector[16] = control_vector[17]
            # control_vector[17] = control_vector[18]
            # control_vector[18] = control_vector[19]
            # control_vector[19] = control_vector[20]
            # control_vector[20] = x[i-1] #position

            # ##for storing the control input
            # control_vector[21] = control_vector[22]
            # control_vector[22] = control_vector[23]
            # control_vector[23] = control_vector[24]
            # control_vector[24] = control_vector[25]
            # control_vector[25] = control_vector[26]
            # control_vector[26] = U[i] #control input (accion de control)

            #Let's perform the control action
            self.U = self.inverse_neuronal_control(control_vector, U[i-1])*0.3 ##Para realizar control Neuronal
            
            ##fill index
            index[i-1] = i

            ##Store the system input 
            U[i-1] = self.U

            ##set initial conditions and timespan
            y0 = [x[i-1], v[i-1]]
            t_span = [t[i-1], t[i]]

            ##solve the system of equations for this iteration
            sol = solve_ivp(self.system_equations, t_span, y0, dense_output = True)

            ##get the position and velocity for the next samplw time
            x[i], v[i] = sol.sol(t[i])

            ##Let's update the graph lines for showing system status
            position_line.set_data(t[:i+1], x[:i+1])
            velocity_line.set_data(t[:i+1], v[:i+1])
            input_line.set_data(t[:i+1], U[:i+1])
            sp_line.set_data(t[:i+1], sp_arr[:i+1])
            err_line.set_data(t[:i+1], err_arr[:i+1])

            ##Let's show the data up until the actual sample time
            plt.xlim(0, t[i])
            plt.ylim(min(min(x),min(v),min(U)) - 0.5, max(max(x),max(v),max(U)) + 0.5)

            ##Let's pause the graph
            #plt.pause(self.N/self.s_t)
            
        plt.show()


        # data = np.vstack((x, U, sp_arr,err_arr))
        # print(data)
        # # Create a DataFrame from the data
        # df = pd.DataFrame(data)
    
        # # Save the DataFrame to an Excel file
        # excel_filename = 'TOMA_DATOS_PID_DIR.xlsx' ##Para Planta usar "PLANTA" y para pid usar "PID_DIR"
        # df.to_excel(excel_filename, index=False)
        # print(f'Data saved to {excel_filename}')
       
        
        

    def pid_control(self,x,sp):
        ##dynamic system output.
        ## x system position
        Kp = 0.005
        Ki = 0.001
        Kd = 0.005

        self.sp = sp
            
        ## error fo proportional part of PID
        error = self.sp - x

        ##error acumulative for integral part of PID
        self.error_sum += error

        ##Error derivative for derivative part of PID
        error_derivative = error - self.error_previous
        
        self.U = Kp*error + Ki*self.error_sum + Kd*error_derivative

    def inverse_neuronal_control(self,control_vector,label):

        label = label.reshape((1,1))
        control_vector = control_vector.reshape((1,self.inputs))
        U = self.model.predict(control_vector) 

        #history = self.model.fit(control_vector,label,eval_metric = 'mae')

        return U[0]


P = Pendulum()
P.run_simulation()