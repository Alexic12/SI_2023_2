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
        self.N = 100
        ##number of sample times
        self.s_t = 100
        ##initial Position
        self.x0 = 0
        ##initial velocity
        self.v0 = 0
        ##Sample time
        self.step_interval = 10 ## ms
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
        xg = xgb(10)
        self.model = xg.load_model(name = 'INDENT_PID_PENDULO3',inputs =  11, alfa = 0.02) #Descomentar si deseo meter el modelo ya entrenado para que no me saque error
        # aqui se modifica los input es decir los feautures 

    def update_force(self, t, type, force):
    #Aqui se decide cual fuerza meter una random para escalones o una ecterna que decide la fuerza.
        if type == 'random':
            if t-self.last_update_time >= self.step_interval: 
                self.U = random.uniform(-4,4)
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
        l = 1.5 # lenght of the pendulum
        m = 1.2 # mass of the pendulum
        g = 9.8 # acceleration due to gravity 

        x, v = y ##actual position and velocity of system
        dxdt = v
        dvdt = -(g/l)*x+self.U*(m+l)
        return [dxdt, dvdt]
    #def system_equations(self,t,y):
        #l = 0.5 # lenght of the pendulum
        #m = 0.2 # mass of the pendulum
        #g = 9.8 # acceleration due to gravity 

        #theta,omega = y # x, v = y 
        #dthetadt=y
        #domegadt= omega
        #domegadt =-(g/l)*theta+self.U/(m*l)
        #return [dthetadt,domegadt]
    
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
        sp_arr = self.fill_sp(len(sp_arr), -4, 4,5) # aqui se llena el aleatorio el setpoint para el PID  se dejo en 100 pero se cambia es decir cada 100 cambia s

        #Create error vector
        err_arr = np.zeros(len(t))

        ##get the position and velocity data for the time points
        x, v = sol.sol(t)

        ##setup the plot for visualizing the system output in real time
        plt.figure(figsize=(12,6))
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.title('Pendulum Simulation System')
        plt.grid(True)

        ##Itialize lines for position, velocity and U
        position_line, = plt.plot([],[],label='Position angular (X)')
        #velocity_line, = plt.plot([],[],label='Velocity angular (V)')
        input_line, = plt.plot([],[],label='Torque (U)')
        sp_line, = plt.plot([],[],label='System SetPoint (SP)')
        err_line, = plt.plot([],[],label='System Error (ERR)')

        ##lets show graph legend
        plt.legend()

        ##initialize position and velocity vector and U
        x = np.zeros(len(t))
        v = np.zeros(len(t))
        U = np.zeros(len(t))

        ##lets create a vector for neural controller
        control_vector = np.zeros(11)    # aqui cambia con los inputs ojo 

        ##lets simulate the system fpr the simulation time
        for i in range(1, len(t)):
            ##if we want to identify the system lets update the force

            #self.update_force(t[i-1], 'random', 0) # Esto se descomenta si se va a identificar la planta. 
            #self.pid_control(x[i-1], sp_arr[i-1]) # Esto se descomenta si se va a identificar el controlador. 

            ##lets store the error for that specific time sample
            err_arr[i-1] = sp_arr[i-1] - x[i-1]

            # esto se hizo para tomar los datos
            # recuerde que donde va setpoint se cruza con X

            control_vector[0] = control_vector[1]
            control_vector[1] = control_vector[2]
            control_vector[2] = sp_arr[i-1]   
            control_vector[3] = control_vector[4]
            control_vector[4] = control_vector[5]
            control_vector[5] = err_arr[i-1]
            control_vector[6] = control_vector[7]
            control_vector[7] = control_vector[8]
            control_vector[8] = x[i-1]
            control_vector[9] = control_vector[10]
            control_vector[10] = U[i] # se va i por que va el anterior. 


            '''
            #for storing the position setpoint
            control_vector[0] = control_vector[1]
            control_vector[1] = control_vector[2]
            control_vector[2] = control_vector[3]
            control_vector[3] = control_vector[4]
            control_vector[4] = control_vector[5]
            control_vector[5] = control_vector[6]
            control_vector[6] = sp_arr[i-1] ##setpoint

            ##for storing the error 
            control_vector[7] = control_vector[8]
            control_vector[8] = control_vector[9]
            control_vector[9] = control_vector[10]
            control_vector[10] = control_vector[11]
            control_vector[11] = control_vector[12]
            control_vector[12] = control_vector[13]
            control_vector[13] = err_arr[i-1] #error

            #for storing the position
            control_vector[14] = control_vector[15]
            control_vector[15] = control_vector[16]
            control_vector[16] = control_vector[17]
            control_vector[17] = control_vector[18]
            control_vector[18] = control_vector[19]
            control_vector[19] = control_vector[20]
            control_vector[20] = x[i-1] ##position
            #for storing the control input
            control_vector[21] = control_vector[22]
            control_vector[22] = control_vector[23]
            control_vector[23] = control_vector[24]
            control_vector[24] = control_vector[25]
            control_vector[25] = control_vector[26]
            control_vector[26] = U[i] ##setpoint
            '''

            ##lets perform the control action
            self.U = self.inverse_neuronal_control(control_vector, U[i-1])*0.1 # este es el control neuronal el programa con el programa cargado arriba. se comenta para cuando se quiere identificar la planta.
            # tambien se le puede modificar el escalon
            
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
            #velocity_line.set_data(t[:i+1], v[:i+1])
            input_line.set_data(t[:i+1], U[:i+1])
            sp_line.set_data(t[:i+1], sp_arr[:i+1])
            err_line.set_data(t[:i+1], err_arr[:i+1])

            ##lets show te data up until the actual sample time
            plt.xlim(0, t[i])
            plt.ylim(min(min(x), min(v), min(U)) - 0.5, max(max(x), max(v), max(U)) + 0.5)

            ##lets pause the graph 
            ##plt.pause(self.N/self.s_t/100000)

        
        data = np.vstack((x, U, sp_arr, err_arr)) # se guarda x , u miu , sp setpoint y el error. 
        print(data)
        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        excel_filename = 'TOMA_DATOS_PENDULO_PID_T.xlsx' # aqui se guarda el archivo de la indentificacio. 
        df.to_excel(excel_filename, index=False)

        print(f'Data saved to {excel_filename}')
        
        plt.show()


    def pid_control(self, x, sp):
        ## dynamic system 
        ## x system position
        Kp = 0.02
        Ki = 0.01
        Kd = 0.01

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

    def inverse_neuronal_control(self, control_vector, label):

        label = label.reshape((1,1))
        control_vector = control_vector.reshape((1,11)) ## aqui toca modifciar esto por la ventana 

        ##lets create an evaluation set
        #history = self.model.fit(control_vector, label)
        U = self.model.predict(control_vector)

        
        
        return U[0]






S = MassDamper()
S.run_simulation()













