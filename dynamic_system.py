import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import random
#from hubs.neural_hub import Neural as N
from hubs.models.xgboost import xgb

class MassDamper:
    def __init__(self):
        ##System Mass
        self.m = 1.2 ##Kg,USADO EN EL FINAL
        ##String Constant
        self.k = 0.5
        ##Friction coefficient
        self.c = 2
        ##Number of Seg for simulation
        self.N = 100 #USADO EN EL FINAL
        ##number of sample times
        self.s_t = 1000 #USADO EN EL FINAL
        ##initial Position angular
        self.x0 = 0 #USADO EN EL FINAL
        ##initial velocity
        self.v0 = 0 #USADO EN EL FINAL
        #ACELERACION
        self.g = 9.8 #m/s,USADO EN EL FINAL
        #Longitud Pendulo
        self.l = 1.5 #m, USADO EN EL FINAL
        ##Sample time
        self.step_interval = 10 ## ms
        ##Last update time for simulation
        self.last_update_time = -self.step_interval
        ##System input
        self.U = 0 #USADO EN EL FINAL

        ##error acummulative for PID integral part
        self.error_sum = 0
        ##Previous error for calculation of change in the error between iterations
        self.error_previous = 0
        ##las value for filling sp arr
        self.last_value = 0
        ##Lets load the neural model
        xg = xgb(10)
        self.model = xg.load_model(name= "DATOS_PLANTA_DIRECTA", inputs = 7, alfa = 0.02)

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



    '''
    def system_equations(self, t, y):
        x, v = y ##actual position and velocity of system
        dxdt = v
        #aqui nos pone otro sistema dinamico y se cambia en dvdt, al cual le deberiamos de hacer identificacion directa e inversa
        dvdt = -(self.k/self.m) * x - (self.c/self.m) * v + self.U/self.m
        return [dxdt, dvdt]'''
    def system_equations(self, t, y):
        x, v = y
        dxdt = v
        dvdt= -(self.g/self.l)* x + self.U * (self.m + self.l)
        #dvdt = -(self.k/self.m) * x - (self.c/self.m) * v + self.U/self.m
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

        ##Create error vector
        err_arr = np.zeros(len(t))
        
        ##get the position and velocity data for the time points
        x, v = sol.sol(t)

        ##setup the plot for visualizing the system output in real time
        plt.figure(figsize=(12,6))
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.title('SIMULACION DE UN PENDULO')
        plt.grid(True)

        ##Itialize lines for position, velocity and U
        position_line, = plt.plot([],[],label='Position (X)')
        velocity_line, = plt.plot([],[],label='Velocity (V)')
        input_line, = plt.plot([],[],label='System Input (U)')
        sp_line, = plt.plot([],[],label='System SetPoint (SP)')
        err_line, = plt.plot([],[],label='ERROR LINE (ERR)')
        ##lets show graph legend
        plt.legend()

        ##initialize position and velocity vector and U
        x = np.zeros(len(t))
        v = np.zeros(len(t))
        U = np.zeros(len(t))

        #Vector que se le va a pasar para el control neuronal
        control_vector= np.zeros(7)

        ##lets simulate the system for the simulation time(simular el sistema para todo el tiempo de simulacion)
        for i in range(1, len(t)):

            ##if we want to identify the system lets update (actualizar) the force
            #self.update_force(t[i-1], 'random', 0)#aqui al sistema le entra una fuerza aleatoria, ademas se pone en i-1 porque el rango del for es de (1, len(t)), eso es porque el sistema ya tiene condiciones iniciales, pero para actualizar la fuerza se hace desde el tiempo inicial, por eso toca restar 1
            #self.pid_control(x[i-1], sp_arr[i-1])#aqui se hace el sistema de control directo
            
            ##lets store the error for that specific time sample
            err_arr[i-1]=sp_arr[i-1]-x[i-1]
           
            control_vector[0]= control_vector[1] 
            control_vector[1] =sp_arr[i-1] ##setpoint

            control_vector[2] = control_vector[3]
            control_vector[3] = err_arr[i-1] ##error

            control_vector[4]= control_vector[5] 
            control_vector[5] =sp_arr[i-1] ## X-posicion

            control_vector[6] = U[i] ##error
            

           ##Lets perform the control action
            self.U= self.inverse_neural_control(control_vector)*0.8

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
            x[i-1], v[i] = sol.sol(t[i])

            ##lets udate the graph lines for showing system status
            position_line.set_data(t[:i+1], x[:i+1])
            velocity_line.set_data(t[:i+1], v[:i+1])
            input_line.set_data(t[:i+1], U[:i+1])
            sp_line.set_data(t[:i+1], sp_arr[:i+1])
            err_line.set_data(t[:i+1], err_arr[:i+1])
            ##lets show te data up until the actual sample time
            plt.xlim(0, t[i])
            plt.ylim(min(min(x), min(v), min(U)) - 0.5, max(max(x), max(v), max(U)) + 0.5)

            ##lets pause the graph 
            #plt.pause(self.N/self.s_t)

        '''
        data = np.vstack((x, U, sp_arr, err_arr))
        print(data)
        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file ## AQUI SE GUARDA LA TOMA DE DATOS DE LA IDENTIFICACION DIRECTA E INVERSA DE LA PLANTA SOLA SIN CONTROLADOR
        excel_filename = 'DATOS_PLANTA1_PID.xlsx'
        df.to_excel(excel_filename, index=False)

        print(f'Data saved to {excel_filename}')'''
        
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

    def inverse_neural_control (self, control_vector):
        control_vector = control_vector.reshape((1,7))
        U = self.model.predict(control_vector)#este .predict recibe una tabla
        return U[0]#estaba entregando un array 1,1 y sacaba error, como debe ser un entero pues pedimos que nos de el unico valor que hay en el array
    

S = MassDamper()
S.run_simulation()

        
