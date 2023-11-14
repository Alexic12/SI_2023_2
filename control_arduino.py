import serial
import time
import keyboard
import numpy as np
from hubs.models.xgboost import xgb


def read_position(ser):
    # Read the position from the serial port
    line = ser.readline().decode('utf-8').strip()  # assuming data is terminated by a newline
    try:
        position = float(line)
        return position
    except ValueError:
        print("Received invalid data:", line)
        return None

def calculate_control_output(model,position, setpoint, control_vector, previousControl):
    # Implement your control logic here.
    # For this example, I'm just implementing a dummy control that tries to maintain the position at 50%.
    control_output = inverse_neuronal_control(control_vector, previousControl, model)
    control_output *= 8 #hay que des-normalizar la salida
    if control_output >= 5:
        control_output = 5
    if control_output <= 0:
        control_output = 0
    return control_output

def inverse_neuronal_control(control_vector, label, model):

        #label = label.reshape((1,1))
        control_vector = control_vector.reshape((1,7))

        ##lets create an evaluation set
        #history = self.model.fit(control_vector, label)
        U = model.predict(control_vector)
        
        return U[0]

def main():
    # Set up the serial connection
    ser = serial.Serial(port="COM16", baudrate=115200)  # replace 'COM3' with your port and 9600 with your baud rate
    time.sleep(2)  # give some time for the connection to establish
    setpoint = [0,3,5,2,4,1]
    index = 0
    xg = xgb(10)
    model = xg.load_model(name = 'INDENT_PID_PLANTA',inputs =  7, alfa = 0.02)
    control_vector = np.zeros(7)
    previousControl = 0
    try:
        while True:
            position = read_position(ser)
            if position is not None:
                print(f"Current Position: {position}%")
                error = setpoint[index] - position
                control_vector[0] = control_vector[1]
                control_vector[1] = position
                control_vector[2] = control_vector[3]
                control_vector[3] = error
                control_vector[4] = control_vector[5]
                control_vector[5] = setpoint[index]
                control_vector[6] = previousControl
                control_output = calculate_control_output(model,position,setpoint[index],control_vector,previousControl)
                previousControl = control_output
                print(f"Setpoint: {setpoint[index]}")
                print(f"Sending Control Output: {control_output}")
                ser.write((str(control_output)+'\n').encode('utf-8'))  # send the control output
                time.sleep(0.03)  # delay for stability, adjust as necessary
            
            if keyboard.is_pressed('t'):
                index +=1
                if index == 6:
                    index = 0
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()