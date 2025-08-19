# Import relevant libraries
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
# from LivePlot import Plotting

# Global pressure vector in N/mm^2
kinPressures = np.array([5, 2, 1])/1000
goalCoords = np.array( [0, 20, 20] )

class ArduinoConnect(serial.Serial):

    #Initialize host computer serial communication settings
    def __init__(self, port, baud):
        serial.Serial.__init__(self, port=port, baudrate=baud, bytesize=serial.EIGHTBITS,
                               stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE, timeout=3)


class ArduinoComm:

    def __init__(self, chain):
        self.comm = chain
    
        # Serial port configuration
        self.PORT = 'COM13'  # Adjust to your Teensy port
        self.BAUD_RATE = 115200
        self.TIMEOUT = 0.1
        self.current_data = [0 ,0, 0, 0, 0, 0]


    def read(self):
        data = self.comm.read_until(expected='\r'.encode())
        # print(data.decode())
        return data.decode()
    
    def receive_data(self):
        # clear data buffer, eventually change to queue with pop fucntion 
        data_buffer = []
        

        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('utf-8').strip()
            line_v = line.split(",")

            try:
                # data_buffer.append([float(line_v[0]), float(line_v[1]), float(line_v[2]),
                #                     float(line_v[3]), float(line_v[4]), float(line_v[5])])
                data_buffer.append([float(val) for val in line_v])
            except ValueError:
                print("Invalid data received:", line)
        
        # if the buffer is empty, return the most current data value
        if len(data_buffer) == 0:
            # print("No data received, sending latest value instead")
            return self.current_data
        
        # otherwise, update the current data value and return the buffer
        self.current_data = data_buffer[-1]

        return self.current_data

    def send_data(self, pressures, commands):
        # Format desired values as strings with 3 decimal places
        serial_str = f"{pressures[0]:.3f},{pressures[1]:.3f},{pressures[2]:.3f},{commands[0]:.1f},{commands[1]:.1f},{commands[2]:.1f},{commands[3]:.1f}\n"  

        # print(serial_str)
        # Send serial command to Teensy over serial
        self.comm.write(serial_str.encode())  

    def closePort(self):
        self.comm.close()