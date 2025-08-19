import numpy as np
import time
import csv
from datetime import datetime
from CollocatedDynamics import CollocatedDynamics
from Cam_tracker_class import Cam_Tracker
from Arduino import ArduinoComm, ArduinoConnect
from ForceSensor import SensorComm, SensorConnect

def main():
    np.set_printoptions(precision=3, suppress=True)
    

    # Initialize hardware
    sen_con = SensorConnect('/dev/ttyUSB0', 115200)
    sensor = SensorComm(sen_con)
    time.sleep(.1)
    sensor.calibrate_sensor(100)
    # sensor = []
    
    connect = ArduinoConnect('/dev/ttyACM0', 250000)
    arduino = ArduinoComm(connect)
    time.sleep(.1)
    init_data = np.array( arduino.receive_data() )
    
    tracker = Cam_Tracker()
    dyn = CollocatedDynamics(init_data[3:6])
    path_coords = dyn.generate_circle(10, 30, 50)
    # path_coords = np.array([[0],
    #                         [0],
    #                         [30]])

    # Initialize time
    time_start = time.time()
    time_prev = time_start-.05


    # file_name = input('Enter the filename: ')
    file_name = "t1"
    
    file = open(file_name + '.csv' , 'w')


    while True:
        try:
            # Get current timestamp
            time_now = time.time()
            dt = time_now - time_prev
            time_prev = time_now
            print(f"FPS: {1/dt:.1f}")

            
            # Read data
            th = np.pi*2/3
            robot_data = arduino.receive_data()
            force = sensor.receive_data() @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
            # print(force)
            # force = np.array([0,0,0])
            
            # get pose data
            t_tip, R_tip = tracker.get_pose()

            print(t_tip, R_tip)

            # Run control algorithm
            u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt, q_a, q_u_0  = dyn.CLIK(
                robot_data, t_tip*1000, R_tip, force, path_coords
            )
            
            # Send commands
            # arduino.send_data(u_volumes, commands)

            # update tracker graphics
            tracker.update(rigid_pred)

            # print((t_tip*1000).flatten(), R_tip.flatten(), q_a.flatten(), q_u_0.flatten() )

            # print('Q_no_load: {}' .format(kin.q_no_load(q_a + 5.0)))
            # print('Tip: {}' .format((t_tip*1000).flatten()))
            # print('model_tip: {}' .format(kin.Compute_actual_tip(kin.q_no_load(q_a+5.0))))
            current_datetime = datetime.now()
            Time = np.array([current_datetime.minute, current_datetime.second, (current_datetime.microsecond)//1000])
            # print('Time: {}' .format(Time.flatten()))
            if file.tell() == 0:
                file.write(f"Time; Q_a; q_u_no_load; Tip; Pose\n")
            file.write(f"{Time.flatten()};{(q_a +5.0).flatten()}; {q_u_0.flatten()}; {(t_tip*1000).flatten()}; {R_tip.flatten()}\n")

        except KeyboardInterrupt:
            file.close()


if __name__ == "__main__":
    main()