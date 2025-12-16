import numpy as np
import time
import csv
from datetime import datetime
from CollocatedDynamics import CollocatedDynamics
# from new_dynamics import CollocatedDynamics
# from collocated_dynamics_18 import CollocatedDynamics
from Cam_tracker_class import Cam_Tracker
from Arduino import ArduinoComm, ArduinoConnect
from ForceSensor import ForceSensor, SensorConnect


from Force_plotting_class import Force_Plot

import matplotlib.pyplot as plt
from collections import deque


def Filter(x_val, x_array, median_size):
    x_array = np.delete(x_array, 0, axis = 0)
    x_array = np.vstack((x_array, x_val.reshape(1,-1)))
    x_updated = x_array.copy()
    for i in range(1, len(x_array) - (len(x_array)%median_size + 1)+1):
        median = np.median(x_array[-i:-(median_size + i):-1], axis = 0)
        x_updated[-i,:] = median.reshape(1,-1)
    

    output = np.mean(x_updated, axis = 0)

    return output, x_array

def low_pass_filter(cutoff, input, prev_ouput):
    dt  = 0.033
    alpha = dt/(1/(2*np.pi*cutoff) + dt)

    output = prev_ouput + alpha * (input - prev_ouput)

    return output

def main():
    np.set_printoptions(precision=3, suppress=True)

    # initialize hardware    
    connect = ArduinoConnect('/dev/ttyACM0', 250000)
    arduino = ArduinoComm(connect)
    time.sleep(.2)

    init_volumes = np.array( arduino.receive_data()[3:6] )
    n = 100    
    for i in range(0,100):
        u_volumes = init_volumes*(n-i-1)/n
        print(u_volumes)
        commands = [0,0,0,0]
        arduino.send_data(u_volumes, commands)
        time.sleep(.05)

    init_volumes = np.array( arduino.receive_data()[3:6] )
    
    
    dyn = CollocatedDynamics(init_volumes)
    point_plane = np.array([0,15, 18])

    Plane_transformation = np.eye(4)
    Plane_transformation[:3,:3] = dyn.Rx(15.0*np.pi/180)
    Plane_transformation[:3, 3] = point_plane
    Path_points = np.array([[-2,2,2,-2],[2,2,-2,-2],[0,0,0,0],[1,1,1,1]])
    path_coords = Plane_transformation @ Path_points
    path_coords  = path_coords[:3,:]

    


    #Moving average window size
    Mean_size = 10
    wrench_array = np.zeros((Mean_size, 6))

    # Initialize forse sensor
    sen_con = SensorConnect('/dev/ttyUSB0', 250000)
    sensor = ForceSensor(sen_con, 
        model_path="model_params/force_torque_model.pth",
        n_steps=5,
        input_size=5*9,  # n_steps * 9 magnetometer values
        hidden_dim=512)
    time.sleep(.2)
    sensor.calibrate(300)
    time.sleep(.2)

    plt.ion()


    time_window = 15
    max_len = int(time_window / 0.05)    #~fps = 20

    #Force Calibration
    # th = 5.289448291629661
    th = 5.6983 # sensor fell and readjusted
    force = -sensor.receive_data()[0:3] @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
    print(sensor.receive_data()[0:3])
    calibrate = np.empty((0,6))
    while len(calibrate) < 10000: # TODO: Switch back to 10000
        force  = -sensor.receive_data()[0:3] @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
        moment = -sensor.receive_data()[3:] @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
        wrench = np.concatenate((force.reshape(-1,1), moment.reshape(-1,1)))
        calibrate = np.vstack((calibrate, wrench.reshape(1,-1)))

    mean_wrench = np.mean(calibrate, axis = 0)
    force  = -sensor.receive_data()[0:3] @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
    moment = -sensor.receive_data()[3:] @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
    wrench = np.concatenate((force.reshape(-1,1), moment.reshape(-1,1)))

    prev_wrench = wrench.flatten() - mean_wrench.flatten()


    # bernouli infinity curve, update in collocated to match
    period = 10
    t_vec = np.linspace(0,period)
    path_coords = np.array([5*np.sin(2*np.pi*t_vec/ (period/2) ),
                            10*np.sin(2*np.pi*t_vec/ period),
                            25 + 4*np.cos(2*np.pi*t_vec/ (period/2) )])
    # path_coords = np.array([[0], [0 ], [25]])

    tracker = Cam_Tracker(False, False)
    rigid_pred =np.zeros((3,3))
    tracker.update(rigid_pred, path_coords)
    t_tip_, R_tip_, t_wall_, R_wall_ = tracker.get_pose()

    # generate and rotate circle path
    # radius = 5
    # path_coords = dyn.generate_circle(10, 0, 20)
    # path_coords = ( (R_wall_ @ path_coords).T + t_wall_ ).T
    

    

    plotting = Force_Plot()
    
    file = []
    # file_name = input('Enter the filename: ')
    file_name = 'test_data_saved'
    file = open(file_name + '.csv' , 'w')


    # # Initialize time
    time_start = time.time()
    time_prev = time_start-.04

    while True:
        try:
            # Get current timestamp
            time_now = time.time()
            t_elapsed = time_now - time_start
            dt = time_now - time_prev
            time_prev = time_now
            # print(f"FPS: {1/dt:.1f}, time: {t_elapsed:.2f}", flush=True)

            
            # Read data
            robot_data  = arduino.receive_data()
            force_raw   = -sensor.receive_data()[0:3] @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
            moment_raw  = -sensor.receive_data()[3:] @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
            # moment_raw[0] = moment_raw[0] - (6.67)*force_raw[1]
            # moment_raw[1] = moment_raw[1] + (6.67)*force_raw[0] 
            wrench_raw = np.concatenate((force_raw.reshape(-1,1), moment_raw.reshape(-1,1))).flatten()
            
            wrench = wrench_raw - mean_wrench

            # moment = sensor.receive_data()[3:] @ np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
            # wrench = np.concatenate((force.reshape(-1,1), moment.reshape(-1,1))).reshape(1,-1)
            # wrench = wrench - mean_wrench
            # print(sensor.receive_data()[0:3])
            # force = np.array([0,0,0])
            
            # get pose data
            tracker.update(rigid_pred, path_coords)
            t_tip, R_tip, t_wall, R_wall = tracker.get_pose()

            # path_coords = np.array([[radius*np.cos(2*np.pi*time_now/ period), radius*np.cos(2*np.pi* (time_now + 0.01)/ period)],
            #                         [radius*np.sin(2*np.pi*time_now/ period), radius*np.sin(2*np.pi* (time_now + 0.01)/ period)],
            # path_coords = ((R_wall @ path_coords).T + t_wall).T

            # print(t_tip, R_tip, t_wall, R_wall)

            wrench_transformation = np.eye(6)
            wrench_transformation[:3,:3]  = R_tip
            wrench_transformation[3:,3:]  = R_tip

            wrench = (wrench_transformation @ (wrench.reshape(-1,1))).flatten()
            prev_wrench = (wrench_transformation @ prev_wrench.reshape(-1,1)).flatten()

    
            # wrench[3:6] = np.zeros((1,3))
            filtered_wrench, wrench_array = Filter(wrench, wrench_array, 4)
            # filtered_wrench = low_pass_filter(2.5, wrench.flatten(), prev_wrench.flatten())
            prev_wrench = filtered_wrench.flatten()
            # force_2 = -R_tip @ force_raw
            # print('Wrench Input: {}' .format(filtered_wrench), flush=True)

            #Shresth controller
            # u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt, q_a, q_u_0  = dyn.CLIK(
            #     robot_data, t_tip*1000, R_tip, filtered_wrench, path_coords
            # )

            

            #leo controller
            u_volumes, commands, backbone_pred, rigid_pred, tip_pred, dt, q_a, q_u_0, F_des, e_t, goal_coords  = dyn.CLIK(
                robot_data, t_tip, R_tip, t_wall, R_wall, filtered_wrench[0:3], path_coords, period
            )
            print(f"FPS: {1/dt:.1f}, time: {t_elapsed:.2f}, error: {np.linalg.norm(e_t):.3f}", flush=True)


            F_wall = R_wall.T @ filtered_wrench[:3]
            
            # Send commands
            arduino.send_data(u_volumes, commands)

            # update tracker graphics
            plotting.update(t_elapsed, F_wall, F_des, e_t)



            # print((t_tip*1000).flatten(), R_tip.flatten(), q_a.flatten(), q_u_0.flatten() )

            # print('Q_no_load: {}' .format(kin.q_no_load(q_a + 5.0)))
            # print('Tip: {}' .format((t_tip*1000).flatten()))
            # print('model_tip: {}' .format(kin.Compute_actual_tip(kin.q_no_load(q_a+5.0))))
            current_datetime = datetime.now()
            Time = np.array([current_datetime.minute, current_datetime.second, (current_datetime.microsecond)//1000])
            
            # if plotting.quit:
            #     plotting.finalize('Test1')
            #     break
            
            # print('Time: {}' .format(Time.flatten()))

            # if file.tell() == 0:
            #     file.write(f"Time; Q_a; q_u_no_load; Tip; Pose\n")
            # file.write(f"{Time.flatten()};{(q_a +5.0).flatten()}; {q_u_0.flatten()}; {(t_tip*1000).flatten()}; {R_tip.flatten()}\n")
            # file.write(f"{t_elapsed:.2f};{t_tip:.2f};{goal_coords:.2f};{";".join(map(str, u_volumes))}; \n")

            if file:
               file.write(f"{t_elapsed:.3f};{";".join(map(str, t_tip))};{";".join(map(str, goal_coords))};{";".join(map(str, u_volumes))};{";".join(map(str, F_wall))};{F_des[2]:.4f};{";".join(map(str, backbone_pred[0]))};{";".join(map(str, backbone_pred[1]))};{";".join(map(str, backbone_pred[2]))}\n")

        except KeyboardInterrupt:
            tracker.close()
            plotting.finalize('Test1')
            # file.close()
            break
            


if __name__ == "__main__":
    main()