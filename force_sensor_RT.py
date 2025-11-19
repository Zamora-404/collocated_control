import numpy as np
import torch
import serial
import time
import threading
from collections import deque

class SensorConnect(serial.Serial):
    def __init__(self, port, baud):
        # Increased timeout slightly to prevent thread hanging
        serial.Serial.__init__(self, port=port, baudrate=baud, timeout=0.05) 

class SensorComm:
    def __init__(self, chain):
        self.comm = chain
        self.lock = threading.Lock()
        self.running = True
        
        # Load normalization parameters
        try:
            norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
            self.X_median = norm_params['X_median'].astype(np.float32)
            self.X_iqr = norm_params['X_iqr'].astype(np.float32)
            self.Y_median = norm_params['Y_median'].astype(np.float32)
            self.Y_iqr = norm_params['Y_iqr'].astype(np.float32)
            self.n_lags = norm_params['n_lags'] # Should be 5
        except Exception as e:
            print(f"Error loading normalization params: {e}")
            raise

        # ---  BUFFER ---
        # need n_lags + 1 (6 steps) for one prediction
        self.SEQUENCE_LENGTH = self.n_lags + 1 
        # larger than needed to handle jitter safely
        self.raw_history = deque(maxlen=30) 
        
        # Load model
        try:
            self.model = torch.jit.load('force_calibration_model_optimized.pt')
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.baseForce = np.zeros(3)
        self.current_force = np.zeros(3)
        
        # Start the background data collector thread
        self.thread = threading.Thread(target=self._background_collector)
        self.thread.daemon = True # Thread dies when main program dies
        self.thread.start()
        print("Sensor background thread started.")

    def _background_collector(self):
        """
        This runs in a separate thread. 
        It reads from Serial continuously, ensuring no data is lost 
        and time-steps remain consistent regardless of the main loop speed.
        """
        while self.running:
            if self.comm.in_waiting > 0:
                try:
                    line = self.comm.readline().decode('ascii', errors='ignore').strip()
                    # Expecting 9 values (3 sensors * 3 axes)
                    values = list(map(float, line.split(',')))
                    
                    if len(values) == 9:
                        # Store raw values immediately.
                        # This operation is thread-safe in Python for deques
                        self.raw_history.append(values)
                        
                except ValueError:
                    continue
                except Exception as e:
                    print(f"Serial Error: {e}")
            else:
                # Sleep tiny amount to prevent CPU hogging if buffer empty
                time.sleep(0.0001)

    def get_latest_force(self):
        """
        Called by the Main Controller.
        Retrieves the latest COMPLETE sequence from the history buffer.
        """
        # Check if we have enough history for a prediction
        if len(self.raw_history) < self.SEQUENCE_LENGTH:
            return np.zeros(3) # Not ready yet

        # --- THE JITTER FIX ---
        # We grab the LAST 6 samples from the deque.
        # Because the background thread collected them, they are guaranteed
        # to be consecutive samples from the Teensy (e.g., exactly 5ms apart),
        # even if this function wasn't called for 50ms.
        
        # Convert deque snapshot to numpy array
        # Shape: (6, 9) -> 6 time steps, 9 features
        raw_sequence = np.array(list(self.raw_history)[-self.SEQUENCE_LENGTH:], dtype=np.float32)
        
        # 1. Feature Engineering (Add Magnitudes)
        # Match the training logic: [Bx1..Bz3, B_mag, B2_mag, B3_mag]
        b_mag = np.sqrt(raw_sequence[:,0]**2 + raw_sequence[:,1]**2 + raw_sequence[:,2]**2)
        b2_mag = np.sqrt(raw_sequence[:,3]**2 + raw_sequence[:,4]**2 + raw_sequence[:,5]**2)
        b3_mag = np.sqrt(raw_sequence[:,6]**2 + raw_sequence[:,7]**2 + raw_sequence[:,8]**2)
        
        # Stack to get (6, 12)
        features = np.column_stack((raw_sequence, b_mag, b2_mag, b3_mag))
        
        # 2. Flatten
        X_input = features.flatten().reshape(1, -1) # Shape (1, 72)
        
        # 3. Normalize
        X_norm = (X_input - self.X_median) / (self.X_iqr + 1e-8)
        
        # 4. Predict
        with torch.no_grad():
            # Assuming model outputs 6 values (3 Force + 3 Torque)
            # We only take the first 3 (Force)
            prediction_norm = self.model(torch.from_numpy(X_norm)).numpy()[0, :3]
            
        # 5. Denormalize
        force = prediction_norm * self.Y_iqr[:3] + self.Y_median[:3]
        
        # 6. Apply Tare
        return force - self.baseForce

    def calibrate_sensor(self, num_samples=500):
        print(f"Calibrating... Keep sensor stationary.")
        self.baseForce = np.zeros(3)
        collected_forces = []
        
        # Give the buffer time to fill
        time.sleep(1.0)
        
        start_time = time.time()
        while len(collected_forces) < num_samples:
            # We use get_latest_force directly, but we need to ensure 
            # we aren't reading the *same* sample twice if the loop is too fast.
            # Ideally, wait for new data, but for tare, averaging over time is fine.
            
            force = self.get_latest_force()
            
            # Only add if it's not exactly zero (which implies buffer not ready)
            if not np.all(force == 0):
                # We calculate the RAW force (without tare) to find the offset
                raw_force = force + self.baseForce 
                collected_forces.append(raw_force)
            
            time.sleep(0.005) # Sleep 5ms
            
            if time.time() - start_time > 10:
                print("Calibration timeout!")
                break
                
        if collected_forces:
            self.baseForce = np.mean(collected_forces, axis=0)
            print(f"Calibration complete. Offset: {self.baseForce}")
        else:
            print("Calibration failed. No data.")

    def close(self):
        self.running = False
        self.thread.join()
        self.comm.close()

def main():
    # Use high baudrate
    ser = SensorConnect('/dev/ttyUSB0', 500000)
    sensor = SensorComm(ser)
    
    # Calibrate
    sensor.calibrate_sensor()
    
    print("Starting Loop. Press Ctrl+C to stop.")
    try:
        while True:
            # SLOW KINEMATIC CONTROLLER
            # the sensor prediction will remain accurate because the background thread maintained the 200Hz history structure.
            
            start_time = time.time()
            
            force = sensor.get_latest_force()
            
            print(f"\rFx: {force[0]:.4f} | Fy: {force[1]:.4f} | Fz: {force[2]:.4f}", end="")
            
            # Simulate variable processing time
            time.sleep(0.001) 
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sensor.close()

if __name__ == "__main__":
    main()