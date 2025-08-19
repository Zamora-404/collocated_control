import numpy as np
import torch
import serial
import time
from collections import deque

class SensorConnect(serial.Serial):
    def __init__(self, port, baud):
        serial.Serial.__init__(self, port=port, baudrate=baud, timeout=0.1)

class SensorComm:
    def __init__(self, chain):
        self.comm = chain
        
        # Load normalization parameters
        norm_params = np.load('normalization_params.npy', allow_pickle=True).item()
        self.X_median = norm_params['X_median'].astype(np.float32)  # Shape: (num_features*(n_lags+1),)
        self.X_iqr = norm_params['X_iqr'].astype(np.float32)        # Same shape as X_median
        self.Y_median = norm_params['Y_median'].astype(np.float32)  # Shape: (3,)
        self.Y_iqr = norm_params['Y_iqr'].astype(np.float32)        # Shape: (3,)
        self.n_lags = norm_params['n_lags']
        self.num_features = 8  # 6 magnetic + 2 magnitude
        
        # Initialize buffers
        self.HISTORY_BUFFER_SIZE = self.n_lags + 1
        self.history = deque(maxlen=self.HISTORY_BUFFER_SIZE)
        self.SMOOTHING_WINDOW = 5
        self.bx_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bx2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.by2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)
        self.bz2_buffer = deque(maxlen=self.SMOOTHING_WINDOW)

        # Load model
        self.model = torch.jit.load('force_calibration_model_optimized.pt')
        self.model.eval()
        
        self.baseForce = np.zeros(3)
        self.current_data = np.zeros(3)
        self.initial_samples = 0

    def receive_data(self):
        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('ascii', errors='ignore').strip()
            if line and line.count(',') == 5:
                try:
                    bx, by, bz, bx2, by2, bz2 = map(float, line.split(','))
                    
                    # Apply smoothing
                    self.bx_buffer.append(bx)
                    self.by_buffer.append(by)
                    self.bz_buffer.append(bz)
                    self.bx2_buffer.append(bx2)
                    self.by2_buffer.append(by2)
                    self.bz2_buffer.append(bz2)
                    
                    smoothed = [
                        np.mean(self.bx_buffer) if self.bx_buffer else bx,
                        np.mean(self.by_buffer) if self.by_buffer else by,
                        np.mean(self.bz_buffer) if self.bz_buffer else bz,
                        np.mean(self.bx2_buffer) if self.bx2_buffer else bx2,
                        np.mean(self.by2_buffer) if self.by2_buffer else by2,
                        np.mean(self.bz2_buffer) if self.bz2_buffer else bz2
                    ]
                    
                    # Predict force
                    force = self.predict_force(*smoothed)
                    self.current_data = force - self.baseForce
                    
                except ValueError:
                    continue
        
        return self.current_data

    def predict_force(self, bx, by, bz, bx2, by2, bz2):
        # Calculate derived features
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        b2_mag = np.sqrt(bx2**2 + by2**2 + bz2**2)
        
        # Create current feature vector (8 features)
        current_features = np.array([bx, by, bz, bx2, by2, bz2, b_mag, b2_mag], dtype=np.float32)
        
        # Add to history
        self.history.append(current_features)
        self.initial_samples += 1
        
        # Wait until buffer fills
        if len(self.history) < self.HISTORY_BUFFER_SIZE:
            return np.zeros(3)
        
        # Create input vector by flattening the history
        X_cal = np.concatenate(self.history).reshape(1, -1)  # Shape: (1, num_features*(n_lags+1))
        
        # Verify shapes match
        if X_cal.shape[1] != len(self.X_median):
            raise ValueError(f"Feature dimension mismatch! Input: {X_cal.shape[1]}, Expected: {len(self.X_median)}")
        
        # Normalize input
        X_norm = (X_cal - self.X_median) / (self.X_iqr + 1e-8)
        
        # Predict
        with torch.no_grad():
            Y_norm = self.model(torch.from_numpy(X_norm)).numpy()[0]
        
        # Denormalize output
        return Y_norm * self.Y_iqr + self.Y_median

    def calibrate_sensor(self, num_samples=1000):
        print(f"Calibrating with {num_samples} samples... Keep sensor stationary")
        samples = []
        self.baseForce = np.zeros(3)  # Reset base force
        # Clear all buffers
        self.history.clear()
        self.bx_buffer.clear()
        self.by_buffer.clear()
        self.bz_buffer.clear()
        self.bx2_buffer.clear()
        self.by2_buffer.clear()
        self.bz2_buffer.clear()
        # Collect raw samples
        while len(samples) < num_samples:
            line = self.read_valid_line()
            if line:
                try:
                    values = list(map(float, line.split(',')))
                    if len(values) == 6:
                        samples.append(values)
                        if len(samples) % 100 == 0:
                            print(f"Collected {len(samples)}/{num_samples} samples")
                except ValueError:
                    continue
    # Process collected samples to get forces
        forces = []
        for sample in samples:
            bx, by, bz, bx2, by2, bz2 = sample
            # Apply smoothing
            self.bx_buffer.append(bx)
            self.by_buffer.append(by)
            self.bz_buffer.append(bz)
            self.bx2_buffer.append(bx2)
            self.by2_buffer.append(by2)
            self.bz2_buffer.append(bz2)
            smoothed = [
                np.mean(self.bx_buffer) if self.bx_buffer else bx,
                np.mean(self.by_buffer) if self.by_buffer else by,
                np.mean(self.bz_buffer) if self.bz_buffer else bz,
                np.mean(self.bx2_buffer) if self.bx2_buffer else bx2,
                np.mean(self.by2_buffer) if self.by2_buffer else by2,
                np.mean(self.bz2_buffer) if self.bz2_buffer else bz2
            ]
            # Predict force for this sample
            force = self.predict_force(*smoothed)
            forces.append(force)
    # Calculate average force
        if len(forces) > 0:
            self.baseForce = np.mean(forces, axis=0)
            print(f"Calibration complete. New base force: {self.baseForce}")
        else:
            print("Warning: No valid samples collected during calibration")
        return self.baseForce


    def read_valid_line(self):
        """Read until getting a valid data line"""
        while self.comm.in_waiting > 0:
            line = self.comm.readline().decode('utf-8', errors='ignore').strip()
            if line and not any(x in line for x in ['Initializing', 'sensor', 'ready']):
                if len(line.split(',')) == 6:
                    return line
        return None

def main():
    # Initialize serial connection
    ser = SensorConnect('/dev/ttyUSB0', 115200)
    sensor = SensorComm(ser)
    
    # Calibrate sensor
    sensor.calibrate_sensor()
    
    # Main loop
    try:
        while True:
            force = sensor.receive_data()
            #print(f"Fx: {force[0]:.4f} N, Fy: {force[1]:.4f} N, Fz: {force[2]:.4f} N")
            print(f"\rFx: {force[0]:.5f} N | Fy: {force[1]:.5f} N | Fz: {force[2]:.5f} N")
            time.sleep(0.01)
    except KeyboardInterrupt:
        ser.close()

if __name__ == "__main__":
    main()