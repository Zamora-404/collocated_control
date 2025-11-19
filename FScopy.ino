#include <Wire.h>


// Sensor configuration
const byte I2CAddress1 = 0x60;  // Sensor address
const long freq = 2000000;      // I2C Frequency in Hz (2MHz)

// Calibration parameters for the sensor
// All offsets (mu and base) are set to 0 as requested.
const float mu1[3] = {0, 0, 0};
const float sigma1[3] = {15, 15, 15}; // Using 15 as a non-zero sigma
const int base_x1 = 0, base_y1 = 0, base_z1 = 0;

const float delay_millis = 15; // in milliseconds
const int numReadings = 10;    // Moving average window size

// Register addresses
#define REG_STAT1 0x00
#define REG_X_L 0x01
#define REG_X_H 0x02
#define REG_Y_L 0x03
#define REG_Y_H 0x04
#define REG_Z_L 0x05
#define REG_Z_H 0x06
#define REG_T_L 0x07
#define REG_T_H 0x08
#define REG_STAT2 0x09
#define REG_CTRL1 0x0E
#define REG_CTRL3 0x14
#define REG_CTRL4 0x15

// Data structure for the sensor
struct SensorData {
  int16_t x_val = 0, y_val = 0, z_val = 0, temp_val = 0;
  uint8_t stat1 = 0, stat2 = 0;
  float avg_x = 0, avg_y = 0, avg_z = 0, avg_temp = 0;
  float total_x = 0, total_y = 0, total_z = 0, total_temp = 0;
  int readIndex = 0;
  int read_x[10] = {0};  // Matches numReadings size
  int read_y[10] = {0};
  int read_z[10] = {0};
  int read_temp[10] = {0};
};

SensorData sensor1; // Only one sensor instance

/**
 * @brief Checks if a device is connected at the given I2C address.
 * @param address The I2C address to check.
 * @return true if a device acknowledges, false otherwise.
 */
bool checkSensorConnection(byte address) {
  Wire.beginTransmission(address);
  return (Wire.endTransmission() == 0);
}

/**
 * @brief Reads a single 8-bit register from the sensor.
 * @param address The sensor's I2C address.
 * @param reg The register to read from.
 * @param value Pointer to store the read value.
 * @return true on success, false on I2C error.
 */
bool readRegister(byte address, uint8_t reg, uint8_t* value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  
  Wire.requestFrom(address, (uint8_t)1);
  if (Wire.available()) {
    *value = Wire.read();
    return true;
  }
  return false;
}

/**
 * @brief Writes a single 8-bit value to a sensor register.
 * @param address The sensor's I2C address.
 * @param reg The register to write to.
 * @param value The 8-bit value to write.
 *s * @return true on success, false on I2C error.
 */
bool writeRegister(byte address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission() == 0);
}

/**
 * @brief Checks if new measurement data is ready.
 * @param address The sensor's I2C address.
 * @param stat1 Pointer to store the STAT1 register value.
 * @return true if data is ready, false otherwise.
 */
bool dataReady(byte address, uint8_t* stat1) {
  // Checks STAT1 register, Bit 0 (DRDY)
  return readRegister(address, REG_STAT1, stat1) && (*stat1 & 0x01);
}

/**
 * @brief Reads all sensor data (X, Y, Z, Temp) in a single burst.
 * @param address The sensor's I2C address.
 * @param data Pointer to the SensorData struct to fill.
 * @return true on success, false on I2C error or if data not ready.
 */
bool readSensorData(byte address, SensorData* data) {
  // Check if data is ready
  if (!dataReady(address, &data->stat1)) {
    return false;
  }

  // Read measurement data (X, Y, Z, Temp)
  Wire.beginTransmission(address);
  Wire.write(REG_X_L); // Start reading from X_L (0x01)
  if (Wire.endTransmission(false) != 0) return false;
  
  // Read 9 bytes: X_L, X_H, Y_L, Y_H, Z_L, Z_H, T_L, T_H, STAT2
  uint8_t sensorData[9];
  Wire.requestFrom(address, (uint8_t)9);
  for (int i = 0; i < 9; i++) {
    if (!Wire.available()) return false;
    sensorData[i] = Wire.read();
  }

  // STAT2 is the 9th byte (index 8)
  data->stat2 = sensorData[8];

  // Convert data to 16-bit values
  data->x_val = (int16_t)((sensorData[1] << 8) | sensorData[0]);
  data->y_val = (int16_t)((sensorData[3] << 8) | sensorData[2]);
  data->z_val = (int16_t)((sensorData[5] << 8) | sensorData[4]);
  data->temp_val = (int16_t)((sensorData[7] << 8) | sensorData[6]);

  return true;
}

/**
 * @brief Configures the sensor for continuous measurement with temp.
 * @param address The sensor's I2C address.
 * @return true on success, false on I2C error.
 */
bool configureSensor(byte address) {
  // Reset the sensor by reading STAT1
  uint8_t stat1;
  if (!readRegister(address, REG_STAT1, &stat1)) return false;
  delay(60);
  
  // Configure CTRL3 (0x14) - Set OSR and filters
  // OSR_HALL=1, OSR_TEMP=1, DIG_FILT_HALL_XY=0, DIG_FILT_TEMP=1
  uint8_t ctrl3 = 0xC1; // 1100 0001
  if (!writeRegister(address, REG_CTRL3, ctrl3)) return false;
  delay(1);

  // Configure CTRL4 (0x15) - Enable Temperature (T_EN)
  // Set T_EN (Bit 5) to 1.
  // Bit 7: CTRL4_7 (1)
  // Bit 6: Reserved (0)
  // Bit 5: T_EN (1) <-- This enables temperature
  // Bit 4: CTRL4_4 (1)
  // Bit 3: DRDY_EN (0)
  // Bit 2-0: DIG_FILT_HALL_Z (111)
  uint8_t ctrl4 = 0xB7; // 1011 0111
  if (!writeRegister(address, REG_CTRL4, ctrl4)) return false;
  delay(1);

  // Configure CTRL1 (0x0E)
  // X, Y, Z enabled, Continuous mode 100Hz
  uint8_t ctrl1 = 0x70 | 0x0C; 
  if (!writeRegister(address, REG_CTRL1, ctrl1)) return false;

  delay(40);
  return true;
}

/**
 * @brief Applies a moving average filter to the sensor data.
 * @param data Pointer to the SensorData struct to smooth.
 */
void smooth(SensorData* data) {
  // subtract the last reading:
  data->total_x -= data->read_x[data->readIndex];
  data->total_y -= data->read_y[data->readIndex];
  data->total_z -= data->read_z[data->readIndex];
  data->total_temp -= data->read_temp[data->readIndex];
  
  // read the sensor (values already updated in readSensorData):
  data->read_x[data->readIndex] = data->x_val;
  data->read_y[data->readIndex] = data->y_val;
  data->read_z[data->readIndex] = data->z_val;
  data->read_temp[data->readIndex] = data->temp_val;
  
  // add value to total:
  data->total_x += data->read_x[data->readIndex];
  data->total_y += data->read_y[data->readIndex];
  data->total_z += data->read_z[data->readIndex];
  data->total_temp += data->read_temp[data->readIndex];
  
  // handle index
  data->readIndex = (data->readIndex + 1) % numReadings;
  
  // calculate the average:
  data->avg_x = data->total_x / numReadings;
  data->avg_y = data->total_y / numReadings;
  data->avg_z = data->total_z / numReadings;
  data->avg_temp = data->total_temp / numReadings;
}

// --- Main Setup ---
void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  Wire.begin();
  Wire.setClock(freq);

  Serial.println("Initializing MLX90394 sensor...");
  
  // Initialize sensor 1
  if (!checkSensorConnection(I2CAddress1)) {
    Serial.println("Sensor not found at 0x60!");
    while(1); // Halt
  }
  if (!configureSensor(I2CAddress1)) {
    Serial.println("Sensor configuration failed!");
    while(1); // Halt
  }

  Serial.println("Sensor ready");
  Serial.println("X_Raw\tY_Raw\tZ_Raw\tTemp_Raw");
}

// --- Main Loop ---
void loop() {
  static uint32_t lastSampleTime = 0;
  // Sample interval 3300 micros = ~303Hz
  // Note: Sensor is set to 100Hz continuous, so 303Hz polling is faster
  // than new data is generated. The `dataReady()` check handles this.
  const uint32_t sampleInterval = 3300; 
  
  uint32_t currentTime = micros();
  if (currentTime < lastSampleTime) {
    lastSampleTime = currentTime; // Handle overflow
  }
  
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    
    // Read sensor 1
    if (readSensorData(I2CAddress1, &sensor1)) {
      // Data was successfully read
      smooth(&sensor1);
      
      // Prepare data for neural network
      float B_raw1[3] = {sensor1.avg_x, sensor1.avg_y, sensor1.avg_z};
      float B_norm1[3];
      float F1[3]; // To store the predicted force
      
      // Run normalization and prediction
      //normalize_input(B_raw1, B_norm1, mu1, sigma1);
      //predict_force(B_norm1, F1);
      
      // --- Output Data ---
      // This prints the *raw* (non-averaged) values
      Serial.print(sensor1.x_val); Serial.print(',');
      Serial.print(sensor1.y_val); Serial.print(',');
      Serial.print(sensor1.z_val); Serial.print(',');
      Serial.println(sensor1.temp_val);
      
      // Example: Print averaged values
      // Serial.print(sensor1.avg_x); Serial.print(',');
      // ...
      
      // Example: Print predicted forces
      // Serial.print(F1[0]); Serial.print(',');
      // Serial.print(F1[1]); Serial.print(',');
      // Serial.println(F1[2]);
    }
  }
}