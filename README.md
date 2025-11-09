# RUL prediction using BiLSTM+Attention model
This project implements a **Remaining Useful Life (RUL) prediction** model for turbofan engines using the **NASA C-MAPSS FD003** dataset. A hybrid **Bidirectional LSTM with Attention mechanism** is used to capture both temporal dependencies and feature importance in sensor data.

<img width="434" height="231" alt="image" src="https://github.com/user-attachments/assets/5d48a3c6-ec12-4594-8230-8d41fb1bf73e" />

## Dataset

**Source:** NASA C-MAPSS dataset  
**Subset used:** `FD003`  
- Number of engines: 100  
- Each engine has a variable number of cycles until failure.  
- Each row corresponds to a single time step (cycle) with 21 sensor readings.  
- Additional operational settings are provided.

### Key Preprocessing Steps
1. **RUL Labeling**  
   - For each engine, `RUL = max(cycles) - current(cycle)`.  
2. **Normalization**  
   - Applied **MinMaxScaler** to scale sensor and operational features between 0 and 1.  
3. **Sequence Creation**  
   - Sliding window approach (`window_size = 30`) to create temporal sequences for LSTM input.  
4. **Train-Test Split**  
   - Separate scaling and labeling performed for `train` and `test` datasets.

---

## Model Architecture

### **BiLSTM + Attention**

- **Input:** Time window of 30 cycles × selected features  
- **Layers:**
  1. **Bidirectional LSTM (128 units)** – captures temporal dependencies in both directions.  
  2. **Attention Layer** – focuses on the most informative time steps.  
  3. **Dense (64 units, ReLU)** – feature transformation.  
  4. **Dense (1 unit, linear)** – final RUL prediction.

### **Activation Function**
- `ReLU` used in intermediate layers.

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| **Optimizer** | Adam |
| **Loss Function** | Mean Squared Error (MSE) |
| **Metrics** | MSE |
| **Epochs** | 50 |
| **Batch Size** | 64 |
| **Validation Split** | 0.2 |




