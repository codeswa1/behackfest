import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    start_time = datetime(2024, 1, 1, 10, 0, 0)
    data = []
    
    signals = {
        "motor_temp": {"base": 45, "noise": 1.0},
        "vibration": {"base": 0.5, "noise": 0.05},
        "pressure": {"base": 30, "noise": 0.5},
        "rpm": {"base": 1500, "noise": 10}
    }
    
    # Generate 60 minutes of data
    for i in range(60):
        ts = start_time + timedelta(minutes=i)
        
        for name, params in signals.items():
            val = params["base"] + np.random.normal(0, params["noise"])
            
            # Anomaly 1: Sudden spike at 10:15 - 10:20
            if 15 <= i <= 20:
                if name == "motor_temp": val += 40
                if name == "vibration": val += 2.5
                if name == "pressure": val += 30
                if name == "rpm": val -= 800
            
            # Anomaly 2: Gradual drift at 10:40 - 10:45
            if 40 <= i <= 45:
                if name == "motor_temp": val += (i - 40) * 5
                if name == "vibration": val += (i - 40) * 0.2
            
            data.append({"signal_id": name, "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "value": val})

    pd.DataFrame(data).to_csv("data/timeseries.csv", index=False)
    print("Generated 60 minutes of data with 2 anomalies in data/timeseries.csv")

    logs = [
        {"timestamp": "2024-01-01 10:05:00", "log": "System health check passed. All parameters nominal."},
        {"timestamp": "2024-01-01 10:15:30", "log": "Acoustic sensor picked up unusual grinding noise."},
        {"timestamp": "2024-01-01 10:17:00", "log": "Thermal alerts triggered on motor housing."},
        {"timestamp": "2024-01-01 10:21:00", "log": "Manual override engaged to stabilize RPM."},
        {"timestamp": "2024-01-01 10:42:00", "log": "Operator noted slight increase in housing temperature."},
        {"timestamp": "2024-01-01 10:50:00", "log": "Routine maintenance completed after drift observation."}
    ]
    pd.DataFrame(logs).to_csv("data/operator_logs.csv", index=False)
    print("Updated data/operator_logs.csv")

if __name__ == "__main__":
    generate_data()
