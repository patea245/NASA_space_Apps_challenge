import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv(r'C:\Users\owner\Downloads\Dummy_Data_2018.csv')
date_strings = list(df['datetime'])
date_objects = [datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ") for date_str in date_strings]

timestamps = [date_obj.timestamp() for date_obj in date_objects]
timestamp_array = np.array(timestamps)
timestamp_array = timestamp_array.reshape(-1, 1)
Kp_array = np.array(df['Kp'])
