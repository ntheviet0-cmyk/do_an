import pandas as pd
cal = pd.read_csv("data/raw/calendar.csv")
print("So dong co su kien (event_type_1 notna):", cal["event_type_1"].notna().sum())
print("\nPhan bo event_type_1:")
print(cal["event_type_1"].value_counts(dropna=False))
print("\nTat ca event_name_1 xuat hien (co type):")
print(cal["event_name_1"].value_counts(dropna=True).to_string())
print("\nTong so Halloween/Christmas/Thanksgiving/IndependenceDay trong data:",
      cal["event_name_1"].isin(["Halloween","Christmas","Thanksgiving","IndependenceDay","Easter","LaborDay","MemorialDay","NewYear"]).sum())
# kiem tra cot evt_National sinh ra
import numpy as np
cal["evt_National"] = (cal["event_type_1"]=="National").astype(np.int8)
print("Tong gia tri evt_National =", cal["evt_National"].sum())
