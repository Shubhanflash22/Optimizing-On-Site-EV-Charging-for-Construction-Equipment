import os
import pandas as pd

shrinking_dir = r"C:\Users\shubh\Desktop\MPC\Approach 1\Shrinking_Horizon\data\input_data"
receding_dir  = r"C:\Users\shubh\Desktop\MPC\Approach 1\Receding_Horizon\data\input_data"

files = [
    "time_data.csv",
    "travel_time.csv",
    "work_flexible.csv",
    "ev_data.csv",
    "mcs_data.csv",
    "parameters.csv",
    "place.csv"
]

for file in files:
    print(f"\n{'='*60}")
    print(f"Comparing: {file}")

    f1 = os.path.join(shrinking_dir, file)
    f2 = os.path.join(receding_dir, file)

    if not os.path.exists(f1):
        print("Missing in Shrinking_Horizon")
        continue
    if not os.path.exists(f2):
        print("Missing in Receding_Horizon")
        continue

    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)

    # Check shapes first
    if df1.shape != df2.shape:
        print(f"Different shapes:")
        print(f"  Shrinking: {df1.shape}")
        print(f"  Receding : {df2.shape}")
        continue

    # Exact equality
    if df1.equals(df2):
        print("No differences.")
        continue

    print("Differences found.")

    # Compare values
    diff = df1.compare(df2, keep_shape=False, keep_equal=False)

    if diff.empty:
        print("Only datatype/index differences.")
    else:
        print(f"Number of differing rows: {len(diff)}")
        print("\nFirst few differences:")
        print(diff.head(20))