import pandas as pd
import numpy as np

# ======== FILE PATHS ========
ideal_path = r"C:\Users\shubh\Desktop\New folder\Ideal.xlsx"
pred_files = {
    "Visual_1_50": r"C:\Users\shubh\Desktop\New folder\Activity_Visual_1_50.csv",
    "Visual_2_50": r"C:\Users\shubh\Desktop\New folder\Activity_Visual_2_50.csv",
    "Visual_1_18": r"C:\Users\shubh\Desktop\New folder\Activity_Visual_1_18.csv",
    "Visual_2_18": r"C:\Users\shubh\Desktop\New folder\Activity_Visual_2_18.csv"
}
output_excel = r"C:\Users\shubh\Desktop\New folder\Comparison_0.1sec.xlsx"

# Offset: 55 min 58 sec = 1841 sec
offset_sec = 55*60 + 58

# ======= HELPER FUNCTIONS =======
# Convert HH:MM or HH:MM:SS to seconds
def hms_to_seconds(hms_str):
    parts = hms_str.strip().split(':')
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    elif len(parts) == 2:
        return parts[0]*60 + parts[1]
    else:
        return parts[0]

# Parse ground truth interval
def parse_interval(interval_str):
    start_str, end_str = interval_str.split('-')
    start_sec = hms_to_seconds(start_str) - offset_sec
    end_sec = hms_to_seconds(end_str) - offset_sec
    return start_sec, end_sec

# ======= LOAD GROUND TRUTH =======
df_true = pd.read_excel(ideal_path)
df_true = df_true[['time','activity']]  # remove extra columns

# ======= PROCESS EACH PREDICTION FILE =======
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:

    for sheet_name, pred_path in pred_files.items():
        print("Processing:", sheet_name)
        df_pred = pd.read_csv(pred_path)

        results = []

        # Convert prediction intervals to per 0.1 sec
        pred_intervals = []
        for _, row in df_pred.iterrows():
            start_10 = int(row['start_time_sec']*10)
            end_10 = int(row['end_time_sec']*10)
            pred_intervals.append((start_10, end_10, row['activity']))

        # Expand ground truth intervals per 0.1 sec
        for _, row in df_true.iterrows():
            gt_start_10, gt_end_10 = map(lambda x: int(x*10), parse_interval(row['time']))
            gt_act = row['activity']

            for t in range(gt_start_10, gt_end_10+1):
                # Find prediction covering this 0.1 sec slice
                pred_act = None
                for s,e,a in pred_intervals:
                    if s <= t <= e:
                        pred_act = a
                        break
                match = int(pred_act.lower() == gt_act.lower()) if pred_act else 0
                results.append([t/10, gt_act, pred_act, match])  # convert back to seconds

        # Save to dataframe
        df_res = pd.DataFrame(results, columns=['time_sec','true_activity','pred_activity','match'])
        accuracy = df_res['match'].mean()
        df_res.loc[len(df_res)] = ['', '', 'Accuracy', accuracy]

        # Write to Excel sheet
        df_res.to_excel(writer, sheet_name=sheet_name, index=False)

print("DONE! Saved to:", output_excel)
