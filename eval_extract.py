import pandas as pd
import re
import matplotlib.pyplot as plt
import os

# Read the content of the uploaded file
file_content = "evaluation_results.txt"
with open(file_content, 'r') as file:
        file_content = file.read()
# print(file_content) 
# The regular expression pattern to capture the step and all metrics/values
# 1. Capture the step number: (\d+)/\d+
# 2. Capture all the metric/value pairs that follow: (?: - (\w+): ([\d\.]+e?[\+\-]?\d?))*
#    - (?:...) is a non-capturing group.
#    - \w+ is the metric name (e.g., AbsRel).
#    - [\d\.]+e?[\+\-]?\d? is the value (handling floating point, scientific notation, etc.).
pattern = re.compile(r'\s*(\d+)/\d+.*?(?: - (\w+): ([\d\.]+e?[\+\-]?\d?))*')

# Dictionary to hold the data, structured by column
data = {
    'Step': [],
    'AbsRel': [],
    'SqRel': [],
    'RMSE': [],
    'RMSE_log': [],
    'Delta1': [],
    'Delta2': [],
    'Delta3': []
}

# Process each line
for line in file_content.splitlines():
    # Only process lines that contain a progress indicator (e.g., '1/3849')
    if '/' in line and ' - ' in line:
        try:
            # 1. Extract the step number
            step_match = re.search(r'\s*(\d+)/\d+', line)
            if step_match:
                step = int(step_match.group(1))
                data['Step'].append(step)
            else:
                continue # Skip lines without a step number

            # 2. Extract metrics and values
            # Find all metric: value pairs
            metric_matches = re.findall(r'(\w+): ([\d\.]+e?[\+\-]?\d?)', line)

            # Convert to a temporary dictionary for easy lookup
            metrics_dict = {metric: float(value) for metric, value in metric_matches}

            # 3. Populate the data dictionary
            for key in ['AbsRel', 'SqRel', 'RMSE', 'RMSE_log', 'Delta1', 'Delta2', 'Delta3']:
                # Use .get() in case a metric is missing in a specific line (though unlikely here)
                value = metrics_dict.get(key)
                if value is not None:
                    data[key].append(value)
                else:
                    # Pad missing values if necessary to keep columns aligned (shouldn't happen here)
                    if len(data[key]) < len(data['Step']):
                         data[key].append(None)

        except Exception as e:
            # Optional: print problematic lines for debugging
            # print(f"Error processing line: {line}. Error: {e}")
            pass

# Create the final DataFrame
df = pd.DataFrame(data)

print("DataFrame Head (First 5 Rows):")
print(df.head())
print("\nDataFrame Info:")
df.info()

# Assuming you have the 'df' DataFrame from the extraction step above
# You typically want to plot 'RMSE' and/or 'AbsRel' against 'Step'

# Create a figure and an axes object
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Plot 1: Root Mean Squared Error (RMSE) ---
color = 'tab:red'
ax1.set_xlabel('Evaluation Step')
ax1.set_ylabel('RMSE (Lower is Better)', color=color)
ax1.plot(df['Step'], df['RMSE'], color=color, label='RMSE', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Plot 2: Root Mean Squared Error log (RMSE_log) on a secondary Y-axis ---
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('RMSE_log (Lower is Better)', color=color)
ax2.plot(df['Step'], df['RMSE_log'], color=color, label='RMSE_log', linestyle='--', alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)

# Add a title and show the plot
plt.title('Model Performance Metrics Over Evaluation Steps')
fig.tight_layout()  # ensures the labels don't overlap
plt.show()
