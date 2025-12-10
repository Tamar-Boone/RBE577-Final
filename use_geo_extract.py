import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_training_log(file_path):
    """
    Parses the training log file to extract Epoch, Loss, and RMSE_log 
    based on the 'Batch 0' lines.
    """
    data = []
    current_epoch = None
    
    # Regex patterns based on the user's example and file content
    epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
    # Pattern matches: Batch 0: Loss = 3.6536, RMSE_log = 2.1283
    metrics_pattern = re.compile(r"Batch 0: Loss = ([\d\.]+), RMSE_log = ([\d\.]+)")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check for Epoch header
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            
            # Check for Metrics line (Batch 0)
            metrics_match = metrics_pattern.search(line)
            if metrics_match and current_epoch is not None:
                loss = float(metrics_match.group(1))
                rmse = float(metrics_match.group(2))
                
                data.append({
                    "Epoch": current_epoch,
                    "Loss": loss,
                    "RMSE_log": rmse
                })

    return pd.DataFrame(data)

def plot_metrics(df):
    """
    Creates a plot of Loss and RMSE_log vs Epoch.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Loss
    plt.plot(df["Epoch"], df["Loss"], label="Loss", color="blue", marker="o", markersize=3)
    
    # Plot RMSE_log
    plt.plot(df["Epoch"], df["RMSE_log"], label="RMSE_log", color="orange", marker="x", markersize=3)
    
    plt.title("Training Progress: Loss & RMSE_log vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    file_name = 'use_geo_results.txt'
    
    try:
        # Extract data
        df_results = parse_training_log(file_name)
        
        if not df_results.empty:
            print("Extracted Data (First 5 rows):")
            print(df_results.head())
            
            # Generate Plot
            plot_metrics(df_results)
            
            # Optional: Save extracted data to CSV
            # df_results.to_csv("extracted_training_metrics.csv", index=False)
        else:
            print("No matching data found. Please check the file format.")
            
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found in the current directory.")