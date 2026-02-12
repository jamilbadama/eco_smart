import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_results(modes=['audio', 'video', 'text', 'multimodal']):
    sns.set_theme(style="whitegrid")
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    for mode in modes:
        log_file = f"experiments/log_{mode}.csv"
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            plt.plot(df['epoch'], df['train_loss'], label=f"{mode} Loss")
    
    plt.title("Training Loss per Modality")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("experiments/loss_comparison.png")
    plt.close()
    
    # Plot F1
    plt.figure(figsize=(10, 6))
    for mode in modes:
        log_file = f"experiments/log_{mode}.csv"
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            plt.plot(df['epoch'], df['dev_f1'], label=f"{mode} F1")
            
    plt.title("Validation F1 Score per Modality")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig("experiments/f1_comparison.png")
    plt.close()
    
    # Plot RMSE
    plt.figure(figsize=(10, 6))
    for mode in modes:
        log_file = f"experiments/log_{mode}.csv"
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            plt.plot(df['epoch'], df['dev_rmse'], label=f"{mode} RMSE")
            
    plt.title("Validation RMSE per Modality")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("experiments/rmse_comparison.png")
    plt.close()
    
    print("Graphs saved to experiments/ folder.")

if __name__ == "__main__":
    plot_results()
