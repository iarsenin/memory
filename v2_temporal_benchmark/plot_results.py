import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_killer_figures(results_dir: Path):
    sweep_path = results_dir / "sweep_results.csv"
    rag_path = results_dir / "rag_results.csv"
    
    if not sweep_path.exists():
        print(f"Error: {sweep_path} not found.")
        return

    df = pd.read_csv(sweep_path)
    
    # Calculate Task-Normalized Accuracy per row
    df['task_accuracy'] = (
        df['current_state_pct_current'] + 
        df['stale_premise_rejection_pct_current'] + 
        df['historical_state_pct_stale'] + 
        df['relational_after_update_pct_current']
    ) / 4.0
    
    # Calculate Forward-Time Distribution (Excluding Historical)
    df['fw_pct_current'] = (df['current_state_pct_current'] + df['stale_premise_rejection_pct_current'] + df['relational_after_update_pct_current']) / 3.0
    df['fw_pct_stale'] = (df['current_state_pct_stale'] + df['stale_premise_rejection_pct_stale'] + df['relational_after_update_pct_stale']) / 3.0
    df['fw_pct_both'] = (df['current_state_pct_both'] + df['stale_premise_rejection_pct_both'] + df['relational_after_update_pct_both']) / 3.0
    df['fw_pct_distractor'] = (df['current_state_pct_distractor'] + df['stale_premise_rejection_pct_distractor'] + df['relational_after_update_pct_distractor']) / 3.0

    agg_df = df.groupby(['sweep_type', 'volume_pct']).agg(
        task_accuracy=('task_accuracy', 'mean'),
        std_accuracy=('task_accuracy', 'std'),
        fw_pct_current=('fw_pct_current', 'mean'),
        fw_pct_stale=('fw_pct_stale', 'mean'),
        fw_pct_both=('fw_pct_both', 'mean'),
        fw_pct_distractor=('fw_pct_distractor', 'mean')
    ).reset_index()

    # ---------------------------------------------------------
    # Figure 1: The Volume-Fidelity Saturation Curve
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    random_df = agg_df[agg_df['sweep_type'] == 'random'].sort_values('volume_pct')
    ax.errorbar(random_df['volume_pct'] * 100, random_df['task_accuracy'], yerr=random_df['std_accuracy'],
                 marker='o', linestyle='-', linewidth=2, capsize=5, markersize=8,
                 label='Sweep A: Random Volume', color='#d95f02')
    
    salience_df = agg_df[agg_df['sweep_type'] == 'salience'].sort_values('volume_pct')
    ax.errorbar(salience_df['volume_pct'] * 100, salience_df['task_accuracy'], yerr=salience_df['std_accuracy'],
                 marker='s', linestyle='-', linewidth=2, capsize=5, markersize=8,
                 label='Sweep B: Salience-Ordered', color='#1b9e77')
    
    if rag_path.exists():
        rag_df = pd.read_csv(rag_path)
        rag_df['task_accuracy'] = (rag_df['current_state_pct_current'] + rag_df['stale_premise_rejection_pct_current'] + rag_df['historical_state_pct_stale'] + rag_df['relational_after_update_pct_current']) / 4.0
        rag_mean = rag_df['task_accuracy'].mean()
        rag_std = rag_df['task_accuracy'].std()
        ax.axhline(y=rag_mean, color='#7570b3', linestyle='--', linewidth=2, label=f'BM25 RAG (Top-3) [{rag_mean:.1f}%]')
        ax.fill_between([-5, 105], rag_mean - rag_std, rag_mean + rag_std, color='#7570b3', alpha=0.1)

    ax.set_title('Task-Normalized Accuracy vs. Parametric Memory Volume', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Parametric Memory Volume (% of Stable Facts Retained)', fontsize=12)
    ax.set_ylabel('Task-Normalized Accuracy (%)', fontsize=12)
    ax.set_ylim(40, 65) 
    ax.set_xlim(-5, 105)
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    fig.savefig(results_dir / 'fig1_volume_fidelity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Figure 2: Forward-Time Behavioral Superposition 
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    volumes = salience_df['volume_pct'] * 100
    x = np.arange(len(volumes) + 1)
    width = 0.6
    
    c_vals = list(salience_df['fw_pct_current'].values)
    s_vals = list(salience_df['fw_pct_stale'].values)
    b_vals = list(salience_df['fw_pct_both'].values)
    d_vals = list(salience_df['fw_pct_distractor'].values)
    labels_x = [f"LoRA\n{int(v)}% Vol" for v in volumes]

    if rag_path.exists():
        rag_fw_c = (rag_df['current_state_pct_current'].mean() + rag_df['stale_premise_rejection_pct_current'].mean() + rag_df['relational_after_update_pct_current'].mean()) / 3.0
        rag_fw_s = (rag_df['current_state_pct_stale'].mean() + rag_df['stale_premise_rejection_pct_stale'].mean() + rag_df['relational_after_update_pct_stale'].mean()) / 3.0
        rag_fw_b = (rag_df['current_state_pct_both'].mean() + rag_df['stale_premise_rejection_pct_both'].mean() + rag_df['relational_after_update_pct_both'].mean()) / 3.0
        rag_fw_d = (rag_df['current_state_pct_distractor'].mean() + rag_df['stale_premise_rejection_pct_distractor'].mean() + rag_df['relational_after_update_pct_distractor'].mean()) / 3.0
        
        c_vals.append(rag_fw_c)
        s_vals.append(rag_fw_s)
        b_vals.append(rag_fw_b)
        d_vals.append(rag_fw_d)
        labels_x.append("RAG\nTop-3")

    ax.bar(x, c_vals, width, label='Current State (Update Fidelity)', color='#2ca02c', edgecolor='white')
    ax.bar(x, s_vals, width, bottom=c_vals, label='Stale State (Endorsement)', color='#d62728', edgecolor='white')
    ax.bar(x, b_vals, width, bottom=np.array(c_vals) + np.array(s_vals), label='Behavioral Superposition (Both)', color='#ff7f0e', edgecolor='white')
    ax.bar(x, d_vals, width, bottom=np.array(c_vals) + np.array(s_vals) + np.array(b_vals), label='Distractor / Confusion', color='#7f7f7f', edgecolor='white')

    ax.set_title('Distribution of Beliefs on Forward-Time Queries\n(Salience-Ordered LoRA vs RAG)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Percentage of Answers (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)
    plt.tight_layout()
    fig.savefig(results_dir / 'fig2_superposition_stack.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Corrected charts saved to {results_dir}")

if __name__ == "__main__":
    RESULTS_DIR = Path(__file__).resolve().parent / "results"
    plot_killer_figures(RESULTS_DIR)
