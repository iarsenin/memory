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
    
    # Aggregate over seeds
    agg_df = df.groupby(['sweep_type', 'volume_pct']).agg(
        pct_current=('overall_pct_current', 'mean'),
        std_current=('overall_pct_current', 'std'),
        pct_stale=('overall_pct_stale', 'mean'),
        pct_both=('overall_pct_both', 'mean'),
        pct_distractor=('overall_pct_distractor', 'mean')
    ).reset_index()

    # ---------------------------------------------------------
    # Figure 1: The Volume-Fidelity Frontier
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sweep A (Random)
    random_df = agg_df[agg_df['sweep_type'] == 'random'].sort_values('volume_pct')
    ax.errorbar(random_df['volume_pct'] * 100, random_df['pct_current'], yerr=random_df['std_current'],
                 marker='o', linestyle='-', linewidth=2, capsize=5, markersize=8,
                 label='Sweep A: Random Volume', color='#d95f02')
    
    # Sweep B (Salience)
    salience_df = agg_df[agg_df['sweep_type'] == 'salience'].sort_values('volume_pct')
    ax.errorbar(salience_df['volume_pct'] * 100, salience_df['pct_current'], yerr=salience_df['std_current'],
                 marker='s', linestyle='-', linewidth=2, capsize=5, markersize=8,
                 label='Sweep B: Salience-Ordered', color='#1b9e77')
    
    # RAG Baseline
    if rag_path.exists():
        rag_df = pd.read_csv(rag_path)
        rag_mean = rag_df['overall_pct_current'].mean()
        rag_std = rag_df['overall_pct_current'].std()
        ax.axhline(y=rag_mean, color='#7570b3', linestyle='--', linewidth=2, 
                   label=f'BM25 RAG (Top-3) [{rag_mean:.1f}%]')
        ax.fill_between([-5, 105], rag_mean - rag_std, rag_mean + rag_std, color='#7570b3', alpha=0.1)

    ax.set_title('Update Fidelity vs. Parametric Memory Volume', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Parametric Memory Volume (% of Stable Facts Retained)', fontsize=12)
    ax.set_ylabel('Update Fidelity (Overall % Current)', fontsize=12)
    ax.set_ylim(35, 55) 
    ax.set_xlim(-5, 105)
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    fig.savefig(results_dir / 'fig1_volume_fidelity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # Figure 2: Behavioral Superposition (Stacked Bar Chart)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    volumes = random_df['volume_pct'] * 100
    x = np.arange(len(volumes) + 1) # +1 for RAG
    width = 0.6
    
    # Use Sweep B (Salience) for the stack since it shows the peak Superposition best
    c_vals = list(salience_df['pct_current'].values)
    s_vals = list(salience_df['pct_stale'].values)
    b_vals = list(salience_df['pct_both'].values)
    d_vals = list(salience_df['pct_distractor'].values)
    labels_x = [f"LoRA\n{int(v)}% Vol" for v in volumes]

    if rag_path.exists():
        c_vals.append(rag_mean)
        s_vals.append(rag_df['overall_pct_stale'].mean())
        b_vals.append(rag_df['overall_pct_both'].mean())
        d_vals.append(rag_df['overall_pct_distractor'].mean())
        labels_x.append("RAG\nTop-3")

    c_vals = np.array(c_vals)
    s_vals = np.array(s_vals)
    b_vals = np.array(b_vals)
    d_vals = np.array(d_vals)
    
    c_color, s_color, b_color, d_color = '#2ca02c', '#d62728', '#ff7f0e', '#7f7f7f' 
    
    ax.bar(x, c_vals, width, label='Current State (Update Fidelity)', color=c_color, edgecolor='white')
    ax.bar(x, s_vals, width, bottom=c_vals, label='Stale State (Endorsement)', color=s_color, edgecolor='white')
    ax.bar(x, b_vals, width, bottom=c_vals + s_vals, label='Behavioral Superposition (Both)', color=b_color, edgecolor='white')
    ax.bar(x, d_vals, width, bottom=c_vals + s_vals + b_vals, label='Distractor / Confusion', color=d_color, edgecolor='white')

    ax.set_title('Distribution of Beliefs under Fact Drift\n(Salience-Ordered LoRA vs RAG)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Percentage of Answers (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Reverse legend order to match stack visual (top to bottom)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)
    
    plt.tight_layout()
    fig.savefig(results_dir / 'fig2_superposition_stack.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Charts saved to {results_dir}")

if __name__ == "__main__":
    RESULTS_DIR = Path(__file__).resolve().parent / "results"
    plot_killer_figures(RESULTS_DIR)
