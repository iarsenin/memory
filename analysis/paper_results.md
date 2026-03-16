# MemLoRA v1 — Paper Results

Seeds: [42, 123, 456] · Personas: ['alice', 'bob', 'charlie', 'diana', 'ethan', 'fiona', 'george', 'hannah', 'ian', 'julia']
Accuracy = mean ± std (%) across seeds. Frozen/RAG std = 0 (deterministic).

## Main Results Table

| Condition | Stable | Updated | Superseded | Relational | Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MemLoRA (ours)** | 24.7 ±4.7 | 41.7 ±4.6 | 25.6 ±6.9 | 44.4 ±4.8 | **35.5 ±4.6** |
| Naïve LoRA | 26.1 ±3.4 | 32.8 ±3.4 | 11.7 ±5.8 | 25.0 | 31.0 ±2.1 |
| Unfiltered LoRA | 45.6 ±9.9 | 50.1 ±0.6 | 20.3 ±19.3 | 47.2 ±12.7 | 44.8 ±0.7 |
| Oracle-Data LoRA (upper bound) | 38.1 ±19.8 | 60.0 ±9.6 | 51.1 ±16.9 | 52.8 ±12.7 | 50.1 ±5.8 |
| Ablation: No Salience Weighting | 37.9 ±10.0 | 39.8 ±3.8 | 9.2 ±1.2 | 33.3 ±11.8 | 36.1 ±6.1 |
| Ablation: No Replay Buffer | 21.7 ±10.6 | 41.7 ±2.4 | 18.3 ±14.1 | 50.0 | 34.0 ±2.4 |
| Ablation: No Anti-Memory Pairs | 32.5 ±2.4 | 39.4 ±5.0 | 17.5 ±1.2 | 33.3 ±11.8 | 35.8 ±3.5 |

### Per-Seed Overall Accuracy

| Condition | Seed 42 | Seed 123 | Seed 456 |
|:---|:---:|:---:|:---:|
| **MemLoRA (ours)** | 33.7% | 40.7% | 32.0% |
| Naïve LoRA | 33.3% | 29.5% | 30.2% |
| Unfiltered LoRA | 44.8% | 44.1% | 45.4% |
| Oracle-Data LoRA (upper bound) | 43.4% | 53.8% | 53.0% |
| Ablation: No Salience Weighting | 40.5% | 31.8% | — |
| Ablation: No Replay Buffer | 35.7% | 32.3% | — |
| Ablation: No Anti-Memory Pairs | 33.3% | 38.2% | — |

## Bucket Detail (mean % ± std)

### **MemLoRA (ours)**
  - Stable: 24.7 ±4.7
  - Updated: 41.7 ±4.6
  - Superseded: 25.6 ±6.9
  - Relational: 44.4 ±4.8
  - Overall: 35.5 ±4.6

### Naïve LoRA
  - Stable: 26.1 ±3.4
  - Updated: 32.8 ±3.4
  - Superseded: 11.7 ±5.8
  - Relational: 25.0
  - Overall: 31.0 ±2.1

### Unfiltered LoRA
  - Stable: 45.6 ±9.9
  - Updated: 50.1 ±0.6
  - Superseded: 20.3 ±19.3
  - Relational: 47.2 ±12.7
  - Overall: 44.8 ±0.7

### Oracle-Data LoRA (upper bound)
  - Stable: 38.1 ±19.8
  - Updated: 60.0 ±9.6
  - Superseded: 51.1 ±16.9
  - Relational: 52.8 ±12.7
  - Overall: 50.1 ±5.8

### Ablation: No Salience Weighting
  - Stable: 37.9 ±10.0
  - Updated: 39.8 ±3.8
  - Superseded: 9.2 ±1.2
  - Relational: 33.3 ±11.8
  - Overall: 36.1 ±6.1

### Ablation: No Replay Buffer
  - Stable: 21.7 ±10.6
  - Updated: 41.7 ±2.4
  - Superseded: 18.3 ±14.1
  - Relational: 50.0
  - Overall: 34.0 ±2.4

### Ablation: No Anti-Memory Pairs
  - Stable: 32.5 ±2.4
  - Updated: 39.4 ±5.0
  - Superseded: 17.5 ±1.2
  - Relational: 33.3 ±11.8
  - Overall: 35.8 ±3.5