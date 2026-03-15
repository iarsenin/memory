# MemLoRA v1 — Paper Results

Seeds: [42, 123, 456] · Personas: ['alice', 'bob']
Accuracy = mean ± std (%) across seeds. Frozen/RAG std = 0 (deterministic).

## Main Results Table

| Condition | Stable | Updated | Superseded | Relational | Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| Frozen (base) | 0.0 | 0.0 | 41.7 | 8.3 | 17.9 |
| RAG (no adapter) | 58.3 | 58.3 | 33.3 | 33.3 | 39.3 |
| Naïve LoRA | 11.1 ±4.8 | 4.2 ±7.2 | 8.3 ±5.8 | 25.0 | 11.9 ±2.7 |
| Unfiltered LoRA | 22.2 ±25.5 | 41.0 ±17.7 | 22.2 ±19.9 | 47.2 ±12.7 | 31.0 ±1.0 |
| Gold LoRA (upper bound) | 27.8 ±34.7 | 20.8 ±18.2 | 9.4 ±7.7 | 47.2 ±17.3 | 22.0 ±8.8 |
| **MemLoRA (ours)** | 22.2 ±25.5 | 36.8 ±7.9 | 15.0 ±13.2 | 38.9 ±12.7 | **25.0 ±3.1** |

### Per-Seed Overall Accuracy

| Condition | Seed 42 | Seed 123 | Seed 456 |
|:---|:---:|:---:|:---:|
| Frozen (base) | 17.9% | (=seed42) | (=seed42) |
| RAG (no adapter) | 39.3% | (=seed42) | (=seed42) |
| Naïve LoRA | 14.3% | 8.9% | 12.5% |
| Unfiltered LoRA | 30.4% | 32.1% | 30.4% |
| Gold LoRA (upper bound) | 32.1% | 16.1% | 17.9% |
| **MemLoRA (ours)** | 21.4% | 26.8% | 26.8% |

## Bucket Detail (mean % ± std)

### Frozen (base)
  - Stable: 0.0
  - Updated: 0.0
  - Superseded: 41.7
  - Relational: 8.3
  - Overall: 17.9

### RAG (no adapter)
  - Stable: 58.3
  - Updated: 58.3
  - Superseded: 33.3
  - Relational: 33.3
  - Overall: 39.3

### Naïve LoRA
  - Stable: 11.1 ±4.8
  - Updated: 4.2 ±7.2
  - Superseded: 8.3 ±5.8
  - Relational: 25.0
  - Overall: 11.9 ±2.7

### Unfiltered LoRA
  - Stable: 22.2 ±25.5
  - Updated: 41.0 ±17.7
  - Superseded: 22.2 ±19.9
  - Relational: 47.2 ±12.7
  - Overall: 31.0 ±1.0

### Gold LoRA (upper bound)
  - Stable: 27.8 ±34.7
  - Updated: 20.8 ±18.2
  - Superseded: 9.4 ±7.7
  - Relational: 47.2 ±17.3
  - Overall: 22.0 ±8.8

### **MemLoRA (ours)**
  - Stable: 22.2 ±25.5
  - Updated: 36.8 ±7.9
  - Superseded: 15.0 ±13.2
  - Relational: 38.9 ±12.7
  - Overall: 25.0 ±3.1