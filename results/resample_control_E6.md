| Model | mode | runs | C-25 | DATA-1 | DATA-2 | DATA-3 | mean | first-try | retries/box | boxes w/ retry | exhausted/run | retry = previous (pick · path) | 1st decision = plain | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-4o-mini | plain | 20 | 0.729 ± 0.065 | 0.653 ± 0.061 | 0.676 ± 0.030 | 0.609 ± 0.060 | **0.667** | 1.00 | 0.00 | 0.0% | 0.00 | – · – | 20/20 | 0.082 |
| openai/gpt-4o-mini | nofb | 20 | 0.693 ± 0.109 | 0.587 ± 0.056 | 0.687 ± 0.049 | 0.613 ± 0.069 | **0.645** | 1.00 | 0.00 | 0.0% | 0.00 | – · – | 15/20 | 0.079 |
| openai/gpt-4o-mini | resample | 20 | 0.706 ± 0.102 | 0.634 ± 0.051 | 0.696 ± 0.046 | 0.638 ± 0.112 | **0.668** | 1.00 | 0.00 | 0.0% | 0.00 | – · – | 16/20 | 0.082 |
| meta-llama/llama-4-maverick | plain | 20 | 0.689 ± 0.062 | 0.632 ± 0.114 | 0.705 ± 0.090 | 0.652 ± 0.067 | **0.669** | 0.84 | 0.20 | 16.5% | 0.65 | 29/60 · 2/52 | 20/20 | 0.122 |
| meta-llama/llama-4-maverick | nofb | 20 | 0.696 ± 0.067 | 0.587 ± 0.085 | 0.643 ± 0.083 | 0.589 ± 0.068 | **0.629** | 0.80 | 0.86 | 22.3% | 4.00 | 167/188 · 187/212 | 20/20 | 0.167 |
| meta-llama/llama-4-maverick | resample | 20 | 0.680 ± 0.102 | 0.564 ± 0.076 | 0.708 ± 0.030 | 0.627 ± 0.066 | **0.645** | 0.83 | 0.44 | 16.5% | 1.85 | 78/107 · 44/87 | 20/20 | 0.148 |
| meta-llama/llama-4-maverick | resample-t1 | 20 | 0.672 ± 0.092 | 0.609 ± 0.084 | 0.674 ± 0.105 | 0.643 ± 0.034 | **0.649** | 0.87 | 0.32 | 13.3% | 1.20 | 70/77 · 30/60 | 20/20 | 0.141 |
| deepseek/deepseek-v3.1-terminus | plain | 20 | 0.694 ± 0.056 | 0.615 ± 0.060 | 0.768 ± 0.054 | 0.671 ± 0.119 | **0.687** | 0.90 | 0.17 | 9.8% | 0.05 | 11/15 · 12/51 | 20/20 | 0.149 |
| deepseek/deepseek-v3.1-terminus | nofb | 20 | 0.696 ± 0.062 | 0.685 ± 0.060 | 0.771 ± 0.072 | 0.668 ± 0.054 | **0.705** | 0.93 | 0.15 | 8.0% | 0.00 | 11/18 · 10/40 | 14/20 | 0.147 |
| deepseek/deepseek-v3.1-terminus | resample | 20 | 0.706 ± 0.089 | 0.643 ± 0.076 | 0.696 ± 0.108 | 0.664 ± 0.071 | **0.677** | 0.90 | 0.22 | 10.8% | 0.10 | 10/29 · 8/60 | 11/20 | 0.152 |

Paired utilization differences over shared sequences (mean diff, Wilcoxon p, paired-t p):
  openai/gpt-4o-mini                 plain    - resample n=20  -0.002  p_W=0.845  p_t=0.914
  openai/gpt-4o-mini                 plain    - nofb     n=20  +0.022  p_W=0.133  p_t=0.122
  openai/gpt-4o-mini                 resample - nofb     n=20  +0.023  p_W=0.052  p_t=0.047
  meta-llama/llama-4-maverick        plain    - resample n=20  +0.025  p_W=0.126  p_t=0.127
  meta-llama/llama-4-maverick        plain    - nofb     n=20  +0.041  p_W=0.033  p_t=0.045
  meta-llama/llama-4-maverick        resample - nofb     n=20  +0.016  p_W=0.245  p_t=0.388
  meta-llama/llama-4-maverick        plain    - resample-t1 n=20  +0.020  p_W=0.046  p_t=0.051
  meta-llama/llama-4-maverick        resample-t1 - nofb     n=20  +0.021  p_W=0.306  p_t=0.265
  meta-llama/llama-4-maverick        resample-t1 - resample n=20  +0.005  p_W=0.727  p_t=0.786
  deepseek/deepseek-v3.1-terminus    plain    - resample n=20  +0.010  p_W=0.756  p_t=0.563
  deepseek/deepseek-v3.1-terminus    plain    - nofb     n=20  -0.018  p_W=0.255  p_t=0.253
  deepseek/deepseek-v3.1-terminus    resample - nofb     n=20  -0.028  p_W=0.207  p_t=0.160

Paired budget-exhausted boxes per run differences over shared sequences (mean diff, Wilcoxon p, paired-t p):
  openai/gpt-4o-mini                 plain    - resample n=20  +0.00  p_W=1.000  p_t=1.000
  openai/gpt-4o-mini                 plain    - nofb     n=20  +0.00  p_W=1.000  p_t=1.000
  openai/gpt-4o-mini                 resample - nofb     n=20  +0.00  p_W=1.000  p_t=1.000
  meta-llama/llama-4-maverick        plain    - resample n=20  -1.20  p_W=0.009  p_t=0.023
  meta-llama/llama-4-maverick        plain    - nofb     n=20  -3.35  p_W=0.004  p_t=0.006
  meta-llama/llama-4-maverick        resample - nofb     n=20  -2.15  p_W=0.040  p_t=0.032
  meta-llama/llama-4-maverick        plain    - resample-t1 n=20  -0.55  p_W=0.061  p_t=0.110
  meta-llama/llama-4-maverick        resample-t1 - nofb     n=20  -2.80  p_W=0.015  p_t=0.016
  meta-llama/llama-4-maverick        resample-t1 - resample n=20  -0.65  p_W=0.473  p_t=0.251
  deepseek/deepseek-v3.1-terminus    plain    - resample n=20  -0.05  p_W=0.655  p_t=0.666
  deepseek/deepseek-v3.1-terminus    plain    - nofb     n=20  +0.05  p_W=0.317  p_t=0.330
  deepseek/deepseek-v3.1-terminus    resample - nofb     n=20  +0.10  p_W=0.317  p_t=0.330
