"""
Evaluation harness for the Packi revision (T0.7, R1.3/R1.5/R1.6).

    python evaluate.py --method greedy --dataset data1 --seed 0
    python aggregate.py results/

Every run reads a fixed sequence file from data/sequences/, runs one policy
through the same validator and retry budget, and writes one JSON to results/.
"""
