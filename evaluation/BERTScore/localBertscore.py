import argparse
import csv
import sys
import numpy as np
from bert_score import score


def main():
    parser = argparse.ArgumentParser(description="Compute BERTScore for pilot output columns")
    parser.add_argument('--input', required=True, help='Path to results CSV file')
    parser.add_argument('--pred-cols', nargs='+', type=int, required=True,
                        help='Column indices (0-based) of predicted answers to evaluate')
    parser.add_argument('--ref-col', type=int, default=1,
                        help='Column index (0-based) of reference answers (default: 1)')
    parser.add_argument('--model', default='bert-base-uncased',
                        help='BERT model path or HuggingFace name for BERTScore')
    parser.add_argument('--layers', type=int, default=12,
                        help='Number of layers to use (default: 12)')
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)

    num_columns = len(args.pred_cols)
    P_scores = [[] for _ in range(num_columns)]
    R_scores = [[] for _ in range(num_columns)]
    F1_scores = [[] for _ in range(num_columns)]

    for i, row in enumerate(data):
        try:
            ref = row[args.ref_col]
            for j, col in enumerate(args.pred_cols):
                P, R, F1 = score([row[col]], [ref], model_type=args.model,
                                 num_layers=args.layers, idf=False)
                P_scores[j].append(P.numpy()[0])
                R_scores[j].append(R.numpy()[0])
                F1_scores[j].append(F1.numpy()[0])

            print(f"Row {i}:")
            for j in range(num_columns):
                print(f"  Col {args.pred_cols[j]}: P={P_scores[j][-1]:.4f}  "
                      f"R={R_scores[j][-1]:.4f}  F1={F1_scores[j][-1]:.4f}")
        except Exception as e:
            print(f"Row {i}: Error {e}. Skipping.")

    print("\n=== Average BERTScores ===")
    for j, col in enumerate(args.pred_cols):
        col_name = header[col] if col < len(header) else str(col)
        avg_P = np.mean(P_scores[j]) if P_scores[j] else None
        avg_R = np.mean(R_scores[j]) if R_scores[j] else None
        avg_F1 = np.mean(F1_scores[j]) if F1_scores[j] else None
        print(f"{col_name}: Precision={avg_P:.4f}  Recall={avg_R:.4f}  F1={avg_F1:.4f}")


if __name__ == "__main__":
    main()
