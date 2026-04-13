import argparse
import csv
import sys
import numpy as np
from rouge_score import rouge_scorer


def compute_rouge(reference, predictions):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    rouge_scores = []
    for pred in predictions:
        scores = scorer.score(reference, pred)
        rouge_scores.append([
            scores['rouge1'].fmeasure,
            scores['rouge2'].fmeasure,
            scores['rougeL'].fmeasure
        ])
    return np.array(rouge_scores)


def main():
    parser = argparse.ArgumentParser(description="Compute English ROUGE scores for pilot output columns")
    parser.add_argument('--input', required=True, help='Path to results CSV file')
    parser.add_argument('--pred-cols', nargs='+', type=int, required=True,
                        help='Column indices (0-based) of predicted answers to evaluate')
    parser.add_argument('--ref-col', type=int, default=1,
                        help='Column index (0-based) of reference answers (default: 1)')
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)

    num_columns = len(args.pred_cols)
    rouge1_scores = [[] for _ in range(num_columns)]
    rouge2_scores = [[] for _ in range(num_columns)]
    rougeL_scores = [[] for _ in range(num_columns)]

    for i, row in enumerate(data):
        try:
            reference = row[args.ref_col]
            candidate_answers = [row[c] for c in args.pred_cols]

            scores = compute_rouge(reference, candidate_answers)

            for j in range(num_columns):
                rouge1_scores[j].append(scores[j, 0])
                rouge2_scores[j].append(scores[j, 1])
                rougeL_scores[j].append(scores[j, 2])

            print(f"Row {i}: ROUGE-1: {[round(r, 4) for r in scores[:, 0]]}")
        except Exception as e:
            print(f"Row {i}: Error {e}. Skipping.")

    print("\n=== Average ROUGE Scores ===")
    for j, col in enumerate(args.pred_cols):
        col_name = header[col] if col < len(header) else str(col)
        avg_r1 = np.mean(rouge1_scores[j]) if rouge1_scores[j] else None
        avg_r2 = np.mean(rouge2_scores[j]) if rouge2_scores[j] else None
        avg_rL = np.mean(rougeL_scores[j]) if rougeL_scores[j] else None
        print(f"{col_name} - ROUGE-1: {avg_r1:.4f}  ROUGE-2: {avg_r2:.4f}  ROUGE-L: {avg_rL:.4f}")


if __name__ == "__main__":
    main()
