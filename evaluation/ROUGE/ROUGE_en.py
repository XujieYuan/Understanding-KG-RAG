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


csv_file = './output/chatdoctor5k/output_all_7b.csv'

csv.field_size_limit(sys.maxsize)

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  
    data = list(reader)

columns = [5, 8, 10, 12, 13, 16, 18, 20, 21, 24, 26, 28, 29]

num_columns = len(columns)

# 初始化存储 ROUGE 结果的数组
rouge1_scores = [[] for _ in range(num_columns)]
rouge2_scores = [[] for _ in range(num_columns)]
rougeL_scores = [[] for _ in range(num_columns)]

for i, row in enumerate(data):
    try:
        reference = row[1]  
        candidate_answers = [row[c] for c in columns]  

        rouge_scores = compute_rouge(reference, candidate_answers)

        for j in range(num_columns):
            rouge1_scores[j].append(rouge_scores[j, 0])
            rouge2_scores[j].append(rouge_scores[j, 1])
            rougeL_scores[j].append(rouge_scores[j, 2])

        print(f"Row {i}: ROUGE-1: {[round(r, 4) for r in rouge_scores[:, 0]]}")

    except Exception as e:
        print(f"Row {i}: Error {e}. Skipping this row.")

for j in range(num_columns):
    avg_r1 = np.mean(rouge1_scores[j]) if len(rouge1_scores[j]) > 0 else None
    avg_r2 = np.mean(rouge2_scores[j]) if len(rouge2_scores[j]) > 0 else None
    avg_rL = np.mean(rougeL_scores[j]) if len(rougeL_scores[j]) > 0 else None
    print(f"Method {j + 1} - Avg ROUGE-1: {avg_r1:.4f}, Avg ROUGE-2: {avg_r2:.4f}, Avg ROUGE-L: {avg_rL:.4f}")
