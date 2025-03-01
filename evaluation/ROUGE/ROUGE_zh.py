import numpy as np
import jieba
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer
import csv
import sys

class SpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split()

def preprocess(text):
    return " ".join(jieba.cut(text.strip()))

def compute_rouge(reference, predictions):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        tokenizer=SpaceTokenizer()  
    )
    rouge_scores = []
    for pred in predictions:
        scores = scorer.score(preprocess(reference), preprocess(pred))
        rouge_scores.append([
            scores['rouge1'].fmeasure,
            scores['rouge2'].fmeasure,
            scores['rougeL'].fmeasure
        ])
    return np.array(rouge_scores)


csv_file = './output/explainpe/output_all_7b.csv'

csv.field_size_limit(sys.maxsize)

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  
    data = list(reader)

columns = [4, 7, 10] 

num_columns = len(columns)

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