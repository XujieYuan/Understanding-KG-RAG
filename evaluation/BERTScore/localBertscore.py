# # BERT-base-uncased
from bert_score import score
from transformers import AutoModel, AutoTokenizer
import csv
import sys

def load_model(model_path):
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def calculate_scores(row, columns, model_path):
    P_scores = []
    R_scores = []
    F1_scores = []

    for column in columns:
        try:
            P, R, F1 = score([row[column]], [row[1]], model_type=model_path, num_layers=12, idf=False)
            P_scores.append(P.numpy()[0])
            R_scores.append(R.numpy()[0])
            F1_scores.append(F1.numpy()[0])
        except Exception as e:
            print(f"Error {e}. Skipping this column.")

    return P_scores, R_scores, F1_scores

# BERT model path
# model_path = "../bert-base-uncased"
model_path = "../bert-base-chinese"  

model, tokenizer = load_model(model_path)

csv.field_size_limit(sys.maxsize)
# Open the CSV file
with open('./output/chatdoctor5k/output_KG-RAG_7b.csv', 'r', encoding='utf-8') as f:

    reader = csv.reader(f)
    header = next(reader)  # Skip the header
    data = list(reader)

# Define the number of columns to process
num_columns =3  # Change this value as needed

# Initialize the score lists
P_scores = [[] for _ in range(num_columns)]
R_scores = [[] for _ in range(num_columns)]
F1_scores = [[] for _ in range(num_columns)]

# Evaluate each row and add the result to the new column
for i, row in enumerate(data):
    try:
        # Calculate BERTScore for each column
        columns = [4, 7, 10]
        
        for j in range(num_columns):  
            P, R, F1 = score([row[columns[j]]], [row[1]], model_type=model_path, num_layers=12, idf=False)
            
            # Add the scores to the lists
            P_scores[j].append(P.numpy()[0])
            R_scores[j].append(R.numpy()[0])
            F1_scores[j].append(F1.numpy()[0])

        print(f"Row {i}:")
        for j in range(num_columns):  
            print(f"Column {j + 1}: Precision {P_scores[j][-1]}, Recall {R_scores[j][-1]}, F1 {F1_scores[j][-1]}")
    except Exception as e:
        print(f"Row {i}: Error {e}. Skipping this row.")

for j in range(num_columns):  
    avg_P = sum(P_scores[j]) / len(P_scores[j]) if P_scores[j] else None
    avg_R = sum(R_scores[j]) / len(R_scores[j]) if R_scores[j] else None
    avg_F1 = sum(F1_scores[j]) / len(F1_scores[j]) if F1_scores[j] else None
    print(f"Column {j + 1} Average Precision: {avg_P}")
    print(f"Column {j + 1} Average Recall: {avg_R}")
    print(f"Column {j + 1} Average F1: {avg_F1}")