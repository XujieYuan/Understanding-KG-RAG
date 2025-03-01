import csv
from llama_cpp import Llama
import numpy as np
import pandas as pd
from tqdm import tqdm

def chat_enllm(prompt):
    llm = Llama(
        model_path="../llama2-70b-chat",
        n_gpu_layers=200,
        n_batch=512,
        n_ctx=2048,
        chat_format="llama-2",
        repeat_penalty=1.0,
        verbose=False,
    )
    completion = llm.create_chat_completion(
      messages=[
          {"role": "system", "content": "You are an AI assistant specialized in evaluating medical question-answering quality."},
          {"role": "user", "content": prompt}
      ]
    )
    response = completion["choices"][0]["message"]["content"]
    print("english llm response:", response)
    return response


def create_evaluation_prompt_english(question, answer, method):
    prompt = f"""You will evaluate a summary (answer) written for a medical question. Your task is to rate the summary on four metrics. Please read and understand these instructions carefully.

                    Question: {question}

                    {method} method's answer: {answer}

                    Evaluation Criteria:
                    1. Context Relevance (1-5) - How well does the answer relate to the context of the question? Does it address the specific medical issue raised?
                    2. Comprehensiveness (1-5) - Does the answer cover all key aspects of the question? Is it thorough in its explanation?
                    3. Correctness (1-5) - Is the medical information provided in the answer accurate and up-to-date? Are there any factual errors?
                    4. Empowerment (1-5) - Does the answer provide actionable insights or advice? Does it empower the reader to make informed decisions about their health?

                    Evaluation Steps:
                    1. Read the question carefully and identify the main medical issue and key points.
                    2. Read the answer and compare it to the question. Check if the answer addresses the main issue and key points of the question.
                    3. Assign a score for each criterion on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

                    Please provide your evaluation in the following format:
                    Context Relevance: [score]
                    Comprehensiveness: [score]
                    Correctness: [score]
                    Empowerment: [score]

                    Only include the scores, no additional explanation is needed.
                    """
    return prompt


def parse_scores(evaluation_text):
    lines = evaluation_text.split('\n')
    scores = {}
    score_mapping = {
        'Context Relevance': 'Context Relevance',
        'Comprehensiveness': 'Comprehensiveness',
        'Correctness': 'Correctness',
        'Empowerment': 'Empowerment'
    }
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':')
            if key.strip() in score_mapping:
                scores[score_mapping[key.strip()]] = float(value.strip())
    return scores

def evaluate_answers():
    input_file = './output/chatdoctor5k/output_all_7b.csv'
    output_file = './output/chatdoctor5k/output_all_7b_g_eval_results.csv'

    methods = [
        'PreAblation-summary', 'Subgraph+cot_summary', 
        'Subgraph+tot_summary', 'Subgraph+mindmap_summary', 
        'Subgraph_noPrompt', 'path+cot_summary', 'path+tot_summary', 
        'path+mindmap_summary', 'path_noPrompt', 'facts+cot_summary', 
        'facts+tot_summary', 'facts+mindmap_summary', 'facts_noPrompt'
    ]

    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)

        new_fields = []
        for method in methods:
            for metric in ['Context_Relevance', 'Comprehensiveness', 'Correctness', 'Empowerment']:
                new_fields.append(f'{method}_{metric}')
        
        fieldnames = reader.fieldnames + new_fields
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="evaluating"):
            question = row['Question']
            

            for method in methods:
                if method in row and row[method].strip():  
                    scores = parse_scores(chat_enllm(create_evaluation_prompt_english(question, row[method], method)))
                    
                    for metric, score in scores.items():
                        metric_key = f'{method}_{metric.replace(" ", "_")}'
                        row[metric_key] = score

            writer.writerow(row)


def calculate_stats(scores):
    mean = np.mean(scores) * 20 
    std = np.std(scores) * 20
    return f"{mean:.2f}% ± {std:.2f}%"

def analyze_results(input_file):
    df = pd.read_csv(input_file)
    
    methods = [
        'PreAblation-summary', 'Subgraph+cot_summary', 
        'Subgraph+tot_summary', 'Subgraph+mindmap_summary', 
        'Subgraph_noPrompt', 'path+cot_summary', 'path+tot_summary', 
        'path+mindmap_summary', 'path_noPrompt', 'facts+cot_summary', 
        'facts+tot_summary', 'facts+mindmap_summary', 'facts_noPrompt'
    ]

    metrics = ['Context_Relevance', 'Comprehensiveness', 'Correctness', 'Empowerment']
    
    results = []
    
    for method in methods:
        method_results = [method]
        for metric in metrics:
            column = f"{method}_{metric}"
            scores = df[column].dropna().values
            method_results.append(calculate_stats(scores))
        results.append(method_results)
    
    headers = ["Method", "Context_Relevance", "Comprehensiveness", "Correctness", "Empowerment"]
    table = [headers] + results
    
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    
    print_table(table, col_widths)

def print_table(table, col_widths):
    header_str = "| " + " | ".join(f"{'':<{w}}" for w in col_widths) + " |"
    print("-" * len(header_str))
    
    header = table[0]
    header_str = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)) + " |"
    print(header_str)
    print("-" * len(header_str))
    
    for row in table[1:]:
        row_str = "| " + " | ".join(f"{str(cell):<{w}}" for cell, w in zip(row, col_widths)) + " |"
        print(row_str)
    
    print("-" * len(header_str))

if __name__ == "__main__":
    evaluate_answers()
    
    input_file = './output/chatdoctor5k/output_all_7b_g_eval_results.csv'
    analyze_results(input_file)