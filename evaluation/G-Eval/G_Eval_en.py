import argparse
import csv
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama


def build_llm(model_path):
    return Llama(
        model_path=model_path,
        n_gpu_layers=200,
        n_batch=512,
        n_ctx=2048,
        chat_format="llama-2",
        repeat_penalty=1.0,
        verbose=False,
    )


def chat_llm(llm, prompt):
    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in evaluating medical question-answering quality."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion["choices"][0]["message"]["content"]
    print("english llm response:", response)
    return response


def create_evaluation_prompt(question, answer, method):
    return f"""You will evaluate a summary (answer) written for a medical question. Your task is to rate the summary on four metrics. Please read and understand these instructions carefully.

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
            key, value = line.split(':', 1)
            if key.strip() in score_mapping:
                try:
                    scores[score_mapping[key.strip()]] = float(value.strip())
                except ValueError:
                    pass
    return scores


def calculate_stats(scores):
    mean = np.mean(scores) * 20
    std = np.std(scores) * 20
    return f"{mean:.2f}% ± {std:.2f}%"


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


def main():
    parser = argparse.ArgumentParser(description="G-Eval (English) for pilot output columns")
    parser.add_argument('--input', required=True, help='Path to results CSV file')
    parser.add_argument('--output', required=True, help='Path to output CSV file with G-Eval scores')
    parser.add_argument('--methods', nargs='+', required=True,
                        help='Column names (headers) of methods to evaluate')
    parser.add_argument('--question-col', default='Question',
                        help='Column name for questions (default: Question)')
    parser.add_argument('--model', required=True, help='Path to GGUF model file for evaluation')
    args = parser.parse_args()

    llm = build_llm(args.model)
    csv.field_size_limit(sys.maxsize)
    metrics = ['Context_Relevance', 'Comprehensiveness', 'Correctness', 'Empowerment']

    with open(args.input, 'r', newline='', encoding='utf-8') as infile, \
         open(args.output, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        new_fields = [f'{m}_{metric}' for m in args.methods for metric in metrics]
        fieldnames = list(reader.fieldnames) + new_fields
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="evaluating"):
            question = row[args.question_col]
            for method in args.methods:
                if method in row and row[method].strip():
                    scores = parse_scores(chat_llm(llm, create_evaluation_prompt(question, row[method], method)))
                    for metric, val in scores.items():
                        row[f'{method}_{metric.replace(" ", "_")}'] = val
            writer.writerow(row)

    # Analyze results
    df = pd.read_csv(args.output)
    results = []
    for method in args.methods:
        method_results = [method]
        for metric in metrics:
            column = f"{method}_{metric}"
            if column in df.columns:
                vals = df[column].dropna().values
                method_results.append(calculate_stats(vals) if len(vals) > 0 else "N/A")
            else:
                method_results.append("N/A")
        results.append(method_results)

    headers = ["Method", "Context_Relevance", "Comprehensiveness", "Correctness", "Empowerment"]
    table = [headers] + results
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    print_table(table, col_widths)


if __name__ == "__main__":
    main()
