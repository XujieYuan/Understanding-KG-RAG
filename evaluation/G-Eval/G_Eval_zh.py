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
        n_ctx=4096,
        chat_format="qwen",
        repeat_penalty=1.0,
        verbose=False,
    )


def chat_llm(llm, prompt):
    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "你是一位专门评估医疗问答质量的AI助手。"},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion["choices"][0]["message"]["content"]
    print("chinese llm response:", response)
    return response


def create_evaluation_prompt(question, answer, method):
    return f"""请你评估一个针对医疗问题的回答（摘要）。你需要从四个维度对这个回答进行评分。请仔细阅读并理解以下指示。

                问题：{question}

                {method}方法的回答：{answer}

                评估标准：
                1. 上下文相关性（1-5分）- 回答与问题的上下文相关程度如何？是否准确针对所提出的医疗问题？
                2. 全面性（1-5分）- 回答是否涵盖了问题的所有关键方面？解释是否充分详尽？
                3. 准确性（1-5分）- 回答中提供的信息是否准确和最新？是否存在任何事实错误？
                4. 实用性（1-5分）- 回答是否提供了可行的见解或建议？是否能帮助读者对自身健康做出明智的决定？

                评估步骤：
                1. 仔细阅读问题，确定主要医疗问题和关键点。
                2. 阅读回答并与问题进行对比，检查回答是否解决了问题的主要内容和关键点。
                3. 根据评估标准，对每个维度进行1-5分的评分，其中1分最低，5分最高。

                请按以下格式提供你的评分：
                上下文相关性：[分数]
                全面性：[分数]
                准确性：[分数]
                实用性：[分数]

                只需提供分数，无需额外解释。
                """


def parse_scores(evaluation_text):
    lines = evaluation_text.split('\n')
    scores = {}
    score_mapping = {
        '上下文相关性': 'Context Relevance',
        '全面性': 'Comprehensiveness',
        '准确性': 'Correctness',
        '实用性': 'Empowerment'
    }

    for line in lines:
        if '：' in line:
            key, value = line.split('：', 1)
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
    parser = argparse.ArgumentParser(description="G-Eval (Chinese) for pilot output columns")
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

        for row in tqdm(reader, desc="正在评估"):
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

    headers = ["方法", "上下文相关性", "全面性", "准确性", "实用性"]
    table = [headers] + results
    col_widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    print_table(table, col_widths)


if __name__ == "__main__":
    main()
