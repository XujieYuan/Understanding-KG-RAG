import json
import re
from collections import defaultdict

def classify_question_type(question, question_kg):
    main_question = question.split('\n')[0]
    
    l2_features = [
        lambda q: "患者" in q,  
        lambda q: len(re.findall(r'[，。、；]', q)) >= 2,  
        lambda q: "合用" in q or "配伍" in q,  
        lambda q: any(word in q for word in ["禁忌", "注意", "选用", "建议"]), 
        lambda q: len(question_kg.split(',')) >= 4  
    ]
    
    l2_score = sum(1 for feature in l2_features if feature(main_question))
    
    return "L2" if l2_score >= 2 else "L1"

def analyze_dataset(dataset_path):
    counts = {
        'L1': 0,
        'L2': 0,
        'total': 0
    }
    
    with open(dataset_path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                if 'question' in data and 'question_kg' in data:
                    question_type = classify_question_type(data['question'], data['question_kg'])
                    counts[question_type] += 1
                    counts['total'] += 1
            except json.JSONDecodeError:
                continue
    
    return counts

def analyze_all_datasets():
    datasets = [
        "Medical_Practitioner",
        "Medical_Technology",
        "Nursing",
        "Pharmacy",
        "Postgraduate",
        "Professional"
    ]
    
    all_results = {}
    total_stats = defaultdict(int)
    
    for dataset in datasets:
        input_file = f"./data/CMB-Exam/{dataset}/{dataset}.json"
        try:
            results = analyze_dataset(input_file)
            all_results[dataset] = results
            
            for key, value in results.items():
                total_stats[key] += value
                
        except FileNotFoundError:
            continue
    
    print("\n=== Results ===")
    print(f"{'Datasets':<20} {'Total':<8} {'L1 amount':<8} {'L1 ratio':<8} {'L2 amount':<8} {'L2 ratio':<8}")
    print("-" * 70)
    
    for dataset, counts in all_results.items():
        total = counts['total']
        if total > 0:
            l1_percentage = (counts['L1'] / total) * 100
            l2_percentage = (counts['L2'] / total) * 100
            print(f"{dataset:<20} {total:<8} {counts['L1']:<8} {l1_percentage:>6.2f}% {counts['L2']:<8} {l2_percentage:>6.2f}%")

    total = total_stats['total']
    if total > 0:
        l1_percentage = (total_stats['L1'] / total) * 100
        l2_percentage = (total_stats['L2'] / total) * 100
        print(f"Total question: {total}")
        print(f"L1 amount: {total_stats['L1']} ({l1_percentage:.2f}%)")
        print(f"L2 amount: {total_stats['L2']} ({l2_percentage:.2f}%)")

if __name__ == "__main__":
    analyze_all_datasets()