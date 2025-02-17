import json
import re
from collections import defaultdict

def classify_question_type(question, question_kg):
    # 移除选项部分便于判断问题本身
    main_question = question.split('\n')[0]
    
    # L2类型的特征
    l2_features = [
        lambda q: "患者" in q,  # 包含患者描述
        lambda q: len(re.findall(r'[，。、；]', q)) >= 2,  # 包含多个分句
        lambda q: "合用" in q or "配伍" in q,  # 涉及药物相互作用
        lambda q: any(word in q for word in ["禁忌", "注意", "选用", "建议"]),  # 涉及用药决策
        lambda q: len(question_kg.split(',')) >= 4  # 知识图谱中包含较多概念
    ]
    
    # 计算符合L2特征的数量
    l2_score = sum(1 for feature in l2_features if feature(main_question))
    
    # 如果符合2个或以上L2特征，则归类为L2
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
    
    # 存储所有数据集的统计结果
    all_results = {}
    total_stats = defaultdict(int)
    
    # 分析每个数据集
    for dataset in datasets:
        input_file = f"./data/CMB-Exam/{dataset}/{dataset}.json"
        try:
            results = analyze_dataset(input_file)
            all_results[dataset] = results
            
            # 累加总计
            for key, value in results.items():
                total_stats[key] += value
                
        except FileNotFoundError:
            print(f"警告: 未找到数据集 {dataset}")
            continue
    
    # 打印详细报告
    print("\n=== 详细统计报告 ===")
    print(f"{'数据集':<20} {'总数':<8} {'L1数量':<8} {'L1比例':<8} {'L2数量':<8} {'L2比例':<8}")
    print("-" * 70)
    
    for dataset, counts in all_results.items():
        total = counts['total']
        if total > 0:
            l1_percentage = (counts['L1'] / total) * 100
            l2_percentage = (counts['L2'] / total) * 100
            print(f"{dataset:<20} {total:<8} {counts['L1']:<8} {l1_percentage:>6.2f}% {counts['L2']:<8} {l2_percentage:>6.2f}%")
    
    # 打印汇总统计
    print("\n=== 汇总统计 ===")
    total = total_stats['total']
    if total > 0:
        l1_percentage = (total_stats['L1'] / total) * 100
        l2_percentage = (total_stats['L2'] / total) * 100
        print(f"总问题数量: {total}")
        print(f"L1类型总数: {total_stats['L1']} ({l1_percentage:.2f}%)")
        print(f"L2类型总数: {total_stats['L2']} ({l2_percentage:.2f}%)")

if __name__ == "__main__":
    analyze_all_datasets()