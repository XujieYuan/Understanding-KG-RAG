import json
import re
from collections import defaultdict

def extract_entities(text):
    """Extract entities from the text between 'entities are' and '<EOS>'"""
    match = re.search(r'entities are (.*?)<EOS>', text)
    if match:
        return match.group(1).split(', ')
    return []

def classify_dialogue_type(question, answer):
    """Classify the dialogue as L1 or L2 based on specific features"""
    
    # Extract entities from question and answer
    question_entities = extract_entities(question)
    answer_entities = extract_entities(answer)
    
    # L2 features
    l2_features = [
        # Complex symptoms (multiple symptoms or conditions)
        lambda q, a: len(question_entities) >= 3,
        
        # Multiple recommended procedures/treatments
        lambda q, a: len(answer_entities) >= 5,
        
        # Contains multiple diagnostic steps or complex treatment plan
        lambda q, a: any(word in answer.lower() for word in [
            "first", "then", "additionally", "finally", "depending on"
        ]),
        
        # Involves multiple body systems or complex conditions
        lambda q, a: len(re.findall(r'and|or', answer)) >= 3,
        
        # Requires additional tests or evaluations
        lambda q, a: any(word in answer.lower() for word in [
            "evaluation", "assessment", "examination", "test", "screen"
        ])
    ]
    
    # Calculate L2 score
    l2_score = sum(1 for feature in l2_features if feature(question, answer))
    
    # Classify as L2 if meets at least 2 features
    return "L2" if l2_score >= 4 else "L1"

def analyze_dataset(file_path, limit=700):
    counts = {
        'L1': 0,
        'L2': 0,
        'total': 0
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
                
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                question = data['qustion_output']  # Note: keeping the typo as it's in the data
                answer = data['answer_output']
                
                dialogue_type = classify_dialogue_type(question, answer)
                counts[dialogue_type] += 1
                counts['total'] += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line {i+1}: {str(e)}")
                continue
    
    return counts

def print_statistics(counts):
    print("\n=== Dialogue Classification Statistics ===")
    total = counts['total']
    if total > 0:
        l1_percentage = (counts['L1'] / total) * 100
        l2_percentage = (counts['L2'] / total) * 100
        
        print(f"Total dialogues analyzed: {total}")
        print(f"L1 (Simple) dialogues: {counts['L1']} ({l1_percentage:.2f}%)")
        print(f"L2 (Complex) dialogues: {counts['L2']} ({l2_percentage:.2f}%)")
        print("\nClassification criteria:")
        print("L1: Simple, straightforward medical queries")
        print("L2: Complex cases involving multiple symptoms, treatments, or requiring additional evaluation")

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "./data/chatdoctor5k/NER_chatgpt.json"
    results = analyze_dataset(file_path)
    print_statistics(results)