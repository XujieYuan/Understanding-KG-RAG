import argparse
import json
from llama_cpp import Llama


def build_llm(model_path):
    return Llama(
        model_path=model_path,
        n_gpu_layers=200,
        n_batch=512,
        n_ctx=2048,
        repeat_penalty=1.0,
        verbose=False,
    )


def chat_llm(llm, prompt, system_content):
    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion["choices"][0]["message"]["content"]
    print("llm response:", response)
    return response


SYSTEM_PROMPTS = {
    'commonsense': (
        "You are a professional knowledge extraction assistant, helping to construct commonsense "
        "knowledge graphs. You excel at extracting knowledge triples from text while maintaining "
        "high accuracy and consistency."
    ),
    'medical': (
        "You are a professional medical knowledge extraction assistant, helping to construct "
        "medical knowledge graphs. You excel at extracting knowledge triples from medical text "
        "while maintaining high accuracy and consistency."
    ),
    'legal': (
        "You are a professional legal knowledge extraction assistant, helping to construct legal "
        "knowledge graphs. You excel at extracting knowledge triples from legal text while "
        "maintaining high accuracy and consistency."
    ),
}

EXTRACTION_PROMPTS = {
    'commonsense': """As a professional knowledge extraction assistant, your task is to extract knowledge triples from the given multiple-choice question. Please follow these rules strictly:

    1. Carefully read the question description, all options, and the correct answer.
    2. Focus on the core concept "{question_concept}" in the question.
    3. Extract commonsense knowledge triples related to the question.
    4. Each triple should be in the format: subject\tpredicate\tobject
    5. Focus on the following types of relationships:
       - Conceptual relations (e.g., dog\tis a\tanimal)
       - Object properties (e.g., sun\tproperty\tbright)
       - Object functions (e.g., key\tused for\tunlocking doors)
       - Spatial relations (e.g., fish\tlives in\twater)
       - Temporal relations (e.g., breakfast\ttime is\tmorning)
       - Causal relations (e.g., exercise\tcauses\tsweating)
    6. Each triple must be concrete and valuable commonsense knowledge.
    7. Avoid subjective or controversial knowledge.
    8. Ensure triples are logically sound and align with common sense.
    9. Don't just copy option content, extract the implied knowledge.

    Please extract knowledge triples from this multiple-choice question:

    Question: {question}
    Core Concept: {question_concept}
    Correct Answer: {correct_option}

    Please output knowledge triples directly, one per line, in the format: subject\tpredicate\tobject. If no valid knowledge triples can be found, reply with "No valid knowledge triples found".""",

    'medical': """As a professional medical knowledge extraction assistant, your task is to extract knowledge triples from the given question. Please follow these rules strictly:

    1. Carefully read the question and the correct answer.
    2. Focus on the core medical concept "{question_concept}" in the question.
    3. Extract medical knowledge triples related to the question.
    4. Each triple should be in the format: subject\tpredicate\tobject
    5. Focus on the following types of relationships:
       - Disease-symptom (e.g., diabetes\tsymptom\thigh blood sugar)
       - Drug-effect (e.g., aspirin\ttreats\tpain)
       - Disease-treatment (e.g., hypertension\ttreated by\tantihypertensives)
       - Anatomy relations (e.g., heart\tlocated in\tchest)
    6. Each triple must be accurate medical knowledge.
    7. Ensure triples are clinically relevant.

    Please extract knowledge triples from this question:

    Question: {question}
    Core Concept: {question_concept}
    Correct Answer: {correct_option}

    Please output knowledge triples directly, one per line, in the format: subject\tpredicate\tobject. If no valid knowledge triples can be found, reply with "No valid knowledge triples found".""",
}


def extract_triples(question_data, llm, domain):
    question = question_data['question']
    answer = question_data['answer']
    question_concept = question_data.get('question_kg', '')
    correct_option = answer.strip()

    prompt_template = EXTRACTION_PROMPTS.get(domain, EXTRACTION_PROMPTS['commonsense'])
    prompt = prompt_template.format(
        question=question,
        question_concept=question_concept,
        correct_option=correct_option,
    )
    system = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS['commonsense'])

    result = chat_llm(llm, prompt, system)
    return result.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge triples from QA dataset")
    parser.add_argument('--model', required=True, help='Path to GGUF model file')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', required=True, help='Output knowledge graph TSV file path')
    parser.add_argument('--domain', default='commonsense',
                        choices=['commonsense', 'medical', 'legal'],
                        help='Domain type for prompt selection')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint file to resume from (stores processed line count)')
    args = parser.parse_args()

    llm = build_llm(args.model)

    start_line = 0
    if args.checkpoint:
        try:
            with open(args.checkpoint, 'r') as ckpt:
                start_line = int(ckpt.read().strip())
            print(f"Resuming from line {start_line}")
        except FileNotFoundError:
            pass

    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'a', encoding='utf-8') as fout:

        for i, line in enumerate(fin):
            if i < start_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                question_data = json.loads(line)
                triples = extract_triples(question_data, llm, args.domain)
                if triples and triples != "No valid knowledge triples found":
                    fout.write(triples + '\n')
                    fout.flush()
            except Exception as e:
                print(f"Line {i}: Error {e}. Skipping.")

            if args.checkpoint:
                with open(args.checkpoint, 'w') as ckpt:
                    ckpt.write(str(i + 1))

    print("Extraction complete.")


if __name__ == "__main__":
    main()
