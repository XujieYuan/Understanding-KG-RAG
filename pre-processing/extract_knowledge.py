from llama_cpp import Llama


def chat_llm(prompt):
    llm = Llama(
        model_path="",
        n_gpu_layers=200,
        n_batch=512,
        n_ctx=2048,
        repeat_penalty=1.0,
        verbose=False,
    )
    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a professional knowledge extraction assistant, helping to construct commonsense knowledge graphs. You excel at extracting knowledge triples from text while maintaining high accuracy and consistency."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion["choices"][0]["message"]["content"]
    print("llm response:", response)
    return response


def extract_commonsense_triples(question_data: dict) -> str:
    question = question_data['question']
    answer = question_data['answer']
    question_concept = question_data['question_kg']
    correct_option = answer.strip()

    prompt = f"""As a professional knowledge extraction assistant, your task is to extract knowledge triples from the given multiple-choice question. Please follow these rules strictly:

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

    Please output knowledge triples directly, one per line, in the format: subject\tpredicate\tobject. If no valid knowledge triples can be found, reply with "No valid knowledge triples found"."""

    result = chat_llm(prompt)
    return result.strip()