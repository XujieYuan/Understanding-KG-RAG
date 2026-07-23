import time
import heapq
import re
import os
from typing import List, Tuple
import pickle
import json
import csv

# Module-level globals — initialized in __main__ or via external init
np = None
PromptTemplate = None
ChatPromptTemplate = None
SystemMessagePromptTemplate = None
HumanMessagePromptTemplate = None
AIMessage = None
HumanMessage = None
SystemMessage = None
ChatOpenAI = None
GraphDatabase = None
ServiceUnavailable = None
torch = None
BertTokenizer = None
BertModel = None
SentenceTransformer = None
sentence_model = None
chat = None
bert_tokenizer = None
bert_model = None
driver = None
uri = None
username = None
password = None


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def encode_text(text):
    tokens = bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings

def encode_text_sentence_transformer(text):
    if isinstance(text, list):
        return sentence_model.encode(text, normalize_embeddings=True)
    else:
        return sentence_model.encode([text], normalize_embeddings=True)[0]

def extract_keywords(input_question, prompts, max_retries=3):
    try:
        messages = [
            SystemMessage(content=prompts['keyword_system']),
            HumanMessage(content=prompts['keyword_instruction'].format(question=input_question))
        ]
        result = chat(messages)
        keywords = result.content.strip().split(',')
        print(f"Extracted keywords: {keywords}")
        return [kw.strip() for kw in keywords]
    except Exception as e:
        words = re.findall(r'\b\w+\b', input_question.lower())
        return list(set(words))[:5]

def match_entities(input_question, entity_embeddings, prompts):
    keywords = extract_keywords(input_question, prompts)

    match_kg = []

    for keyword in keywords:
        keyword_embedding = encode_text_sentence_transformer(keyword)

        similarities = []
        for i, entity_embedding in enumerate(entity_embeddings["embeddings"]):
            similarity = np.dot(keyword_embedding, entity_embedding) / (
                np.linalg.norm(keyword_embedding) * np.linalg.norm(entity_embedding)
            )
            similarities.append((similarity, i))

        similarities.sort(reverse=True)

        for similarity, idx in similarities[:5]:
            entity = entity_embeddings["entities"][idx]
            entity_formatted = entity.replace(" ", "_")

            if entity_formatted not in match_kg and similarity > 0.6:
                match_kg.append(entity_formatted)
                break

    return match_kg


def preprocess_output(output):
    output = re.sub(r"[\n\r]+", " ", output)
    output = output.replace(",", ".")
    output = re.sub(r"[:.]", " ", output)
    output = re.sub(r"\s+", " ", output)
    output = output.strip()
    return output


def extract_topics(input_question, match_kg, prompts):
    for _ in range(3):
        messages = [
            SystemMessage(content=prompts['topic_system']),
            HumanMessage(content="Based on the question" + input_question),
            AIMessage(
                content="You have some relevant entities in the following:\n\n" + '###' + str(match_kg)),
            HumanMessage(content=prompts['topic_instruction'])
        ]
        result = chat(messages)
        output_all = result.content
        output_all = preprocess_output(output_all)
        print("llm topics:", output_all)

        pattern = prompts.get('topic_pattern', r"[Tt]he main topic (?:is|can be summarized as) '(.*?)'")
        match = re.search(pattern, output_all, re.IGNORECASE)
        if match:
            return match.group(1)

    return ""


def calculate_score(entity, input_question, main_topic):
    input_question_embeddings = encode_text(input_question)
    entity_embeddings = encode_text(entity)
    similarity_score = torch.cosine_similarity(input_question_embeddings, entity_embeddings, dim=0).item()

    if main_topic:
        topic_score = torch.cosine_similarity(encode_text(main_topic), entity_embeddings, dim=0).item()
    else:
        topic_score = 0

    total_score = similarity_score * 0.5 + topic_score * 0.5
    return total_score


def calculate_score_noTopic(entity, input_question):
    input_question_embeddings = encode_text(input_question)
    entity_embeddings = encode_text(entity)
    similarity_score = torch.cosine_similarity(input_question_embeddings, entity_embeddings, dim=0).item()

    return similarity_score


def heuristic_search(input_question, match_kg, topic, max_depth=2, max_width=3):
    max_attempts = 3

    visited = set()
    subgraph = {}
    priority_queue = []

    for entity in match_kg:
        score = calculate_score(entity, input_question, topic)
        heapq.heappush(priority_queue, (-score, entity, 0))

    while priority_queue and len(visited) < max_width:
        current_score, current_node, current_depth = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)
        print(f"visited: {visited}, current_node: {current_node}, current_score: {current_score}, current_depth: {current_depth}")

        if current_depth >= max_depth:
            break

        attempt = 0
        while attempt < max_attempts:
            try:
                neighbors = get_entity_neighbors_ours(current_node)
                break
            except ServiceUnavailable:
                print(f"Attempt {attempt + 1} failed to get neighbors for entity: {current_node}")
                time.sleep(2)
                attempt += 1

        if attempt == max_attempts:
            print(f"Max attempts reached for entity {current_node}. Skipping.")
            continue

        filtered_neighbors = []
        for neighbor, relationship in neighbors:
            neighbor_score = calculate_score(neighbor, input_question, topic)
            if neighbor_score > 0.5:
                heapq.heappush(priority_queue, (-neighbor_score, neighbor, current_depth + 1))
                filtered_neighbors.append((neighbor, relationship))

        subgraph[current_node] = filtered_neighbors
        print(f"subgraph: {subgraph}")

    return subgraph


def heuristic_search_preAblation(input_question, match_kg, max_depth=2, max_width=3):
    max_attempts = 3

    visited = set()
    subgraph = {}
    priority_queue = []

    for entity in match_kg:
        score = calculate_score_noTopic(entity, input_question)
        heapq.heappush(priority_queue, (-score, entity, 0))

    while priority_queue and len(visited) < max_width:
        current_score, current_node, current_depth = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)
        print(f"visited: {visited}, current_node: {current_node}, current_score: {current_score}, current_depth: {current_depth}")

        if current_depth >= max_depth:
            break

        attempt = 0
        while attempt < max_attempts:
            try:
                neighbors = get_entity_neighbors_ours(current_node)
                break
            except ServiceUnavailable:
                print(f"Attempt {attempt + 1} failed to get neighbors for entity: {current_node}")
                time.sleep(2)
                attempt += 1

        if attempt == max_attempts:
            print(f"Max attempts reached for entity {current_node}. Skipping.")
            continue

        filtered_neighbors = []
        for neighbor, relationship in neighbors:
            neighbor_score = calculate_score_noTopic(neighbor, input_question)
            if neighbor_score > 0.5:
                heapq.heappush(priority_queue, (-neighbor_score, neighbor, current_depth + 1))
                filtered_neighbors.append((neighbor, relationship))

        subgraph[current_node] = filtered_neighbors
        print(f"subgraph: {subgraph}")

    return subgraph


def get_entity_neighbors_ours(entity_name: str, max_retries=3) -> List[Tuple[str, str]]:
    global driver
    attempt = 0
    while attempt < max_retries:
        try:
            with driver.session() as session:
                query = """
                    MATCH (e:Entity)-[r]->(n)
                    WHERE e.name = $entity_name
                    RETURN n.name AS neighbor, type(r) AS relationship_type
                """
                result = session.run(query, entity_name=entity_name)

                neighbor_list = []
                for record in result:
                    relationship_type = record["relationship_type"]
                    neighbor = record["neighbor"]
                    neighbor_list.append((neighbor, relationship_type))

                return neighbor_list

        except ServiceUnavailable as e:
            print(f"Attempt {attempt + 1} failed with error: {e}. Database connection lost. Attempting to reconnect...")
            attempt += 1
            time.sleep(2)
            if attempt < max_retries:
                try:
                    driver.close()
                    driver = GraphDatabase.driver(uri, auth=(username, password))
                except ServiceUnavailable:
                    print("Failed to reconnect to the database.")
                    continue

    print("All retry attempts failed. Please check your database.")
    raise Exception("Max retries reached, unable to get entity neighbors.")


def extract_entities(subgraph: dict) -> List[Tuple[str, str, str]]:
    entity_relations = []

    for entity, neighbors in subgraph.items():
        for neighbor, relation in neighbors:
            entity_relations.append((entity, relation, neighbor))

    return entity_relations


def format_knowledge(entity_relations: List[Tuple[str, str, str]]) -> str:
    formatted_lines = []

    for head, relation, tail in entity_relations:
        formatted_lines.append(f"'{head}' -> '{relation}' -> '{tail}'")

    return '\n'.join(formatted_lines)


def generate_natural_language(subgraph: str) -> str:
    entity_relations = extract_entities(subgraph)
    knowledge = format_knowledge(entity_relations)
    response = prompt_knowledge(knowledge)
    return response


def path_search(input_question, match_kg, topic, max_depth=3, similarity_threshold=0.7):
    visited = set()
    path = []
    priority_queue = []

    for entity in match_kg:
        score = calculate_score(entity, input_question, topic)
        heapq.heappush(priority_queue, (-score, entity, 0, None))

    while priority_queue:
        current_score, current_node, current_depth, relationship = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        if current_depth >= max_depth:
            break

        visited.add(current_node)

        if relationship:
            path.append((current_node, relationship))
        else:
            path.append((current_node, None))

        print(f"visited: {visited}, current_node: {current_node}, current_score: {current_score}, current_depth: {current_depth}")

        try:
            neighbors = get_entity_neighbors_ours(current_node)
        except ServiceUnavailable:
            print("Failed to get neighbors for entity:", current_node)
            continue

        best_neighbor = None
        best_score = -float('inf')
        best_relationship = None

        for neighbor, relationship in neighbors:
            if neighbor not in visited:
                neighbor_score = calculate_score(neighbor, input_question, topic)
                if neighbor_score > similarity_threshold and neighbor_score > best_score:
                    best_score = neighbor_score
                    best_neighbor = neighbor
                    best_relationship = relationship

        if best_neighbor:
            heapq.heappush(priority_queue, (-best_score, best_neighbor, current_depth + 1, best_relationship))

    return path


def retrieve_top_triples(input_question, match_kg, main_topic, top_n):
    top_triples = []

    for entity in match_kg:
        try:
            neighbors = get_entity_neighbors_ours(entity)
        except ServiceUnavailable:
            print(f"Failed to retrieve neighbors for entity: {entity}")
            continue

        for neighbor, relationship in neighbors:
            triple_text = f"{entity} {relationship} {neighbor}"
            score = calculate_score(triple_text, input_question, main_topic)
            top_triples.append((score, (entity, relationship, neighbor)))

    top_triples.sort(reverse=True, key=lambda x: x[0])

    return [triple for _, triple in top_triples[:top_n]]


def prompt_knowledge(knowledge):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {knowledge}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. And name them as Evidence 1, Evidence 2,...\n\n

    output:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["knowledge"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    system_message_prompt.format(knowledge=knowledge)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(knowledge=knowledge,
                                                        text={})

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response_of_KG = chat(chat_prompt_with_values.to_messages()).content
            pattern = r'Evidence \d+:.*?(?=Evidence \d+:|$)'
            evidences = re.findall(pattern, response_of_KG, re.DOTALL)

            pattern_remove = r'I hope.*'
            evidences = [re.sub(pattern_remove, '', evidence, flags=re.DOTALL) for evidence in evidences]

            if not evidences:
                raise ValueError("No evidence found in response")
            break
        except ValueError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 == max_attempts:
                return "Failed to get a valid subgraph response after maximum attempts"

    result = '\n'.join(evidences)
    return result


def generate_path_description(path):
    if not path:
        return "No path found."

    valid_segments = []
    current_start = path[0][0]

    for i in range(1, len(path)):
        node, relationship = path[i]
        if relationship:
            valid_segments.append({
                'start': current_start,
                'relation': relationship,
                'end': node
            })
            current_start = node

    if not valid_segments:
        return f"Found start node:'{path[0][0]}'"

    description = "Found related path: \n"
    for i, segment in enumerate(valid_segments):
        if i == 0:
            description += f"'{segment['start']}' {segment['relation']} '{segment['end']}'"
        else:
            description += f", {segment['relation']} '{segment['end']}'"

    return description


def generate_natural_language_facts(triples):
    if not triples:
        return ""

    formatted_lines = []
    for head, relation, tail in triples:
        formatted_lines.append(f"'{head}' -> '{relation}' -> '{tail}'")

    return "\n".join([f"Evidence{i+1}：'{head}' {relation} '{tail}'" for i, (head, relation, tail) in enumerate(triples)])


# Prompt design
def final_answer_noPrompt(question, subgraph, prompts):
    messages = [
        SystemMessage(content=prompts['system']),
        HumanMessage(content=prompts['input_label'] + question),
        AIMessage(content=prompts['evidence_label'] + '\n\n' + '###' + subgraph),
        HumanMessage(content=prompts['task_noPrompt'])
    ]

    attempt_count = 0
    max_attempts = 3
    while attempt_count < max_attempts:
        try:
            result = chat(messages)
            output_all = result.content
            print("generate no prompt final output success!")
            return output_all

        except Exception as e:
            print(f"An error occurred in no prompt output: {e}. ")
            time.sleep(5)
            attempt_count += 1

    return "Request failed after multiple attempts."


def final_answer_cot(question, subgraph, prompts):
    messages = [
        SystemMessage(content=prompts['system']),
        HumanMessage(content=prompts['input_label'] + question),
        AIMessage(content=prompts['evidence_label'] + '\n\n' + '###' + subgraph),
        HumanMessage(content=prompts['task_cot'])
    ]

    attempt_count = 0
    max_attempts = 3
    while attempt_count < max_attempts:
        try:
            result = chat(messages)
            output_all_cot = result.content
            print("generate cot final output success!")
            return output_all_cot

        except Exception as e:
            print(f"An error occurred in cot output: {e}. ")
            time.sleep(5)
            attempt_count += 1

    return "Request failed after multiple attempts."


def final_answer_tot(question, subgraph, prompts):
    messages = [
        SystemMessage(content=prompts['system']),
        HumanMessage(content=prompts['input_label'] + question),
        AIMessage(content=prompts['evidence_label'] + '\n\n' + '###' + subgraph),
        HumanMessage(content=prompts['task_tot'])
    ]

    attempt_count = 0
    max_attempts = 3
    while attempt_count < max_attempts:
        try:
            result = chat(messages)
            output_all_tot = result.content
            print("generate tot final output success!")
            return output_all_tot

        except Exception as e:
            print(f"An error occurred in tot output: {e}. ")
            time.sleep(5)
            attempt_count += 1

    return "Request failed after multiple attempts."


def final_answer_mindmap(question, subgraph, prompts):
    messages = [
        SystemMessage(content=prompts['system']),
        HumanMessage(content=prompts['input_label'] + question),
        AIMessage(content=prompts['evidence_label'] + '\n\n' + '###' + subgraph),
        HumanMessage(content=prompts['task_mindmap'])
    ]

    attempt_count = 0
    max_attempts = 3

    while attempt_count < max_attempts:
        try:
            result = chat(messages)
            output_ours = result.content
            print("generate OURS final output success!")
            return output_ours

        except Exception as e:
            print(f"An error occurred in output_all: {e}. ")
            time.sleep(5)
            attempt_count += 1

    return "Request failed after multiple attempts."


# Generate final output
def generate_output_all_noPrompt(question, natural_language, prompts):
    try:
        output_all_ours = final_answer_noPrompt(question, natural_language, prompts)
    except Exception as e:
        print(f"An error occurred: {e}. Failed to generate no prompt final answer")
        return None
    return output_all_ours

def generate_output_all_cot(question, natural_language, prompts):
    try:
        output_all_ours = final_answer_cot(question, natural_language, prompts)
    except Exception as e:
        print(f"An error occurred: {e}. Failed to generate cot final answer")
        return None
    return output_all_ours

def generate_output_all_tot(question, natural_language, prompts):
    try:
        output_all_ours = final_answer_tot(question, natural_language, prompts)
    except Exception as e:
        print(f"An error occurred: {e}. Failed to generate tot final answer")
        return None
    return output_all_ours

def generate_output_all_mindmap(question, natural_language, prompts):
    try:
        output_all_ours = final_answer_mindmap(question, natural_language, prompts)
    except Exception as e:
        print(f"An error occurred: {e}. Failed to generate mindmap final answer")
        return None
    return output_all_ours

def extract_output_summary(output, pattern):
    if not isinstance(output, str):
        try:
            output = ' '.join(map(str, output))
        except Exception as e:
            return f"Error: Failed to convert output to string. Original error: {e}"
    extracted_output = re.findall(pattern, output, flags=re.DOTALL)
    if len(extracted_output) > 0:
        return extracted_output[0]
    else:
        return None


def get_required_config(cfg, path):
    current = cfg
    walked = []
    for key in path:
        walked.append(key)
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(walked)
            raise ValueError(f"Missing required config field: {dotted}")
        current = current[key]
    return current


def resolve_repo_path(repo_root, path_value):
    if not path_value:
        raise ValueError("Config path value must not be empty.")
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(repo_root, path_value)


def validate_config(cfg, repo_root, supported_datasets):
    dataset = get_required_config(cfg, ["dataset"])
    if dataset not in supported_datasets:
        supported = ", ".join(sorted(supported_datasets))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported values: {supported}")

    get_required_config(cfg, ["neo4j", "uri"])
    get_required_config(cfg, ["neo4j", "username"])
    get_required_config(cfg, ["neo4j", "password"])
    get_required_config(cfg, ["llm", "api_base"])
    get_required_config(cfg, ["llm", "api_key"])
    get_required_config(cfg, ["bert", "model_path"])
    get_required_config(cfg, ["embedding", "model_path"])

    input_path = resolve_repo_path(repo_root, get_required_config(cfg, ["data", "input"]))
    entity_embeddings_path = resolve_repo_path(repo_root, get_required_config(cfg, ["data", "entity_embeddings"]))
    output_path = resolve_repo_path(repo_root, get_required_config(cfg, ["data", "output"]))

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input dataset file not found: {input_path}"
        )

    if not os.path.isfile(entity_embeddings_path):
        raise FileNotFoundError(
            "Entity embedding file not found: "
            f"{entity_embeddings_path}. Generate it first with "
            "pre-processing/encode_entities.py."
        )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if not output_dir:
        raise ValueError("Output path must include a parent directory.")

    return {
        "dataset": dataset,
        "input_path": input_path,
        "entity_embeddings_path": entity_embeddings_path,
        "output_path": output_path,
    }


if __name__ == "__main__":
    import argparse
    import yaml
    import sys

    parser = argparse.ArgumentParser(description="Pilot KG-RAG pipeline")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Add repo root to sys.path for prompts import
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import prompts as prompts_module

    DATASET_PROMPT_MAP = {
        'genmedgpt': prompts_module.MEDICAL_DIAGNOSIS,
        'cmcqa': prompts_module.MEDICAL_DIAGNOSIS,
        'commonsenseqa': prompts_module.COMMONSENSE,
        'explainpe': prompts_module.MEDICAL_EXAM,
        'cmb_exam': prompts_module.MEDICAL_EXAM,
    }
    resolved = validate_config(cfg, repo_root, DATASET_PROMPT_MAP.keys())
    dataset = resolved["dataset"]
    prompts = DATASET_PROMPT_MAP.get(dataset, prompts_module.MEDICAL_DIAGNOSIS)

    import numpy as np
    import torch
    from langchain import PromptTemplate
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
    from sentence_transformers import SentenceTransformer
    from transformers import BertModel, BertTokenizer

    # Initialize Neo4j
    uri = cfg['neo4j']['uri']
    username = cfg['neo4j']['username']
    password = cfg['neo4j']['password']
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Initialize BERT
    BERT_PATH = cfg['bert']['model_path']
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    bert_model = BertModel.from_pretrained(BERT_PATH)

    # Initialize sentence transformer
    sentence_model = SentenceTransformer(cfg['embedding']['model_path'])

    # Initialize LLM
    chat = ChatOpenAI(
        openai_api_key=cfg['llm']['api_key'],
        openai_api_base=cfg['llm']['api_base'],
    )

    output_path = resolved["output_path"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow([
            'Question', 'Label',
            'Subgraph_preAblation', 'NL subgraph_preAblation', 'PreAblation', 'PreAblation-summary',
            'NL subgraph', 'Subgraph+cot', 'Subgraph+cot_summary',
            'Subgraph+tot', 'Subgraph+tot_summary',
            'Subgraph+mindmap', 'Subgraph+mindmap_summary',
            'Subgraph_noPrompt',
            'Path', 'path+cot', 'path+cot_summary',
            'path+tot', 'path+tot_summary',
            'path+mindmap', 'path+mindmap_summary',
            'path_noPrompt',
            'Facts', 'facts+cot', 'facts+cot_summary',
            'facts+tot', 'facts+tot_summary',
            'facts+mindmap', 'facts+mindmap_summary',
            'facts_noPrompt',
        ])

    with open(resolved["entity_embeddings_path"], 'rb') as f1:
        entity_embeddings = pickle.load(f1)

    output_re = prompts.get('output_pattern', r"Output 1:(.*?)Output 2:")

    with open(resolved["input_path"], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)

            question_raw = x['question']
            answer_raw = x['answer']
            if isinstance(question_raw, list):
                question = question_raw[0]
                label = answer_raw[0] if isinstance(answer_raw, list) else answer_raw
            else:
                question = question_raw
                label = answer_raw

            if not question:
                continue
            print('Question:', question)
            print('Answer:', label)

            match_kg = match_entities(question, entity_embeddings, prompts)

            # Pre-retrieval: topic extraction
            attempt_count = 0
            max_attempts = 3
            while attempt_count < max_attempts:
                try:
                    topic = extract_topics(question, match_kg, prompts)
                    print("Extracted topic:", topic)
                    break
                except Exception as e:
                    print("Failed to extract topic:", e)
                    time.sleep(5)
                    attempt_count += 1
                    topic = ""

            # Subgraph retrieval
            subgraph = heuristic_search(question, match_kg, topic)
            print("Final Subgraph:")
            for entity, neighbors in subgraph.items():
                print(f"Entity: {entity}, Neighbors: {neighbors}")

            entity_relations = extract_entities(subgraph)
            triples = format_knowledge(entity_relations)
            print("Formatted triples:", triples)

            natural_language_subgraph = generate_natural_language(subgraph)
            if natural_language_subgraph.startswith("Failed"):
                natural_language_subgraph = triples
            print("Natural language:\n", natural_language_subgraph)

            # Ablation subgraph (no topic)
            subgraph_preAblation = heuristic_search_preAblation(question, match_kg)
            print("Final Subgraph ablation:")
            for entity, neighbors in subgraph_preAblation.items():
                print(f"Entity: {entity}, Neighbors: {neighbors}")

            entity_relations_preAblation = extract_entities(subgraph_preAblation)
            triples_preAblation = format_knowledge(entity_relations_preAblation)
            print("Formatted triples ablation:", triples_preAblation)

            natural_language_subgraph_preAblation = generate_natural_language(subgraph_preAblation)
            if natural_language_subgraph_preAblation.startswith("Failed"):
                natural_language_subgraph_preAblation = triples_preAblation
            print("Natural language ablation:\n", natural_language_subgraph_preAblation)

            # Path retrieval
            path = path_search(question, match_kg, topic)
            print("Final path:")
            for node, relationship in path:
                if relationship:
                    print(f"{node} --[{relationship}]--> ", end='')
                else:
                    print(f"{node}", end='')
            natural_language_path = generate_path_description(path)
            print("Natural language description path:\n", natural_language_path)

            # Facts retrieval
            facts = retrieve_top_triples(question, match_kg, topic, 4)
            print("TopN related facts:", facts)
            print("Final Selected top N facts:")
            for head, relation, tail in facts:
                print(f"'{head}' -> '{relation}' -> '{tail}'")
            natural_language_facts = generate_natural_language_facts(facts)
            print("Natural language description facts:\n", natural_language_facts)

            # Final output generation
            for _ in range(3):
                output_all_preAblation = generate_output_all_mindmap(question, natural_language_subgraph_preAblation, prompts)
                print('\nPreAblation:\n', output_all_preAblation)
                output1_preAblation = extract_output_summary(output_all_preAblation, output_re)
                if output1_preAblation is not None:
                    print('\nPreAblation-summary:\n', output1_preAblation)
                    break
            else:
                print("Failed to extract output1_preAblation from output_all_preAblation after 3 attempts")
                continue

            for _ in range(3):
                output_all_subgraph_cot = generate_output_all_cot(question, natural_language_subgraph, prompts)
                if output_all_subgraph_cot is not None:
                    print('\nSubgraph+cot:\n', output_all_subgraph_cot)
                    output1_subgraph_cot = extract_output_summary(output_all_subgraph_cot, output_re)
                    print('\nSubgraph+cot_summary:\n', output1_subgraph_cot)
                    break
            else:
                print("Failed to generate Subgraph+cot after 3 attempts")
                continue

            for _ in range(3):
                output_all_subgraph_tot = generate_output_all_tot(question, natural_language_subgraph, prompts)
                if output_all_subgraph_tot is not None:
                    print('\nSubgraph+tot:\n', output_all_subgraph_tot)
                    output1_subgraph_tot = extract_output_summary(output_all_subgraph_tot, output_re)
                    print('\nSubgraph+tot_summary:\n', output1_subgraph_tot)
                    break
            else:
                print("Failed to generate Subgraph+tot after 3 attempts")
                continue

            for _ in range(3):
                output_all_subgraph_mindmap = generate_output_all_mindmap(question, natural_language_subgraph, prompts)
                if output_all_subgraph_mindmap is not None:
                    print('\nSubgraph+mindmap:\n', output_all_subgraph_mindmap)
                    output1_subgraph_mindmap = extract_output_summary(output_all_subgraph_mindmap, output_re)
                    print('\nSubgraph+mindmap_summary:\n', output1_subgraph_mindmap)
                    break
            else:
                print("Failed to generate Subgraph+mindmap after 3 attempts")
                continue

            for _ in range(3):
                output_all_noPrompt = generate_output_all_noPrompt(question, natural_language_subgraph, prompts)
                if output_all_noPrompt is not None:
                    print('\nSubgraph_noPrompt:\n', output_all_noPrompt)
                    break
            else:
                print("Failed to generate Subgraph_noPrompt after 3 attempts")
                continue

            for _ in range(3):
                output_all_path_cot = generate_output_all_cot(question, natural_language_path, prompts)
                if output_all_path_cot is not None:
                    print('\nPath+cot:\n', output_all_path_cot)
                    output1_path_cot = extract_output_summary(output_all_path_cot, output_re)
                    print('\nPath+cot_summary:\n', output1_path_cot)
                    break
            else:
                print("Failed to generate Path+cot after 3 attempts")
                continue

            for _ in range(3):
                output_all_path_tot = generate_output_all_tot(question, natural_language_path, prompts)
                if output_all_path_tot is not None:
                    print('\nPath+tot:\n', output_all_path_tot)
                    output1_path_tot = extract_output_summary(output_all_path_tot, output_re)
                    print('\nPath+tot_summary:\n', output1_path_tot)
                    break
            else:
                print("Failed to generate Path+tot after 3 attempts")
                continue

            for _ in range(3):
                output_all_path_mindmap = generate_output_all_mindmap(question, natural_language_path, prompts)
                if output_all_path_mindmap is not None:
                    print('\nPath+mindmap:\n', output_all_path_mindmap)
                    output1_path_mindmap = extract_output_summary(output_all_path_mindmap, output_re)
                    print('\nPath+mindmap_summary:\n', output1_path_mindmap)
                    break
            else:
                print("Failed to generate Path+mindmap after 3 attempts")
                continue

            for _ in range(3):
                output_all_path_noPrompt = generate_output_all_noPrompt(question, natural_language_path, prompts)
                if output_all_path_noPrompt is not None:
                    print('\nPath_noPrompt:\n', output_all_path_noPrompt)
                    break
            else:
                print("Failed to generate Path_noPrompt after 3 attempts")
                continue

            for _ in range(3):
                output_all_fact_cot = generate_output_all_cot(question, natural_language_facts, prompts)
                if output_all_fact_cot is not None:
                    print('\nFact+cot:\n', output_all_fact_cot)
                    output1_fact_cot = extract_output_summary(output_all_fact_cot, output_re)
                    print('\nFact+cot_summary:\n', output1_fact_cot)
                    break
            else:
                print("Failed to generate Fact+cot after 3 attempts")
                continue

            for _ in range(3):
                output_all_fact_tot = generate_output_all_tot(question, natural_language_facts, prompts)
                if output_all_fact_tot is not None:
                    print('\nFact+tot:\n', output_all_fact_tot)
                    output1_fact_tot = extract_output_summary(output_all_fact_tot, output_re)
                    print('\nFact+tot_summary:\n', output1_fact_tot)
                    break
            else:
                print("Failed to generate Fact+tot after 3 attempts")
                continue

            for _ in range(3):
                output_all_fact_mindmap = generate_output_all_mindmap(question, natural_language_facts, prompts)
                if output_all_fact_mindmap is not None:
                    print('\nFact+mindmap:\n', output_all_fact_mindmap)
                    output1_fact_mindmap = extract_output_summary(output_all_fact_mindmap, output_re)
                    print('\nFact+mindmap_summary:\n', output1_fact_mindmap)
                    break
            else:
                print("Failed to generate Fact+mindmap after 3 attempts")
                continue

            for _ in range(3):
                output_all_fact_noPrompt = generate_output_all_noPrompt(question, natural_language_facts, prompts)
                if output_all_fact_noPrompt is not None:
                    print('\nFact_noPrompt:\n', output_all_fact_noPrompt)
                    break
            else:
                print("Failed to generate Fact_noPrompt after 3 attempts")
                continue

            # Save results
            with open(output_path, 'a+', newline='') as f6:
                writer = csv.writer(f6)
                writer.writerow([question, label,
                                 triples_preAblation, natural_language_subgraph_preAblation, output_all_preAblation, output1_preAblation,
                                 natural_language_subgraph, output_all_subgraph_cot, output1_subgraph_cot,
                                 output_all_subgraph_tot, output1_subgraph_tot,
                                 output_all_subgraph_mindmap, output1_subgraph_mindmap,
                                 output_all_noPrompt,
                                 natural_language_path, output_all_path_cot, output1_path_cot,
                                 output_all_path_tot, output1_path_tot,
                                 output_all_path_mindmap, output1_path_mindmap,
                                 output_all_path_noPrompt,
                                 natural_language_facts, output_all_fact_cot, output1_fact_cot,
                                 output_all_fact_tot, output1_fact_tot,
                                 output_all_fact_mindmap, output1_fact_mindmap,
                                 output_all_fact_noPrompt])
                f6.flush()

    print("Finished!")
