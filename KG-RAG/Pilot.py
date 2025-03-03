import time
import heapq
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import pandas as pd
from typing import List, Tuple
import pickle
import json
import csv
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


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

def extract_keywords(input_question, max_retries=3):
    try:
        messages = [
            SystemMessage(content="You are a medical keyword extraction specialist. Your task is to extract important keywords from medical questions."),
            HumanMessage(content=f"Please extract important keywords from the following medical question, separated by commas. Return only the keyword list without any explanation:\n\n{input_question}")
        ]
        result = chat(messages)
        keywords = result.content.strip().split(',')
        print(f"Extracted keywords: {keywords}")
        return [kw.strip() for kw in keywords]
    except Exception as e:
        words = re.findall(r'\b\w+\b', input_question.lower())
        return list(set(words))[:5]     

def match_entities(input_question, entity_embeddings):
    keywords = extract_keywords(input_question)
    
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


def extract_topics(input_question, match_kg):
    for _ in range(3):  
        messages = [
            SystemMessage(
                content="You are an excellent AI assistant, and you can summarize main topic based on input question and corresponding entities."),
            HumanMessage(content="Based on the question" + input_question),
            AIMessage(
                content="You have some relevant entities in the following:\n\n" + '###' + str(match_kg)),
            HumanMessage(
                content="What specific disease or medical condition is this question mainly about? Summarize the main topic of this question in one word or phrase. Extract the main topic and indicate it in single quotation marks. Make sure to follow this exact format and only output once: The main topic is '...'.\n\n\n"
                        + "There is a sample, refer to the format:\n"
                        + """Based on the question "Doctor, I have been experiencing sudden and frequent panic attacks. I don't know what to do." and the relevant entities "Panic_disorder", "Anxiety_and_nervousness", "Palpitations". The main topic is 'mental health'."""
                        # + "The main topic is 'mental health'."
            )
        ]
        result = chat(messages)
        output_all = result.content
        output_all = preprocess_output(output_all)
        print("llm topics:", output_all)

        pattern = r"[Tt]he main topic (?:is|can be summarized as) '(.*?)'"
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
        template = template,
        input_variables = ["knowledge"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(knowledge = knowledge)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(knowledge=knowledge,\
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


# # path
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


# # fact
def generate_natural_language_facts(triples):
    if not triples:
        return ""
    
    formatted_lines = []
    for head, relation, tail in triples:
        formatted_lines.append(f"'{head}' -> '{relation}' -> '{tail}'")
    
    return "\n".join([f"Evidence{i+1}：'{head}' {relation} '{tail}'" for i, (head, relation, tail) in enumerate(triples)])


# # Prompt design
def final_answer_noPrompt(str, subgraph):
    messages = [
            SystemMessage(
            content="You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "),
            HumanMessage(content="Patient input:" + input_text[0]),
            AIMessage(content="You have some medical knowledge information in the following:\n\n" + '###' + subgraph),
            HumanMessage(
                content="What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? "
            )
        ]
        
    # Add a try/except block to handle the exception
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


def final_answer_cot(str, subgraph):
    messages = [
    SystemMessage(
        content="You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "
    ),
    HumanMessage(content="Patient input:" + input_text[0]),
    AIMessage(content="You have some medical knowledge information in the following:\n\n" + '###' + subgraph),
    HumanMessage(
        content="Based on the given question and knowledge graph evidence, please use chain of thought reasoning to provide a comprehensive medical analysis. Follow these steps strictly:\n\n"
        "【Step 1: Information Analysis】\n"
        "- Analyze patient's symptoms and presentation\n"
        "- Identify key clinical information\n"
        "- Determine relevant medical history\n\n"
        "【Step 2: Evidence Connection】\n"
        "- Identify relevant evidence from knowledge graph\n"
        "- Connect symptoms to potential conditions\n"
        "- Link conditions to appropriate tests and treatments\n\n"
        "【Step 3: Reasoning Chain】\n"
        "Develop a logical reasoning chain that shows:\n"
        "- How symptoms connect to possible diagnoses\n"
        "- Why certain tests are necessary\n"
        "- How treatment recommendations follow from diagnosis\n\n"
        "【Step 4: Structured Output】\n"
        "Please provide your response in two distinct parts:\n\n"
        "Output 1: Clinical Conclusion\n"
        "Provide a clear summary of:\n"
        "- Most likely diagnosis\n"
        "- Recommended diagnostic tests\n"
        "- Suggested treatment plan\n\n"
        "Output 2: Reasoning Process\n"
        "Show the complete chain of thought as:\n"
        "Symptom -> Evidence -> Reasoning -> Conclusion\n"
        "Format each step as:\n"
        "1. Initial observation: [symptom/condition]\n"
        "2. Supporting evidence: [knowledge graph reference]\n"
        "3. Reasoning: [logical connection]\n"
        "4. Intermediate conclusion: [diagnostic step]\n"
        "5. Final assessment: [comprehensive diagnosis and plan]\n\n"
        "Example format:\n"
        "Output 1:\n"
        "Based on the analysis, the patient likely has [diagnosis]. Recommended tests include [tests]. Treatment plan should consist of [medications/interventions].\n\n"
        "Output 2:\n"
        "1. Patient presents with [symptom] → Evidence shows [connection] → This suggests [conclusion]\n"
        "2. Given [finding] → Knowledge graph indicates [evidence] → Therefore [reasoning]\n"
        "3. Considering [factor] → Medical evidence supports [connection] → Leading to [final diagnosis]\n"
        )
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


def final_answer_tot(str, subgraph):
    messages = [
    SystemMessage(
        content="You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "
    ),
    HumanMessage(content="Patient input:" + input_text[0]),
    AIMessage(
        content="You have some medical knowledge information in the following:\n\n" + '###' + subgraph
    ),
    HumanMessage(
        content="Based on the patient's description and medical evidence provided, please conduct a detailed analysis using tree-structured thinking and provide diagnosis and treatment recommendations. Follow these steps strictly:\n\n"
        "【Step 1: Information Extraction】\n"
        "Extract all key information from the patient's description, including:\n"
        "- Patient demographics\n"
        "- Symptoms and signs\n"
        "- Examination results\n"
        "- Medical history\n"
        "- Current medications\n\n"
        "【Step 2: Medical Evidence Correlation】\n"
        "Link the extracted information to relevant medical evidence, possible conditions, or pathological states. Identify typical symptoms, necessary examinations, and standard treatment protocols for each potential condition.\n\n"
        "【Step 3: Tree-Structured Analysis】\n"
        "Construct a decision tree where each node represents a key reasoning step or decision point. Branches should show different reasoning paths and alternatives. Each node must specify the supporting symptoms, examination data, or medical knowledge, along with justification for selecting or excluding that path.\n\n"
        "【Step 4: Output Format】\n"
        "Please divide your response into two distinct outputs:\n\n"
        "**Output 1: \n"
        "Provide a concise summary of:\n"
        "- Probable diagnosis\n"
        "- Recommended further examinations\n"
        "- Suggested treatment plan\n\n"
        "**Output 2: \n"
        "Present the complete reasoning process in a tree structure, with:\n"
        "- Each node showing key evidence\n"
        "- Branches displaying different diagnostic pathways\n"
        "- Clear rationale for each decision point\n\n"
        "【Sample Output Format】:\n"
        "**Output 1:\n"
        "Based on the patient's presentation, the initial diagnosis suggests upper respiratory tract infection with mild pharyngitis. Further blood work and throat swab are recommended to rule out bacterial infection. Symptomatic treatment with antipyretics and antitussives is advised, along with adequate hydration and rest.\n\n"
        "**Output 2:\n"
        "Patient Presentation (Root)\n"
        "├─ Branch 1: Upper Respiratory Tract Infection\n"
        "│    ├─ Node: Based on fever and sore throat\n"
        "│    └─ Node: Based on persistent cough after symptom improvement\n"
        "├─ Branch 2: Mild Pharyngitis\n"
        "│    ├─ Node: Based on throat inflammation and local discomfort\n"
        "│    └─ Node: Based on examination supporting non-severe infection\n"
        "└─ Branch 3: Bacterial Infection Ruled Out\n"
        "     └─ Node: Based on normal WBC count and clear lung examination\n\n"
        "Please ensure your output follows this format with rigorous logic and sufficient supporting evidence."
    )
]
        
    # Add a try/except block to handle the exception
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


def final_answer_mindmap(str, subgraph):
    messages = [
        SystemMessage(
            content="You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "),
        HumanMessage(content="Patient input:" + input_text[0]),
        AIMessage(
            content="You have some medical knowledge information in the following:\n\n" + '###' + subgraph),
        HumanMessage(
            content="What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
                    + "Output strictly according to the format of 'Output1, Output2, Output3'"
                    + "Output1: The answer includes disease and tests and recommened medications.\n\n"
                    + "Output2: Show me inference process as a string about extract what knowledge from which Evidence, and in the end infer what result. \n Transport the inference process into the following format:\n Evidence number('entity name'->'relation name'->...)->Evidence number('entity name'->'relation name'->...)->Evidence number('entity name'->'relation name'->...)->result number('entity name')->Evidence number('entity name'->'relation name'->...)->Evidence number('entity name'->'relation name'->...). \n\n"
                    + "Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
                    + "There is a sample, refer to the format:\n"
                    + """
Output 1:
Based on the symptoms described, the patient may have ..., which is inflammation of.... To confirm the diagnosis, the patient should undergo .... It is also recommended to....

Output 2:
Evidence 1('...'->'...'->'...')->Evidence 2('...'->'...'->'...')->Evidence 1('...'->'...'->'...')->Evidence 2('...'->'...'->'...')->result 1('...')->Evidence 3('...'->'...'->'...')->Evidence 3('...'->'...'->'...').

Output 3: 
Patient(Evidence 1)
└── has been experiencing(Evidence 1)
    └── ...(Evidence 1)(Evidence 2)
        └── could be caused by(Evidence 2)
            └── ...(Evidence 2)(Evidence 1)
                ├── requires(Evidence 1)
                │   └── ...(Evidence 1)(Evidence 2)
                │       └── may include(Evidence 2)
                │           └──...(Evidence 2)(result 1)(Evidence 3)
                ├── can be treated with(Evidence 3)
                │   └── ...(Evidence 3)(Evidence 3)
                └── should be accompanied by(Evidence 3)
                    └── ...(Evidence 3)
                                    """
        )
    ]
    # Add a try/except block to handle the exception
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


# # # Generate final output
def generate_output_all_noPrompt(input_text, natural_language):
    try:
        output_all_ours = final_answer_noPrompt(input_text, natural_language)
    except Exception as e:
        print(f"An error occurred: {e}. Failed to generate no prompt final answer")
        return None
    return output_all_ours

def generate_output_all_cot(input_text, natural_language):
    try:
        output_all_ours = final_answer_cot(input_text, natural_language)
    except Exception as e:
        print(f"An error occurred: {e}. Failed to generate cot final answer")
        return None
    return output_all_ours

def generate_output_all_tot(input_text, natural_language):
    try:
        output_all_ours = final_answer_tot(input_text, natural_language)
    except Exception as e:
        print(f"An error occurred: {e}. Failed to generate tot final answer")
        return None
    return output_all_ours

def generate_output_all_mindmap(input_text, natural_language):
    try:
        output_all_ours = final_answer_mindmap(input_text, natural_language)
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
    

if __name__ == "__main__":
    # 1. build neo4j knowledge graph datasets
    uri = "your_neo4j_uri"
    username = "your_neo4j_username"
    password = "your_neo4j_password"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    # Initialize the BERT model and tokenizer
    BERT_PATH = '../bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    bert_model = BertModel.from_pretrained(BERT_PATH)

    # # 2. OpenAI API based chat
    chat = ChatOpenAI(openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1")


    with open('./output/GenMedGPT-5K/output_pilot.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(
            ['Question', 'Label', 
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

    with open('./data/GenMedGPT-5K/entity_embeddings.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)

    with open("./data/GenMedGPT-5K/genmedgpt-5k.json", "r") as f:
        for line in f.readlines()[:]:
            x = json.loads(line)
            input_text = x["question"]           
            if input_text == []:
                continue
            print('Question:',input_text[0])

            output_text = x["answer"]
            print('Answer:',output_text)

            match_kg = match_entities(input_text[0], entity_embeddings)

            # # # Pre-retrieval
            attempt_count = 0
            max_attempts = 3 
            while attempt_count < max_attempts:
                try:
                    topic = extract_topics(input_text[0], match_kg)
                    print("Extracted topic:", topic)
                    break
                except Exception as e:
                    print("Failed to extract topic:", e)
                    time.sleep(5)  
                    attempt_count += 1 
                    topic = ""


            subgraph = heuristic_search(input_text[0], match_kg, topic)
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

            # # Ablation
            subgraph_preAblation = heuristic_search_preAblation(input_text[0], match_kg)
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

            # # Path level
            path = path_search(input_text[0], match_kg, topic)  
            print("Final path:")
            for node, relationship in path:
                if relationship:
                    print(f"{node} --[{relationship}]--> ", end='')
                else:
                    print(f"{node}", end='')
            natural_language_path = generate_path_description(path)
            print("Natural language description path:\n", natural_language_path)

            # # Facts level 
            facts = retrieve_top_triples(input_text[0], match_kg, topic, 4)
            print("TopN related facts:", facts)
            print("Final Selected top N facts:")
            for head, relation, tail in facts:
                print(f"'{head}' -> '{relation}' -> '{tail}'")
            natural_language_facts = generate_natural_language_facts(facts)
            print("Natural language description facts:\n", natural_language_facts)

            # 13. Final output and Extract output summary
            re = r"Output 1:(.*?)Output 2:"

            for _ in range(3):
                output_all_preAblation = generate_output_all_mindmap(input_text[0], natural_language_subgraph_preAblation)
                print('\nPreAblation:\n', output_all_preAblation)
                output1_preAblation = extract_output_summary(output_all_preAblation, re)
                if output1_preAblation is not None:
                    print('\nPreAblation-summary:\n', output1_preAblation)
                    break
            else:
                print("Failed to extract output1_preAblation from output_all_preAblation after 3 attempts")
                continue

            for _ in range(3):
                output_all_subgraph_cot = generate_output_all_cot(input_text[0], natural_language_subgraph)     
                if output_all_subgraph_cot is not None:  
                    print('\nSubgraph+cot:\n', output_all_subgraph_cot)
                    output1_subgraph_cot = extract_output_summary(output_all_subgraph_cot, re)
                    print('\nSubgraph+cot_summary:\n', output1_subgraph_cot)
                    if output1_subgraph_cot is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Subgraph+cot after 3 attempts")
                continue

            for _ in range(3):  
                output_all_subgraph_tot = generate_output_all_tot(input_text[0], natural_language_subgraph)     
                if output_all_subgraph_tot is not None:  
                    print('\nSubgraph+tot:\n', output_all_subgraph_tot)
                    output1_subgraph_tot = extract_output_summary(output_all_subgraph_tot, re)
                    print('\nSubgraph+tot_summary:\n', output1_subgraph_tot)
                    if output1_subgraph_tot is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Subgraph+tot after 3 attempts")
                continue

            for _ in range(3):  
                output_all_subgraph_mindmap = generate_output_all_mindmap(input_text[0], natural_language_subgraph)
                if output_all_subgraph_mindmap is not None:
                    print('\nSubgraph+mindmap:\n', output_all_subgraph_mindmap)
                    output1_subgraph_mindmap = extract_output_summary(output_all_subgraph_mindmap, re)
                    print('\nSubgraph+mindmap_summary:\n', output1_subgraph_mindmap)
                    if output1_subgraph_mindmap is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Subgraph+mindmap after 3 attempts")
                continue

            for _ in range(3):
                output_all_noPrompt = generate_output_all_noPrompt(input_text[0], natural_language_subgraph)
                if output_all_noPrompt is not None:
                    print('\nSubgraph_noPrompt:\n', output_all_noPrompt)
                    break
            else:
                print("Failed to generate Subgraph_noPrompt after 3 attempts")
                continue

            for _ in range(3):  
                output_all_path_cot = generate_output_all_cot(input_text[0], natural_language_path)     
                if output_all_path_cot is not None:  
                    print('\nPath+cot:\n', output_all_path_cot)
                    output1_path_cot = extract_output_summary(output_all_path_cot, re)
                    print('\nPath+cot_summary:\n', output1_subgraph_cot)
                    if output1_path_cot is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Path+cot after 3 attempts")
                continue

            for _ in range(3): 
                output_all_path_tot = generate_output_all_tot(input_text[0], natural_language_path)
                if output_all_path_tot is not None:
                    print('\nPath+tot:\n', output_all_path_tot)
                    output1_path_tot = extract_output_summary(output_all_path_tot, re)
                    print('\nPath+tot_summary:\n', output1_path_tot)
                    if output1_path_tot is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Path+tot after 3 attempts")
                continue

            for _ in range(3): 
                output_all_path_mindmap = generate_output_all_mindmap(input_text[0], natural_language_path)
                if output_all_path_mindmap is not None:
                    print('\nPath+mindmap:\n', output_all_path_mindmap)
                    output1_path_mindmap = extract_output_summary(output_all_path_mindmap, re)
                    print('\nPath+mindmap_summary:\n', output1_path_mindmap)
                    if output1_path_mindmap is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Path+mindmap after 3 attempts")
                continue

            for _ in range(3): 
                output_all_path_noPrompt = generate_output_all_noPrompt(input_text[0], natural_language_path)
                if output_all_path_noPrompt is not None:
                    print('\nPath_noPrompt:\n', output_all_path_noPrompt)
                    break
            else:
                print("Failed to generate Path_noPrompt after 3 attempts")
                continue

            for _ in range(3):
                output_all_fact_cot = generate_output_all_cot(input_text[0], natural_language_facts)
                if output_all_fact_cot is not None:
                    print('\nFact+cot:\n', output_all_fact_cot)
                    output1_fact_cot = extract_output_summary(output_all_fact_cot, re)
                    print('\nFact+cot_summary:\n', output1_fact_cot)
                    if output1_fact_cot is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Fact+cot after 3 attempts")
                continue

            for _ in range(3):
                output_all_fact_tot = generate_output_all_tot(input_text[0], natural_language_facts)
                if output_all_fact_tot is not None:
                    print('\nFact+tot:\n', output_all_fact_tot)
                    output1_fact_tot = extract_output_summary(output_all_fact_tot, re)
                    print('\nFact+tot_summary:\n', output1_fact_tot)
                    if output1_fact_tot is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Fact+tot after 3 attempts")
                continue

            for _ in range(3):
                output_all_fact_mindmap = generate_output_all_mindmap(input_text[0], natural_language_facts)
                if output_all_fact_mindmap is not None:
                    print('\nFact+mindmap:\n', output_all_fact_mindmap)
                    output1_fact_mindmap = extract_output_summary(output_all_fact_mindmap, re)
                    print('\nFact+mindmap_summary:\n', output1_fact_mindmap)
                    if output1_fact_mindmap is None:
                        flag_wrong = 1
                    break
            else:
                print("Failed to generate Fact+mindmap after 3 attempts")
                continue

            for _ in range(3):
                output_all_fact_noPrompt = generate_output_all_noPrompt(input_text[0], natural_language_facts)
                if output_all_fact_noPrompt is not None:
                    print('\nFact_noPrompt:\n', output_all_fact_noPrompt)
                    break
            else:
                print("Failed to generate Fact_noPrompt after 3 attempts")
                continue

            ### save the final result
            with open('./output/GenMedGPT-5K/output_pilot.csv', 'a+', newline='') as f6:
                writer = csv.writer(f6)
                writer.writerow([input_text[0], output_text[0], 
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
                                 output_all_fact_noPrompt,])
                f6.flush()

    # Finish the session
    print("Finished!")