"""
Dataset-type-specific prompt templates for the Pilot KG-RAG pipeline.

Each dict must contain:
  keyword_system, keyword_instruction (with {question} placeholder)
  topic_system, topic_instruction, topic_pattern
  system, input_label, evidence_label
  task_noPrompt, task_cot, task_tot, task_mindmap
  output_pattern
"""

MEDICAL_DIAGNOSIS = {
    'keyword_system': (
        "You are a medical keyword extraction specialist. "
        "Your task is to extract important keywords from medical questions."
    ),
    'keyword_instruction': (
        "Please extract important keywords from the following medical question, "
        "separated by commas. Return only the keyword list without any explanation:\n\n{question}"
    ),
    'topic_system': (
        "You are an excellent AI assistant, and you can summarize main topic "
        "based on input question and corresponding entities."
    ),
    'topic_instruction': (
        "What specific disease or medical condition is this question mainly about? "
        "Summarize the main topic of this question in one word or phrase. "
        "Extract the main topic and indicate it in single quotation marks. "
        "Make sure to follow this exact format and only output once: The main topic is '...'.\n\n\n"
        "There is a sample, refer to the format:\n"
        "Based on the question \"Doctor, I have been experiencing sudden and frequent panic attacks. "
        "I don't know what to do.\" and the relevant entities \"Panic_disorder\", "
        "\"Anxiety_and_nervousness\", \"Palpitations\". The main topic is 'mental health'."
    ),
    'topic_pattern': r"[Tt]he main topic (?:is|can be summarized as) '(.*?)'",
    'system': (
        "You are an excellent AI doctor, and you can diagnose diseases and recommend "
        "medications based on the symptoms in the conversation."
    ),
    'input_label': "Patient input:",
    'evidence_label': "You have some medical knowledge information in the following:",
    'task_noPrompt': (
        "What disease does the patient have? What tests should patient take to confirm "
        "the diagnosis? What recommened medications can cure the disease?"
    ),
    'task_cot': (
        "Based on the given question and knowledge graph evidence, please use chain of thought "
        "reasoning to provide a comprehensive medical analysis. Follow these steps strictly:\n\n"
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
        "Based on the analysis, the patient likely has [diagnosis]. Recommended tests include "
        "[tests]. Treatment plan should consist of [medications/interventions].\n\n"
        "Output 2:\n"
        "1. Patient presents with [symptom] → Evidence shows [connection] → This suggests [conclusion]\n"
        "2. Given [finding] → Knowledge graph indicates [evidence] → Therefore [reasoning]\n"
        "3. Considering [factor] → Medical evidence supports [connection] → Leading to [final diagnosis]"
    ),
    'task_tot': (
        "Based on the patient's description and medical evidence provided, please conduct a "
        "detailed analysis using tree-structured thinking and provide diagnosis and treatment "
        "recommendations. Follow these steps strictly:\n\n"
        "【Step 1: Information Extraction】\n"
        "Extract all key information from the patient's description, including:\n"
        "- Patient demographics\n"
        "- Symptoms and signs\n"
        "- Examination results\n"
        "- Medical history\n"
        "- Current medications\n\n"
        "【Step 2: Medical Evidence Correlation】\n"
        "Link the extracted information to relevant medical evidence, possible conditions, or "
        "pathological states. Identify typical symptoms, necessary examinations, and standard "
        "treatment protocols for each potential condition.\n\n"
        "【Step 3: Tree-Structured Analysis】\n"
        "Construct a decision tree where each node represents a key reasoning step or decision "
        "point. Branches should show different reasoning paths and alternatives. Each node must "
        "specify the supporting symptoms, examination data, or medical knowledge, along with "
        "justification for selecting or excluding that path.\n\n"
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
        "Based on the patient's presentation, the initial diagnosis suggests upper respiratory "
        "tract infection with mild pharyngitis. Further blood work and throat swab are recommended "
        "to rule out bacterial infection. Symptomatic treatment with antipyretics and antitussives "
        "is advised, along with adequate hydration and rest.\n\n"
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
        "Please ensure your output follows this format with rigorous logic and sufficient "
        "supporting evidence."
    ),
    'task_mindmap': (
        "What disease does the patient have? What tests should patient take to confirm the "
        "diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
        "Output strictly according to the format of 'Output1, Output2, Output3'\n"
        "Output1: The answer includes disease and tests and recommened medications.\n\n"
        "Output2: Show me inference process as a string about extract what knowledge from which "
        "Evidence, and in the end infer what result. \n"
        " Transport the inference process into the following format:\n"
        " Evidence number('entity name'->'relation name'->...)->Evidence number('entity name'->"
        "'relation name'->...)->Evidence number('entity name'->'relation name'->...)->"
        "result number('entity name')->Evidence number('entity name'->'relation name'->...)->"
        "Evidence number('entity name'->'relation name'->...). \n\n"
        "Output3: Draw a decision tree. The entity or relation in single quotes in the inference "
        "process is added as a node with the source of evidence, which is followed by the entity "
        "in parentheses.\n\n"
        "There is a sample, refer to the format:\n"
        "Output 1:\n"
        "Based on the symptoms described, the patient may have ..., which is inflammation of.... "
        "To confirm the diagnosis, the patient should undergo .... It is also recommended to....\n\n"
        "Output 2:\n"
        "Evidence 1('...'->'...'->'...')->Evidence 2('...'->'...'->'...')->Evidence 1('...'->"
        "'...'->'...')->Evidence 2('...'->'...'->'...')->result 1('...')->Evidence 3('...'->"
        "'...'->'...')->Evidence 3('...'->'...'->'...').\n\n"
        "Output 3: \n"
        "Patient(Evidence 1)\n"
        "└── has been experiencing(Evidence 1)\n"
        "    └── ...(Evidence 1)(Evidence 2)\n"
        "        └── could be caused by(Evidence 2)\n"
        "            └── ...(Evidence 2)(Evidence 1)\n"
        "                ├── requires(Evidence 1)\n"
        "                │   └── ...(Evidence 1)(Evidence 2)\n"
        "                │       └── may include(Evidence 2)\n"
        "                │           └──...(Evidence 2)(result 1)(Evidence 3)\n"
        "                ├── can be treated with(Evidence 3)\n"
        "                │   └── ...(Evidence 3)(Evidence 3)\n"
        "                └── should be accompanied by(Evidence 3)\n"
        "                    └── ...(Evidence 3)"
    ),
    'output_pattern': r"Output 1:(.*?)Output 2:",
}


MEDICAL_EXAM = {
    'keyword_system': (
        "You are a medical exam question keyword extraction specialist. "
        "Your task is to extract the key medical concepts from exam questions."
    ),
    'keyword_instruction': (
        "Please extract important medical keywords from the following exam question, "
        "separated by commas. Return only the keyword list without any explanation:\n\n{question}"
    ),
    'topic_system': (
        "You are an excellent AI assistant, and you can identify the main medical topic "
        "of an exam question based on the question and relevant entities."
    ),
    'topic_instruction': (
        "What is the main medical topic of this exam question? "
        "Summarize it in one word or phrase and indicate it in single quotation marks. "
        "Make sure to follow this exact format: The main topic is '...'.\n\n"
        "Example: The main topic is 'pharmacology'."
    ),
    'topic_pattern': r"[Tt]he main topic (?:is|can be summarized as) '(.*?)'",
    'system': (
        "You are an excellent medical AI assistant specializing in multiple-choice exam questions. "
        "You can select the correct answer based on medical knowledge and evidence."
    ),
    'input_label': "Question:",
    'evidence_label': "You have some relevant medical knowledge information in the following:",
    'task_noPrompt': (
        "Based on the above information, which option is correct? "
        "Provide only the letter of the correct answer (A, B, C, D, or E)."
    ),
    'task_cot': (
        "Based on the question and knowledge graph evidence, use chain of thought reasoning "
        "to select the correct answer. Follow these steps:\n\n"
        "【Step 1: Question Analysis】\n"
        "- Identify the key medical concept being tested\n"
        "- Note any specific conditions or constraints\n\n"
        "【Step 2: Evidence Evaluation】\n"
        "- Match each option against the knowledge graph evidence\n"
        "- Identify which option is supported by the evidence\n\n"
        "【Step 3: Structured Output】\n"
        "Output 1: Answer\n"
        "State the correct answer letter and a brief explanation.\n\n"
        "Output 2: Reasoning Process\n"
        "Show the reasoning for selecting or rejecting each option:\n"
        "Option A: [evidence-based assessment]\n"
        "Option B: [evidence-based assessment]\n"
        "...\n"
        "Conclusion: The correct answer is [letter] because [reasoning]."
    ),
    'task_tot': (
        "Based on the question and medical evidence, analyze all options using "
        "tree-structured thinking. Follow these steps:\n\n"
        "【Step 1: Option Extraction】\n"
        "List all options and their key claims.\n\n"
        "【Step 2: Evidence Matching】\n"
        "For each option, identify supporting or contradicting evidence.\n\n"
        "【Step 3: Tree-Structured Analysis】\n"
        "Build a decision tree evaluating each option branch.\n\n"
        "【Step 4: Output Format】\n"
        "**Output 1:\n"
        "The correct answer is [letter]. [Brief explanation]\n\n"
        "**Output 2:\n"
        "Question (Root)\n"
        "├─ Option A: [assessment based on evidence]\n"
        "├─ Option B: [assessment based on evidence]\n"
        "├─ Option C: [assessment based on evidence]\n"
        "└─ Option D/E: [assessment based on evidence]\n"
        "Conclusion: [letter] is correct because [evidence-based rationale]"
    ),
    'task_mindmap': (
        "Which option is correct? Think step by step.\n\n"
        "Output strictly according to the format of 'Output1, Output2, Output3'\n"
        "Output1: The correct answer is [letter]. [Brief explanation]\n\n"
        "Output2: Show the inference process indicating which Evidence supports or refutes "
        "each option, in the format:\n"
        "Evidence number('entity'->'relation'->...)->...->result('correct answer letter')\n\n"
        "Output3: Draw a decision tree evaluating each option with evidence sources.\n\n"
        "Example:\n"
        "Output 1:\n"
        "The correct answer is C. [Explanation based on evidence]\n\n"
        "Output 2:\n"
        "Evidence 1('drug'->'mechanism'->...)->Evidence 2('condition'->'treatment'->...)->result('C')\n\n"
        "Output 3:\n"
        "Question(Evidence 1)\n"
        "├── Option A: incorrect(Evidence 2)\n"
        "├── Option B: incorrect(Evidence 1)\n"
        "├── Option C: correct(Evidence 1)(Evidence 2)\n"
        "└── Option D: incorrect(Evidence 3)"
    ),
    'output_pattern': r"Output 1:(.*?)Output 2:",
}


COMMONSENSE = {
    'keyword_system': (
        "You are a keyword extraction specialist. "
        "Your task is to extract the key concepts from commonsense questions."
    ),
    'keyword_instruction': (
        "Please extract important keywords from the following question, "
        "separated by commas. Return only the keyword list without any explanation:\n\n{question}"
    ),
    'topic_system': (
        "You are an excellent AI assistant. You can identify the main concept "
        "of a question based on the question and relevant entities."
    ),
    'topic_instruction': (
        "What is the main topic or concept of this question? "
        "Summarize it in one word or phrase and indicate it in single quotation marks. "
        "Make sure to follow this exact format: The main topic is '...'.\n\n"
        "Example: The main topic is 'transportation'."
    ),
    'topic_pattern': r"[Tt]he main topic (?:is|can be summarized as) '(.*?)'",
    'system': (
        "You are an excellent AI assistant with strong commonsense reasoning ability. "
        "You can select the most appropriate answer to commonsense questions."
    ),
    'input_label': "Question:",
    'evidence_label': "You have some relevant commonsense knowledge in the following:",
    'task_noPrompt': (
        "Based on the above information, which option is the most appropriate answer? "
        "Provide only the letter of the correct answer (A, B, C, D, or E)."
    ),
    'task_cot': (
        "Based on the question and commonsense knowledge evidence, use chain of thought "
        "reasoning to select the best answer. Follow these steps:\n\n"
        "【Step 1: Question Analysis】\n"
        "- Identify the commonsense scenario being asked about\n"
        "- Note what knowledge is needed to answer it\n\n"
        "【Step 2: Evidence Evaluation】\n"
        "- Match each option against the commonsense knowledge evidence\n"
        "- Identify which option best fits common sense\n\n"
        "【Step 3: Structured Output】\n"
        "Output 1: Answer\n"
        "State the correct answer letter and a brief explanation.\n\n"
        "Output 2: Reasoning Process\n"
        "Show the reasoning for each option:\n"
        "Option A: [commonsense assessment]\n"
        "Option B: [commonsense assessment]\n"
        "...\n"
        "Conclusion: The correct answer is [letter] because [reasoning]."
    ),
    'task_tot': (
        "Based on the question and commonsense evidence, analyze all options using "
        "tree-structured thinking. Follow these steps:\n\n"
        "【Step 1: Scenario Understanding】\n"
        "Describe the commonsense scenario and what the question is asking.\n\n"
        "【Step 2: Option Analysis】\n"
        "For each option, evaluate its plausibility using common sense and evidence.\n\n"
        "【Step 3: Tree-Structured Analysis】\n"
        "Build a decision tree evaluating each option branch.\n\n"
        "【Step 4: Output Format】\n"
        "**Output 1:\n"
        "The correct answer is [letter]. [Brief explanation]\n\n"
        "**Output 2:\n"
        "Question (Root)\n"
        "├─ Option A: [plausibility assessment]\n"
        "├─ Option B: [plausibility assessment]\n"
        "├─ Option C: [plausibility assessment]\n"
        "└─ Option D/E: [plausibility assessment]\n"
        "Conclusion: [letter] is most plausible because [evidence-based rationale]"
    ),
    'task_mindmap': (
        "Which option is the best answer? Think step by step.\n\n"
        "Output strictly according to the format of 'Output1, Output2, Output3'\n"
        "Output1: The correct answer is [letter]. [Brief explanation]\n\n"
        "Output2: Show the inference process indicating which Evidence supports each option, "
        "in the format:\n"
        "Evidence number('concept'->'relation'->...)->...->result('correct answer letter')\n\n"
        "Output3: Draw a decision tree evaluating each option with evidence sources.\n\n"
        "Example:\n"
        "Output 1:\n"
        "The correct answer is A. [Explanation]\n\n"
        "Output 2:\n"
        "Evidence 1('work'->'goal'->...)->result('A')\n\n"
        "Output 3:\n"
        "Question(Evidence 1)\n"
        "├── Option A: correct(Evidence 1)\n"
        "├── Option B: less likely(Evidence 2)\n"
        "├── Option C: incorrect(Evidence 1)\n"
        "└── Option D: incorrect(Evidence 2)"
    ),
    'output_pattern': r"Output 1:(.*?)Output 2:",
}
