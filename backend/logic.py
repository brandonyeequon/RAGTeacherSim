import os
import pickle
import json
import openai
import faiss
import numpy as np
from huggingface_hub import hf_hub_download

# Set OpenAI API key from environment or fallback
openai.api_key = os.getenv("OPENAI_API_KEY")

# Model config
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_STUDENT_MODEL = "gpt-4.1-mini-2025-04-14"
OPENAI_EXPERT_MODEL = "gpt-4.1-2025-04-14"
HUGGINGFACE_REPO_ID = "brandonyeequon/teacher_faiss"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_FAISS_INDEX_PATH = "vectorized_textbooks.faiss"
HF_TEXTBOOK_PASSAGES_PATH = "textbook_passages.pkl"  

try:
    downloaded_path = hf_hub_download(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=HF_TEXTBOOK_PASSAGES_PATH,
        local_dir=".", # Directory to save the file
        local_dir_use_symlinks=False, # Ensure the actual file is copied
        token=HUGGINGFACE_TOKEN,
        repo_type="dataset",
        cache_dir=None # Avoid using HF cache, download directly

    )
    # hf_hub_download might place it in a subdirectory structure based on the filename
except Exception as e:
    print(f"Error downloading FAISS index: {e}")
    raise

try:
    downloaded_path = hf_hub_download(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=HF_FAISS_INDEX_PATH,        # Path within the repo
        local_dir=".", # Directory to save the file
        local_dir_use_symlinks=False, # Ensure the actual file is copied
        token=HUGGINGFACE_TOKEN,
        repo_type="dataset",
        cache_dir=None # Avoid using HF cache, download directly

    )
    # hf_hub_download might place it in a subdirectory structure based on the filename
except Exception as e:
    print(f"Error downloading FAISS index: {e}")
    raise

# Load textbook context
with open("textbook_passages.pkl", "rb") as f:
    textbook_passages = pickle.load(f)
index = faiss.read_index("vectorized_textbooks.faiss")

# Load scenarios
with open("data/scenarios.json") as f:
    scenario_data = json.load(f)
scenarios_dict = {s["scenario_id"]: s for s in scenario_data}

def retrieve_textbook_context(query: str, top_k: int = 3) -> list:
    embed_resp = openai.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=[query])
    embedding = np.array([embed_resp.data[0].embedding], dtype="float32")
    distances, indices = index.search(embedding, top_k)
    return [textbook_passages[i] for i in indices[0] if 0 <= i < len(textbook_passages)]

def generate_student_response(user_input: str, chat_history: list, scenario_id: str = None) -> str:
    #We could implement a caching system that would store the context and append to the chat history for each session ID. For now this will be a simple implementation.
    scenario_context = ""
    if scenario_id and scenario_id in scenarios_dict: 
        s = scenarios_dict[scenario_id]
        student_name = s.get("student_name", "a student")
        if student_name != "Whole Class":
            scenario_context += f"Your name is {student_name}. "
            if "student_details" in s:
                scenario_context += f"Your personality: {s['student_details']} "
        if s.get("type") == "Student Emotion" and "emotion" in s:
            scenario_context += f"You are feeling {s['emotion'].lower()}. "
        if "classroom_situation" in s:
            scenario_context += f"Situation: {s['classroom_situation']} "
        if "description" in s:
            scenario_context += f"Topic: {s['description']}"

    system_prompt = {
        "role": "system",
        "content": f"You are a 2nd grade student (7-8 years old). Respond simply, sometimes distractedly, in a childlike manner, using simple vocabulary. Keep responses short (1-3 sentences). {scenario_context} You always respond as the second grader, never break character, and never act as the teacher. Address the teacher naturally based on their input."
    }
    messages = [system_prompt] + [
        {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
        for m in chat_history
    ] + [{"role": "user", "content": user_input}]

    response = openai.chat.completions.create(
        model=OPENAI_STUDENT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

def generate_expert_advice(question: str, chat_history: list, scenario_id: str = None) -> str:
    transcript = "\n".join(
        f"Teacher (User): {m['content']}" if m["role"] == "user" else f"Student (Assistant): {m['content']}"
        for m in chat_history
    )
    context = ""
    retrieval_query = question
    if scenario_id and scenario_id in scenarios_dict:
        s = scenarios_dict[scenario_id]
        context += f"Scenario: {s.get('title', '')}\n{s.get('description', '')}\n"
        retrieval_query += " " + s.get("title", "") + " " + s.get("description", "")
        if s.get("student_details"):
            context += f"Student Details: {s['student_details']}\n"
            retrieval_query += " " + s["student_details"]
        if s.get("classroom_situation"):
            context += f"Situation: {s['classroom_situation']}\n"
            retrieval_query += " " + s["classroom_situation"]
        if s.get("teacher_objective"):
            context += f"Objective: {s['teacher_objective']}\n"

    passages = retrieve_textbook_context(retrieval_query)
    passages_text = "\n".join(f"- {p}" for p in passages)

    prompt = f"""
{context}
Teacher's Question: {question}

Conversation:
{transcript}

Teaching Principles:
{passages_text}

What should the teacher do next?
"""
    messages = [
        {"role": "system", "content": "You are a helpful expert in teaching strategy for 2nd grade."},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=OPENAI_EXPERT_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def generate_expert_advice(question: str, chat_history: list, scenario_id: str = None) -> str:
    transcript = "\n".join(
        f"Teacher (User): {m['content']}" if m["role"] == "user" else f"Student (Assistant): {m['content']}"
        for m in chat_history
    )
    context = ""
    retrieval_query = question
    if scenario_id and scenario_id in scenarios_dict:
        s = scenarios_dict[scenario_id]
        context += f"Scenario: {s.get('title', '')}\n{s.get('description', '')}\n"
        retrieval_query += " " + s.get("title", "") + " " + s.get("description", "")
        if s.get("student_details"):
            context += f"Student Details: {s['student_details']}\n"
            retrieval_query += " " + s["student_details"]
        if s.get("classroom_situation"):
            context += f"Situation: {s['classroom_situation']}\n"
            retrieval_query += " " + s["classroom_situation"]
        if s.get("teacher_objective"):
            context += f"Objective: {s['teacher_objective']}\n"

    passages = retrieve_textbook_context(retrieval_query)
    passages_text = "\n".join(f"- {p}" for p in passages)

    prompt = f"""
{context}
Teacher's Question: {question}

Conversation:
{transcript}

Teaching Principles:
{passages_text}

What should the teacher do next?
"""
    messages = [
        {"role": "system", "content": "You are a helpful expert in teaching strategy for 2nd grade."},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=OPENAI_EXPERT_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def evaluate_teacher_effectiveness(chat_history: list, scenario_id: str = None) -> tuple:

    context = ""
    if scenario_id and scenario_id in scenarios_dict:
        s = scenarios_dict[scenario_id]
        context += f"Scenario: {s.get('title', '')}\n{s.get('description', '')}\n"
        if s.get("student_details"):
            context += f"Student Details: {s['student_details']}\n"
        if s.get("teacher_objective"):
            context += f"Objective: {s['teacher_objective']}\n"

    transcript = "\n".join(
        f"Teacher: {m['content']}" if m["role"] == "user" else f"Student: {m['content']}"
        for m in chat_history
    )
    prompt = f"""
Evaluate the effectiveness of this teacher-student interaction in resolving the objective included in the context. Generous in your rating.:
Context: {context}
Teacher-Student interaction: {transcript}

Respond with a valid JSON object only — no explanation, no markdown, and no code block formatting the JSON object should contain:
1. A score from 1 to 10 based on how well the teacher resolved the student's issue based on the Score Rubric.
2. A brief list of positive things the teacher did, This should be returned in a comma delinated string.
3. Advice for improvement.

Score rubric:
1-3: Objective was poorly achieved or the teacher used ineffective teaching strategies 
4-6: The teacher attempted to use good teaching pratices but did not achieve the teaching objective 
7-8: The teaching objective was mostly achieved and the teacher used good teaching practices
9-10: The teaching objective was fully achieved and the teacher used excellent teaching practices

Format your response as JSON with keys: "score", "feedback", and "advice".
"""
    messages = [
        {"role": "system", "content": "You are a teaching evaluator for elementary school simulations."},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=OPENAI_EXPERT_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=400
    )

    try:
        result = json.loads(response.choices[0].message.content)
        score = result.get("score", 5)
        feedback = result.get("feedback", "No feedback provided.")
        advice = result.get("advice", "No advice provided.")
    except Exception as e:
        print(f"Failed to parse evaluation response: {e}")
        score = 5
        feedback = "Unable to evaluate conversation."
        advice = "Make sure to engage clearly and follow through on the student's confusion."

    return score, feedback, advice
