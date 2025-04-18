import streamlit as st
import faiss
import numpy as np
import pickle
import json
import os
import openai # Use OpenAI for chat and embeddings
from typing import Optional, List, Dict, Tuple, Generator

import tempfile # No longer needed
import requests
from huggingface_hub import hf_hub_download, snapshot_download
import random # For testing the evaluation, can be removed later
import re # Import regex for parsing

# --- Configuration & Initialization ---
# Set page config first - Must be the first Streamlit command
st.set_page_config(page_title="AcademIQ", layout="wide", page_icon="🎓") # Changed page title

# --- Constants and Configuration ---
# Hugging Face dataset info
HUGGINGFACE_REPO_ID = "brandonyeequon/teacher_faiss"
HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", None)

# Define local file paths (where the app expects files to be, or downloads them to)
# Files will be checked/downloaded into the same directory as the script, or a subdirectory
LOCAL_DATA_DIR = "data" # Subdirectory for JSON files
FAISS_INDEX_PATH = "vectorized_textbooks.faiss" # In the root directory
TEXTBOOK_PASSAGES_PATH = "textbook_passages.pkl" # In the root directory
SCENARIO_MENU_PATH = os.path.join(LOCAL_DATA_DIR, "scenario_menu.json")
SCENARIOS_DATA_PATH = os.path.join(LOCAL_DATA_DIR, "scenarios.json")

# HF file paths - paths within HF dataset (used for downloading)
HF_FAISS_INDEX_PATH = "vectorized_textbooks.faiss"  # Relative path in the repo
HF_TEXTBOOK_PASSAGES_PATH = "textbook_passages.pkl"  # Relative path in the repo

# Embedding and model configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_STUDENT_MODEL = "gpt-4o-mini"
OPENAI_EXPERT_MODEL = "gpt-4o" # Use the more powerful model for evaluation

# --- OpenAI API Key Setup ---
try:
    # Ensure openai library is recent enough for this attribute
    if hasattr(openai, 'api_key'):
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        print("OpenAI API key configured from Streamlit secrets.")
    else:
        # Handle older versions or different client initialization if needed
        # For newer versions (>=1.0.0), client initialization is preferred
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        # You might need to adjust functions below to use 'client' instead of 'openai' directly
        print("OpenAI API key configured using OpenAI client.")
        # Example adjustment: Replace openai.embeddings.create with client.embeddings.create
        # Example adjustment: Replace openai.chat.completions.create with client.chat.completions.create
except KeyError:
    st.error("ERROR: OpenAI API key not found in Streamlit secrets. Please create .streamlit/secrets.toml with your key.", icon="🚨")
    st.stop() # Stop execution if key is missing
except Exception as e:
    st.error(f"Error configuring OpenAI API key: {e}", icon="🚨")
    st.stop()

# Initialize OpenAI client (if using newer library version)
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None # Fallback or handle error if client init fails

# --- Caching Resources ---
@st.cache_resource # Cache the FAISS index
def load_faiss_index():
    print(f"Attempting to load FAISS index from: {FAISS_INDEX_PATH}")
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"Successfully loaded FAISS index. Size: {index.ntotal} vectors, Dimension: {index.d}")
            return index
        except Exception as e:
            st.error(f"Error reading FAISS index file '{FAISS_INDEX_PATH}': {e}. Was it created with the correct model ({OPENAI_EMBEDDING_MODEL})?", icon="🚨")
            st.stop()
    else:
        st.error(f"FAISS index file not found at {FAISS_INDEX_PATH}. Please ensure it exists or can be downloaded.", icon="🚨")
        st.stop()

@st.cache_data # Cache the textbook passages
def load_textbook_passages():
    print(f"Attempting to load textbook passages from: {TEXTBOOK_PASSAGES_PATH}")
    if os.path.exists(TEXTBOOK_PASSAGES_PATH):
        try:
            with open(TEXTBOOK_PASSAGES_PATH, "rb") as f:
                passages = pickle.load(f)
            print(f"Successfully loaded {len(passages)} textbook passages.")
            return passages
        except Exception as e:
            st.error(f"Error reading passages file '{TEXTBOOK_PASSAGES_PATH}': {e}", icon="🚨")
            st.stop()
    else:
        st.error(f"Textbook passages file not found at {TEXTBOOK_PASSAGES_PATH}. Please ensure it exists or can be downloaded.", icon="🚨")
        st.stop()

@st.cache_data # Cache scenario data
def load_scenario_data():
    print("Loading scenario data...")
    try:
        # Ensure data directory exists for JSON files
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        
        if not os.path.exists(SCENARIO_MENU_PATH):
            st.error(f"Error: Scenario menu file not found at '{SCENARIO_MENU_PATH}'. Please ensure 'data/scenario_menu.json' exists.", icon="🚨")
            st.stop()
        if not os.path.exists(SCENARIOS_DATA_PATH):
            st.error(f"Error: Scenarios data file not found at '{SCENARIOS_DATA_PATH}'. Please ensure 'data/scenarios.json' exists.", icon="🚨")
            st.stop()
            
        with open(SCENARIO_MENU_PATH, "r") as f:
            menu_data = json.load(f)
        with open(SCENARIOS_DATA_PATH, "r") as f:
            scenarios_content = json.load(f)
        # Convert scenarios list to dict for easy lookup
        scenarios_dict = {s["scenario_id"]: s for s in scenarios_content}
        print("Successfully loaded scenario menu and data.")
        return menu_data.get("scenario_menu", []), scenarios_dict
    except FileNotFoundError as e:
        st.error(f"Error loading scenario file: {e}", icon="🚨")
        return [], {}
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON in scenario files: {e}. Please check the file format.", icon="🚨")
        return [], {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading scenario data: {e}", icon="🚨")
        return [], {}

def ensure_files_downloaded():
    """Check for local files, download from Hugging Face if missing."""
    files_to_check = [
        {"local_path": FAISS_INDEX_PATH, "hf_path": HF_FAISS_INDEX_PATH, "desc": "FAISS index"},
        {"local_path": TEXTBOOK_PASSAGES_PATH, "hf_path": HF_TEXTBOOK_PASSAGES_PATH, "desc": "textbook passages"},
    ]
    
    # Determine the target directory for downloads (current directory for these files)
    target_directory = "."
    
    for file_info in files_to_check:
        local_path = file_info["local_path"]
        hf_path = file_info["hf_path"]
        desc = file_info["desc"]
        
        if os.path.exists(local_path):
            print(f"Found local {desc} file: {local_path}")
        else:
            st.warning(f"Local {desc} file not found at '{local_path}'. Attempting to download from Hugging Face repo: {HUGGINGFACE_REPO_ID}...", icon="⏳")
            print(f"Downloading {desc} from Hugging Face ({hf_path}) to {target_directory}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=HUGGINGFACE_REPO_ID,
                    filename=hf_path,        # Path within the repo
                    local_dir=target_directory, # Directory to save the file
                    local_dir_use_symlinks=False, # Ensure the actual file is copied
                    token=HUGGINGFACE_TOKEN,
                    repo_type="dataset",
                    cache_dir=None # Avoid using HF cache, download directly
                )
                expected_download_location = os.path.join(target_directory, hf_path)
                
                if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
                     if os.path.exists(downloaded_path):
                        import shutil
                        print(f"  Moving downloaded file from {downloaded_path} to {local_path}")
                        try:
                             shutil.move(downloaded_path, local_path)
                             try:
                                 if os.path.dirname(hf_path):
                                     os.removedirs(os.path.dirname(downloaded_path))
                             except OSError:
                                 pass
                        except Exception as move_err:
                             st.error(f"Failed to move downloaded file to {local_path}: {move_err}", icon="🚨")
                             st.stop()
                     else:
                         if not os.path.exists(local_path):
                             st.error(f"Download completed, but the file is not at the expected location: {local_path}. Downloaded path reported as: {downloaded_path}", icon="🚨")
                             st.stop()
                
                if os.path.exists(local_path):
                     st.success(f"Successfully downloaded {desc} to {local_path}", icon="✅")
                     print(f"  Successfully downloaded {desc} to {local_path}")
                else:
                     st.error(f"Download attempted, but {desc} file still not found at {local_path}.", icon="🚨")
                     st.stop()
                
            except Exception as e:
                st.error(f"Failed to download {desc} from Hugging Face: {e}", icon="🚨")
                st.info(f"Please ensure the file '{hf_path}' exists in the repo '{HUGGINGFACE_REPO_ID}' or place the file manually at '{local_path}'.")
                st.stop()
    
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    if not os.path.exists(SCENARIO_MENU_PATH):
        st.error(f"Essential scenario menu file not found locally at {SCENARIO_MENU_PATH}. Please ensure '{LOCAL_DATA_DIR}' directory exists and contains 'scenario_menu.json'.", icon="🚨")
        st.stop()
    if not os.path.exists(SCENARIOS_DATA_PATH):
        st.error(f"Essential scenarios data file not found locally at {SCENARIOS_DATA_PATH}. Please ensure '{LOCAL_DATA_DIR}' directory exists and contains 'scenarios.json'.", icon="🚨")
        st.stop()
    
    print("All required files checked/downloaded.")

# --- Load Resources ---
ensure_files_downloaded()
index = load_faiss_index()
textbook_passages = load_textbook_passages()
scenario_menu, scenarios_dict = load_scenario_data()

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "expert_chat_history" not in st.session_state:
    st.session_state.expert_chat_history = []
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None
if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False
if "expert_first_message_sent" not in st.session_state:
    st.session_state.expert_first_message_sent = False
if 'scenario_ended' not in st.session_state:
    st.session_state.scenario_ended = False
if 'evaluation_submitted' not in st.session_state: # Kept for potential future use, not strictly needed now
    st.session_state.evaluation_submitted = False
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# --- Core Logic Functions ---
def get_openai_client():
    """Returns an initialized OpenAI client or None."""
    if client:
        return client
    try:
        return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}", icon="🚨")
        return None

def retrieve_textbook_context(query: str, top_k: int = 3) -> list[str]:
    """Retrieves relevant textbook passages using OpenAI embeddings and FAISS."""
    if not query:
        return []
    local_client = get_openai_client()
    if not local_client:
        st.error("OpenAI client not available for context retrieval.", icon="🚨")
        return []
    
    print(f"Retrieving context for query (first 50 chars): '{query[:50]}...'")
    try:
        print(f"  Embedding query using OpenAI model: {OPENAI_EMBEDDING_MODEL}")
        response = local_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = response.data[0].embedding
        embedding_dim = len(query_embedding)
        print(f"  Query embedded successfully. Dimension: {embedding_dim}")
        
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        if index is None:
             st.error("FAISS index is not loaded. Cannot perform search.", icon="🚨")
             return []
        if index.d != embedding_dim:
             st.error(f"Dimension mismatch! FAISS index dimension ({index.d}) != Query embedding dimension ({embedding_dim}). Ensure the index file '{FAISS_INDEX_PATH}' was created using the embedding model '{OPENAI_EMBEDDING_MODEL}'.", icon="🚨")
             return []
             
        print(f"  Searching FAISS index (size: {index.ntotal} vectors, dimension: {index.d})...")
        distances, indices = index.search(query_embedding_np, top_k)
        print(f"  FAISS search complete. Found indices: {indices[0]}")
        
        # Debugging output removed for brevity, was present in previous step
        
        valid_indices = [i for i in indices[0] if 0 <= i < len(textbook_passages)]
        retrieved = [textbook_passages[i] for i in valid_indices]
        
        print(f"  Retrieved {len(retrieved)} valid passages.")
        return retrieved
    except openai.APIError as e:
        st.error(f"OpenAI API Error during query embedding: {e}", icon="⚠️")
        return []
    except AttributeError as e:
        if "'NoneType' object has no attribute 'search'" in str(e):
            st.error("FAISS index is not loaded correctly (is None). Cannot perform search.", icon="🚨")
        elif "'NoneType' object has no attribute 'embeddings'" in str(e):
            st.error("OpenAI client is not initialized correctly (is None). Cannot create embeddings.", icon="🚨")
        else:
            st.error(f"Attribute Error during context retrieval: {e}", icon="⚠️")
        return []
    except Exception as e:
        st.error(f"Error retrieving textbook context: {e}", icon="⚠️")
        import traceback
        traceback.print_exc()
        return []

# --- MODIFIED FOR STREAMING ---
def generate_student_response(user_input: str, chat_history: list[dict], scenario_id: Optional[str] = None) -> Generator[str, None, None]:
    """Generates a student response using OpenAI Chat Completion with streaming."""
    local_client = get_openai_client()
    if not local_client:
        yield "Uh oh, my connection is fuzzy! (OpenAI client unavailable)"
        return # Stop generation
    
    scenario_context = ""
    if scenario_id and scenario_id in scenarios_dict:
        scenario = scenarios_dict[scenario_id]
        scenario_type = scenario.get("type", "")
        student_name = scenario.get("student_name", "a student")
        if student_name != "Whole Class":
            scenario_context = f"Your name is {student_name}. "
        if "student_details" in scenario:
            scenario_context += f"Your personality: {scenario['student_details']} "
        if scenario_type == "Student Emotion" and "emotion" in scenario:
            scenario_context += f"You are feeling {scenario['emotion'].lower()}. "
        if "classroom_situation" in scenario:
            scenario_context += f"Current situation: {scenario['classroom_situation']} "
        if "description" in scenario:
            scenario_context += f"This interaction is about: {scenario['description']}"
    
    system_prompt_content = f"You are a 2nd grade student (7-8 years old). Respond simply, sometimes distractedly, in a childlike manner, using simple vocabulary. Keep responses short (1-3 sentences). {scenario_context} You always respond as the second grader, never break character, and never act as the teacher. Address the teacher naturally based on their input."
    system_prompt = {"role": "system", "content": system_prompt_content}
    
    messages = [system_prompt]
    
    # Add previous messages from history
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content")
        if role and content:
            api_role = "assistant" if role == "assistant" else "user"
            messages.append({"role": api_role, "content": content})
    
    # Add the latest user input (which triggered this call) - NO, history already includes it before calling
    messages.append({"role": "user", "content": user_input}) # This would duplicate the last user message
    try:
        stream = local_client.chat.completions.create(
            model=OPENAI_STUDENT_MODEL,
            messages=messages, # Pass the history as constructed
            temperature=0.7,
            max_tokens=600,
            stream=True, # Enable streaming
        )
        # Yield content chunks from the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    except openai.APIError as e:
        st.error(f"OpenAI API Error (Student Response): {e}", icon="🚨")
        yield "Uh oh, my brain got fuzzy!"
    except Exception as e:
        st.error(f"Error generating student response: {e}", icon="🚨")
        yield "Something went wrong with my thinking."

# --- MODIFIED FOR STREAMING AND CONVERSATION HISTORY ---
def generate_expert_advice(question: str, conversation_history: list[dict], expert_chat_history: list[dict], scenario_id: Optional[str] = None) -> Generator[str, None, None]:
    """Generates expert teacher advice using OpenAI Chat Completion with streaming, RAG with OpenAI embeddings."""
    local_client = get_openai_client()
    if not local_client:
        yield "There was an issue connecting with the expert advisor AI (OpenAI client unavailable)."
        return

    transcript = "\n".join(
        f"{'Teacher (User)' if m.get('role') == 'user' else 'Student (Assistant)'}: {m.get('content', '')}"
        for m in conversation_history # Using the student conversation history for context
    )

    scenario_context = ""
    retrieval_query = question
    if scenario_id and scenario_id in scenarios_dict:
        scenario = scenarios_dict[scenario_id]
        scenario_context = f"Scenario Context:\nTitle: {scenario.get('title', 'N/A')}\nDescription: {scenario.get('description', 'N/A')}\n"
        retrieval_query += f" scenario: {scenario.get('title', '')} {scenario.get('description', '')}"
        if "student_name" in scenario and scenario["student_name"] != "Whole Class":
            scenario_context += f"Student: {scenario['student_name']}\n"
        if "student_details" in scenario:
            scenario_context += f"Student Profile: {scenario['student_details']}\n"
            retrieval_query += f" student profile: {scenario['student_details']}"
        if "classroom_situation" in scenario:
            scenario_context += f"\nClassroom Situation:\n{scenario['classroom_situation']}\n"
            retrieval_query += f" situation: {scenario['classroom_situation']}"
        if "teacher_objective" in scenario:
            scenario_context += f"\nTeaching Objective:\n{scenario['teacher_objective']}\n"
            retrieval_query += f" objective: {scenario['teacher_objective']}"
        scenario_context += "\n---\n"

    passages = retrieve_textbook_context(retrieval_query)
    passages_text = "\n".join(f"- {p}" for p in passages) if passages else "No specific teaching principles automatically retrieved for this query."

    system_prompt_content = "You are an expert teacher trainer AI specializing in elementary education (specifically 2nd grade). Provide specific, actionable, and concise advice based on educational best practices and the provided context (scenario, conversation transcript, retrieved teaching principles). Focus on practical strategies the teacher can implement next in this specific interaction. If a 'Teaching Objective' is provided, ensure your advice aligns with achieving it. Use clear, direct language suitable for a busy teacher. Directly reference relevant retrieved principles from UVU textbooks when applicable."
    
    # Construct messages for the API call - Now including previous conversation history
    messages = [{"role": "system", "content": system_prompt_content}]
    
    # Add the previous expert chat history for continuity
    for msg in expert_chat_history:
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "content": msg["content"]})
    
    # Add context as a note to the latest user message
    context_note = f"{scenario_context}\nConversation Transcript So Far:\n{transcript}\n\nRetrieved Teaching Principles:\n{passages_text}\n\n"
    
    # Add the current question with context
    messages.append({"role": "user", "content": context_note + question})

    try:
        stream = local_client.chat.completions.create(
            model=OPENAI_EXPERT_MODEL,
            messages=messages,
            temperature=0.9,
            max_tokens=4000,
            stream=True # Enable streaming
        )
        # Yield content chunks from the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except openai.APIError as e:
        st.error(f"OpenAI API Error (Expert Advice): {e}", icon="🚨")
        yield "There was an issue connecting with the expert advisor AI." # Yield error message
    except Exception as e:
        st.error(f"Error generating expert advice: {e}", icon="🚨")
        import traceback
        traceback.print_exc()
        yield "An unexpected error occurred while generating advice." # Yield error message

# --- IMPROVED EVALUATION FUNCTION WITH BETTER PARSING AND CONTEXT HANDLING ---
def generate_ai_assessment(chat_history: List[Dict[str, str]], scenario: Optional[Dict] = None) -> Tuple[str, str, str, str]:
    """Generates a performance evaluation using the OpenAI API (streaming internally) based on chat history and scenario."""
    local_client = get_openai_client()
    if not local_client:
        return "Evaluation Unavailable", "Could not connect to the assessment AI.", "-", "-"
    
    if not chat_history:
        return "N/A", "No conversation took place.", "-", "-"
    
    # Format the conversation transcript with clear role indicators
    transcript = ""
    for i, m in enumerate(chat_history):
        role = "Teacher" if m.get("role") == "user" else "Student"
        content = m.get("content", "").strip()
        if content:  # Only add if there's actual content
            transcript += f"{role}: {content}\n\n"
    
    # Build a more detailed scenario context
    scenario_context = "No specific scenario context provided."
    if scenario:
        scenario_context = f"Scenario Title: {scenario.get('title', 'N/A')}\n\n"
        if "description" in scenario and scenario["description"]:
            scenario_context += f"Description: {scenario['description']}\n\n"
        if "student_name" in scenario:
            scenario_context += f"Student: {scenario['student_name']}"
            if scenario["student_name"] != "Whole Class" and "student_details" in scenario:
                scenario_context += f" (Profile: {scenario['student_details']})"
            scenario_context += "\n\n"
        if "teacher_objective" in scenario and scenario["teacher_objective"]:
            scenario_context += f"Teacher's Objective: {scenario['teacher_objective']}\n\n"
        if "classroom_situation" in scenario and scenario["classroom_situation"]:
            scenario_context += f"Classroom Situation: {scenario['classroom_situation']}\n\n"
    
    # Use a very explicit system prompt with formatting requirements
    system_prompt_content = """You are an expert educational assessor specializing in evaluating teacher-student interactions in elementary school (specifically 2nd grade, age 7-8).

Your task is to carefully analyze the ENTIRE conversation transcript between a teacher and a simulated student. Pay close attention to all exchanges and reference specific examples from the conversation in your evaluation.

IMPORTANT: You MUST provide your evaluation in EXACTLY the following format using these EXACT headings:

**Score:** [number] / 10

**Rationale:**
[Your explanation here]

**Strengths:**
[List specific strengths with examples from the transcript]

**Areas for Improvement:**
[List specific actionable suggestions with examples from the transcript]

Note: Please maintain this EXACT format with these EXACT headings to ensure your evaluation can be properly processed.
"""
    system_prompt = {"role": "system", "content": system_prompt_content}
    
    # Clearer user prompt with explicit instructions to reference the conversation
    user_input_content = f"""SCENARIO CONTEXT:
{scenario_context}

CONVERSATION TRANSCRIPT:
{transcript}

Please evaluate the teacher's effectiveness based on the SPECIFIC CONVERSATION above. 
Look at the actual exchanges between teacher and student, and reference specific examples from this transcript.
Evaluate based on clarity, engagement, questioning techniques, responsiveness, patience, and alignment with the teaching objective.

Remember to format your response EXACTLY as specified with the headings: Score, Rationale, Strengths, and Areas for Improvement.
"""
    
    user_prompt = {"role": "user", "content": user_input_content}
    
    messages = [system_prompt, user_prompt]
    
    print("Generating AI assessment (streaming internally)...")
    print(f"Sending transcript with {len(chat_history)} exchanges and {len(transcript)} characters")
    
    raw_evaluation = ""
    try:
        stream = local_client.chat.completions.create(
            model=OPENAI_EXPERT_MODEL,
            messages=messages,
            temperature=0.9,  # Lower temperature for more consistent formatting
            max_tokens=8000,   # Increased token limit for more detailed evaluation
            stream=True       # Use stream=True
        )
        
        # Collect the streamed response internally
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                raw_evaluation += chunk.choices[0].delta.content
        
        print(f"Raw evaluation received: {len(raw_evaluation)} characters")
        
        # Improved regex patterns with better handling of variations
        # These patterns are more flexible in matching the headings and content
        score_pattern = r"\*\*Score:\*\*\s*(\d+(\.\d+)?)\s*\/\s*10"
        rationale_pattern = r"\*\*Rationale:\*\*([\s\S]*?)(?=\*\*Strengths:\*\*|\Z)"
        strengths_pattern = r"\*\*Strengths:\*\*([\s\S]*?)(?=\*\*Areas for Improvement:\*\*|\Z)"
        improvement_pattern = r"\*\*Areas for Improvement:\*\*([\s\S]*)"
        
        score_match = re.search(score_pattern, raw_evaluation, re.MULTILINE)
        rationale_match = re.search(rationale_pattern, raw_evaluation, re.MULTILINE)
        strengths_match = re.search(strengths_pattern, raw_evaluation, re.MULTILINE)
        improvement_match = re.search(improvement_pattern, raw_evaluation, re.MULTILINE)
        
        # Better handling of parsing results
        if score_match:
            score_str = f"{score_match.group(1).strip()} / 10"
        else:
            # Fallback parsing for alternative formats
            alt_score_match = re.search(r"(\d+(\.\d+)?)\s*\/\s*10", raw_evaluation)
            score_str = f"{alt_score_match.group(1).strip()} / 10" if alt_score_match else "N/A"
        
        rationale = rationale_match.group(1).strip() if rationale_match else ""
        strengths = strengths_match.group(1).strip() if strengths_match else ""
        improvement_areas = improvement_match.group(1).strip() if improvement_match else ""
        
        # If any section is missing, try to recover by displaying the raw response
        if not rationale or not strengths or not improvement_areas or score_str == "N/A":
            print("Warning: Could not parse one or more sections of the evaluation")
            
            # Attempt to clean up the raw response for display
            cleaned_response = raw_evaluation.strip()
            
            # Display appropriate warning
            if not raw_evaluation or len(raw_evaluation) < 50:
                st.warning("Evaluation returned empty or incomplete response.", icon="⚠️")
                return "N/A", "The evaluation system returned an incomplete response. Please try again.", "-", "-"
            else:
                st.warning("Could not parse the evaluation format. Displaying raw response.", icon="⚠️")
                return "N/A", f"Raw Evaluation:\n\n{cleaned_response}", "-", "-"
        
        return score_str, rationale, strengths, improvement_areas
        
    except openai.APIError as e:
        st.error(f"OpenAI API Error (Assessment): {e}", icon="🚨")
        return "Evaluation Error", f"API Error: {e}", "-", "-"
    except Exception as e:
        st.error(f"Error generating assessment: {e}", icon="🚨")
        import traceback
        traceback.print_exc()
        return "Evaluation Error", f"Unexpected Error: {e}", "-", "-"

# --- Streamlit UI ---
st.markdown(
"""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem; /* Reduced bottom padding to make space */
}
.stSidebar {
    padding: 15px;
}
.stSidebar h1 {
    font-size: 1.75rem;
    margin-bottom: 1rem;
    color: var(--sidebar-text-color);
}
div[data-testid="stChatMessage"] {
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
    max-width: 100%;
}
.stExpander {
    border: 1px solid #ddd;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.stExpander header {
    font-weight: bold;
}
div.stSpinner > div > div { /* More specific selector for spinner animation */
    /* display: none !important; */ /* Keep spinner for assessment */
}
.processing-active [data-testid="stChatInput"] {
    opacity: 0.6;
    pointer-events: none;
}
/* Ensure sidebar itself doesn't push content down unnecessarily */
section[data-testid="stSidebar"] > div:first-child {
    padding-bottom: 60px; /* Add padding at the bottom of sidebar content area to avoid overlap with input */
}
</style>
""",
unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
    """
    <style>
    /* Base sidebar styling */
    .stSidebar {
        padding: 10px;
        border-radius: 4px;
    }
    
    /* Dark theme - white text */
    body[data-theme="dark"] .stSidebarHeader,
    body[data-theme="dark"] .stSidebar .stTextInput input,
    body[data-theme="dark"] .stSidebar .stButton,
    body[data-theme="dark"] .stSidebar .stMarkdown,
    body[data-theme="dark"] .st-emotion-cache-1mw54nq.egexzqm0,
    body[data-theme="dark"] .st-emotion-cache-fsammq.egexzqm0 {
        color: white !important;
    }
    
    /* Light theme - dark text */
    body[data-theme="light"] .stSidebarHeader,
    body[data-theme="light"] .stSidebar .stTextInput input,
    body[data-theme="light"] .stSidebar .stButton,
    body[data-theme="light"] .stSidebar .stMarkdown,
    body[data-theme="light"] .st-emotion-cache-1mw54nq.egexzqm0,
    body[data-theme="light"] .st-emotion-cache-fsammq.egexzqm0 {
        color: #333333 !important;
    }
    </style>
    """, unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown("<h1 style='text-align: left; '>Expert Teacher</h1>", unsafe_allow_html=True)
    
    # Add the note about UVU textbooks here with green background
    st.markdown("""
    <div class="info-box">
    📚 <span class="info-text">Expert teacher references textbooks curated for UVU students</span>
    </div>
    <style>
    div.info-box { background-color: #2E8B57 !important; padding: 12px 18px; border-radius: 8px; font-weight: bold; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .info-text { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.current_scenario and not st.session_state.scenario_ended:
        # --- MODIFIED: Container for messages WITHOUT fixed height ---
        # Let the sidebar scroll naturally if content overflows
        expert_chat_container = st.container(border=False)
        with expert_chat_container:
            # Display existing history
            for msg in st.session_state.expert_chat_history:
                with st.chat_message(name=msg["role"]):
                    st.markdown(msg["content"])
            
            # New user messages and streaming responses will also be added within this container context below

        # --- Input for new expert question - Placed *after* the container definition ---
        # This allows the input to stay at the bottom of the sidebar's content flow
        expert_prompt = st.chat_input(
            "Ask the expert a question...",
            key="expert_sidebar_input",
            disabled=st.session_state.scenario_ended
        )
        
        if expert_prompt:
            # Add user message to history FIRST
            st.session_state.expert_chat_history.append({"role": "user", "content": expert_prompt})
            
            # Display user prompt immediately *within the container*
            with expert_chat_container:
                with st.chat_message("user"):
                    st.markdown(expert_prompt)
            
            # Generate and stream expert response *within the container*
            with expert_chat_container:
                 with st.chat_message("assistant"):
                     # st.write_stream adds content progressively here
                     full_expert_response = st.write_stream(generate_expert_advice(
                         expert_prompt,
                         st.session_state.chat_history, # Pass student chat history for context
                         st.session_state.expert_chat_history, # Pass expert chat history for continuity
                         st.session_state.current_scenario["scenario_id"]
                     ))
                     # Add the complete response to history *after* streaming is done
                     st.session_state.expert_chat_history.append({"role": "assistant", "content": full_expert_response})
                     st.session_state.expert_first_message_sent = True
                     # No explicit rerun here, state update handles refresh on next interaction
    elif st.session_state.scenario_ended:
        st.info("Expert advice is paused during evaluation.")
        # Show disabled input at the bottom
        st.chat_input("Ask the expert a question...", key="expert_sidebar_input_disabled", disabled=True)
    else:
        # Initial state before scenario selection
        st.markdown("""
        <div class="expert-help-box">
        💡 <span class="expert-help-text">Ask for advice here once a teaching scenario has been selected from the main chat</span>
        </div>
        <style>
        /* Styles moved inside for locality if preferred, or keep global */
        div.expert-help-box { background-color: #2E8B57 !important; padding: 12px 18px; border-radius: 8px; font-weight: bold; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .expert-help-text { color: white !important; }
        </style>
        """, unsafe_allow_html=True)
        # Show disabled input at the bottom
        st.chat_input("Ask the expert a question...", key="expert_sidebar_input_disabled_initial", disabled=True)

st.markdown(
"""
<style>
.stSelectbox > div > div > div { font-size: 20px }
.stSelectbox > div > div { height: 50px }
</style>
""",
unsafe_allow_html=True
)

# --- Scenario Selection Area ---
if not st.session_state.current_scenario and not st.session_state.scenario_ended:
    st.markdown( # Changed logo text and structure
    """
    <style>
    .academIQ-logo {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
    }
    .academIQ-logo .iq-green { /* Changed class name */
        color: #2E8B57; /* Green color */
    }
    </style>
    <div class="academIQ-logo">Academ<span class="iq-green">IQ</span></div>
    """,
    unsafe_allow_html=True
    )
    st.write("""
    Transform the way you prepare for the classroom with our AI-powered teaching assistant!
    This interactive tool helps elementary school teachers refine their skills by simulating real classroom interactions.
    The AI behaves like a real second-grader, responding dynamically to your teaching style, questions, and guidance.
    """)
    st.write("")
    
    def handle_scenario_change():
        selected_title = st.session_state.scenario_selector
        if selected_title != "Select a scenario...":
            scenario_id_found = None
            for menu_item in scenario_menu:
                if menu_item.get("title") == selected_title:
                    scenario_id_found = menu_item.get("scenario_id")
                    break
            if scenario_id_found and scenario_id_found in scenarios_dict:
                st.session_state.current_scenario = scenarios_dict[scenario_id_found]
                # Reset relevant states
                st.session_state.chat_history = []
                st.session_state.expert_chat_history = []
                st.session_state.first_message_sent = False
                st.session_state.expert_first_message_sent = False
                st.session_state.scenario_ended = False
                st.session_state.evaluation_submitted = False
                st.session_state.evaluation_results = None
                print(f"Scenario selected: {selected_title} (ID: {scenario_id_found})")
            else:
                # Handle error case: Reset if scenario details not found
                st.session_state.current_scenario = None
                st.session_state.chat_history = []
                st.session_state.expert_chat_history = []
                st.session_state.first_message_sent = False
                st.session_state.expert_first_message_sent = False
                st.session_state.scenario_ended = False
                st.session_state.evaluation_submitted = False
                st.session_state.evaluation_results = None
                print(f"Warning: Scenario details not found for title '{selected_title}' or ID '{scenario_id_found}'. Resetting.")
                st.warning(f"Could not load details for scenario '{selected_title}'. Please check data files or select another scenario.", icon="⚠️")
        else:
            # Reset if "Select a scenario..." is chosen
            st.session_state.current_scenario = None
            st.session_state.chat_history = []
            st.session_state.expert_chat_history = []
            st.session_state.first_message_sent = False
            st.session_state.expert_first_message_sent = False
            st.session_state.scenario_ended = False
            st.session_state.evaluation_submitted = False
            st.session_state.evaluation_results = None
            print("Scenario deselected.")
        # No explicit rerun needed, state changes handled by Streamlit
    
    scenario_options = ["Select a scenario..."] + sorted([s.get('title', f"Untitled Scenario ID: {s.get('scenario_id', 'Unknown')}") for s in scenario_menu])
    st.selectbox(
        "",
        scenario_options,
        index=0,
        key="scenario_selector",
        on_change=handle_scenario_change, # Callback handles state changes
        help="Select a classroom situation to practice."
    )
    st.write("")
    st.chat_input("Your message to the student...", key="student_chat_input_disabled_initial", disabled=True) # Disable input initially

# --- Scenario Active Area ---
elif st.session_state.current_scenario and not st.session_state.scenario_ended:
    st.markdown( # Changed logo text and structure
    """
    <style>
    .academIQ-logo {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        text-align: center;
        font-weight: bold;
        margin: 20px 0 1.5rem 0;
    }
    .academIQ-logo .iq-green { color: #2E8B57; } /* Changed class name */
    </style>
    <div class="academIQ-logo">Academ<span class="iq-green">IQ</span></div>
    """,
    unsafe_allow_html=True
    )
    with st.expander("Current Scenario Details", expanded=True):
        scenario = st.session_state.current_scenario
        st.subheader(f"{scenario.get('title', 'Unnamed Scenario')}")
        info_cols = st.columns(2)
        with info_cols[0]:
            if "student_name" in scenario:
                st.markdown(f"Student: {scenario['student_name']}")
                if scenario["student_name"] != "Whole Class" and "student_details" in scenario:
                    st.caption(f"Profile: {scenario['student_details']}")
                elif scenario["student_name"] == "Whole Class":
                    st.caption("Interaction involves the whole class.")
            if "type" in scenario:
                scenario_type = scenario["type"]
                if scenario_type == "Student Emotion" and "emotion" in scenario:
                    st.markdown(f"Student Emotion: {scenario['emotion']}")
        with info_cols[1]:
            if "teacher_objective" in scenario:
                st.markdown(f"Your Objective:")
                st.markdown(f"{scenario['teacher_objective']}")
            if "classroom_situation" in scenario:
                st.markdown("Classroom Situation:")
                st.markdown(f"{scenario['classroom_situation']}")
    st.write("")
    # Display chat history
    # Use a container, but allow its height to be flexible (no fixed height)
    chat_container = st.container(border=False)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        # New messages will be added here dynamically by the input handling below
    
    # Handle student chat input and response streaming
    # Place chat input after the container where messages are displayed
    prompt = st.chat_input("Your message to the student...", key="student_chat_input_widget")
    
    if prompt:
        # Add user message to history and display it immediately in the container
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Generate and stream student response within the container
        with chat_container:
            with st.chat_message("assistant"):
                # st.write_stream handles displaying the generator output
                # It also returns the full concatenated string when finished
                full_student_response = st.write_stream(generate_student_response(
                    prompt, # Pass only the new prompt here, function adds history
                    st.session_state.chat_history, # Pass the full history
                    st.session_state.current_scenario["scenario_id"]
                ))
                # Add the complete student response to history *after* streaming
                st.session_state.chat_history.append({"role": "assistant", "content": full_student_response})
                st.session_state.first_message_sent = True
                # No explicit rerun needed
    
    # End Scenario Button - Place below chat elements
    cols = st.columns([3, 1]) # Or adjust layout as needed
    with cols[1]: # Place button in the right column
        def end_scenario_and_evaluate():
            st.session_state.scenario_ended = True
            st.session_state.evaluation_submitted = False # Reset evaluation state
            st.session_state.evaluation_results = None
            print("Scenario ended by user. Proceeding to evaluation.")
            # No st.rerun() needed here, button click + state change triggers it
        
        st.button(
            "End Scenario and Get Feedback",
            key="end_chat_button",
            on_click=end_scenario_and_evaluate, # Define callback
            use_container_width=True,
            disabled=not st.session_state.first_message_sent # Disable if no interaction yet
        )

# --- Evaluation Screen ---
elif st.session_state.scenario_ended:
    st.title("Scenario Evaluation")
    
    # Generate evaluation if not already done
    if st.session_state.evaluation_results is None:
        with st.spinner("Evaluating your interaction... Please wait."): # Added message
            # Call the assessment function (which streams internally)
            score_str, rationale, strengths, improvement_areas = generate_ai_assessment(
                st.session_state.chat_history,
                st.session_state.current_scenario
            )
            # Store the results
            st.session_state.evaluation_results = {
                "score": score_str,
                "rationale": rationale,
                "strengths": strengths,
                "improvement": improvement_areas
            }
            # No rerun needed here, spinner context manager handles it
    
    # Display evaluation results
    results = st.session_state.evaluation_results
    if results:
        st.subheader(f"Score: {results['score']}")
        with st.expander("Rationale", expanded=(results['score'] == "N/A")): # Expand rationale if score parsing failed
            st.markdown(results['rationale'])
        st.subheader("Strengths")
        st.markdown(results['strengths'])
        st.subheader("Areas for Improvement")
        st.markdown(results['improvement'])
    else:
        # This case should ideally not happen if the spinner logic above works
        st.warning("Evaluation is being processed or encountered an error.")
    
    if st.button("Select New Scenario"):
        # Reset session state for a new scenario selection
        st.session_state.current_scenario = None
        st.session_state.first_message_sent = False
        st.session_state.expert_first_message_sent = False
        st.session_state.scenario_ended = False
        st.session_state.chat_history = []
        st.session_state.expert_chat_history = []
        st.session_state.evaluation_results = None
        st.session_state.evaluation_submitted = False
        st.rerun() # Rerun to go back to the scenario selection screen

# --- Footer ---
# Footer HTML remains unchanged
footer_html = """
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 5px 175px;
    box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    z-index: 999;
    display: flex;
    justify-content: flex-end;
    color: var(--st-text-color);
}
body[data-theme="light"] .footer { background-color: #ffffff; }
body[data-theme="dark"] .footer { background-color: #1e1e1e; }
.footer details summary { list-style: none; }
.footer details summary::-webkit-details-marker { display: none; }
.footer details summary { cursor: pointer; font-weight: bold; padding: 3px; color: inherit; font-size: 14px; }
.footer details[open] summary { content: "Back"; }
.footer details[open] { position: absolute; bottom: 50px; right: 20px; border-radius: 8px; padding: 10px; width: 300px; box-shadow: 0 0 10px rgba(0,0,0,0.1); background-color: #f8f8f8; color: #333; }
body[data-theme="light"] .footer details[open] { background-color: #ffffff; color: #333; }
body[data-theme="dark"] .footer details[open] { background-color: #333; color: #f0f2f6; }
.footer details[open] p { text-align: left; margin: 5px 0; color: inherit; }
</style>
<div class="footer">
    <details>
        <summary>❓ Help</summary>
        <p>👩‍🏫 <b>Want to start the chat?</b> Pick a scenario from the "Select a scenario..." dropdown and begin chatting with the student.</p>
        <p>💡 <b>Need expert advice?</b> The Teacher Expert panel on the left offers real-time strategies.</p>
        <p>📈 <b>Get personalized feedback!</b> Your chats are evaluated to improve your teaching techniques. Click "End Scenario and Get Feedback" for the assessment.</p>
        <p>💬 <b>Want to start a new chat?</b> After getting feedback, click the "Select New Scenario" button.</p>
        <p>⚙️ <b>Want to change the look of the page?</b> Click the three dots in the top right corner then "Settings".</p>
    </details>
</div>
<script>
    document.querySelectorAll('.footer details').forEach((details) => {
        details.addEventListener('toggle', () => {
            const summary = details.querySelector('summary');
            summary.innerHTML = details.open ? "⬅️ Back" : "❓ Help";
        });
    });
</script>
"""
# Conditional rendering of footer based on state
if not st.session_state.scenario_ended: # Show footer during selection and active scenario
    st.markdown(footer_html, unsafe_allow_html=True)