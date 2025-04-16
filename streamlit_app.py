# streamlit_app.py
import streamlit as st
import faiss
import numpy as np
import pickle
import json
import os
import openai # Use OpenAI for chat and embeddings
from typing import Optional, List
# import tempfile # No longer needed
import requests
from huggingface_hub import hf_hub_download, snapshot_download
import random # For testing the evaluation, can be removed later

# --- Configuration & Initialization ---

# Set page config first - Must be the first Streamlit command
st.set_page_config(page_title="AcademiQ AI", layout="wide", page_icon="🎓")

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
OPENAI_EXPERT_MODEL = "gpt-4o"

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
    st.error("ERROR: OpenAI API key not found in Streamlit secrets. Please create `.streamlit/secrets.toml` with your key.", icon="🚨")
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
            # Basic dimension check (can be refined if needed)
            # Note: Correct dimension for text-embedding-3-small is 1536
            # expected_dim = 1536 # Or derive dynamically if possible
            # if index.d != expected_dim:
            #     st.warning(f"Warning: Loaded FAISS index dimension ({index.d}) might not match expected embedding dimension ({expected_dim} for {OPENAI_EMBEDDING_MODEL}). Ensure index was created with the correct model.", icon="⚠️")
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
             # Attempt to continue without scenarios if absolutely necessary, or stop
             # return [], {} # Option 1: Continue without scenarios
             st.stop() # Option 2: Stop execution
        if not os.path.exists(SCENARIOS_DATA_PATH):
            st.error(f"Error: Scenarios data file not found at '{SCENARIOS_DATA_PATH}'. Please ensure 'data/scenarios.json' exists.", icon="🚨")
            # return [], {} # Option 1: Continue without scenarios
            st.stop() # Option 2: Stop execution

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
                # hf_hub_download might place it in a subdirectory structure based on the filename
                # We need to ensure it's exactly at local_path
                expected_download_location = os.path.join(target_directory, hf_path)

                # Check if the downloaded file is where we expect it
                if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
                     # This case might happen if hf_path includes subdirs, e.g. "data/file.pkl"
                     # If local_path is just "file.pkl" and target_dir is "."
                     # hf_hub_download might create "./data/file.pkl"
                     # We need to move it to the expected location "."
                     if os.path.exists(downloaded_path):
                        import shutil
                        print(f"  Moving downloaded file from {downloaded_path} to {local_path}")
                        try:
                             shutil.move(downloaded_path, local_path)
                             # Clean up potential empty directories left by hf_hub_download
                             try:
                                 # Only remove if the original download path had subdirs relative to target_dir
                                 if os.path.dirname(hf_path):
                                     os.removedirs(os.path.dirname(downloaded_path))
                             except OSError:
                                 pass # Directory not empty or doesn't exist, ignore
                        except Exception as move_err:
                             st.error(f"Failed to move downloaded file to {local_path}: {move_err}", icon="🚨")
                             st.stop()
                     else:
                         # This check handles cases where the download path reported doesn't match
                         # but the file might already be at the target location due to library logic.
                         if not os.path.exists(local_path):
                             st.error(f"Download completed, but the file is not at the expected location: {local_path}. Downloaded path reported as: {downloaded_path}", icon="🚨")
                             st.stop()


                # Verify again after potential move
                if os.path.exists(local_path):
                     st.success(f"Successfully downloaded {desc} to {local_path}", icon="✅")
                     print(f"  Successfully downloaded {desc} to {local_path}")
                else:
                     # If after all attempts the file isn't there, stop.
                     st.error(f"Download attempted, but {desc} file still not found at {local_path}.", icon="🚨")
                     st.stop()


            except Exception as e:
                st.error(f"Failed to download {desc} from Hugging Face: {e}", icon="🚨")
                st.info(f"Please ensure the file '{hf_path}' exists in the repo '{HUGGINGFACE_REPO_ID}' or place the file manually at '{local_path}'.")
                st.stop()

    # Check for essential JSON files (assumed to be local, not downloaded)
    # Ensure the data directory exists first
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    if not os.path.exists(SCENARIO_MENU_PATH):
        st.error(f"Essential scenario menu file not found locally at {SCENARIO_MENU_PATH}. Please ensure '{LOCAL_DATA_DIR}' directory exists and contains 'scenario_menu.json'.", icon="🚨")
        st.stop()
    if not os.path.exists(SCENARIOS_DATA_PATH):
        st.error(f"Essential scenarios data file not found locally at {SCENARIOS_DATA_PATH}. Please ensure '{LOCAL_DATA_DIR}' directory exists and contains 'scenarios.json'.", icon="🚨")
        st.stop()

    print("All required files checked/downloaded.")


# --- Load Resources ---
# Ensure all required files are in place before proceeding
ensure_files_downloaded()
# Now load them using the cached functions
index = load_faiss_index()
textbook_passages = load_textbook_passages()
scenario_menu, scenarios_dict = load_scenario_data()

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # For student-teacher chat
if "expert_chat_history" not in st.session_state:
    st.session_state.expert_chat_history = [] # For expert advice chat
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None
if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False
if "expert_first_message_sent" not in st.session_state:
    st.session_state.expert_first_message_sent = False

# --- Core Logic Functions ---

def get_openai_client():
    """Returns an initialized OpenAI client or None."""
    # Use the globally initialized client if available
    if client:
        return client
    # Otherwise, try to initialize again (should not happen if initial setup worked)
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
        return [] # Client initialization failed

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

        # Ensure index is loaded and dimensions match before searching
        if index is None:
             st.error("FAISS index is not loaded. Cannot perform search.", icon="🚨")
             return []
        if index.d != embedding_dim:
             st.error(f"Dimension mismatch! FAISS index dimension ({index.d}) != Query embedding dimension ({embedding_dim}). Ensure the index file '{FAISS_INDEX_PATH}' was created using the embedding model '{OPENAI_EMBEDDING_MODEL}'.", icon="🚨")
             # You might want to log more details here or try to re-initialize
             return []

        print(f"  Searching FAISS index (size: {index.ntotal} vectors, dimension: {index.d})...")
        distances, indices = index.search(query_embedding_np, top_k)
        print(f"  FAISS search complete. Found indices: {indices[0]}")

        # Debug output: Add this section to see the passages
        print("### Debug: Retrieved Passages")
        print(f"Query: {query}")
        print(f"Top {top_k} relevant passages:")

        valid_indices = [i for i in indices[0] if 0 <= i < len(textbook_passages)]
        retrieved = [textbook_passages[i] for i in valid_indices]

        # Display each passage with its score and index
        for i, passage_idx in enumerate(valid_indices):
            print(f"**Passage {i+1}** (Index: {passage_idx}, Distance: {distances[0][i]:.4f})")
            # Print only the first 100 chars to keep logs concise
            passage_preview = textbook_passages[passage_idx][:100] + "..." if len(textbook_passages[passage_idx]) > 100 else textbook_passages[passage_idx]
            print(f"```\n{passage_preview}\n```")
            print("---")

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
        traceback.print_exc() # Print full traceback for debugging
        return []


def generate_student_response(user_input: str, chat_history: list[dict], scenario_id: Optional[str] = None) -> str:
    """Generates a student response using OpenAI Chat Completion."""
    local_client = get_openai_client()
    if not local_client:
        return "Uh oh, my connection is fuzzy! (OpenAI client unavailable)"

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
    for msg in chat_history:
         role = msg.get("role")
         content = msg.get("content")
         if role and content:
            # Use 'assistant' for the student's role in the API call
            api_role = "assistant" if role == "assistant" else "user"
            messages.append({"role": api_role, "content": content})

    messages.append({"role": "user", "content": user_input}) # Teacher's latest message

    try:
        response = local_client.chat.completions.create(
            model=OPENAI_STUDENT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=60
        )
        reply = response.choices[0].message.content.strip()
        return reply if reply else "Hmm, I don't know."
    except openai.APIError as e:
        st.error(f"OpenAI API Error (Student Response): {e}", icon="🚨")
        return "Uh oh, my brain got fuzzy!"
    except Exception as e:
        st.error(f"Error generating student response: {e}", icon="🚨")
        return "Something went wrong with my thinking."


def generate_expert_advice(question: str, conversation_history: list[dict], scenario_id: Optional[str] = None) -> str:
    """Generates expert teacher advice using OpenAI Chat Completion, RAG with OpenAI embeddings."""
    local_client = get_openai_client()
    if not local_client:
        return "There was an issue connecting with the expert advisor AI (OpenAI client unavailable)."

    # Format transcript for the expert model
    transcript = "\n".join(
        # Label turns clearly for the expert
        f"{'Teacher (User)' if m.get('role') == 'user' else 'Student (Assistant)'}: {m.get('content', '')}"
        for m in conversation_history
    )

    scenario_context = ""
    retrieval_query = question # Start retrieval query with the teacher's specific question
    if scenario_id and scenario_id in scenarios_dict:
        scenario = scenarios_dict[scenario_id]
        scenario_context = f"**Scenario Context:**\nTitle: {scenario.get('title', 'N/A')}\nDescription: {scenario.get('description', 'N/A')}\n"
        # Add more scenario details to the retrieval query to get relevant passages
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
        scenario_context += "\n---\n" # Separator

    # Retrieve context based on the combined query
    passages = retrieve_textbook_context(retrieval_query)
    passages_text = "\n".join(f"- {p}" for p in passages) if passages else "No specific teaching principles automatically retrieved for this query."

    # Construct the prompt for the expert model
    system_prompt_content = "You are an expert teacher trainer AI specializing in elementary education (specifically 2nd grade). Provide specific, actionable, and concise advice based on educational best practices and the provided context (scenario, conversation transcript, retrieved teaching principles). Focus on practical strategies the teacher can implement *next* in *this specific interaction*. If a 'Teaching Objective' is provided, ensure your advice aligns with achieving it. Use clear, direct language suitable for a busy teacher. Directly reference relevant retrieved principles if applicable."
    system_prompt = {"role": "system", "content": system_prompt_content}

    user_input_content = f"{scenario_context}" \
                         f"**Teacher's Question:** {question}\n\n" \
                         f"**Conversation Transcript So Far:**\n{transcript}\n\n" \
                         f"**Retrieved Teaching Principles (Consider these):**\n{passages_text}\n\n" \
                         f"**Expert Advice Request:** Based on all the above, what specific advice or next steps would you recommend for the teacher?"

    user_prompt = {"role": "user", "content": user_input_content}

    messages = [system_prompt, user_prompt]

    try:
        response = local_client.chat.completions.create(
            model=OPENAI_EXPERT_MODEL,
            messages=messages,
            temperature=0.4, # Slightly lower temperature for more focused advice
            max_tokens=400  # Allow slightly longer advice if needed
        )
        reply = response.choices[0].message.content.strip()
        return reply if reply else "I need more specific context from the conversation or scenario to provide tailored advice."
    except openai.APIError as e:
        st.error(f"OpenAI API Error (Expert Advice): {e}", icon="🚨")
        return "There was an issue connecting with the expert advisor AI."
    except Exception as e:
        st.error(f"Error generating expert advice: {e}", icon="🚨")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return "An unexpected error occurred while generating advice."


# --- Streamlit UI ---

# Apply custom styling (Updated UI with image)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
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
    /* Hide spinners */
    div.stSpinner {
        display: none !important;
    }
    /* Disable chat inputs while processing */
    .processing-active [data-testid="stChatInput"] {
        opacity: 0.6;
        pointer-events: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stSidebar {
        background-color: #2d3e34;
        padding: 10px;
        border-radius: 4px;
    }
    .stSidebarHeader, .stSidebar .stTextInput input, .stSidebar .stButton, .stSidebar .stMarkdown {color: white }
    .st-emotion-cache-1mw54nq.egexzqm0 {color: white}
    .st-emotion-cache-fsammq.egexzqm0 {color: white}
    </style>
    """, unsafe_allow_html=True
)

# --- Sidebar for Expert Advisor --- (Original version from first prompt)
with st.sidebar:
    st.markdown(
    "<h1 style='text-align: left; '>Expert Teacher</h1>", unsafe_allow_html=True
)

    # Only show expert chat if scenario is selected
    if st.session_state.current_scenario:
        # Only show scrollable container after first message has been sent
        if st.session_state.expert_first_message_sent and st.session_state.expert_chat_history:
            # Create a scrollable container for expert chat messages
            expert_chat_container = st.container(height=700, border=False)

            # Display expert chat history in chronological order
            with expert_chat_container:
                for msg in st.session_state.expert_chat_history:
                    # Use role directly ('user' or 'assistant' as stored)
                    with st.chat_message(name=msg["role"]):
                        st.markdown(msg["content"])

        # Expert chat input area at the bottom of sidebar
        if expert_prompt := st.chat_input("Ask the expert a question...", key="expert_sidebar_input"):
            # Add user message to chat history
            st.session_state.expert_chat_history.append({"role": "user", "content": expert_prompt})

            # Generate expert response without spinner (UI reflects this by not showing one)
            expert_response = generate_expert_advice(
                expert_prompt,
                st.session_state.chat_history, # Pass student chat history
                st.session_state.current_scenario["scenario_id"]
            )

            # Add expert response to chat history
            st.session_state.expert_chat_history.append({"role": "assistant", "content": expert_response})

            # Mark first message as sent
            st.session_state.expert_first_message_sent = True

            # Force a rerun to update the UI properly
            st.rerun()
    else:
        st.markdown("""
        <div class="expert-help-box">
            💡 Ask for advice here once a teaching scenario has been selected from the main chat
        </div>

        <style>
        .expert-help-box {
            background-color: #2D3E34;
            color: white;
            padding: 12px 18px;
            border-radius: 8px;
            font-weight: bold;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

# --- Main Panel --- (Original version from first prompt)

st.markdown(
    """
    <style>
    .stSelectbox > div > div > div { font-size: 20px }
    .stSelectbox > div > div { height: 50px }
    </style>
    """,
    unsafe_allow_html=True
)

if not st.session_state.current_scenario:
    col1, col2, col3 = st.columns([1, 4, 1]) 
    with col2:
        st.image("assets/academiq_logo.png", use_container_width=True)

    st.write("""
    Transform the way you prepare for the classroom with our AI-powered teaching assistant!
    This interactive tool helps elementary school teachers refine their skills by simulating real classroom interactions. 
    The AI behaves like a real second-grader, responding dynamically to your teaching style, questions, and guidance.
    """)
    st.write("")

    # Scenario selection dropdown
    def handle_scenario_change():
        selected_title = st.session_state.scenario_selector
        if selected_title != "Select a scenario...":
            # Find the scenario ID from the menu list based on title
            scenario_id_found = None
            for menu_item in scenario_menu:
                if menu_item.get("title") == selected_title:
                    scenario_id_found = menu_item.get("scenario_id")
                    break

            if scenario_id_found and scenario_id_found in scenarios_dict:
                 st.session_state.current_scenario = scenarios_dict[scenario_id_found]
                 # Reset states for new scenario
                 st.session_state.chat_history = []
                 st.session_state.expert_chat_history = []
                 st.session_state.first_message_sent = False
                 st.session_state.expert_first_message_sent = False
                 print(f"Scenario selected: {selected_title} (ID: {scenario_id_found})")
            else:
                 # Handle case where title is selected but ID not found or not in dict
                 st.session_state.current_scenario = None
                 st.session_state.chat_history = []
                 st.session_state.expert_chat_history = []
                 st.session_state.first_message_sent = False
                 st.session_state.expert_first_message_sent = False
                 print(f"Warning: Scenario details not found for title '{selected_title}' or ID '{scenario_id_found}'.")

        else:
             # Reset if "Select a scenario..." is chosen
             st.session_state.current_scenario = None
             st.session_state.chat_history = []
             st.session_state.expert_chat_history = []
             st.session_state.first_message_sent = False
             st.session_state.expert_first_message_sent = False
             print("Scenario deselected.")
        # No explicit rerun needed here, Streamlit handles it on widget change

    # Prepare dropdown options using the scenario_menu list
    scenario_options = ["Select a scenario..."] + sorted([s.get('title', f"Untitled Scenario ID: {s.get('scenario_id', 'Unknown')}") for s in scenario_menu])

    st.selectbox(
        "",
        scenario_options,
        index=0, # Default to "Select a scenario..."
        key="scenario_selector",
        on_change=handle_scenario_change,
        help="Select a classroom situation to practice."
    )
    st.write("")

# --- Scenario Active Area --- (Original version from first prompt)
if st.session_state.current_scenario:
    st.markdown(
    "<h2 style='text-align: center; margin-bottom: 1.5rem;'>AcademiQ AI</h2>", unsafe_allow_html=True
    ) 
    with st.expander("Current Scenario Details", expanded=True):
        scenario = st.session_state.current_scenario
        st.subheader(f"{scenario.get('title', 'Unnamed Scenario')}")
        info_cols = st.columns(2)
        with info_cols[0]:
            if "student_name" in scenario:
                st.markdown(f"**Student:** {scenario['student_name']}")
                if scenario["student_name"] != "Whole Class" and "student_details" in scenario:
                    st.caption(f"Profile: {scenario['student_details']}")
                elif scenario["student_name"] == "Whole Class":
                     st.caption("Interaction involves the whole class.")
            if "type" in scenario:
                scenario_type = scenario["type"]
                if scenario_type == "Student Emotion" and "emotion" in scenario:
                    st.markdown(f"**Student Emotion:** {scenario['emotion']}")
        with info_cols[1]:
            if "teacher_objective" in scenario:
                st.markdown(f"**Your Objective:**")
                st.markdown(f"{scenario['teacher_objective']}")
        if "classroom_situation" in scenario:
            st.markdown("**Classroom Situation:**")
            st.markdown(f"{scenario['classroom_situation']}")
    st.write("")

    # Only show scrollable container after first message has been sent
    if st.session_state.first_message_sent and st.session_state.chat_history:
        # Create a scrollable container with fixed height for chat messages
        chat_container = st.container(height=400, border=False)

        # Display chat messages in the fixed-height container
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): # Uses 'user' or 'assistant' as stored
                    st.markdown(msg["content"])

    # Chat input area (always visible, but positioned differently based on whether first message sent)
    if not st.session_state.scenario_ended:

        if prompt := st.chat_input("Your message to the student...", key="student_chat_input_widget"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Generate student response without spinner (UI reflects this)
            student_reply = generate_student_response(
                prompt,
                st.session_state.chat_history,
                st.session_state.current_scenario["scenario_id"]
            )

            # Add student response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": student_reply})

            # Mark first message as sent
            st.session_state.first_message_sent = True

            # Force a rerun to update the UI properly
            st.rerun()

    # End scenario button (Updated version)
    if not st.session_state.scenario_ended:
        cols = st.columns([3, 1])
        with cols[1]:
            def end_scenario():
                st.session_state.chat_history = []
                st.session_state.expert_chat_history = []
                st.session_state.scenario_ended = True
                st.session_state.evaluation_submitted = False
                print("Scenario ended by user.")
                # No rerun needed here, state change handles it
            st.button("End Scenario", key="end_chat_button", on_click=end_scenario, use_container_width=True)


def generate_assessment(chat_history):
    # For the sake of this example, we'll randomly generate a score, feedback, and advice
    score = random.randint(5, 10)  # Replace with actual model score
    score_description = f"Score: {score}/10 - Your interaction was {score * 10}% effective."
    
    # AI feedback and advice
    feedback = f"Your conversation was well-structured and on-topic." if score > 5 else f"Try to engage the student more actively."
    advice = f"Consider asking more open-ended questions to encourage student participation." if score < 5 else f"Great job! Keep the conversation flowing naturally."
    
    return score_description, feedback, advice

# Initializing session state if not present
if 'scenario_ended' not in st.session_state:
    st.session_state.scenario_ended = False
    st.session_state.chat_history = []
    st.session_state.evaluation_submitted = False
    st.session_state.first_message_sent = False
    st.session_state.expert_first_message_sent = False
    
    
if st.session_state.scenario_ended and not st.session_state.evaluation_submitted:
    st.title("Scenario Evaluation")

    # Simulate an AI evaluation of the chat history
    score_description, feedback, advice = generate_assessment(st.session_state.chat_history)
    
    # Show the evaluation to the user
    st.subheader("Your Score")
    st.write(score_description)
    
    st.subheader("Feedback")
    st.write(feedback)
    
    st.subheader("Advice for Improvement")
    st.write(advice)
    
    # Button to go back to the chat and select a new scenario
    if st.button("Close Evaluation"):
        # Reset session state for a new scenario
        st.session_state.current_scenario = None
        st.session_state.first_message_sent = False
        st.session_state.expert_first_message_sent = False
        st.session_state.scenario_ended = False  
        st.session_state.chat_history = []  
        st.rerun()  # Refresh the app to show the chat interface again


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

/* Light mode footer background */
body[data-theme="light"] .footer {
    background-color: #ffffff;
}

/* Dark mode footer background */
body[data-theme="dark"] .footer {
    background-color: #1e1e1e;
}

.footer details summary {
    list-style: none;
}
.footer details summary::-webkit-details-marker {
    display: none;
}
.footer details summary {
    cursor: pointer;
    font-weight: bold;
    padding: 3px;
    color: inherit;
    font-size: 14px;
}

.footer details[open] summary {
    content: "Back";
}

/* Background for the expanded help content */
.footer details[open] {
    position: absolute;
    bottom: 50px;
    right: 20px;
    border-radius: 8px;
    padding: 10px;
    width: 300px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    background-color: #f8f8f8;
    color: #333;
}

body[data-theme="light"] .footer details[open] {
    background-color: #ffffff;
    color: #333;
}

body[data-theme="dark"] .footer details[open] {
    background-color: #333;
    color: #f0f2f6;
}

.footer details[open] p {
    text-align: left;
    margin: 5px 0;
    color: inherit;
}
</style>

<div class="footer">
    <details>
        <summary>❓ Help</summary>
        <p>👩‍🏫 <b>Want to start the chat?</b> Pick a scenario from the "Select a scenario..." dropdown and begin chatting with the student.</p>
        <p>💡 <b>Need expert advice?</b> The Teacher Expert panel on the left offers real-time strategies.</p>
        <p>📈 <b>Get personalized feedback!</b> Your chats are evaluated to improve your teaching techniques. Click "end scenario" for feedback.</p>
        <p>💬 <b>Want to start a new chat?</b> End the scenario and click the "Close Evaluation" button.</p>
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

st.markdown(footer_html, unsafe_allow_html=True)