import streamlit as st
import requests
import json
import os


# --- Constants and Configuration ---
API_URL = "http://localhost:8000"
# HUGGINGFACE_REPO_ID = "brandonyeequon/teacher_faiss"
# HUGGINGFACE_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", None)

LOCAL_DATA_DIR = "..\\data" # Subdirectory for JSON files
SCENARIO_MENU_PATH = os.path.join(LOCAL_DATA_DIR, "scenario_menu.json")
SCENARIOS_DATA_PATH = os.path.join(LOCAL_DATA_DIR, "scenarios.json")

st.set_page_config(page_title="AcademiQ", layout="wide")
st.title("AcademiQ Teaching Companion")


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


scenario_menu, scenarios_dict = load_scenario_data()

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
    "<h2 style='text-align: center; margin-bottom: 1.5rem;'>AcademIQ AI</h2>", unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("<h1>Expert Teacher Advisor</h1>", unsafe_allow_html=True)
    st.markdown("Ask for advice on how to handle the current teaching scenario.")

    # Only show expert chat if scenario is selected
    if st.session_state.current_scenario:
        # Only show scrollable container after first message has been sent
        if st.session_state.expert_first_message_sent and st.session_state.expert_chat_history:
            # Create a scrollable container for expert chat messages
            expert_chat_container = st.container(height=350, border=False)

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
            expert_response = requests.post(f"{API_URL}/expert-advice", json={
            "question": expert_prompt,
            "chat_history": st.session_state.chat_history,
            "scenario_id": st.session_state.current_scenario["scenario_id"]
            })
            expert_response = expert_response.json()["response"]

            # Add expert response to chat history
            st.session_state.expert_chat_history.append({"role": "assistant", "content": expert_response})

            # Mark first message as sent
            st.session_state.expert_first_message_sent = True

            # Force a rerun to update the UI properly
            st.rerun()
    else:
        st.info("Select a scenario from the main panel to enable the Expert Advisor.")


#--Main Content--
if not st.session_state.current_scenario:
    st.info("Welcome! Please select a scenario below to begin the simulation.")
    st.write("""
    This interactive tool helps elementary school teachers refine their skills by simulating real classroom interactions.
    Interact with an AI student who behaves like a second-grader, responding to your teaching style.
    Use the **Expert Teacher Advisor** panel on the left for real-time teaching strategies and best practices.
    Practice navigating discussions, engaging students, and sharpening your approach in a risk-free environment.
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
        "Choose a Teaching Scenario:",
        scenario_options,
        index=0, # Default to "Select a scenario..."
        key="scenario_selector",
        on_change=handle_scenario_change,
        help="Select a classroom situation to practice."
    )
    st.write("")


# --- Scenario Active Area --- (Original version from first prompt)
if st.session_state.current_scenario:
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
    if prompt := st.chat_input("Your message to the student...", key="student_chat_input_widget"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        student_reply = requests.post(f"{API_URL}/student-response", json={
        "user_input": prompt,
        "chat_history": st.session_state.chat_history,
        "scenario_id": st.session_state.current_scenario["scenario_id"]
        })
        student_reply = student_reply.json()["response"]

        # Add student response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": student_reply})

        # Mark first message as sent
        st.session_state.first_message_sent = True

        # Force a rerun to update the UI properly
        st.rerun()

    # End scenario button (Original version)
    cols = st.columns([3, 1])
    with cols[1]:
        def end_scenario():
            st.session_state.current_scenario = None
            st.session_state.chat_history = []
            st.session_state.expert_chat_history = []
            st.session_state.first_message_sent = False
            st.session_state.expert_first_message_sent = False
            print("Scenario ended by user.")
            # No rerun needed here, state change handles it
        st.button("End Scenario", key="end_chat_button", on_click=end_scenario, use_container_width=True)










































# # Load scenario menu
# with open("..\\data/scenario_menu.json") as f:
#     scenario_menu = json.load(f)
# scenario_ids = list(scenario_menu.keys())

# # State setup
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "scenario_id" not in st.session_state:
#     st.session_state.scenario_id = scenario_ids[0]

# with st.sidebar:
#     st.subheader("🧠 Scenario")
#     selected_id = st.selectbox("Choose a classroom scenario", scenario_ids, format_func=lambda x: scenario_menu[x]["title"])
#     st.session_state.scenario_id = selected_id
#     st.markdown(scenario_menu[selected_id]["description"])

#     st.subheader("🗣️ Teacher's Question")
#     question = st.text_input("What would you like to ask the expert coach?")
#     if st.button("💡 Get Expert Advice"):
#         res = requests.post(f"{API_URL}/expert-advice", json={
#             "question": question,
#             "chat_history": st.session_state.chat_history,
#             "scenario_id": st.session_state.scenario_id
#         })
#         st.session_state.expert_advice = res.json()["response"]

#     st.subheader("🧪 Evaluate Teacher")
#     if st.button("📋 Evaluate Conversation"):
#         res = requests.post(f"{API_URL}/evaluate-teacher", json={
#             "chat_history": st.session_state.chat_history
#         })
#         st.session_state.evaluation = res.json()["evaluation"]

# st.header("👩‍🏫 Teacher ↔ Student Chat")

# if prompt := st.chat_input("Message the student..."):
#     st.session_state.chat_history.append({"role": "user", "content": prompt})
#     res = requests.post(f"{API_URL}/student-response", json={
#         "user_input": prompt,
#         "chat_history": st.session_state.chat_history,
#         "scenario_id": st.session_state.scenario_id
#     })
#     student_reply = res.json()["response"]
#     st.session_state.chat_history.append({"role": "assistant", "content": student_reply})

# for msg in st.session_state.chat_history:
#     st.chat_message(msg["role"]).markdown(msg["content"])

# if "expert_advice" in st.session_state:
#     st.divider()
#     st.subheader("📘 Expert Advice")
#     st.markdown(st.session_state.expert_advice)

# if "evaluation" in st.session_state:
#     st.divider()
#     st.subheader("📋 Teacher Evaluation")
#     st.markdown(st.session_state.evaluation)