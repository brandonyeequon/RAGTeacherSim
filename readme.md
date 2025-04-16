# Final Project
## Project Summary
Our project was created to provide an immersive chatbot that simulates a second grader for UVU students in the education program. This chatbot was developed at the request of the professors in the UVU education department to help their students practice handling real-life classroom challenges. Users can practice specific scenarios and techniques by selecting from our list of scenarios. If a user ever feels stuck during a scenario, they can get help from the expert teacher chat we have provided. After ending a scenario, users are provided feedback on how they dealt with the situation.

## Team Contributions
**Andrew Buckland(Documentation Specialist/QA)**: Helped with initial data collection. Created readme files for each submission/iteration of our project. Commented code to make it more clear. Quality testing on the chatbot with normal/abnormal use to find bugs. Created help section for chatbot. Created project walkthrough for one of our milestones.  

**Brandon Lee(Product Owner)**: Communicated with the subject matter expert. Researched, programmed, and deployed initial idea and prototype. Created scenarios and deployed the final product on streamlit.  

**Cade Anderson(Data Engineer)**: Organized and led in collecting data, cleaning it, and formatting it in the way that was needed for the model. Collected knowledgebase from the server. Refined scenarios from feedback. Created powerpoint presentations.  

**Collin Anderson(UI/UX Developer)**: Focused on creating a simple and intuitive experience for UVU Education students. Created wireframes using Figma to define the app’s structure and user flow for easy navigation. Translated wireframes into an interactive UI, emphasizing simplicity and consistency with clear navigation between the teacher and student chatbots. Incorporated UVU’s school colors to align with the university’s branding and provide a familiar environment. Also created a fun AI-generated logo. Conducted testing to ensure accessibility and ease of use, pushing to host the app as a Streamlit hosted app. Designed efficient user flows for selecting scenarios and interacting with the chatbots, and receiving the evaluation.  

**Kate Bennett (Project Manager)**: Helped with initial data collection, set up project management tool and initial sprints/tasks in Microsoft Teams, contacted and communicated with the subject matter expert to set up initial feedback meeting, delegated tasks I couldn't complete due to outside circumstances.  

**Porter Nelson(AI Developer)**: Collected and cleaned the base textbooks.
Built embedding pipeline. Experimented with various methods of fine-tuning our local llama AI model. Built frontend/backend architecture for both llama and chatgpt. Added backend support for student responses, expert advice, and chat evaluation. Fine tunes prompts to address qa issues.  

## Technical Overview

This project leverages a variety of tools, frameworks, and APIs to create an interactive teaching simulation application:

#### Tools and Frameworks
- **Streamlit**: Used for building the web-based user interface.
- **FastAPI**: Provides a backend API for handling requests related to student responses, expert advice, and teacher evaluations.
- **FAISS**: Utilized for efficient similarity search and clustering of vector embeddings.
- **NumPy**: For numerical operations and handling embeddings.
- **Hugging Face Hub**: For downloading and managing datasets and models.

#### Models
- **OpenAI GPT Models**: 
  - `text-embedding-3-small`: Used for generating vector embeddings of text.
  - `gpt-4o-mini` and `gpt-4o`: Used for generating student responses, expert advice, and teacher evaluations.

#### APIs
- **OpenAI API**: For accessing GPT models to generate embeddings, chat completions, and evaluations.
- **Hugging Face API**: For downloading required files such as FAISS indices and textbook passages.

#### Backend Logic
- **Student Response Generation**: Simulates a second grader's behavior and responses based on the provided scenario and chat history.
- **Expert Advice Generation**: Provides actionable teaching strategies and advice based on the teacher's question, scenario context, and chat history.
- **Teacher Effectiveness Evaluation**: Evaluates the teacher's performance in handling the scenario, providing a score, feedback, and improvement advice.

#### Additional Libraries
- **Requests**: For making HTTP requests.
- **Pickle**: For loading and saving serialized data files.
- **JSON**: For handling configuration and scenario data.

This combination of tools and technologies enables the simulation of classroom interactions, the provision of expert teaching advice, and the evaluation of teaching effectiveness.

## Instructions to use our app
#### Use our website (Not up to date).
1. **Open your browser**
2. **Go to our website**
```
https://teachersimulation.streamlit.app
```

#### Local Startup Instructions (Up to date) **OPENAI AND HUGGINGFACE KEYS NEEDED**
In Evaluation-Endpoint branch.  
1. **Install dependencies**:
```bash
pip install -r requirements.txt
```
2. **Create .streamlit directory in the project directory**
3. **Create a secrets.toml file inside of the .streamlit directory** Insert this into your secrets.toml file
```
HUGGINGFACE_TOKEN = "YOUR KEY HERE"
OPENAI_API_KEY = "YOUR KEY HERE"
```
4. **Run the backend**:
```bash
uvicorn backend.main:app --reload
```

5. **Run the Streamlit frontend** on line 13 you may need to change the ..\\ to ../:
```bash
streamlit run streamlit_app.py
```