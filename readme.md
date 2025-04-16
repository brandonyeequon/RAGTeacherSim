# Final Project
## Project Summary
Our project was created to provide an immersive chatbot that simulates a second grader for UVU students in the education program. This chatbot was developed at the request of the professors in the UVU education department to help their students practice handling real-life classroom challenges. Users can practice specific scenarios and techniques by selecting from our list of scenarios. If a user ever feels stuck during a scenario, they can get help from the expert teacher chat we have provided. After ending a scenario, users are provided feedback on how they dealt with the situation.

## Team Contributions
Andrew Buckland(Documentation Specialist/QA): Helped with initial data collection, created readme files for submissions, commented code, quality testing on chatbot(regular use and stress testing), created help section for chatbot, and created walkthrough for one of our milestones.  

Brandon Lee(Product Owner):  

Cade Anderson(Data Engineer):  

Collin Anderson(UI/UX Developer):  

Kate Bennett(Project Manager):  

Porter Nelson(AI Developer):  

## Technical Overview

This project leverages a variety of tools, frameworks, and APIs to create an interactive teaching simulation application:

#### Tools and Frameworks
- **Streamlit**: Used for building the web-based user interface.
- **FAISS**: Utilized for efficient similarity search and clustering of vector embeddings.
- **NumPy**: For numerical operations and handling embeddings.
- **Hugging Face Hub**: For downloading and managing datasets and models.

#### Models
- **OpenAI GPT Models**: 
  - `text-embedding-3-small`: Used for generating vector embeddings of text.
  - `gpt-4o-mini` and `gpt-4o`: Used for generating student responses and expert advice.

#### APIs
- **OpenAI API**: For accessing GPT models to generate embeddings and chat completions.
- **Hugging Face API**: For downloading required files such as FAISS indices and textbook passages.

#### Additional Libraries
- **Requests**: For making HTTP requests.
- **Pickle**: For loading and saving serialized data files.
- **JSON**: For handling configuration and scenario data.

This combination of tools and technologies enables the simulation of classroom interactions and the provision of expert teaching advice.

## Instructions to use our app
#### Use our website
1. **Open your browser**
2. **Go to our website**
```
https://teachersimulation.streamlit.app
```

#### Local Startup Instructions **OPENAI AND HUGGINGFACE KEYS NEEDED**

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

4. **Run the Streamlit frontend**:
```bash
streamlit run streamlit_app.py
```