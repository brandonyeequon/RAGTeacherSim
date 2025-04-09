from fastapi import FastAPI
from pydantic import BaseModel
from logic import generate_student_response, generate_expert_advice

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str
    chat_history: list[dict]
    scenario_id: str = None

class AdviceRequest(BaseModel):
    question: str
    chat_history: list[dict]
    scenario_id: str = None

class EvaluationRequest(BaseModel):
    chat_history: list[dict]

@app.post("/student-response")
def student_response(req: ChatRequest):
    response = generate_student_response(req.user_input, req.chat_history, req.scenario_id)
    return {"response": response}

@app.post("/expert-advice")
def expert_advice(req: AdviceRequest):
    response = generate_expert_advice(req.question, req.chat_history, req.scenario_id)
    return {"response": response}

# @app.post("/evaluate-teacher")
# def evaluate_teacher(req: EvaluationRequest):
#     response = evaluate_teacher_effectiveness(req.chat_history)
#     return {"evaluation": response}