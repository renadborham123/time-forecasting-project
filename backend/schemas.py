from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    conversation: list[dict] = []

class ActionRequest(BaseModel):
    action: str

