from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    conversation: list[dict] = Field(default_factory=list)
    quick: bool = False

class ActionRequest(BaseModel):
    action: str

class GuideRequest(BaseModel):
    stage: int = Field(ge=0, le=4)
    completed: bool = False
    customer_id: str | None = Field(default=None, max_length=32)
