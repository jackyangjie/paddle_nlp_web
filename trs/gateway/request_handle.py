from abc import  abstractmethod
from pydantic import BaseModel,Field

url = "http://127.0.0.1:8989"
headers = {"Content-Type": "application/json"}
class ChatRequest(BaseModel):
    query: str = Field(..., description="用来分析的文本")
    prompt: str|None = Field(..., description="如果没有该参数就是 文本分类，如果是 ['人物'] 表示实体提取")
    modelParams: dict|None = None


class ChatResponse(BaseModel):
    result: str
    response: str
    history: str


class BaseRequest:


    @abstractmethod
    def support(self,request: ChatRequest):
        pass

    @abstractmethod
    def send_request(self,request: ChatRequest):
        pass