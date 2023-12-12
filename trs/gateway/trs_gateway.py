from fastapi import FastAPI

from entity_ext import EntityExtHandle
from text_class_handle import TextClassHandle,labels_init
import json,uvicorn
from request_handle import ChatRequest,ChatResponse



chat_handle={
    "entity_handle":EntityExtHandle(),
    "text_class_handle":TextClassHandle()
}

# labels_init()

app = FastAPI()
RESP_SUCCESS = "SUCCESS"
RESP_FAILURE = "FAILURE"
@app.post("/chat")
def home(request : ChatRequest ):

    for handle in chat_handle.values():
        if handle.support(request):
            send_request = handle.send_request(request)
            break
    return {"result": RESP_SUCCESS, "response": json.dumps(send_request,ensure_ascii=False), "history":[]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)