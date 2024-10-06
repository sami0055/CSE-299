from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from rag import chat, get_vault_embeddings_tensors, load_vault_content
from fastapi.middleware.cors import CORSMiddleware

# Create an instance of FastAPI
app = FastAPI()


class Message(BaseModel):
    msg:str

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, but you can restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

vault_content=load_vault_content()
vault_tensor_embeddings=get_vault_embeddings_tensors()

@app.post("/message/")
def send_message(message: Message):
    

    
    response=chat(message.msg,vault_tensor_embeddings,vault_content)
    return {"message": response}