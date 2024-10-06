import torch
import ollama
import os
from openai import OpenAI
import argparse
import json
import pickle

import requests
from dotenv import load_dotenv

load_dotenv()


# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def load_vault_content():
    vault_content = []
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()
    return vault_content

def get_vault_embeddings_tensors():
    vault_embeddings_tensor=None
    
    if os.path.exists("vault_embeddings.pkl"):
        
        with open('vault_embeddings.pkl','rb') as f:
            vault_embeddings_tensor=pickle.load(f)
    else:
        
        # Generate embeddings for the vault content using Ollama
        print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
        vault_content=load_vault_content()
        vault_embeddings = []
        for content in vault_content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
            vault_embeddings.append(response["embedding"])

        # Convert to tensor and print embeddings
        print("Converting embeddings to tensor...")
        vault_embeddings_tensor = torch.tensor(vault_embeddings) 
        with open('vault_embeddings.pkl', 'wb') as f:
            pickle.dump(vault_embeddings_tensor, f)
        print("Embeddings for each line in the vault:")
    return vault_embeddings_tensor






# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    print(input_embedding)
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def chat(user_input, vault_embeddings, vault_content):
    

    
    
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content)
    print(relevant_context)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        # print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        # user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
        system="You are a helpful assistant that is an expert at extracting the most useful information from a given text."
        user_input_with_context =system+"\n\nRelevant Context:\n" + context_str+"\n\nUser Input: "+user_input
    
    
    
    
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGING_FACE_KEY')}"
    }

    

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    # Example request
    data = query({
        "inputs": user_input_with_context
    })
    print(data)

    
    result=data[0]['generated_text'].split(user_input)[1]
    
    return result






