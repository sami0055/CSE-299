import torch
import ollama
import os
import pickle
import wikipediaapi
import requests
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
grq_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Load vault content from file
def load_vault_content():
    try:
        with open("vault_content.pkl", 'rb') as f:
            vault_content = pickle.load(f)
    except FileNotFoundError:
        vault_content = []
    return vault_content

# Load or generate vault embeddings
def get_vault_embeddings_tensors():
    try:
        with open('vault_embeddings.pkl', 'rb') as f:
            vault_embeddings_tensor = pickle.load(f)
    except FileNotFoundError:
        vault_embeddings_tensor = torch.tensor([])
    return vault_embeddings_tensor

# Add information to vault content and embeddings
def add_info(new_info, vault_content, vault_embeddings_tensor):
    # Embed the new information
    new_data_embedding = ollama.embeddings(model='nomic-embed-text', prompt=new_info)["embedding"]

    # Append new information to vault content
    vault_content.append(new_info)

    # Append new embedding to the vault embeddings tensor
    new_embedding_tensor = torch.tensor([new_data_embedding])
    # if vault_embeddings_tensor.nelement() == 0:
    #     vault_embeddings_tensor = new_embedding_tensor
    # else:
    vault_embeddings_tensor = torch.cat((vault_embeddings_tensor, new_embedding_tensor), dim=0)
    

    # # Save updated vault content and embeddings
    # with open("vault_embeddings.pkl", 'wb') as f:
    #     pickle.dump(vault_embeddings_tensor, f)
    # with open("vault_content.pkl", 'wb') as f:
    #     pickle.dump(vault_content, f)

    # print(NEON_GREEN + "New information added to the vault!" + RESET_COLOR)
    return vault_embeddings_tensor

# Get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3, similarity_threshold=0.5):
    if vault_embeddings.nelement() == 0:
        return []

    input_embedding = ollama.embeddings(model='nomic-embed-text', prompt=rewritten_input)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)

    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = [vault_content[idx].strip() for idx in top_indices]

    if torch.max(cos_scores) <= similarity_threshold:
        print("inside wiki")
        relevant_context = fetch_from_wikipedia(rewritten_input)

    return relevant_context

import urllib.parse

def fetch_from_wikipedia(query):
    headers = {'User-Agent': "WikiBot/1.0"}
    encoded_query = urllib.parse.quote(query)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data["extract"][:1000] if "extract" in data else "i didn't find anything relevant on Wikipedia."
    elif response.status_code == 404:
        return "i didn't find a Wikipedia page with that title."
    else:
        return f"There was a problem fetching data from Wikipedia (status code: {response.status_code})."

# Main chat function
def chat(user_input, vault_embeddings, vault_content):
    # Check if the input is a command to add new info
    if user_input.startswith("/addinfo "):
        new_info = user_input[len("/addinfo "):]
        vault_embeddings = add_info(new_info, vault_content, vault_embeddings)
        return "New information added to the vault."

    # Get relevant context for the user's question
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content)
    context_str = "\n".join(relevant_context) if relevant_context else ""
    
    user_input_with_context = user_input
    if relevant_context:
        system = "You are a helpful assistant that is an expert at extracting the most useful information from a given text."
        user_input_with_context = system + "\n\nRelevant Context:\n" + context_str + "\n\nUser Input: " + user_input

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input_with_context}],
        model="llama3-8b-8192",
    )

    result = chat_completion.choices[0].message.content
    return result
