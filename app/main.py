import time
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse, HTMLResponse
from app.model import generate_text, calculate_perplexity,calculate_bleu, calculate_rouge, calculate_meteor
from app.metrics import REQUEST_TIME, GENERATION_LENGTH
from prometheus_client import start_http_server
import uvicorn
from prometheus_client import Counter, Gauge, Summary, generate_latest, REGISTRY
from uuid import uuid4
from typing import Dict
from pydantic import BaseModel

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Initialize Prometheus metrics
BLEU_SCORE = Gauge('bleu_score', 'BLEU score of generated text')
PERPLEXITY = Gauge('perplexity_score', 'Perplexity of generated text')
ROUGE_SCORE = Gauge('rouge_score', 'ROUGE score of generated text')
METEOR_SCORE = Gauge('meteor_score', 'METEOR score of generated text')


app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
#templates = Jinja2Templates(directory="app/templates")

reference = ["The cat is sitting on the mat."]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate/")

async def generate(prompt: str = Form(...)):  # The prompt is expected as form data
    start_time = time.time()
    generated_text = generate_text(prompt)  # Call the synchronous generate_text function

    # Calculate metrics
    bleu_score = calculate_bleu(reference[0], generated_text)  # BLEU score
    perplexity = calculate_perplexity(generated_text)  # Perplexity
    rouge_score = calculate_rouge(reference[0], generated_text)  # ROUGE score
    meteor_score_value = calculate_meteor(reference[0], generated_text)  # METEOR score


    # Update Prometheus metrics
    BLEU_SCORE.set(bleu_score)
    PERPLEXITY.set(perplexity)
    ROUGE_SCORE.set(rouge_score['rouge1'].fmeasure)  # Set ROUGE-1 F1 score
    METEOR_SCORE.set(meteor_score_value)

# Record the time it took to process the request
    processing_time = time.time() - start_time
    REQUEST_TIME.observe(processing_time)  # Manually observe the time spent

 # Return the generated text along with the metrics
    return {
        "generated_text": generated_text,
        "BLEU": bleu_score,
        "Perplexity": perplexity,
        "ROUGE": rouge_score,
        "METEOR": meteor_score_value
    }

# Expose /metrics endpoint for Prometheus scraping
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    # Return the metrics in the Prometheus format
    return generate_latest(REGISTRY)


# Loop conversation with session ------------------------------------- start
# Store chat sessions (in-memory)
sessions: Dict[str, list] = {}

# Pydantic model for incoming prompts
class ChatPrompt(BaseModel):
    prompt: str
    session_id: str = None

@app.get("/convo", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request})

# Function to generate text with limited context

def generate_text_with_context(prompt: str, history: list, max_history=3, max_new_tokens=100) -> str:
    MODEL_NAME = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Limit history to the last N exchanges (max_history * 2 for User and Bot turns)
    trimmed_history = history[-(max_history * 2):]

    # Create structured input for the model
    full_prompt = "\n".join(trimmed_history) + f"\nUser: {prompt}\nBot:"

    # Encode and generate response
    inputs = tokenizer.encode(full_prompt, return_tensors='pt')
    outputs = model.generate(
        inputs, 
        max_new_tokens=max_new_tokens,  # Control output length
        no_repeat_ngram_size=2,  # Prevent repetition
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Creativity balance
        top_k=50,  # Limit token selection
        top_p=0.9  # Nucleus sampling
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the latest bot response
    if "Bot:" in response:
        response = response.split("Bot:")[-1].strip()

    return response

# Route to start a new conversation
@app.post("/start/")
async def start_conversation():
    session_id = str(uuid4())  # Generate unique session ID
    sessions[session_id] = []  # Initialize empty chat history
    return {"session_id": session_id, "message": "New conversation started. Please send your first prompt."}

# Route to continue conversation
@app.post("/chat/")
async def chat(prompt: ChatPrompt):
    if not prompt.session_id or prompt.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session. Start a new conversation.")

    # Start measuring processing time
    start_time = time.time()

    # Get existing conversation history
    history = sessions[prompt.session_id]

    # Generate new response with limited context
    response = generate_text_with_context(prompt.prompt, history)

    # Update conversation history with the new prompt and response
    history.append(f"User: {prompt.prompt}")
    history.append(f"Bot: {response}")

    # Record processing time
    processing_time = time.time() - start_time

    # Return the generated response to the user
    return {
        "session_id": prompt.session_id,
        "response": response
    }

# Loop conversation with session ---------------------------------------- end


if __name__ == "__main__":
    start_http_server(8000)
    uvicorn.run(app, host="0.0.0.0", port=8080)
