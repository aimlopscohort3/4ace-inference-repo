import time
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
from prometheus_client import start_http_server, Gauge, Summary, generate_latest, REGISTRY
import uvicorn

from app.model import generate_text, calculate_perplexity, calculate_bleu, calculate_rouge, calculate_meteor
from app.metrics import REQUEST_TIME

# Initialize Prometheus metrics
BLEU_SCORE = Gauge('bleu_score', 'BLEU score of generated text')
PERPLEXITY = Gauge('perplexity_score', 'Perplexity of generated text')
ROUGE_SCORE = Gauge('rouge_score', 'ROUGE score of generated text')
METEOR_SCORE = Gauge('meteor_score', 'METEOR score of generated text')

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

reference = ["The cat is sitting on the mat."]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def update_metrics(bleu_score, perplexity, rouge_score, meteor_score_value):
    BLEU_SCORE.set(bleu_score)
    PERPLEXITY.set(perplexity)
    ROUGE_SCORE.set(rouge_score['rouge1'].fmeasure)
    METEOR_SCORE.set(meteor_score_value)

@app.post("/generate/")
async def generate(prompt: str = Form(...)) -> dict:
    try:
        start_time = time.time()
        generated_text = generate_text(prompt)

        # Calculate metrics
        bleu_score = calculate_bleu(reference[0], generated_text)
        perplexity = calculate_perplexity(generated_text)
        rouge_score = calculate_rouge(reference[0], generated_text)
        meteor_score_value = calculate_meteor(reference[0], generated_text)

        # Update metrics
        update_metrics(bleu_score, perplexity, rouge_score, meteor_score_value)

        # Record processing time
        processing_time = time.time() - start_time
        REQUEST_TIME.observe(processing_time)

        return {
            "generated_text": generated_text,
            "BLEU": bleu_score,
            "Perplexity": perplexity,
            "ROUGE": rouge_score,
            "METEOR": meteor_score_value
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return generate_latest(REGISTRY)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    start_http_server(8000)
    uvicorn.run(app, host="0.0.0.0", port=8080)