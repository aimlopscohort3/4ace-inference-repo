import time
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse
from app.model import generate_text, calculate_perplexity,calculate_bleu, calculate_rouge, calculate_meteor
from app.metrics import REQUEST_TIME, GENERATION_LENGTH
from prometheus_client import start_http_server
import uvicorn
from prometheus_client import Counter, Gauge, Summary, generate_latest, REGISTRY

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

if __name__ == "__main__":
    start_http_server(8000)
    uvicorn.run(app, host="0.0.0.0", port=8080)
