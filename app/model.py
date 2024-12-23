import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')

MODEL_NAME = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

# Changed to synchronous function
def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    # Use model.generate() synchronously
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define Perplexity calculation function
def calculate_perplexity(text: str) -> float:
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    nlls = []

    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

# Define function to compute BLEU score
def calculate_bleu(reference: str, generated: str) -> float:
    return sentence_bleu([reference.split()], generated.split())

# Define function to compute ROUGE score
def calculate_rouge(reference: str, generated: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

# Define function to compute METEOR score
def calculate_meteor(reference: str, generated: str) -> float:
    return meteor_score([reference.split()], generated.split())
