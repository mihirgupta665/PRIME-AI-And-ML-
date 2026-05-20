# pip install fastapi uvicorn
from fastapi import FastAPI, Requests
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# App Initialization
app = FastAPI(title="Text Summarizer Application", description="Text Summarization using T5 Transformer", version="1.0")

# model and Tokenization
model = T5ForConditionalGeneration.from_pretrained("./saved_summarizer_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summarizer_model")