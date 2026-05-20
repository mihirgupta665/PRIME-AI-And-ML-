# uvicorn app:app --reload   : to run the website at the anaconda command prompt
# pip install fastapi uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel  # formating the input for validation of inputs
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


# Device Selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)    
print(f"Device : {device}")


# templating
templates = Jinja2Templates(directory=".") 


# Input Schema Defined for Dialogue to be Strings
class DialogueInput(BaseModel):
    dialogue: str


# Clean Data Function
def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"<.*?>", " ", text) # html
    text = re.sub(r"\s+", " ", text) # spaces
    text = text.strip().lower()
    return text


# Summarization Mechanism
def summarize_dialogue(dialogue : str) -> str:   # dialogue is a str and it returns a str
    # clean user dialogue
    dialogue = clean_data(dialogue)  # pre-process

    # tokenization
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(
        device
    )  # input token_ids of dialogues

    # Summary Generation inform of tokens first
    # print("Device : ",device)
    model.to(device)
    targets = model.generate(  # summary (target) token id is generated
        input_ids=inputs["input_ids"],  # dialogue tokens
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,  # total no. of summaries generation and out of them one best is selected
        early_stopping=True,  # if all outputs formed and End Of Sequence received then stop generation
    )
    print(f"Targets : {targets}")

    # Converion of tokens of summary to real human level summary by decoding
    summary = tokenizer.decode(
        targets[0], skip_special_tokens=True
    )  # EOS, SEP(Separators)   # targets[0] : is a tokenizer list
    return summary


# API EndPoints Creation
@app.post("/summarize/")
async def summarizer(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary" : summary} # JSON Obj


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )
