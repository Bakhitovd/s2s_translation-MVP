# file: app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

app = FastAPI(title="M2M100 Translation Service")

# Load once at startup
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

class TranslateIn(BaseModel):
    src_lang: str = Field(..., description="Source language code, e.g. 'ru', 'en', 'es'")
    tgt_lang: str = Field(..., description="Target language code, e.g. 'en', 'ru', 'pt'")
    text: str = Field(..., min_length=1)

class TranslateOut(BaseModel):
    translation: str

@app.post("/translate", response_model=TranslateOut)
def translate(req: TranslateIn):
    try:
        tokenizer.src_lang = req.src_lang
        inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True).to(device)
        forced_id = tokenizer.get_lang_id(req.tgt_lang)
        with torch.no_grad():
            generated = model.generate(**inputs, forced_bos_token_id=forced_id, max_length=512)
        out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return TranslateOut(translation=out)
    except KeyError:
        # typically raised if a language code isn't supported by this checkpoint
        raise HTTPException(status_code=400, detail="Unsupported language code for M2M100.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
