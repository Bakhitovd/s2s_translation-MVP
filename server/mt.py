from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from transformers.models.m2m_100 import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# MTManager: loads model/tokenizer once, provides in-process translation
class MTManager:
    def __init__(self, model_name="facebook/m2m100_418M"):
        self.model_name = model_name
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def translate_text(self, src_lang: str, tgt_lang: str, text: str) -> str:
        try:
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            forced_id = self.tokenizer.get_lang_id(tgt_lang)
            with torch.no_grad():
                generated = self.model.generate(**inputs, forced_bos_token_id=forced_id, max_length=512)
            out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return out
        except KeyError:
            raise ValueError("Unsupported language code for M2M100.")
        except Exception as e:
            raise RuntimeError(str(e))

# Optional: FastAPI router for /mt/translate (for debugging/benchmarks)
router = APIRouter()

class TranslateIn(BaseModel):
    src_lang: str = Field(..., description="Source language code, e.g. 'ru', 'en', 'es'")
    tgt_lang: str = Field(..., description="Target language code, e.g. 'en', 'ru', 'pt'")
    text: str = Field(..., min_length=1)

class TranslateOut(BaseModel):
    translation: str

# Singleton instance for in-process use
mt_mgr = MTManager()

@router.post("/mt/translate", response_model=TranslateOut)
def translate(req: TranslateIn):
    try:
        out = mt_mgr.translate_text(req.src_lang, req.tgt_lang, req.text)
        return TranslateOut(translation=out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


