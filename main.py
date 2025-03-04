from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from transformers import MarianMTModel, MarianTokenizer
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from io import BytesIO
from dotenv import load_dotenv 
import uvicorn
import os

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
login(token=HUGGING_FACE_TOKEN)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

ROMANCE_LANGS = ["pt", "pt-pt", "es", "fr", "it", "ro", "ca", "gl", "en"]

NON_ROMANCE_MODELS = {
    "hu": {
        "en": "Helsinki-NLP/opus-mt-hu-en",
        "pt": "Helsinki-NLP/opus-mt-hu-ROMANCE"
    },
    "de": {
        "en": "Helsinki-NLP/opus-mt-de-en",
        "pt": "Helsinki-NLP/opus-mt-de-ROMANCE"
    },
    "ja": {
        "en": "Helsinki-NLP/opus-mt-ja-en"
    },
    "zh": {
        "en": "Helsinki-NLP/opus-mt-zh-en"
    }
}

def traduzir(texto, src_lang="pt", tgt_lang="en"):
    if src_lang == tgt_lang:
        return texto  

    if src_lang in ROMANCE_LANGS and tgt_lang in ROMANCE_LANGS:
        modelo_nome = f"Helsinki-NLP/opus-mt-ROMANCE-{tgt_lang}"

    elif src_lang in ROMANCE_LANGS and tgt_lang in NON_ROMANCE_MODELS:
        if "pt" in tgt_lang: 
            tgt_lang = "pt"
        modelo_nome = NON_ROMANCE_MODELS[tgt_lang].get(src_lang)
        if not modelo_nome:
            raise ValueError(f"Tradução de {src_lang} para {tgt_lang} não é suportada.")

    elif src_lang in NON_ROMANCE_MODELS and tgt_lang in ROMANCE_LANGS:
        modelo_nome = NON_ROMANCE_MODELS[src_lang].get(tgt_lang)
        if not modelo_nome:
            raise ValueError(f"Tradução de {src_lang} para {tgt_lang} não é suportada.")

    elif src_lang in NON_ROMANCE_MODELS and tgt_lang in NON_ROMANCE_MODELS[src_lang]:
        modelo_nome = NON_ROMANCE_MODELS[src_lang][tgt_lang]

    else:
        raise ValueError(f"Tradução de {src_lang} para {tgt_lang} não é suportada.")
    
    tokenizer = MarianTokenizer.from_pretrained(modelo_nome)
    model = MarianMTModel.from_pretrained(modelo_nome)

    tokens = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    traducao = model.generate(**tokens)
    return tokenizer.batch_decode(traducao, skip_special_tokens=True)[0]

@app.post("/traduzir-arquivo/")
async def traduzir_arquivo(
    file: UploadFile = File(...),
    src_lang: str = Query(..., description="Idioma de origem (pt, en, es, hu, de, ja, zh...)"),
    tgt_lang: str = Query(..., description="Idioma de destino (pt, en, es, hu, de, ja, zh...)")
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Apenas arquivos .txt são suportados.")
    
    todas_linguas = set(ROMANCE_LANGS) | set(NON_ROMANCE_MODELS.keys())
    if src_lang not in todas_linguas or tgt_lang not in todas_linguas:
        raise HTTPException(
            status_code=400,
            detail=f"Idiomas suportados: {', '.join(todas_linguas)}"
        )

    conteudo = await file.read()
    texto = conteudo.decode("utf-8")

    paragrafos = texto.split("\n")
    paragrafos_traduzidos = [traduzir(p.strip(), src_lang, tgt_lang) for p in paragrafos if p.strip()]

    buffer = BytesIO()
    buffer.write("\n".join(paragrafos_traduzidos).encode("utf-8"))
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="text/plain", headers={
        "Content-Disposition": f"attachment; filename=traduzido_{file.filename}"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
