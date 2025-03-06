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
    allow_origins=["https://liaai-frontend.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LANGS_LIST = ["pt", "pt-pt", "ro", "ca", "gl", "en"]

LANGS_LIST_DIFFERENT_MODEL = {
    "hu": {
        "en": "Helsinki-NLP/opus-mt-en-hu"
    },
    "de": {
        "en": "Helsinki-NLP/opus-mt-en-de"
    },
    "ja": {
        "en": "Helsinki-NLP/opus-mt-en-jap"
    },
    "zh": {
        "en": "Helsinki-NLP/opus-mt-en-zh"
    }, 
    "es": {
        "en": "Helsinki-NLP/opus-mt-en-es"
    },
    "fr": {
        "en": "Helsinki-NLP/opus-mt-en-fr"
    },
    "it": {
        "en": "Helsinki-NLP/opus-mt-en-it"
    },
}

def translate(text, src_lang="pt", tgt_lang="hu"):
    if src_lang == tgt_lang:
        return text
    
    if src_lang == "pt" and tgt_lang == "pt-pt":
        return text

    if src_lang == "pt" and tgt_lang == "hu":
        text_in_english = translate(text, src_lang="pt", tgt_lang="en")
        return translate(text_in_english, src_lang="en", tgt_lang="hu")
    
    if src_lang == "pt" and tgt_lang == "es":
        text_in_english = translate(text, src_lang="pt", tgt_lang="en")
        return translate(text_in_english, src_lang="en", tgt_lang="es")
    
    if src_lang == "pt" and tgt_lang == "fr":
        text_in_english = translate(text, src_lang="pt", tgt_lang="en")
        return translate(text_in_english, src_lang="en", tgt_lang="fr")
    
    if src_lang == "pt" and tgt_lang == "it":
        text_in_english = translate(text, src_lang="pt", tgt_lang="en")
        return translate(text_in_english, src_lang="en", tgt_lang="it")
    
    if src_lang == "pt" and tgt_lang == "de":
        text_in_english = translate(text, src_lang="pt", tgt_lang="en")
        return translate(text_in_english, src_lang="en", tgt_lang="de")
    
    if src_lang == "pt" and tgt_lang == "ja":
        text_in_english = translate(text, src_lang="pt", tgt_lang="en")
        return translate(text_in_english, src_lang="en", tgt_lang="ja")
    
    if src_lang == "pt" and tgt_lang == "zh":
        text_in_english = translate(text, src_lang="pt", tgt_lang="en")
        return translate(text_in_english, src_lang="en", tgt_lang="zh")

    if src_lang in LANGS_LIST and tgt_lang in LANGS_LIST:
        model_name = f"Helsinki-NLP/opus-mt-ROMANCE-{tgt_lang}"

    elif src_lang in LANGS_LIST and tgt_lang in LANGS_LIST_DIFFERENT_MODEL:
        model_name = LANGS_LIST_DIFFERENT_MODEL[tgt_lang].get(src_lang)
        if not model_name:
            raise ValueError(f"Tradução de {src_lang} para {tgt_lang} não é suportada.")

    elif src_lang in LANGS_LIST_DIFFERENT_MODEL and tgt_lang in LANGS_LIST:
        model_name = LANGS_LIST_DIFFERENT_MODEL[src_lang].get(tgt_lang)
        if not model_name:
            raise ValueError(f"Tradução de {src_lang} para {tgt_lang} não é suportada.")

    elif src_lang in LANGS_LIST_DIFFERENT_MODEL and tgt_lang in LANGS_LIST_DIFFERENT_MODEL[src_lang]:
        model_name = LANGS_LIST_DIFFERENT_MODEL[src_lang][tgt_lang]

    else:
        raise ValueError(f"Tradução de {src_lang} para {tgt_lang} não é suportada.")

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**tokens)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

@app.post("/traduzir-arquivo/")
async def translate_archive(
    file: UploadFile = File(...),
    src_lang: str = Query(..., description="Idioma de origem (pt, en, es, hu, de, ja, zh...)"),
    tgt_lang: str = Query(..., description="Idioma de destino (pt, en, es, hu, de, ja, zh...)")
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Apenas arquivos .txt são suportados.")
    
    todas_linguas = set(LANGS_LIST) | set(LANGS_LIST_DIFFERENT_MODEL.keys())
    if src_lang not in todas_linguas or tgt_lang not in todas_linguas:
        raise HTTPException(
            status_code=400,
            detail=f"Idiomas suportados: {', '.join(todas_linguas)}"
        )

    conteudo = await file.read()
    texto = conteudo.decode("utf-8")

    paragrafos = texto.split("\n")
    paragrafos_traduzidos = [translate(p.strip(), src_lang, tgt_lang) for p in paragrafos if p.strip()]

    buffer = BytesIO()
    buffer.write("\n".join(paragrafos_traduzidos).encode("utf-8"))
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="text/plain", headers={
        "Content-Disposition": f"attachment; filename=traduzido_{file.filename}"
    })

@app.get("/")
async def read_root():
    return {"message": "Bem vindo"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
