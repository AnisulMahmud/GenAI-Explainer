from __future__ import annotations
from typing import Union, Tuple, List, Optional
import io, re, zipfile
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, SpeechT5ForTextToSpeech, SpeechT5HifiGan

#  models
TEXT_TO_SPEECH_MODEL = "microsoft/speecht5_tts"
VOCODER_MODEL = "microsoft/speecht5_hifigan"


XVECTOR_REPO = "Matthijs/cmu-arctic-xvectors"
XVECTOR_ZIP = "spkrec-xvect.zip"
PREFERRED_SPEAKER_FILE: Optional[str] = None  

def _device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
DEVICE = _device()

tokenizer = AutoTokenizer.from_pretrained(TEXT_TO_SPEECH_MODEL)
tts_model = SpeechT5ForTextToSpeech.from_pretrained(TEXT_TO_SPEECH_MODEL).to(DEVICE).eval()
vocoder  = SpeechT5HifiGan.from_pretrained(VOCODER_MODEL).to(DEVICE).eval()

def _load_xvector() -> torch.Tensor:
    
    try:
        path = hf_hub_download(repo_id=XVECTOR_REPO, filename=XVECTOR_ZIP)
        with zipfile.ZipFile(path, "r") as zf:
            files = [n for n in zf.namelist() if n.lower().endswith(".npy")]
            chosen = PREFERRED_SPEAKER_FILE if PREFERRED_SPEAKER_FILE in files else files[0]
            with zf.open(chosen) as f:
                x = np.load(io.BytesIO(f.read())).astype(np.float32)
        if x.shape != (512,): raise ValueError(f"Bad xvector shape {x.shape}")
        return torch.from_numpy(x).unsqueeze(0)  # [1,512]
    except Exception:
        return torch.zeros((1,512), dtype=torch.float32)

_SPK_CPU = _load_xvector()

def synthesize(text: str) -> Tuple[int, Union[bytes, np.ndarray]]:
    """Short text → audio (np.float32) at 16 kHz."""
    if not text or not text.strip():
        return 16000, np.zeros(0, dtype=np.float32)
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        wav = tts_model.generate_speech(
            ids, speaker_embeddings=_SPK_CPU.to(DEVICE), vocoder=vocoder
        )
    return 16000, wav.detach().cpu().numpy().astype(np.float32)

# long text support to avoid token limit
def _max_text_tokens() -> int:
    return int(getattr(tts_model.config, "text_encoder_max_position_embeddings", 600))

def _tok_len(s: str) -> int:
    return len(tokenizer(s, add_special_tokens=False)["input_ids"])

def _split_windows(text: str, max_tokens: int) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text: return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], []
    for s in sents:
        if not s: continue
        if _tok_len(s) > max_tokens:
            words, buf = s.split(), []
            for w in words:
                cand = (" ".join(buf+[w])).strip()
                if _tok_len(cand) <= max_tokens: buf.append(w)
                else: chunks.append(" ".join(buf)); buf = [w]
            if buf: chunks.append(" ".join(buf))
            continue
        cand = (" ".join(cur+[s])).strip()
        if cand and _tok_len(cand) <= max_tokens: cur.append(s)
        else:
            if cur: chunks.append(" ".join(cur))
            cur = [s]
    if cur: chunks.append(" ".join(cur))
    return chunks

def synthesize_long(text: str, pause_sec: float = 0.25) -> Tuple[int, np.ndarray]:
    max_tokens = max(50, _max_text_tokens() - 2)
    windows = _split_windows(text, max_tokens)
    if not windows: return 16000, np.zeros(0, dtype=np.float32)
    sr = 16000
    pause = np.zeros(int(sr * pause_sec), dtype=np.float32)
    parts = []
    for i, w in enumerate(windows):
        _, a = synthesize(w)
        if i: parts.append(pause)
        parts.append(a.astype(np.float32))
    return sr, np.concatenate(parts) if parts else (sr, np.zeros(0, dtype=np.float32))

if __name__ == "__main__":
    sr, audio = synthesize_long("Hello! This is a quick GeneraVoice speech test using SpeechT5. " * 4)
    print(f"Generated audio at {sr} Hz. Samples: {len(audio)} Type: {type(audio)}")
