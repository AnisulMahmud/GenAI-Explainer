# text_gen.py
from transformers import pipeline

# Using a Hugging Face model (FLAN-T5) for text generation
_MODEL = "google/flan-t5-base"  

_script_pipe = pipeline(
    "text2text-generation",
    model=_MODEL,
)

PROMPT_TMPL = (
    "You are a science communicator. Rewrite the following abstract as a clear, "
    "engaging one-minute narration (150-180 words). Avoid jargon, keep it factual, "
    "use present tense, and include a brief motivation, method, and takeaway.\n\n"
    "Abstract:\n{abstract}\n\nNarration:"
)

def make_script(abstract: str, max_new_tokens: int = 220) -> str:
    prompt = PROMPT_TMPL.format(abstract=abstract.strip())
    out = _script_pipe(prompt, max_length=None, max_new_tokens=max_new_tokens, truncation=True)
    return out[0]["generated_text"].strip()

def _demo():
    sample = (
    "Generative AI refers to models that can create new content such as text, images, or music. "
    "Recent advances in large language models show how these systems learn patterns from vast datasets "
    "and produce coherent, context-aware outputs, enabling applications from chatbots to creative writing."
)
    print(make_script(sample))

if __name__ == "__main__":
    _demo()
