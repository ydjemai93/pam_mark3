# llm.py
import openai

def gpt4_stream(messages):
    """
    messages: liste [ {"role":"system","content":...}, {"role":"user","content":...} ]
    Générateur: yields token par token
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # "gpt-4o" d'après la doc fournie
        messages=messages,
        stream=True,
        temperature=0.7
    )
    for chunk in response:
        if "choices" in chunk:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]

