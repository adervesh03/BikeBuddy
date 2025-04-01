# logic.py
import requests

# Query Ollama LLM for instructions based on object and zone
def query_ollama(prompt, model="gemma3:1b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return "Move away from danger."
    except Exception as e:
        print(f"Ollama query error: {e}")
        return "Move away from danger."

# Construct the prompt and get avoidance instruction
def get_avoidance_instructions(object_name, zone):
    prompt = f"""
    You are a bicycle safety assistant. A {object_name} has been detected in the following zone:
    {zone}

    Provide a short, clear instruction (10 words or less) to help the cyclist avoid the {object_name}.
    """
    return query_ollama(prompt)
