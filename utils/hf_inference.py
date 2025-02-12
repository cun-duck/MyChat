from huggingface_hub import InferenceApi

def generate_response(question, context, prompt, hf_token, model_name):
    inference = InferenceApi(repo_id=model_name, token=hf_token)
    
    # Format input sesuai kebutuhan model
    if "DeepSeek" in model_name:
        inputs = {
            "prompt": f"{prompt} {question}\nContext: {context}",
            "max_tokens": 1500
        }
    elif "Mistral" in model_name:
        inputs = {
            "inputs": f"{prompt} {question}\nContext: {context}",
            "parameters": {"max_length": 1500}
        }
    elif "Qwen" in model_name:
        # Format input khusus untuk Qwen 2.5 Coder
        inputs = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"{question}\nContext: {context}"}
            ],
            "max_tokens": 1500
        }
    else:
        # Default format untuk model lain
        inputs = {
            "prompt": prompt,
            "context": context,
            "question": question
        }
    
    response = inference(inputs)
    return response.get("generated_text", response.get("result", "No response generated."))
