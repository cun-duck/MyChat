from huggingface_hub import InferenceClient

def generate_response(question, context, prompt, hf_token, model_name):
    # Initialize the InferenceClient with the appropriate provider
    if "DeepSeek" in model_name:
        client = InferenceClient(provider="together", api_key=hf_token)
    elif "Mistral" in model_name:
        client = InferenceClient(provider="together", api_key=hf_token)
    elif "Qwen" in model_name:
        client = InferenceClient(provider="sambanova", api_key=hf_token)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Prepare messages for chat-based models
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{question}\nContext: {context}"}
    ]

    try:
        # Generate response using the appropriate model
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1500
        )
        # Extract and return the generated message
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"
