from huggingface_hub import InferenceClient

def generate_response(question, context, custom_prompt, hf_token, model_name):
    """
    Generates a response using the Hugging Face Inference API.
    """
    client = InferenceClient(api_key=hf_token)
    messages = [
        {"role": "system", "content": custom_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1500,
    )
    return completion.choices[0].message.content
