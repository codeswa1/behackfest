import ollama

def get_ai_diagnosis(api_key, context):
    """
    Hybrid diagnostic service supporting both local Ollama and Cloud OpenAI.
    """
    
    # 1. Prepare common prompt
    logs_str = "\n".join([f"- {l['timestamp']}: {l['log']}" for l in context['logs']]) if context['logs'] else "No operator logs available."
    prompt = f"""
    Analyze these system data points:
    - CLASSIFICATION: {context['severity']}
    - RECONSTRUCTION ERROR: {context['behavior_score']}
    - DURATION: {context['duration']} min
    
    LOGS:
    {logs_str}
    
    TASK: Brief Root Cause, Reasoning, and Actions. Keep it under 150 words.
    """

    # 2. Routing Logic
    try:
        if api_key == "local_ollama":
            import ollama
            response = ollama.chat(
                model='llama3.2:latest', # Lighter & faster model (3B)
                messages=[
                    {'role': 'system', 'content': 'You are a concise technical diagnostic assistant.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response['message']['content']
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"**AI Service Error**: {str(e)}"
    
