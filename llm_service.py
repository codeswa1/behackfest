def get_ai_diagnosis(api_key, context, provider="groq"):
    """
    Hybrid diagnostic service supporting Groq (free), Gemini (free), and OpenAI.
    """

    # 1. Prepare prompt
    logs_str = "\n".join([f"- {l['timestamp']}: {l['log']}" for l in context['logs']]) if context['logs'] else "No operator logs available."
    prompt = f"""Analyze these system anomaly data points:
- CLASSIFICATION: {context['severity']}
- RECONSTRUCTION ERROR: {context['behavior_score']}
- DURATION: {context['duration']} min

OPERATOR LOGS:
{logs_str}

TASK: Provide a brief Root Cause, Reasoning, and Recommended Actions. Keep it under 150 words."""

    # 2. Routing Logic
    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Free, fast Llama 3.1 on Groq
                messages=[
                    {"role": "system", "content": "You are a concise technical diagnostic assistant for industrial anomaly detection systems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content

        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")  # Free tier
            response = model.generate_content(prompt)
            return response.text

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Cheaper than gpt-4-turbo
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            return response.choices[0].message.content

        elif provider == "ollama":
            try:
                import ollama
                response = ollama.chat(
                    model='llama3.2:latest',
                    messages=[
                        {'role': 'system', 'content': 'You are a concise technical diagnostic assistant.'},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                return response['message']['content']
            except Exception:
                return "⚠️ **Ollama is not available in this environment.** Ollama runs locally only. Please switch to Groq or Gemini in the sidebar."

    except Exception as e:
        return f"**AI Service Error**: {str(e)}"
