import os

def get_ai_diagnosis(api_key, context, provider="groq"):
    """
    Hybrid diagnostic service supporting Groq, Gemini, OpenAI, and local Ollama.
    """

    # 1. Prepare common prompt
    logs_str = "\n".join([f"- {l['timestamp']}: {l['log']}" for l in context['logs']]) if context['logs'] else "No operator logs available."
    prompt = f"""Analyze these system anomaly data points:
- CLASSIFICATION: {context['severity']}
- RECONSTRUCTION ERROR: {context['behavior_score']}
- DURATION: {context['duration']} min

OPERATOR LOGS:
{logs_str}

TASK: Provide a brief Root Cause, Reasoning, and Recommended Actions. Keep it under 150 words.
"""

    # 2. Routing Logic
    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Updated to current Llama 3.1
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
            model = genai.GenerativeModel("gemini-1.5-flash") # Current standard for fast/free
            response = model.generate_content(prompt)
            return response.text

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # requested by user
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            return response.choices[0].message.content

        elif provider == "ollama":
            # Attempt to connect to local Ollama instance
            try:
                import ollama
                response = ollama.chat(
                    model='llama3.2:latest', # requested by user
                    messages=[
                        {'role': 'system', 'content': 'You are a concise technical diagnostic assistant for industrial anomaly detection systems.'},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                return response['message']['content']
            except ImportError:
                return "⚠️ **Ollama library not installed.** Run `pip install ollama` locally."
            except Exception as e:
                return f"⚠️ **Ollama Connection Error**: {str(e)}. Ensure Ollama is running locally and the `llama3.2:latest` model is downloaded (`ollama pull llama3.2`)."

        else:
            return f"❌ **Unknown Provider**: {provider}"

    except Exception as e:
        return f"❌ **AI Service Error ({provider})**: {str(e)}"
