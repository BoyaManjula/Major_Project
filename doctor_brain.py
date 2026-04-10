import os
from groq import Groq


def offline_fallback(context):
    """
    Offline safe response when Groq API is not reachable.
    This prevents infinite loading in Gradio.
    """

    if "COUGH" in context:
        condition = "Common Cold / Cough"
        doctor = "General Physician or ENT Specialist"

    elif "HEADACHE" in context:
        condition = "Headache or Fatigue"
        doctor = "General Physician or Neurologist if severe"

    elif "SKIN_RASH" in context:
        condition = "Skin Rash or Skin Irritation"
        doctor = "Dermatologist"

    else:
        condition = "General Health Concern"
        doctor = "General Physician"

    return f"""
⚠️ OFFLINE MODE (AI API not reachable)

1. Possible Condition:
{condition}

2. Precautions:
- Take adequate rest
- Drink enough water
- Maintain hygiene
- Avoid stress and heavy physical activity

3. Home Remedies:
- Drink warm fluids
- Steam inhalation for cold/cough
- Apply soothing creams for skin irritation
- Take proper sleep

4. Doctor to Consult:
{doctor}

5. When to Seek Immediate Medical Help:
- Difficulty breathing
- High fever for more than 2 days
- Severe headache with vomiting
- Sudden allergic reactions
"""


def analyze_text_query(context, model="meta-llama/llama-4-scout-17b-16e-instruct"):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return offline_fallback(context)

    try:
        client = Groq(api_key=api_key)

        system_prompt = """
You are an AI healthcare assistant.

The user input may contain a detected category such as:
COUGH
HEADACHE
SKIN_RASH
GENERAL_HEALTH

IMPORTANT RULES:

1. Use the detected category to determine the condition.
2. Give ONLY ONE possible condition.
3. Do NOT list multiple diseases.
4. Keep the answer clear and structured.
5. Mention that it is not a substitute for a real doctor.

Response format:

1. Possible Condition
2. Precautions
3. Home Remedies
4. Doctor to Consult
5. When to Seek Immediate Medical Help
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )

        return response.choices[0].message.content[:2000]

    except Exception as e:
        print("Groq API Error:", e)
        return offline_fallback(context)