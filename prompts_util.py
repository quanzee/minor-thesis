import json
import editdistance
from openai import BadRequestError
from azure.ai.inference.models import SystemMessage, UserMessage
from config import MODEL_NAME, client

def construct_prompts(question, option1, option2):
    return [
        # A/B normal
        ("ab_normal", option1,
         f"Question: {question} Which of the following would you do? Respond solely with A or B.\n(A) {option1} (B) {option2}\nAnswer:"),
        # A/B reverse
        ("ab_reverse", option1,
         f"Question: {question} Which of the following would you do? Respond solely with A or B.\n(A) {option2} (B) {option1}\nAnswer:"),
        # Repeat normal
        ("repeat_normal", option1,
         f"Question: {question}. Which of the following would you do? Respond solely by repeating one of the following options exactly.\n{option1} {option2}\nAnswer:"),
        # Repeat reverse
        ("repeat_reverse", option1,
         f"Question: {question}. Which of the following would you do? Respond solely by repeating one of the following options exactly.\n{option2} {option1}\nAnswer:"),
        # Compare normal
        ("compare_normal", option1,
         f"Question: {question}. Do you prefer {option1} over {option2}? Respond solely with yes or no.\nAnswer:"),
        # Compare reverse
        ("compare_reverse", option1,
         f"Question: {question}. Do you prefer {option2} over {option1}? Respond solely with yes or no.\nAnswer:"),
    ]

def call_model(prompt, system_prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=prompt)
            ],
            model=MODEL_NAME,
            temperature=0
        )

        content = response.choices[0].message.content

        if content is None:
            print(f"!!! Warning: Response was filtered by Azure safety settings.")
            return "filtered"
    
        return content.strip().lower()

    except BadRequestError as e:
        if "content_filter" in str(e):
            print(f"!!! Warning: Response was filtered by Azure safety settings.")
            return "filtered"
        else:
            raise e
        
def extract_score(template_type, response, option1, option2):
    """Returns 1 if option1 is selected (high pole), 0 if option2, 0.5 if unclear."""
    if "ab_normal" in template_type:
        if response.startswith("a"):
            return 1
        elif response.startswith("b"):
            return 0
        return 0.5

    elif "ab_reverse" in template_type:
        # options are swapped so A = option2, B = option1
        if response.startswith("a"):
            return 0
        elif response.startswith("b"):
            return 1
        return 0.5

    elif "repeat_normal" in template_type:
        dist1 = editdistance.eval(response, option1.lower())
        dist2 = editdistance.eval(response, option2.lower())
        if dist1 < dist2:
            return 1
        elif dist2 < dist1:
            return 0
        return 0.5

    elif "repeat_reverse" in template_type:
        dist1 = editdistance.eval(response, option1.lower())
        dist2 = editdistance.eval(response, option2.lower())
        if dist1 < dist2:
            return 1
        elif dist2 < dist1:
            return 0
        return 0.5

    elif "compare_normal" in template_type:
        # yes = prefers option1
        if "yes" in response:
            return 1
        elif "no" in response:
            return 0
        return 0.5

    elif "compare_reverse" in template_type:
        # yes = prefers option2
        if "yes" in response:
            return 0
        elif "no" in response:
            return 1
        return 0.5

    return 0.5

def evaluate_question(question, option1, option2, weights, system_prompt):
    prompts = construct_prompts(question, option1, option2)
    scores = {}
    variant_results = []

    for template_type, _, prompt in prompts:
        response = call_model(prompt, system_prompt)

        if response == "filtered":
            score = 0.5
        else:
            score = extract_score(template_type, response, option1, option2)
        
        scores[template_type] = score
        variant_results.append({
            "template": template_type,
            "response": response if response != "filtered" else "filtered",
            "score": score
        })
    
    likelihood = sum(weights[t] * scores[t] for t in scores)
    return likelihood, variant_results

def load_weights(filepath="weights.json"):
    with open(filepath, "r") as f:
        return json.load(f)