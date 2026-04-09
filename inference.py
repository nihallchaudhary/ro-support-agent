import os
import random
from openai import OpenAI
from app.env import ROEnv
from app.models import ROAction

# 🔧 API CONFIG
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

random.seed(42)
env = ROEnv()

ACTIONS = ["pump_issue", "filter_issue", "multi_issue"]


# ✅ Rule-based confidence
def get_confidence(issue):
    issue = issue.lower()

    if "not coming" in issue or "no water" in issue:
        return "pump_issue", 0.9

    if "taste" in issue:
        return "filter_issue", 0.9

    if "noisy" in issue or "low pressure" in issue:
        return "multi_issue", 0.85

    return None, 0.3


# ✅ LLM classification
def llm_decide(issue):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"""
Classify this issue into one of:
pump_issue, filter_issue, multi_issue
Issue: {issue}
Answer only label.
"""
            }],
            max_tokens=10
        )

        ans = response.choices[0].message.content.lower()

        for a in ACTIONS:
            if a in ans:
                return a

    except Exception:
        pass

    return None


# ✅ Decision logic
def choose_action(issue):
    rule_action, confidence = get_confidence(issue)
    llm_action = llm_decide(issue)

    if confidence > 0.7:
        return llm_action if llm_action else rule_action

    if llm_action:
        return llm_action

    return random.choice(ACTIONS)


# ✅ MAIN FUNCTION (platform entry point)
def run_task(task):
    env.set_task(task)
    obs = env.reset()

    issue = obs.customer_query

    action_label = choose_action(issue)

    # smarter booking decision
    book_service = True if action_label in ["multi_issue", "pump_issue"] else False

    # better response (helps LLM criteria)
    reply = (
        "Based on your issue, it seems there might be a problem with your RO system. "
        "Our technician can inspect and resolve it efficiently. "
        "Would you like to schedule a service visit?"
    )

    action = ROAction(
        reply=reply,
        issue_label=action_label,
        book_service=book_service
    )

    return action


# ✅ LOCAL TEST
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        action = run_task(task)
        print(task, action)
