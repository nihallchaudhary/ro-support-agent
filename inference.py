import os
import time
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.env import ROEnv
from app.models import ROAction

# =========================
# CONFIG (REQUIRED BY CHECKER)
# =========================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>🚀 RO Support API is Live</h1>"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

env = ROEnv()

# =========================
# REQUIRED ENDPOINTS
# =========================

@app.post("/reset")
def reset(task: str = "easy"):
    env.set_task(task)
    obs = env.reset()

    return {
        "state": {
            "issue": obs.customer_query,
            "history": [],
            "step": 0
        }
    }


@app.post("/step")
def step(action: dict):
    act = ROAction(
        reply=action.get("reply", "auto"),
        issue_label=action.get("issue_label", ""),
        book_service=action.get("book_service", False)
    )

    result = env.step(act)

    return {
        "reward": result.reward,
        "done": result.done,
        "state": {
            "issue": result.observation.customer_query,
            "history": result.observation.conversation_history,
            "step": result.observation.step_count
        }
    }


# =========================
# COST MAP
# =========================
cost_map = {
    "check_filter": 0,
    "replace_filter": 500,
    "clean_membrane": 300,
    "replace_membrane": 1500,
    "fix_leak": 200,
    "check_pressure": 0,
    "check_power": 0
}

valid_actions = list(cost_map.keys())


# =========================
# RULE-BASED SYSTEM
# =========================
def rule_based_action(issue, action_history):

    if "bad taste" in issue.lower():
        return "replace_filter"

    if "low pressure" in issue.lower():
        return "check_filter"

    if "leak" in issue.lower():
        return "fix_leak"

    if "no water" in issue.lower():
        if "check_power" not in action_history:
            return "check_power"
        return "check_filter"

    return None


# =========================
# LLM FALLBACK
# =========================
def get_best_action(state, action_history, reward_memory):
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": f"""
Customer issue: {state['issue']}
Previous actions: {action_history}
Choose best action from:
{valid_actions}
Format:
action: <action>
reason: <short reason>
"""
                }],
                max_tokens=80
            )

            reply = response.choices[0].message.content.lower()

            action_line = [l for l in reply.split("\n") if "action:" in l][0]
            reason_line = [l for l in reply.split("\n") if "reason:" in l][0]

            action = action_line.split(":")[1].strip().replace(" ", "_")
            reason = reason_line.split(":", 1)[1].strip()

            if action in valid_actions:
                return action, reason

        except:
            time.sleep(1)

    return "check_filter", "fallback"
