import os
import time
import random
from openai import OpenAI
from app.env import ROEnv
from app.models import ROAction

# =========================
# CONFIG
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

random.seed(42)
env = ROEnv()

# =========================
# SMART POLICY (IMPROVED)
# =========================
def choose_action(issue, history):
    issue = issue.lower()

    # priority mapping
    if "not coming" in issue or "no water" in issue:
        if "pump_issue" not in history:
            return "pump_issue"

    if "taste" in issue:
        return "filter_issue"

    if "noisy" in issue or "low" in issue:
        return "multi_issue"

    # fallback (safe)
    return random.choice(["pump_issue", "filter_issue", "multi_issue"])


# =========================
# SAFE LLM FALLBACK (LOW COST)
# =========================
def llm_refine(issue):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"Classify issue into pump_issue, filter_issue, multi_issue: {issue}"
            }],
            max_tokens=10
        )

        ans = response.choices[0].message.content.lower()

        if "pump" in ans:
            return "pump_issue"
        if "filter" in ans:
            return "filter_issue"
        if "multi" in ans:
            return "multi_issue"

    except:
        pass

    return None


# =========================
# RUN TASK
# =========================
def run_task(task):
    print(f"[START] task={task}")

    env.set_task(task)
    obs = env.reset()

    total_reward = 0.0
    history = []

    for step in range(1, 6):

        issue = obs.customer_query

        # smart rule
        action_label = choose_action(issue, history)

        # refine using LLM only if uncertain
        if step == 1:
            llm_action = llm_refine(issue)
            if llm_action:
                action_label = llm_action

        history.append(action_label)

        action = ROAction(
            reply="Diagnosing issue and applying optimal solution efficiently",
            issue_label=action_label,
            book_service=True
        )

        result = env.step(action)

        reward = result.reward
        done = result.done

        total_reward += reward

        print(f"[STEP] step={step} action={action_label} reward={reward:.2f} done={str(done).lower()}")

        obs = result.observation

        if done:
            break

    success = total_reward > 0.6

    print(f"[END] success={str(success).lower()} total_reward={total_reward:.2f}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
