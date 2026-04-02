import os
import time
import random
from openai import OpenAI
from app.env import ROEnv
from app.models import ROAction

# =========================
# CONFIG (REQUIRED)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# deterministic
random.seed(42)

env = ROEnv()

# =========================
# RULE-BASED POLICY
# =========================
def choose_action(issue):
    issue = issue.lower()

    if "not coming" in issue:
        return "pump_issue"
    if "taste" in issue:
        return "filter_issue"
    if "noisy" in issue:
        return "multi_issue"

    return "pump_issue"


# =========================
# RUN TASK
# =========================
def run_task(task):
    print(f"[START] task={task}")

    env.set_task(task)
    obs = env.reset()

    total_reward = 0.0

    for step in range(1, 6):

        issue = obs.customer_query
        action_label = choose_action(issue)

        action = ROAction(
            reply="Checking and resolving the issue properly",
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

    success = total_reward > 0.5

    print(f"[END] success={str(success).lower()} total_reward={total_reward:.2f}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
