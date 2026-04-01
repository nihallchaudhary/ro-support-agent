import os
from openai import OpenAI
import requests
import time

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

ENV_URL = "http://localhost:8000"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# =========================
# COST MAP 💰
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
# LOGGING
# =========================
def log_start(task):
    print(f"[START] task={task} env=ro_support_env model={MODEL_NAME}")

def log_step(step, action, reward, done, reason, cost):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} cost={cost} done={str(done).lower()} reason={reason}")

def log_end(success, steps, rewards, total_cost):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str} total_cost={total_cost}")

# =========================
# 🏆 FINAL RULE-BASED SYSTEM
# =========================
def rule_based_action(issue, action_history):

    if issue == "bad taste":
        return "replace_filter"

    if issue == "low pressure":
        return "check_filter"

    if issue == "leakage":
        return "fix_leak"

    if issue == "no water":
        if "check_power" not in action_history:
            return "check_power"
        return "check_filter"

    return None

# =========================
# 🧠 LLM (Fallback only)
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

# =========================
# ❌ AVOID BAD ACTIONS
# =========================
def avoid_bad_actions(action, reward_memory):
    if action in reward_memory and reward_memory[action] < 0:
        return "check_filter"
    return action

# =========================
# MAIN TASK
# =========================
def run_task(task):
    log_start(task)

    res = requests.post(f"{ENV_URL}/reset", params={"task": task})
    state = res.json()["state"]

    print(f"[START] task={task} issue={state['issue']}")

    rewards = []
    action_history = []
    reward_memory = {}
    total_cost = 0

    for step in range(1, 7):

        # =========================
        # 🔥 RULE FIRST (FAST WIN)
        # =========================
        action = rule_based_action(state["issue"], action_history)

        if action:
            reason = "Rule-based optimized decision"
        else:
            print("[PLANNER] Using AI planning")
            action, reason = get_best_action(state, action_history, reward_memory)

        # =========================
        # ❌ AVOID BAD ACTIONS
        # =========================
        action = avoid_bad_actions(action, reward_memory)

        action_history.append(action)

        # Cost 💰
        cost = cost_map.get(action, 0)
        total_cost += cost

        # =========================
        # ENV STEP
        # =========================
        step_res = requests.post(
            f"{ENV_URL}/step",
            json={"action": action}
        ).json()

        reward = step_res["reward"]
        done = step_res["done"]

        rewards.append(reward)
        reward_memory[action] = reward

        # =========================
        # LOGGING
        # =========================
        log_step(step, action, reward, done, reason, cost)
        print(f"[THINKING] {reason}")

        if "state" in step_res:
            state = step_res["state"]

        # =========================
        # 🚀 HARD STOP (KEY WINNING LOGIC)
        # =========================
        if reward >= 5 or done or total_cost > 2000:
            break

    # =========================
    # FINAL METRICS
    # =========================
    success = sum(rewards) > 1.0
    efficiency = success / step
    profit_score = max(0, 2000 - total_cost)

    print(f"[KPI] efficiency={efficiency:.2f} profit_score={profit_score}")

    log_end(success, step, rewards, total_cost)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)