# inference.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import random
import os
import time

# Optional: Uncomment if you want LLM fallback
# from openai import OpenAI

app = FastAPI(title="RO Support OpenEnv + Planner")

# -----------------------
# Internal Env State
# -----------------------
class ROAction(BaseModel):
    action: str

state = {"issue": None, "step_count": 0, "last_action": None}

# -----------------------
# Cost Map & Valid Actions
# -----------------------
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

# -----------------------
# Logging
# -----------------------
def log_start(task):
    print(f"[START] task={task} issue={state['issue']}")

def log_step(step, action, reward, done, reason, cost):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} cost={cost} done={str(done).lower()} reason={reason}")

def log_end(success, steps, rewards, total_cost):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str} total_cost={total_cost}")

# -----------------------
# Rule-Based Actions
# -----------------------
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

# -----------------------
# LLM Planner (Optional)
# -----------------------
# Uncomment if you want OpenAI fallback
"""
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
client = OpenAI(api_key=API_KEY)

def get_best_action(state, action_history, reward_memory):
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": f\"\"\"
Customer issue: {state['issue']}
Previous actions: {action_history}

Choose best action from:
{valid_actions}

Format:
action: <action>
reason: <short reason>
\"\"\"
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
"""

# -----------------------
# Avoid Bad Actions
# -----------------------
def avoid_bad_actions(action, reward_memory):
    if action in reward_memory and reward_memory[action] < 0:
        return "check_filter"
    return action

# -----------------------
# Environment API
# -----------------------
@app.post("/reset")
async def reset():
    global state
    state = {
        "issue": random.choice(["bad taste", "low pressure", "leakage", "no water"]),
        "step_count": 0,
        "last_action": None
    }
    return {"status": "ok", "state": state}

@app.post("/step")
async def step(action: ROAction):
    global state
    state["step_count"] += 1

    # Initialize logs and memory
    action_history = [state.get("last_action")] if state.get("last_action") else []
    reward_memory = {}

    # -----------------------
    # Rule-based first
    # -----------------------
    chosen_action = rule_based_action(state["issue"], action_history)
    if chosen_action:
        reason = "Rule-based decision"
    else:
        # Uncomment below for LLM fallback
        # chosen_action, reason = get_best_action(state, action_history, reward_memory)
        chosen_action = action.action
        reason = "Fallback / user action"

    # Avoid bad actions
    chosen_action = avoid_bad_actions(chosen_action, reward_memory)

    state["last_action"] = chosen_action

    # Reward calculation (example)
    reward = 1.0 if chosen_action != "check_filter" else 0.5
    reward_memory[chosen_action] = reward
    cost = cost_map.get(chosen_action, 0)
    done = state["step_count"] >= 6

    # Logging
    log_step(state["step_count"], chosen_action, reward, done, reason, cost)

    return {"state": state, "reward": reward, "done": done, "info": {}}

@app.get("/")
async def home():
    return {"message": "RO Support OpenEnv running!"}

# -----------------------
# Run locally
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
