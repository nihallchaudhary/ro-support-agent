# inference.py
from fastapi import FastAPI
from pydantic import BaseModel
import random
import os
import time

# Optional: only if you want LLM fallback
try:
    from openai import OpenAI
    API_KEY = os.getenv("HF_TOKEN")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    LLM_ENABLED = True
except:
    LLM_ENABLED = False

app = FastAPI(title="RO Support OpenEnv")

# -----------------------
# Models
# -----------------------
class ROAction(BaseModel):
    action: str

# -----------------------
# Environment State
# -----------------------
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
    print(f"[START] task={task} env=ro_support_env")

def log_step(step, action, reward, done, reason, cost):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} cost={cost} done={str(done).lower()} reason={reason}")

def log_end(success, steps, rewards, total_cost):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str} total_cost={total_cost}")

# -----------------------
# Rule-Based Decisions
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
# LLM Fallback Planner
# -----------------------
def get_best_action(state_local, action_history, reward_memory):
    if not LLM_ENABLED:
        return random.choice(valid_actions), "LLM disabled fallback"

    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": f"""
Customer issue: {state_local['issue']}
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

# -----------------------
# Avoid Bad Actions
# -----------------------
def avoid_bad_actions(action, reward_memory):
    if action in reward_memory and reward_memory[action] < 0:
        return "check_filter"
    return action

# -----------------------
# API Endpoints
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
    action_history = [state.get("last_action")] if state.get("last_action") else []
    chosen_action = rule_based_action(state["issue"], action_history)
    if not chosen_action:
        chosen_action = action.action

    chosen_action = avoid_bad_actions(chosen_action, {})

    state["last_action"] = chosen_action
    reward = 1.0 if chosen_action != "check_filter" else 0.5
    done = state["step_count"] >= 6

    return {"state": state, "reward": reward, "done": done, "info": {}}

@app.post("/run_task")
async def run_task(task: str):
    log_start(task)
    global state
    state = {
        "issue": random.choice(["bad taste", "low pressure", "leakage", "no water"]),
        "step_count": 0,
        "last_action": None
    }

    rewards = []
    action_history = []
    reward_memory = {}
    total_cost = 0

    for step_num in range(1, 7):
        action = rule_based_action(state["issue"], action_history)
        if action:
            reason = "Rule-based decision"
        else:
            action, reason = get_best_action(state, action_history, reward_memory)

        action = avoid_bad_actions(action, reward_memory)
        action_history.append(action)
        cost = cost_map.get(action, 0)
        total_cost += cost

        # Step reward & done
        reward = 1.0 if action != "check_filter" else 0.5
        rewards.append(reward)
        reward_memory[action] = reward

        state["last_action"] = action
        state["step_count"] += 1
        done = state["step_count"] >= 6 or reward >= 5 or total_cost > 2000

        log_step(step_num, action, reward, done, reason, cost)
        if done:
            break

    success = sum(rewards) > 1.0
    log_end(success, step_num, rewards, total_cost)

    return {
        "state": state,
        "rewards": rewards,
        "total_cost": total_cost,
        "steps": step_num,
        "action_history": action_history
    }

@app.get("/")
async def home():
    return {"message": "RO Support OpenEnv running!"}

# -----------------------
# Run locally
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
