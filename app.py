from flask import Flask, request, jsonify
import random
import json

app = Flask(__name__)

cost_map = {
    "check_filter": 0,
    "replace_filter": 500,
    "clean_membrane": 300,
    "replace_membrane": 1500,
    "fix_leak": 200,
    "check_pressure": 0,
    "check_power": 0
}

issues_db = {
    "low pressure": ["check_filter", "check_pressure", "clean_membrane"],
    "no water": ["check_power", "check_filter", "replace_membrane"],
    "leakage": ["fix_leak"],
    "bad taste": ["replace_filter", "clean_membrane"]
}

state = {}

@app.route("/reset", methods=["POST"])
def reset():
    issue = random.choice(list(issues_db.keys()))
    
    global state
    state = {
        "issue": issue,
        "steps": 0,
        "solution_path": issues_db[issue]
    }

    return jsonify({
        "state": state,
        "message": f"Issue: {issue}"
    })


@app.route("/step", methods=["POST"])
def step():
    action = request.json.get("action")
    correct_action = state["solution_path"][state["steps"]]

    state["steps"] += 1

    # ✅ FIXED REWARD SYSTEM (STRICTLY BETWEEN 0 AND 1)
    if action == correct_action:
        reward = 0.6
        msg = "Correct action"
    else:
        reward = 0.3
        msg = f"Wrong action. Expected: {correct_action}"

    done = state["steps"] >= len(state["solution_path"])

    if done:
        reward += 0.2  # max = 0.8

    # ✅ FINAL SAFETY CLAMP
    reward = max(0.01, min(0.99, reward))

    return jsonify({
        "reward": reward,
        "done": done,
        "message": msg,
        "state": state
    })


def save_customer(issue, actions):
    data = []
    try:
        with open("database.json", "r") as f:
            data = json.load(f)
    except:
        pass

    data.append({
        "issue": issue,
        "actions": actions
    })

    with open("database.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    app.run(port=8000)
