---
title: RO Support Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---
# RO Support AI Agent

## Overview

This project implements an intelligent AI agent for diagnosing and resolving RO water purifier issues using a hybrid approach (rule-based + LLM fallback).

## Features

* Fast decision-making (mostly 1-step solutions)
* Reward-based optimization
* Early stopping for efficiency
* Cost-aware actions

## Tasks Solved

* No Water
* Bad Taste
* Leakage

## How It Works

1. Environment provides issue
2. Agent selects best action
3. Executes action via API
4. Stops early when optimal result achieved

## Run

```bash
python inference.py
```

## Results

* Efficiency: ~1.0
* Minimal steps
* High reward, low cost
