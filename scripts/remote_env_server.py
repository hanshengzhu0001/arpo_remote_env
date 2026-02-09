#!/usr/bin/env python3
"""
Remote OSWorld env server (run on Mac or AWS CPU).
One env; exposes POST /env/reset, /env/step, /env/evaluate, /env/history_messages.
Cluster EnvWorkers call this over HTTP.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
osworld_root = repo_root / "OSWorld"
for p in (repo_root, osworld_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import base64
import os
import traceback
from io import BytesIO

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import docker
except ImportError:
    docker = None

from desktop_env.desktop_env import DesktopEnv
from verl.trainer.remote_env_protocol import messages_to_wire
from verl.trainer.gui_agent import (
    uitars_system_prompt,
    parse_action_to_structure_output,
    parsing_response_to_pyautogui_code,
    add_box_token,
    FINISH_WORD,
    WAIT_WORD,
    ENV_FAIL_WORD,
    CALL_USER,
)

# --- Server state: one env ---
env: DesktopEnv | None = None
history_messages: list = []
is_done = False
step_counter = 0
max_steps = 16
instruction: str | None = None

app = FastAPI(title="OSWorld Remote Env", version="0.1.0")


class ResetRequest(BaseModel):
    task_config: dict


class StepRequest(BaseModel):
    prediction: str


def _build_init_messages(screenshot_bytes: bytes, instruction_text: str) -> list:
    b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    return [
        {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": uitars_system_prompt.format(instruction=instruction_text)}]},
        {"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{b64}", "min_pixels": 3136, "max_pixels": 2116800}]},
    ]


def _get_env():
    global env
    if env is None:
        # Check KVM availability for logging
        kvm_available = os.path.exists("/dev/kvm")
        if kvm_available:
            print("✓ KVM detected: /dev/kvm exists - VM will use hardware acceleration")
        else:
            print("⚠ KVM not found: /dev/kvm does not exist - VM will use software emulation (slower)")
        
        try:
            env = DesktopEnv(
                provider_name="docker",
                action_space="pyautogui",
                screen_size=(1920, 1080),
                cache_dir="cache_dirs/cache_0",
                headless=True,
                os_type="Ubuntu",
                require_a11y_tree=False,
            )
            print(f"DesktopEnv initialized successfully (KVM: {kvm_available})")
        except HTTPException:
            raise
        except Exception as e:
            is_docker_error = (
                docker is not None
                and isinstance(e, docker.errors.DockerException)
            ) or "docker" in str(e).lower() or "connection aborted" in str(e).lower()
            if is_docker_error:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Docker is not available. Install Docker Desktop and start it, or ensure the Docker "
                        "daemon is running and the socket is available (e.g. /var/run/docker.sock). "
                        f"Original error: {e}"
                    ),
                ) from e
            raise
    return env


@app.post("/env/reset")
def env_reset(body: ResetRequest):
    global history_messages, is_done, step_counter, instruction
    task_config = body.task_config
    instruction = task_config.get("instruction", "")
    step_counter = 0
    is_done = False
    history_messages = []
    env = _get_env()

    trial = 0
    while trial < 8:
        try:
            obs = env.reset(task_config)
            break
        except Exception as e:
            print(f"Env reset exception: {e}")
            print(traceback.format_exc())
            trial += 1

    if trial >= 8:
        is_done = True
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": 0.0}

    env.pause()
    screenshot = obs.get("screenshot")
    if screenshot is None:
        is_done = True
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": 0.0}
    if isinstance(screenshot, bytes):
        pass
    else:
        from PIL import Image
        buf = BytesIO()
        Image.open(BytesIO(screenshot) if isinstance(screenshot, bytes) else screenshot).save(buf, format="JPEG")
        screenshot = buf.getvalue()

    history_messages = _build_init_messages(screenshot, instruction)
    return {
        "env_idx": 0,
        "obs_messages": messages_to_wire(history_messages),
        "is_done": False,
        "format_reward": 0.0,
    }


@app.post("/env/step")
def env_step(body: StepRequest):
    global history_messages, is_done, step_counter
    env = _get_env()
    prediction = body.prediction
    action_parse_res_factor = 1000
    model_type = "qwen25vl"
    max_pixels = 16384 * 28 * 28
    min_pixels = 100 * 28 * 28
    obs_image_height, obs_image_width = 1080, 1920

    try:
        parsed_responses = parse_action_to_structure_output(
            prediction, action_parse_res_factor, obs_image_height, obs_image_width, model_type, max_pixels, min_pixels
        )
        actions = []
        action_types = []
        for pr in parsed_responses:
            if "action_type" in pr:
                action_types.append(pr["action_type"])
                if pr["action_type"] == FINISH_WORD:
                    actions = ["DONE"]
                    break
                if pr["action_type"] in (WAIT_WORD, ENV_FAIL_WORD, CALL_USER):
                    actions = ["WAIT"] if pr["action_type"] == WAIT_WORD else ["FAIL"]
                    break
            code = parsing_response_to_pyautogui_code(pr, obs_image_height, obs_image_width, False)
            actions.append(code)
        
        # ARPO-style format_reward heuristics:
        # - Base reward for successful parsing (0.1)
        # - Bonus for meaningful actions (clicks, types, etc.) vs DONE/WAIT (0.05)
        # - Penalty for parse errors (-1.0)
        format_reward = 0.1  # Base reward for successful LLM action parsing
        
        # Check if we have meaningful GUI actions (not just DONE/WAIT/FAIL)
        if actions and actions[0] not in ["DONE", "WAIT", "FAIL"]:
            format_reward += 0.05  # Bonus for executable GUI actions
        
        # Check for FINISH_WORD (task completion signal from LLM)
        if FINISH_WORD in action_types:
            format_reward += 0.1  # Bonus for LLM indicating task completion
        
    except Exception:
        print("Parse action error:", prediction)
        print(traceback.format_exc())
        format_reward = -1.0
        actions = ["DONE"]

    env.unpause()
    obs = None
    step_successful = False
    for action in actions:
        obs, reward, step_done, info = env.step(action, pause=0.5)
        if step_done:
            is_done = True
        step_counter += 1
        if step_counter >= max_steps:
            is_done = True
        if is_done:
            break
        # Check if step executed successfully (obs is valid)
        if obs is not None and obs.get("screenshot") is not None:
            step_successful = True
    
    env.pause()
    
    # Enhance format_reward based on step execution success
    if step_successful:
        format_reward += 0.05  # Bonus for successful step execution (screenshot obtained)
    
    if obs is None and not actions:
        is_done = True
        format_reward = max(format_reward - 0.1, -1.0)  # Penalty for no observation

    history_messages.append({"role": "assistant", "content": [{"type": "text", "text": add_box_token(prediction)}]})

    if is_done:
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": format_reward}

    if obs is None or obs.get("screenshot") is None:
        is_done = True
        format_reward = max(format_reward - 0.1, -1.0)  # Penalty for missing screenshot
        return {"env_idx": 0, "obs_messages": None, "is_done": True, "format_reward": format_reward}

    screenshot = obs["screenshot"]
    if not isinstance(screenshot, bytes):
        from PIL import Image
        buf = BytesIO()
        Image.open(BytesIO(screenshot) if isinstance(screenshot, bytes) else screenshot).save(buf, format="JPEG")
        screenshot = buf.getvalue()
    b64 = base64.b64encode(screenshot).decode("utf-8")
    history_messages.append({
        "role": "user",
        "content": [{"type": "image", "image": f"data:image/jpeg;base64,{b64}", "min_pixels": 3136, "max_pixels": 2116800}],
    })

    return {
        "env_idx": 0,
        "obs_messages": messages_to_wire(history_messages),
        "is_done": False,
        "format_reward": format_reward,
    }


@app.post("/env/evaluate")
def env_evaluate():
    env = _get_env()
    try:
        env.unpause()
        score = env.evaluate()
        print(f"Evaluation completed: score={score}, instruction={instruction}, step_counter={step_counter}")
        return float(score)
    except Exception as e:
        print("Evaluation error:", e)
        print(traceback.format_exc())
        return 0.0


@app.post("/env/history_messages")
def env_history_messages():
    return {"history_messages": messages_to_wire(history_messages) if history_messages else []}


@app.get("/health")
def health():
    """Health check endpoint that also reports KVM status."""
    kvm_available = os.path.exists("/dev/kvm")
    env_status = "initialized" if env is not None else "not_initialized"
    return {
        "status": "ok",
        "kvm_available": kvm_available,
        "kvm_device": "/dev/kvm" if kvm_available else None,
        "env_status": env_status,
        "message": "KVM hardware acceleration enabled" if kvm_available else "KVM not available, using software emulation"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=15001)
