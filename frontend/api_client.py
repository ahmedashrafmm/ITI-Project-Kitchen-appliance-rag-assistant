import os
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def query_assistant(question: str, appliance_hint: str | None = None, k: int = 3, timeout: int = 60):
    payload = {"question": question, "k": k}
    if appliance_hint:
        payload["appliance_hint"] = appliance_hint
    response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def detect_appliance(image_file, timeout: int = 60):
    files = {"image": (image_file.name, image_file.getvalue())}
    response = requests.post(f"{API_BASE_URL}/detect-appliance", files=files, timeout=timeout)
    response.raise_for_status()
    return response.json()


def check_health(timeout: int = 5):
    response = requests.get(f"{API_BASE_URL}/health", timeout=timeout)
    response.raise_for_status()
    return response.json()
