import requests


def chat():
    # This points to your deployed Docker container's exposed port
    url = "http://localhost:8000/webhook"
    print("Connected to Agent_Core (Docker). Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            break

        payload = {
            "chat_id": "cli_session_1",
            "user_id": "dev_user",
            "text": prompt
        }

        try:
            # Send the request to the backend
            response = requests.post(url, json=payload, timeout=120)
            data = response.json()

            if response.ok:
                message = data.get("reply") or data.get("response") or ""
                print(f"\nAgent: {message}\n")
            else:
                print(f"\nAgent error: {data.get('detail', response.text)}\n")

        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the agent. Is the Docker container running?")
        except requests.exceptions.Timeout:
            print("Error: The agent took too long to respond.")


if __name__ == "__main__":
    chat()
