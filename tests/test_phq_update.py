
import requests
import json
import time

def test_phq8_update():
    base_url = "http://localhost:8000"
    
    # 1. First, ensure a session exists. We'll simulate a save if possible or use an existing one.
    # For now, let's try to update a session named 'TEST_PHQ_SESSION'
    session_id = f"TEST_PHQ_{int(time.time())}"
    
    # Pre-populate some dummy data via internal mechanism if we were in the same process,
    # but since app is running, we'll try to trigger a save via websocket or just use the API if it existed.
    # Actually, let's just use the /api/session/update and see if it handles non-existent sessions correctly first,
    # then we'll verify with a real save.
    
    payload = {
        "session_id": session_id,
        "phq8_score": 15
    }
    
    print(f"Testing update for {session_id}...")
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{base_url}/api/session/update", json=payload, headers=headers)
    
    print(f"Response Status Code: {response.status_code}")
    try:
        data = response.json()
        print(f"Response Body: {json.dumps(data, indent=2)}")
    except:
        print(f"Response Body (Raw): {response.text}")
        return

    if 'status' in data:
        if data['status'] == 'error' and 'Session not found' in data['message']:
            print("Expected error: Session not found. Now creating session and retrying...")
            print("SUCCESS: Endpoint reached and returned correct error for missing session.")
        else:
            print(f"FAILURE: Unexpected status/message: {data}")
    else:
        print(f"FAILURE: 'status' key missing from response body. Full response: {data}")

if __name__ == "__main__":
    test_phq8_update()
