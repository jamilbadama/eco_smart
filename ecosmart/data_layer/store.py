import json
import os
from datetime import datetime

class SessionStore:
    def __init__(self, storage_path="sessions_db.json"):
        self.storage_path = storage_path
        self._db = self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {"sessions": {}}

    def save_session_result(self, session_id, result):
        self._db["sessions"][session_id] = {
            "timestamp": datetime.now().isoformat(),
            "data": result
        }
        with open(self.storage_path, 'w') as f:
            json.dump(self._db, f, indent=4)

    def get_session_result(self, session_id):
        return self._db["sessions"].get(session_id)

    def list_sessions(self):
        return list(self._db["sessions"].keys())
