import os

class AuthService:
    def __init__(self):
        self.api_key = os.getenv("ECOSMART_API_KEY", "ecosmart-secret-dev")

    def verify_key(self, key: str) -> bool:
        return key == self.api_key
