from unittest import TestCase
from fastapi.testclient import TestClient
from users import app

class TestUsers(TestCase):
    def test_register_user(self):
        client = TestClient(app)
        response = client.post("/register", json={"username": "testuser", "password": "testpass"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "User registered successfully"})
