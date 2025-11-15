"""Test file for Python parsing"""

# Import statements
from typing import Dict, Any
import requests
from datetime import datetime

# Top-level variables
API_URL = 'https://api.example.com'
MAX_RETRIES = 3
config = {
    'timeout': 5000,
    'retries': MAX_RETRIES
}

# Type alias (using type hint)
UserRole = str  # 'admin' | 'user' | 'guest'

# Class
class UserService:
    """Service for managing users"""

    def __init__(self, base_url: str):
        """Initialize the service with a base URL"""
        self.base_url = base_url

    async def get_user(self, user_id: int) -> Dict[str, Any]:
        """Fetch a user by ID"""
        response = await fetch_data(f"{self.base_url}/users/{user_id}")
        return response

# Top-level function
def fetch_data(url: str) -> Any:
    """Fetch data from a URL"""
    return requests.get(url).json()

# Another function using arrow-like syntax (lambda)
process_user = lambda user: print(user['name'])
