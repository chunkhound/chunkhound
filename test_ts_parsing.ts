// Test file for TypeScript parsing

// Import statements
import { useState } from 'react';
import type { User } from './types';
import * as utils from './utils';

// Top-level variables
const API_URL = 'https://api.example.com';
const MAX_RETRIES = 3;
const config = {
  timeout: 5000,
  retries: MAX_RETRIES
};

// Type alias
type UserRole = 'admin' | 'user' | 'guest';

// Interface
interface ApiResponse<T> {
  data: T;
  error?: string;
}

// Enum
enum Status {
  Pending = 'pending',
  Success = 'success',
  Error = 'error'
}

// Top-level function
function fetchData(url: string): Promise<any> {
  return fetch(url).then(res => res.json());
}

// Arrow function assigned to const
const processUser = (user: User): void => {
  console.log(user.name);
};

// Class
class UserService {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async getUser(id: number): Promise<User> {
    const response = await fetchData(`${this.baseUrl}/users/${id}`);
    return response;
  }
}

// Export statement
export { UserService, processUser };
export default fetchData;
