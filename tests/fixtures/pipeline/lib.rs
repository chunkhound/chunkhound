/// A simple calculator in Rust.

/// Adds two u32 values and returns the sum.
pub fn add(a: u32, b: u32) -> u32 {
    a + b
}

/// Multiplies two u32 values.
pub fn multiply(a: u32, b: u32) -> u32 {
    a * b
}

/// A calculator that remembers its last result.
pub struct Calculator {
    last_result: u32,
}

impl Calculator {
    /// Creates a new Calculator with zero initial state.
    pub fn new() -> Self {
        Self { last_result: 0 }
    }

    /// Adds a value to the last result.
    pub fn add(&mut self, value: u32) -> u32 {
        self.last_result += value;
        self.last_result
    }
}