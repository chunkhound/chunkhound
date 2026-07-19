"""Calculator module — provides arithmetic operations and a tax calculator."""


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def calculate_tax(income: float, rate: float) -> float:
    """Calculate tax based on income and rate."""
    if income <= 0:
        return 0.0
    return income * rate


class TaxCalculator:
    """Simple tax calculator with configurable default rate."""

    def __init__(self, default_rate: float = 0.25) -> None:
        self.default_rate = default_rate
        self.history: list[float] = []

    def compute_annual_tax(self, salary: float) -> float:
        """Compute annual tax from yearly salary."""
        tax = calculate_tax(salary, self.default_rate)
        self.history.append(tax)
        return tax

    def compute_monthly_tax(self, monthly_salary: float) -> float:
        """Compute monthly tax from monthly salary."""
        annual = monthly_salary * 12
        return self.compute_annual_tax(annual) / 12