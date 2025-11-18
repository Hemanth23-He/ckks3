class Polynomial:
    """
    Polynomial arithmetic in Z_q[x]/(x^N + 1) for arbitrary large modulus q.
    """
    def __init__(self, degree, coeffs):
        self.ring_degree = degree
        self.coeffs = [int(c) for c in coeffs]
        assert len(self.coeffs) == degree, f"Polynomial degree mismatch: {len(self.coeffs)} vs {degree}"

    def add(self, poly, coeff_modulus=None):
        assert isinstance(poly, Polynomial)
        result = [a + b for a, b in zip(self.coeffs, poly.coeffs)]
        if coeff_modulus is not None:
            result = [c % coeff_modulus for c in result]
        return Polynomial(self.ring_degree, result)

    def subtract(self, poly, coeff_modulus=None):
        assert isinstance(poly, Polynomial)
        result = [a - b for a, b in zip(self.coeffs, poly.coeffs)]
        if coeff_modulus is not None:
            result = [c % coeff_modulus for c in result]
        return Polynomial(self.ring_degree, result)

    def scalar_multiply(self, scalar, coeff_modulus=None):
        result = [c * scalar for c in self.coeffs]
        if coeff_modulus is not None:
            result = [c % coeff_modulus for c in result]
        return Polynomial(self.ring_degree, result)

    def mod(self, coeff_modulus):
        result = [c % coeff_modulus for c in self.coeffs]
        return Polynomial(self.ring_degree, result)

    def mul_naive(self, poly, coeff_modulus=None):
        # Schoolbook multiplication with reduction modulo x^N + 1
        result = [0] * self.ring_degree
        for i in range(self.ring_degree):
            for j in range(self.ring_degree):
                idx = (i + j) % self.ring_degree
                sign = 1 if i + j < self.ring_degree else -1
                result[idx] += sign * self.coeffs[i] * poly.coeffs[j]
        if coeff_modulus is not None:
            result = [c % coeff_modulus for c in result]
        return Polynomial(self.ring_degree, result)

    def __str__(self):
        terms = []
        for i in range(self.ring_degree - 1, -1, -1):
            coeff = self.coeffs[i]
            if coeff != 0:
                part = f"{coeff}" if i == 0 or coeff != 1 else ''
                part += '' if i == 0 else f"x^{i}" if i > 1 else 'x'
                terms.append(part)
        return " + ".join(terms) if terms else "0"
