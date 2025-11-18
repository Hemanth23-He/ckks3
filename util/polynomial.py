import numpy as np
from util.ntt import NTTContext, FFTContext

try:
    INT_TYPE = np.int128
except AttributeError:
    INT_TYPE = object  # Fallback for platforms without int128

class Polynomial:
    def __init__(self, degree, coeffs):
        self.ring_degree = degree
        arr = np.asarray(coeffs)
        # Cast to 128-bit integers if possible, otherwise arbitrary precision
        self.coeffs = arr.astype(INT_TYPE)

    def add(self, poly, coeff_modulus=None):
        assert isinstance(poly, Polynomial)
        poly_sum = Polynomial(self.ring_degree, np.add(self.coeffs, poly.coeffs))
        if coeff_modulus is not None:
            poly_sum = poly_sum.mod(coeff_modulus)
        return poly_sum

    def subtract(self, poly, coeff_modulus=None):
        assert isinstance(poly, Polynomial)
        poly_diff = Polynomial(self.ring_degree, np.subtract(self.coeffs, poly.coeffs))
        if coeff_modulus is not None:
            poly_diff = poly_diff.mod(coeff_modulus)
        return poly_diff

    def multiply(self, poly, coeff_modulus, ntt=None, crt=None):
        if crt:
            return self.multiply_crt(poly, crt)
        if ntt:
            a = ntt.ftt_fwd(self.coeffs)
            b = ntt.ftt_fwd(poly.coeffs)
            ab = np.multiply(a, b)
            prod = ntt.ftt_inv(ab)
            return Polynomial(self.ring_degree, prod)
        return self.multiply_naive(poly, coeff_modulus)

    def multiply_crt(self, poly, crt):
        assert isinstance(poly, Polynomial)
        poly_prods = [self.multiply(poly, crt.primes[i], ntt=crt.ntts[i]) for i in range(len(crt.primes))]
        stacked = np.stack([p.coeffs for p in poly_prods], axis=0)
        final_coeffs = np.array([crt.reconstruct(stacked[:, i]) for i in range(self.ring_degree)], dtype=INT_TYPE)
        return Polynomial(self.ring_degree, final_coeffs).mod_small(crt.modulus)

    def multiply_fft(self, poly, round=True):
        assert isinstance(poly, Polynomial)
        fft = FFTContext(self.ring_degree * 8)
        a = fft.fft_fwd(np.concatenate([self.coeffs, np.zeros(self.ring_degree, dtype=INT_TYPE)]))
        b = fft.fft_fwd(np.concatenate([poly.coeffs, np.zeros(self.ring_degree, dtype=INT_TYPE)]))
        ab = np.multiply(a, b)
        prod = fft.fft_inv(ab)
        poly_prod = np.zeros(self.ring_degree, dtype=INT_TYPE)
        for d in range(2 * self.ring_degree - 1):
            index = d % self.ring_degree
            sign = (int(d < self.ring_degree) - 0.5) * 2
            poly_prod[index] += sign * prod[d]
        if round:
            return Polynomial(self.ring_degree, np.round(poly_prod).astype(INT_TYPE))
        else:
            return Polynomial(self.ring_degree, poly_prod)

    def multiply_naive(self, poly, coeff_modulus=None):
        assert isinstance(poly, Polynomial)
        poly_prod = np.zeros(self.ring_degree, dtype=INT_TYPE)
        for d in range(2 * self.ring_degree - 1):
            index = d % self.ring_degree
            sign = int(d < self.ring_degree) * 2 - 1
            valid = np.arange(self.ring_degree)
            mask = (d - valid >= 0) & (d - valid < self.ring_degree)
            indices = valid[mask]
            coeff = np.sum(self.coeffs[indices] * poly.coeffs[d - indices])
            poly_prod[index] += sign * coeff
            if coeff_modulus is not None:
                poly_prod[index] %= coeff_modulus
        return Polynomial(self.ring_degree, poly_prod)

    def scalar_multiply(self, scalar, coeff_modulus=None):
        prod = self.coeffs * scalar
        if coeff_modulus is not None:
            prod = np.mod(prod, coeff_modulus)
        return Polynomial(self.ring_degree, prod.astype(INT_TYPE))

    def scalar_integer_divide(self, scalar, coeff_modulus=None):
        divided = self.coeffs // scalar
        if coeff_modulus is not None:
            divided = np.mod(divided, coeff_modulus)
        return Polynomial(self.ring_degree, divided.astype(INT_TYPE))

    def rotate(self, r):
        k = pow(5, r, 2 * self.ring_degree)
        new_coeffs = np.zeros(self.ring_degree, dtype=INT_TYPE)
        for i in range(self.ring_degree):
            index = (i * k) % (2 * self.ring_degree)
            if index < self.ring_degree:
                new_coeffs[index] = self.coeffs[i]
            else:
                new_coeffs[index - self.ring_degree] = -self.coeffs[i]
        return Polynomial(self.ring_degree, new_coeffs)

    def conjugate(self):
        new_coeffs = np.zeros(self.ring_degree, dtype=INT_TYPE)
        new_coeffs[0] = self.coeffs[0]
        new_coeffs[1:] = -self.coeffs[self.ring_degree - np.arange(1, self.ring_degree)]
        return Polynomial(self.ring_degree, new_coeffs)

    def round(self):
        new_coeffs = np.round(self.coeffs).astype(INT_TYPE)
        return Polynomial(self.ring_degree, new_coeffs)

    def floor(self):
        new_coeffs = np.floor(self.coeffs).astype(INT_TYPE)
        return Polynomial(self.ring_degree, new_coeffs)

    def mod(self, coeff_modulus):
        new_coeffs = np.mod(self.coeffs, coeff_modulus)
        return Polynomial(self.ring_degree, new_coeffs.astype(INT_TYPE))

    def mod_small(self, coeff_modulus):
        new_coeffs = np.mod(self.coeffs, coeff_modulus)
        half_mod = coeff_modulus // 2
        new_coeffs = np.where(new_coeffs > half_mod, new_coeffs - coeff_modulus, new_coeffs)
        return Polynomial(self.ring_degree, new_coeffs.astype(INT_TYPE))

    def base_decompose(self, base, num_levels):
        decomposed = []
        poly = self
        for _ in range(num_levels):
            part = poly.mod(base)
            decomposed.append(part)
            poly = poly.scalar_multiply(1 / base).floor()
        return decomposed

    def evaluate(self, val):
        result = self.coeffs[-1]
        for coeff in reversed(self.coeffs[:-1]):
            result = result * val + coeff
        return result

    def __str__(self):
        terms = []
        for i in range(self.ring_degree - 1, -1, -1):
            c = self.coeffs[i]
            if c != 0:
                term = ''
                if i == 0 or c != 1:
                    term += str(int(c))
                if i != 0:
                    term += 'x'
                if i > 1:
                    term += '^' + str(i)
                terms.append(term)
        return ' + '.join(terms)
