"""
Chebyshev Ring — Exact Arithmetic in Q[s][u] / (u² - su + 1)
==============================================================

This module implements the algebraic number field arising from the
traction algebra's Chebyshev structure.

Generators:
    u = 0^(1/2)     — the half-power of traction zero
    v = ω^(1/2)     — the half-power of traction omega (= u⁻¹)
    s = u + v        — the Chebyshev parameter

Minimal polynomial:
    u² - su + 1 = 0   ⟹   u² = su - 1

Every element has canonical form  a + b·u  where a, b ∈ Q[s].

Arithmetic rules (derived from u² = su - 1):
    (a₁ + b₁u)(a₂ + b₂u) = (a₁a₂ - b₁b₂) + (a₁b₂ + b₁a₂ + b₁b₂s)u
    conj(a + bu) = (a + bs) - bu
    N(a + bu) = a² + abs + b²
    (a + bu)⁻¹ = conj(a + bu) / N(a + bu)

Connection to traction:
    0^n = u^(2n),   ω^n = v^(2n) = u^(-2n)
"""

from fractions import Fraction


# ============================================================
# GeneratorSpec — Pluggable base element specification
# ============================================================

class GeneratorSpec:
    """
    Specifies a base element for Chebyshev ring construction.

    A base element B has:
        - A name (for display)
        - An identity table: B^0, B^1, B^(-1), B^B
        - A sign convention: how B-exponents map to ring generator powers
          (0-base: 0^n → g^(+2n), ω-base: ω^n → g^(-2n) since ω = 0^(-1))

    The ring is always Q[s][g]/(g²-sg+1), s = g + g⁻¹.
    Only the interpretation of g and the exponent mapping change.
    """
    __slots__ = ['name', 'symbol', 'exponent_sign', 'omega_band', 'zero_band']

    def __init__(self, name, symbol, exponent_sign=1, omega_band=False, zero_band=False):
        """
        Args:
            name: Human-readable name ('zero', 'omega')
            symbol: Display symbol ('0', 'ω')
            exponent_sign: +1 for 0-base (0^n → g^{+scale·n}),
                          -1 for ω-base (ω^n → g^{-scale·n})
            omega_band: If True, this is an omega-channel generator
                       (exponents contain ω, θ-independent)
            zero_band: If True, this is a zero-channel generator
                      (exponents contain traction-zero; for n=2, g² = 1, hyperbolic)
        """
        if omega_band and zero_band:
            raise ValueError("A generator cannot be in both omega- and zero-bands")
        self.name = name
        self.symbol = symbol
        self.exponent_sign = exponent_sign
        self.omega_band = omega_band
        self.zero_band = zero_band

    def format_generator(self, scale):
        """Format the generator expression for display.
        scale is the denominator, so g = B^(1/scale) or g = B^(ω/scale)."""
        if self.omega_band:
            if scale == 1:
                return f'{self.symbol}^\u03c9'
            elif scale == 2:
                return f'{self.symbol}^(\u03c9/2)'
            else:
                return f'{self.symbol}^(\u03c9/{scale})'
        elif self.zero_band:
            if scale == 1:
                return f'{self.symbol}^(0/1)'
            elif scale == 2:
                return f'{self.symbol}^(0/2)'
            else:
                return f'{self.symbol}^(0/{scale})'
        else:
            if scale == 1:
                return self.symbol
            elif scale == 2:
                return f'{self.symbol}^(1/2)'
            else:
                return f'{self.symbol}^(1/{scale})'

    def format_s(self, scale):
        """Format the s parameter for display."""
        g = self.format_generator(scale)
        g_inv = self.format_generator(scale).replace('^(', '^(-').replace('^\u03c9', '^(-\u03c9)') \
            if '^(' in self.format_generator(scale) or '^\u03c9' == self.format_generator(scale)[-2:] \
            else f'{self.symbol}^(-1/{scale})'
        # Simpler: just show g + g⁻¹
        return f's = g + g\u207b\u00b9'

    def __repr__(self):
        return f'GeneratorSpec({self.name!r}, {self.symbol!r}, sign={self.exponent_sign})'


# Predefined generator specs
ZERO_SPEC = GeneratorSpec('zero', '0', exponent_sign=+1, omega_band=False)
OMEGA_SPEC = GeneratorSpec('omega', '\u03c9', exponent_sign=-1, omega_band=False)
ZERO_OMEGA_SPEC = GeneratorSpec('zero-omega', '0', exponent_sign=+1, omega_band=True)
OMEGA_OMEGA_SPEC = GeneratorSpec('omega-omega', '\u03c9', exponent_sign=-1, omega_band=True)
ZERO_ZERO_SPEC = GeneratorSpec('zero-zero', '0', exponent_sign=+1, zero_band=True)
OMEGA_ZERO_SPEC = GeneratorSpec('omega-zero', '\u03c9', exponent_sign=-1, zero_band=True)


# ============================================================
# QsPoly — Polynomials in Q[s]
# ============================================================

class QsPoly:
    """
    A polynomial in s with rational coefficients.
    Stored as a tuple of Fraction in ascending degree order:
    coeffs[i] is the coefficient of s^i.
    """
    __slots__ = ['coeffs']

    def __init__(self, coeffs):
        if isinstance(coeffs, QsPoly):
            self.coeffs = coeffs.coeffs
            return
        if isinstance(coeffs, (int, Fraction)):
            self.coeffs = (Fraction(coeffs),)
            return
        raw = tuple(Fraction(c) for c in coeffs)
        # Strip trailing zeros
        while len(raw) > 1 and raw[-1] == 0:
            raw = raw[:-1]
        self.coeffs = raw

    # --- Constructors ---

    @staticmethod
    def zero():
        return QsPoly((0,))

    @staticmethod
    def one():
        return QsPoly((1,))

    @staticmethod
    def s():
        """The indeterminate s itself."""
        return QsPoly((0, 1))

    @staticmethod
    def from_int(n):
        return QsPoly((n,))

    @staticmethod
    def from_fraction(f):
        return QsPoly((Fraction(f),))

    # --- Queries ---

    def degree(self):
        if self.is_zero():
            return -1
        return len(self.coeffs) - 1

    def is_zero(self):
        return all(c == 0 for c in self.coeffs)

    def lead(self):
        """Leading coefficient."""
        return self.coeffs[-1]

    def is_constant(self):
        return self.degree() <= 0

    def constant_value(self):
        """Return the constant term as a Fraction."""
        return self.coeffs[0]

    # --- Arithmetic ---

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            other = QsPoly(other)
        a, b = self.coeffs, other.coeffs
        n = max(len(a), len(b))
        result = []
        for i in range(n):
            ca = a[i] if i < len(a) else Fraction(0)
            cb = b[i] if i < len(b) else Fraction(0)
            result.append(ca + cb)
        return QsPoly(result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            other = QsPoly(other)
        a, b = self.coeffs, other.coeffs
        n = max(len(a), len(b))
        result = []
        for i in range(n):
            ca = a[i] if i < len(a) else Fraction(0)
            cb = b[i] if i < len(b) else Fraction(0)
            result.append(ca - cb)
        return QsPoly(result)

    def __rsub__(self, other):
        return QsPoly(other).__sub__(self)

    def __neg__(self):
        return QsPoly(tuple(-c for c in self.coeffs))

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            return self.scale(Fraction(other))
        if isinstance(other, QsPoly):
            a, b = self.coeffs, other.coeffs
            if self.is_zero() or other.is_zero():
                return QsPoly.zero()
            n = len(a) + len(b) - 1
            result = [Fraction(0)] * n
            for i, ca in enumerate(a):
                if ca == 0:
                    continue
                for j, cb in enumerate(b):
                    result[i + j] += ca * cb
            return QsPoly(result)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, Fraction)):
            return self.scale(Fraction(other))
        return NotImplemented

    def scale(self, c):
        """Multiply all coefficients by scalar c."""
        c = Fraction(c)
        return QsPoly(tuple(ci * c for ci in self.coeffs))

    def shift(self, n=1):
        """Multiply by s^n (prepend n zeros)."""
        if self.is_zero():
            return QsPoly.zero()
        return QsPoly((Fraction(0),) * n + self.coeffs)

    @staticmethod
    def divmod(a, b):
        """Polynomial long division: returns (quotient, remainder) such that a = b*q + r."""
        if b.is_zero():
            raise ZeroDivisionError("Polynomial division by zero")
        if a.degree() < b.degree():
            return QsPoly.zero(), QsPoly(a.coeffs)

        a_coeffs = list(a.coeffs)
        b_lead = b.lead()
        b_deg = b.degree()
        q_coeffs = [Fraction(0)] * (len(a_coeffs) - b_deg)

        for i in range(len(q_coeffs) - 1, -1, -1):
            q_coeffs[i] = a_coeffs[i + b_deg] / b_lead
            for j in range(b_deg + 1):
                a_coeffs[i + j] -= q_coeffs[i] * b.coeffs[j]

        return QsPoly(q_coeffs), QsPoly(a_coeffs[:b_deg])

    # --- Evaluation ---

    def eval_at(self, s_val):
        """Evaluate at a numeric value of s. Returns complex."""
        s_val = complex(s_val)
        result = complex(0)
        s_pow = complex(1)
        for c in self.coeffs:
            result += complex(c) * s_pow
            s_pow *= s_val
        return result

    # --- Comparison ---

    def __eq__(self, other):
        if isinstance(other, (int, Fraction)):
            other = QsPoly(other)
        if not isinstance(other, QsPoly):
            return NotImplemented
        return self.coeffs == other.coeffs

    def __hash__(self):
        return hash(self.coeffs)

    # --- Display ---

    def __repr__(self):
        if self.is_zero():
            return '0'
        terms = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(str(c))
            elif i == 1:
                if c == 1:
                    terms.append('s')
                elif c == -1:
                    terms.append('-s')
                else:
                    terms.append(f'{c}*s')
            else:
                if c == 1:
                    terms.append(f's^{i}')
                elif c == -1:
                    terms.append(f'-s^{i}')
                else:
                    terms.append(f'{c}*s^{i}')
        if not terms:
            return '0'
        result = terms[0]
        for t in terms[1:]:
            if t.startswith('-'):
                result += f' - {t[1:]}'
            else:
                result += f' + {t}'
        return result


# ============================================================
# Element — Q[s][u] / (u² - su + 1)
# ============================================================

class Element:
    """
    An element of the quotient ring Q[s][u] / (u² - su + 1).
    Canonical form: a + b*u where a, b are QsPoly instances.
    """
    __slots__ = ['a', 'b']

    def __init__(self, a, b=None):
        if b is None:
            b = QsPoly.zero()
        self.a = a if isinstance(a, QsPoly) else QsPoly(a)
        self.b = b if isinstance(b, QsPoly) else QsPoly(b)

    # --- Constructors ---

    @staticmethod
    def one():
        return Element(QsPoly.one(), QsPoly.zero())

    @staticmethod
    def zero_el():
        """The additive identity (not traction zero!)."""
        return Element(QsPoly.zero(), QsPoly.zero())

    @staticmethod
    def u():
        """The generator u = 0^(1/2)."""
        return Element(QsPoly.zero(), QsPoly.one())

    @staticmethod
    def v():
        """v = u⁻¹ = s - u."""
        return Element(QsPoly.s(), QsPoly.from_int(-1))

    @staticmethod
    def from_int(n):
        return Element(QsPoly.from_int(n))

    @staticmethod
    def from_fraction(f):
        return Element(QsPoly.from_fraction(f))

    @staticmethod
    def u_power(n):
        """Compute u^n for any integer n."""
        if n == 0:
            return Element.one()
        if n > 0:
            return Element.u() ** n
        return Element.v() ** (-n)

    @staticmethod
    def from_traction_exp(n):
        """
        Convert 0^n to a ring element.
        n may be an int, Fraction, or float (converted to Fraction).
        0^n = u^(2n), so 2n must be an integer.
        """
        if isinstance(n, float):
            n = Fraction(n).limit_denominator(1000)
        n = Fraction(n)
        two_n = n * 2
        if two_n.denominator != 1:
            raise ValueError(
                f"Exponent {n} yields u^({two_n}), which is not an integer power. "
                f"The ring Q[s][u]/(u²-su+1) requires half-integer traction exponents."
            )
        return Element.u_power(int(two_n))

    # --- Arithmetic ---

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Element.from_int(other) if isinstance(other, int) else Element.from_fraction(other)
        return Element(self.a + other.a, self.b + other.b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Element.from_int(other) if isinstance(other, int) else Element.from_fraction(other)
        return Element(self.a - other.a, self.b - other.b)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return Element(-self.a, -self.b)

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(other)
            return Element(self.a * c, self.b * c)
        if not isinstance(other, Element):
            return NotImplemented
        # (a1 + b1*u)(a2 + b2*u)
        # = a1*a2 + a1*b2*u + b1*a2*u + b1*b2*u²
        # = a1*a2 + (a1*b2 + b1*a2)*u + b1*b2*(su - 1)
        # = (a1*a2 - b1*b2) + (a1*b2 + b1*a2 + b1*b2*s)*u
        a1, b1 = self.a, self.b
        a2, b2 = other.a, other.b
        b1b2 = b1 * b2
        new_a = a1 * a2 - b1b2
        new_b = a1 * b2 + b1 * a2 + b1b2 * QsPoly.s()
        return Element(new_a, new_b)

    def __rmul__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(other)
            return Element(self.a * c, self.b * c)
        return NotImplemented

    def conj(self):
        """
        Conjugation: replace u with v = s - u.
        conj(a + bu) = a + b(s - u) = (a + bs) - bu
        """
        return Element(self.a + self.b * QsPoly.s(), -self.b)

    def norm(self):
        """
        Norm: N(a + bu) = (a + bu)(conj(a + bu)) = a² + abs + b²
        Always lands in Q[s].
        """
        a, b = self.a, self.b
        return a * a + a * b * QsPoly.s() + b * b

    def can_invert(self):
        """Check if this element is invertible (norm is a nonzero constant)."""
        n = self.norm()
        return n.is_constant() and not n.is_zero()

    def inv(self):
        """
        Inverse: (a + bu)⁻¹ = conj(a + bu) / N(a + bu).
        Only works when the norm is a nonzero constant.
        """
        n = self.norm()
        if n.is_zero():
            raise ZeroDivisionError("Cannot invert the zero element")
        if not n.is_constant():
            raise ValueError(
                f"Element is not a unit: norm = {n} is not a constant. "
                f"Inversion requires working in the fraction field."
            )
        c = n.constant_value()
        c_inv = Fraction(1, 1) / c
        conj = self.conj()
        return Element(conj.a * c_inv, conj.b * c_inv)

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(1) / Fraction(other)
            return Element(self.a * c, self.b * c)
        if not isinstance(other, Element):
            return NotImplemented
        return self * other.inv()

    def __pow__(self, n):
        """Exponentiation by squaring. Handles negative n via inv()."""
        if not isinstance(n, int):
            raise TypeError(f"Exponent must be an integer, got {type(n)}")
        if n == 0:
            return Element.one()
        if n < 0:
            return self.inv() ** (-n)
        # Binary exponentiation
        result = Element.one()
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    # --- Queries ---

    def is_zero(self):
        return self.a.is_zero() and self.b.is_zero()

    def is_scalar(self):
        """True if this element is in Q[s] (no u component)."""
        return self.b.is_zero()

    def __eq__(self, other):
        if isinstance(other, (int, Fraction)):
            other = Element.from_int(other) if isinstance(other, int) else Element.from_fraction(other)
        if not isinstance(other, Element):
            return NotImplemented
        return self.a == other.a and self.b == other.b

    def __hash__(self):
        return hash((self.a.coeffs, self.b.coeffs))

    # --- Evaluation ---

    def eval_at(self, s_val=None, theta=None):
        """
        Evaluate numerically.
        If theta given: s = 2cos(θ), u = e^{iθ}
        If s_val given: u = (s + √(s²-4))/2
        """
        import cmath, math
        if theta is not None:
            s_num = 2 * math.cos(theta)
            u_num = cmath.exp(1j * theta)
        elif s_val is not None:
            s_num = complex(s_val)
            disc = s_num ** 2 - 4
            u_num = (s_num + cmath.sqrt(disc)) / 2
        else:
            raise ValueError("Provide s_val or theta")

        a_num = self.a.eval_at(s_num)
        b_num = self.b.eval_at(s_num)
        return a_num + b_num * u_num

    # --- Display ---

    def __repr__(self):
        a_zero = self.a.is_zero()
        b_zero = self.b.is_zero()

        if a_zero and b_zero:
            return '0'

        a_str = repr(self.a)
        b_str = repr(self.b)

        if b_zero:
            return a_str

        # Format the u-coefficient part
        if self.b == QsPoly.one():
            u_part = 'u'
        elif self.b == QsPoly.from_int(-1):
            u_part = '-u'
        elif self.b.is_constant():
            u_part = f'{b_str}*u'
        else:
            u_part = f'({b_str})*u'

        if a_zero:
            return u_part

        # Combine a + b*u
        if u_part.startswith('-'):
            return f'{a_str} - {u_part[1:]}'
        return f'{a_str} + {u_part}'


# ============================================================
# TowerElement — Level 2: Q[s][t,w] / (t²=s+2, w²=tw-1)
# ============================================================

# Precomputed: S2 = s + 2 as a QsPoly
_S2 = QsPoly((2, 1))  # 2 + s


class TowerElement:
    """
    An element of the Level 2 tower ring:
        Q[s][t, w] / (t² - s - 2,  w² - tw + 1)

    Canonical form: a + b·t + c·w + d·tw
    where a, b, c, d ∈ Q[s] (QsPoly instances).

    Basis: {1, t, w, tw}
    Reduction rules: t² = s+2,  w² = tw-1
    Minimal polynomial of w over Q[s]: w⁴ - sw² + 1 = 0

    Generators:
        w = 0^(1/4)      — quarter-power of traction zero
        t = w + w⁻¹      — Level 2 Chebyshev parameter
        u = w² = tw - 1  — Level 1 generator (half-power)
        s = t² - 2       — Level 1 Chebyshev parameter
    """
    __slots__ = ['a', 'b', 'c', 'd']

    def __init__(self, a, b=None, c=None, d=None):
        self.a = a if isinstance(a, QsPoly) else QsPoly(a)
        self.b = b if isinstance(b, QsPoly) else QsPoly(b or 0)
        self.c = c if isinstance(c, QsPoly) else QsPoly(c or 0)
        self.d = d if isinstance(d, QsPoly) else QsPoly(d or 0)

    # --- Constructors ---

    @staticmethod
    def one():
        z = QsPoly.zero()
        return TowerElement(QsPoly.one(), z, z, z)

    @staticmethod
    def zero_el():
        z = QsPoly.zero()
        return TowerElement(z, z, z, z)

    @staticmethod
    def t():
        """The generator t = w + w⁻¹ (Level 2 Chebyshev parameter)."""
        z = QsPoly.zero()
        return TowerElement(z, QsPoly.one(), z, z)

    @staticmethod
    def w():
        """The generator w = 0^(1/4)."""
        z = QsPoly.zero()
        return TowerElement(z, z, QsPoly.one(), z)

    @staticmethod
    def w_inv():
        """w⁻¹ = t - w."""
        z = QsPoly.zero()
        return TowerElement(z, QsPoly.one(), QsPoly.from_int(-1), z)

    @staticmethod
    def from_int(n):
        z = QsPoly.zero()
        return TowerElement(QsPoly.from_int(n), z, z, z)

    @staticmethod
    def from_element(el):
        """Lift a Level 1 Element (a + bu) into Level 2.
        u = w² = tw - 1, so a + bu = a + b(tw - 1) = (a - b) + 0·t + 0·w + b·tw."""
        return TowerElement(el.a - el.b, QsPoly.zero(), QsPoly.zero(), el.b)

    @staticmethod
    def w_power(n):
        """Compute w^n for any integer n."""
        if n == 0:
            return TowerElement.one()
        if n > 0:
            return TowerElement.w() ** n
        return TowerElement.w_inv() ** (-n)

    @staticmethod
    def from_traction_exp(n):
        """
        Convert 0^n to a Level 2 ring element.
        0^n = w^(4n), so 4n must be an integer.
        Supports integer, half-integer, and quarter-integer traction exponents.
        """
        if isinstance(n, float):
            n = Fraction(n).limit_denominator(1000)
        n = Fraction(n)
        four_n = n * 4
        if four_n.denominator != 1:
            raise ValueError(
                f"Exponent {n} yields w^({four_n}), which is not an integer power. "
                f"The Level 2 ring requires quarter-integer traction exponents."
            )
        return TowerElement.w_power(int(four_n))

    # --- Arithmetic ---

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            other = TowerElement.from_int(other) if isinstance(other, int) else TowerElement(QsPoly.from_fraction(other))
        return TowerElement(self.a + other.a, self.b + other.b,
                            self.c + other.c, self.d + other.d)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            other = TowerElement.from_int(other) if isinstance(other, int) else TowerElement(QsPoly.from_fraction(other))
        return TowerElement(self.a - other.a, self.b - other.b,
                            self.c - other.c, self.d - other.d)

    def __neg__(self):
        return TowerElement(-self.a, -self.b, -self.c, -self.d)

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(other)
            return TowerElement(self.a * c, self.b * c, self.c * c, self.d * c)
        if not isinstance(other, TowerElement):
            return NotImplemented

        # (a1 + b1·t + c1·w + d1·tw)(a2 + b2·t + c2·w + d2·tw)
        # Reduction: t² = s+2 (=S2), w² = tw-1
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d

        new_a = a1*a2 + _S2*(b1*b2 - d1*d2) - c1*c2
        new_b = a1*b2 + b1*a2 - c1*d2 - d1*c2
        new_c = a1*c2 + c1*a2 + _S2*(b1*d2 + d1*b2 + c1*d2 + d1*c2)
        new_d = a1*d2 + d1*a2 + b1*c2 + c1*b2 + c1*c2 + _S2*d1*d2

        return TowerElement(new_a, new_b, new_c, new_d)

    def __rmul__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(other)
            return TowerElement(self.a * c, self.b * c, self.c * c, self.d * c)
        return NotImplemented

    # --- Conjugations ---

    def sigma(self):
        """σ-conjugation: w → w⁻¹ = t - w (fixes t).
        σ(a + bt + cw + dtw) = (a + d·S2) + (b+c)t + (-c)w + (-d)tw"""
        return TowerElement(
            self.a + self.d * _S2,
            self.b + self.c,
            -self.c,
            -self.d
        )

    def tau(self):
        """τ-conjugation: w → -w, t → -t.
        τ(a + bt + cw + dtw) = a - bt - cw + dtw"""
        return TowerElement(self.a, -self.b, -self.c, self.d)

    def sigma_tau(self):
        """στ-conjugation: w → -(t - w) = w - t, t → -t."""
        return self.tau().sigma()

    # --- Norms ---

    def half_norm(self):
        """N₂(x) = x · σ(x). Result has c=d=0 (lives in Q[s][t]/(t²=s+2))."""
        return self * self.sigma()

    def full_norm(self):
        """N₄(x) = x · σ(x) · τ(x) · στ(x). Result is in Q[s] (a scalar).
        Computed as N₂(half_norm)."""
        hn = self.half_norm()
        # hn should have c=d=0. Its τ-conjugate negates b.
        # N₄ = hn * τ(hn) = (a + bt)(a - bt) = a² - b²t² = a² - b²(s+2)
        return hn.a * hn.a - hn.b * hn.b * _S2

    def can_invert(self):
        """Check if this element is invertible (full norm is a nonzero constant)."""
        n = self.full_norm()
        return n.is_constant() and not n.is_zero()

    def inv(self):
        """
        Inverse: x⁻¹ = σ(x)·τ(x)·στ(x) / N₄(x).
        Only works when the full norm is a nonzero constant.
        """
        fn = self.full_norm()
        if fn.is_zero():
            raise ZeroDivisionError("Cannot invert the zero element")
        if not fn.is_constant():
            raise ValueError(
                f"Element is not a unit: full norm = {fn} is not a constant. "
                f"Inversion requires working in the fraction field."
            )
        # x⁻¹ = σ(x) · τ(x) · στ(x) / N₄
        c_inv = Fraction(1) / fn.constant_value()
        numerator = self.sigma() * self.tau() * self.sigma_tau()
        return TowerElement(numerator.a * c_inv, numerator.b * c_inv,
                            numerator.c * c_inv, numerator.d * c_inv)

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(1) / Fraction(other)
            return TowerElement(self.a * c, self.b * c, self.c * c, self.d * c)
        if not isinstance(other, TowerElement):
            return NotImplemented
        return self * other.inv()

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError(f"Exponent must be an integer, got {type(n)}")
        if n == 0:
            return TowerElement.one()
        if n < 0:
            return self.inv() ** (-n)
        result = TowerElement.one()
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    # --- Queries ---

    def is_zero(self):
        return self.a.is_zero() and self.b.is_zero() and self.c.is_zero() and self.d.is_zero()

    def is_scalar(self):
        """True if in Q[s] (no t, w, or tw components)."""
        return self.b.is_zero() and self.c.is_zero() and self.d.is_zero()

    def __eq__(self, other):
        if isinstance(other, (int, Fraction)):
            other = TowerElement.from_int(other) if isinstance(other, int) else TowerElement(QsPoly.from_fraction(other))
        if not isinstance(other, TowerElement):
            return NotImplemented
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    def __hash__(self):
        return hash((self.a.coeffs, self.b.coeffs, self.c.coeffs, self.d.coeffs))

    # --- Evaluation ---

    def eval_at(self, s_val=None, theta=None):
        """
        Evaluate numerically.
        If theta given: s = 2cos(2θ), t = 2cos(θ), w = e^{iθ/2}...
        Actually: w = 0^(1/4) = e^{iθ} at Level 1's θ corresponds to
        w being at half that angle. Let's use theta as the Level 2 angle:
        w = e^{iθ}, t = 2cos(θ), s = t² - 2 = 2cos(2θ).
        """
        import cmath, math
        if theta is not None:
            s_num = 2 * math.cos(2 * theta)
            t_num = 2 * math.cos(theta)
            w_num = cmath.exp(1j * theta)
        elif s_val is not None:
            s_num = complex(s_val)
            t_num = (s_num + 2) ** 0.5  # positive root
            disc = t_num ** 2 - 4
            w_num = (t_num + cmath.sqrt(disc)) / 2
        else:
            raise ValueError("Provide s_val or theta")

        a_num = self.a.eval_at(s_num)
        b_num = self.b.eval_at(s_num)
        c_num = self.c.eval_at(s_num)
        d_num = self.d.eval_at(s_num)
        return a_num + b_num * t_num + c_num * w_num + d_num * t_num * w_num

    # --- Display ---

    def __repr__(self):
        parts = []
        for coeff, basis_name in [(self.a, ''), (self.b, 't'), (self.c, 'w'), (self.d, 'tw')]:
            if coeff.is_zero():
                continue
            c_str = repr(coeff)
            if not basis_name:
                parts.append(c_str)
            elif coeff == QsPoly.one():
                parts.append(basis_name)
            elif coeff == QsPoly.from_int(-1):
                parts.append(f'-{basis_name}')
            elif coeff.is_constant():
                parts.append(f'{c_str}*{basis_name}')
            else:
                parts.append(f'({c_str})*{basis_name}')

        if not parts:
            return '0'

        result = parts[0]
        for p in parts[1:]:
            if p.startswith('-'):
                result += f' - {p[1:]}'
            else:
                result += f' + {p}'
        return result


# ============================================================
# BiPoly — Bivariate Polynomials in Q[s₁, s₂]
# ============================================================

class BiPoly:
    """
    A bivariate polynomial in s₁ and s₂ with rational coefficients.
    Stored as a tuple of QsPoly: coeffs[j] is the coefficient of s₂^j,
    where each coefficient is a QsPoly in s₁.
    """
    __slots__ = ['coeffs']

    def __init__(self, coeffs):
        if isinstance(coeffs, BiPoly):
            self.coeffs = coeffs.coeffs
            return
        if isinstance(coeffs, (int, Fraction)):
            self.coeffs = (QsPoly(coeffs),)
            return
        if isinstance(coeffs, QsPoly):
            self.coeffs = (coeffs,)
            return
        raw = tuple(c if isinstance(c, QsPoly) else QsPoly(c) for c in coeffs)
        while len(raw) > 1 and raw[-1].is_zero():
            raw = raw[:-1]
        self.coeffs = raw

    @staticmethod
    def zero():
        return BiPoly(0)

    @staticmethod
    def one():
        return BiPoly(1)

    @staticmethod
    def s1():
        """The indeterminate s₁."""
        return BiPoly(QsPoly.s())

    @staticmethod
    def s2():
        """The indeterminate s₂."""
        return BiPoly((QsPoly.zero(), QsPoly.one()))

    @staticmethod
    def from_fraction(f):
        return BiPoly(QsPoly.from_fraction(f))

    def is_zero(self):
        return all(c.is_zero() for c in self.coeffs)

    def is_constant(self):
        if len(self.coeffs) > 1:
            for c in self.coeffs[1:]:
                if not c.is_zero():
                    return False
        return self.coeffs[0].is_constant()

    def constant_value(self):
        return self.coeffs[0].constant_value()

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            other = BiPoly(other)
        if isinstance(other, QsPoly):
            other = BiPoly(other)
        a, b = self.coeffs, other.coeffs
        n = max(len(a), len(b))
        result = []
        for j in range(n):
            ca = a[j] if j < len(a) else QsPoly.zero()
            cb = b[j] if j < len(b) else QsPoly.zero()
            result.append(ca + cb)
        return BiPoly(result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            other = BiPoly(other)
        if isinstance(other, QsPoly):
            other = BiPoly(other)
        a, b = self.coeffs, other.coeffs
        n = max(len(a), len(b))
        result = []
        for j in range(n):
            ca = a[j] if j < len(a) else QsPoly.zero()
            cb = b[j] if j < len(b) else QsPoly.zero()
            result.append(ca - cb)
        return BiPoly(result)

    def __neg__(self):
        return BiPoly(tuple(-c for c in self.coeffs))

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(other)
            return BiPoly(tuple(coeff * c for coeff in self.coeffs))
        if isinstance(other, QsPoly):
            return BiPoly(tuple(coeff * other for coeff in self.coeffs))
        if isinstance(other, BiPoly):
            if self.is_zero() or other.is_zero():
                return BiPoly.zero()
            a, b = self.coeffs, other.coeffs
            n = len(a) + len(b) - 1
            result = [QsPoly.zero()] * n
            for i, ca in enumerate(a):
                if ca.is_zero():
                    continue
                for j, cb in enumerate(b):
                    result[i + j] = result[i + j] + ca * cb
            return BiPoly(result)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, Fraction)):
            return self.__mul__(other)
        if isinstance(other, QsPoly):
            return self.__mul__(other)
        return NotImplemented

    def eval_at(self, s1_val, s2_val):
        """Evaluate at numeric values of s₁ and s₂."""
        s2_val = complex(s2_val)
        result = complex(0)
        s2_pow = complex(1)
        for c in self.coeffs:
            result += c.eval_at(s1_val) * s2_pow
            s2_pow *= s2_val
        return result

    def __eq__(self, other):
        if isinstance(other, (int, Fraction)):
            other = BiPoly(other)
        if isinstance(other, QsPoly):
            other = BiPoly(other)
        if not isinstance(other, BiPoly):
            return NotImplemented
        # Normalize and compare
        a = self.coeffs
        b = other.coeffs
        n = max(len(a), len(b))
        for j in range(n):
            ca = a[j] if j < len(a) else QsPoly.zero()
            cb = b[j] if j < len(b) else QsPoly.zero()
            if ca != cb:
                return False
        return True

    def __hash__(self):
        return hash(tuple(c.coeffs for c in self.coeffs))

    def __repr__(self):
        if self.is_zero():
            return '0'
        terms = []
        for j, coeff in enumerate(self.coeffs):
            if coeff.is_zero():
                continue
            c_str = repr(coeff)
            if j == 0:
                terms.append(c_str)
            elif j == 1:
                if coeff == QsPoly.one():
                    terms.append('s\u2082')
                elif coeff == QsPoly.from_int(-1):
                    terms.append('-s\u2082')
                elif coeff.is_constant():
                    terms.append(f'{c_str}\u00b7s\u2082')
                else:
                    terms.append(f'({c_str})\u00b7s\u2082')
            else:
                if coeff == QsPoly.one():
                    terms.append(f's\u2082^{j}')
                elif coeff == QsPoly.from_int(-1):
                    terms.append(f'-s\u2082^{j}')
                elif coeff.is_constant():
                    terms.append(f'{c_str}\u00b7s\u2082^{j}')
                else:
                    terms.append(f'({c_str})\u00b7s\u2082^{j}')
        if not terms:
            return '0'
        result = terms[0]
        for t in terms[1:]:
            if t.startswith('-'):
                result += f' - {t[1:]}'
            else:
                result += f' + {t}'
        # Rename 's' to 's₁' for clarity in bivariate context
        import re
        result = re.sub(r'\bs\b', 's\u2081', result)
        return result


# ============================================================
# MultiBandElement — Q[s₁,s₂][g₁,g₂] / (g₁²=s₁g₁-1, g₂²=s₂g₂-1)
# ============================================================

class MultiBandElement:
    """
    An element of a two-generator Chebyshev ring with independent bands.

    Generators:
        g₁ — first band (e.g., 0^(1/2) for rational exponents)
        g₂ — second band (e.g., 0^(ω/7) for omega-scaled exponents)

    Reduction rules:
        g₁² = s₁·g₁ - 1    where s₁ = g₁ + g₁⁻¹
        g₂² = s₂·g₂ - 1    where s₂ = g₂ + g₂⁻¹

    Canonical form: a + b·g₁ + c·g₂ + d·g₁g₂
    where a, b, c, d ∈ Q[s₁, s₂] (BiPoly instances).
    """
    __slots__ = ['a', 'b', 'c', 'd']

    def __init__(self, a, b=None, c=None, d=None):
        self.a = a if isinstance(a, BiPoly) else BiPoly(a)
        self.b = b if isinstance(b, BiPoly) else BiPoly(b or 0)
        self.c = c if isinstance(c, BiPoly) else BiPoly(c or 0)
        self.d = d if isinstance(d, BiPoly) else BiPoly(d or 0)

    # --- Constructors ---

    @staticmethod
    def one():
        z = BiPoly.zero()
        return MultiBandElement(BiPoly.one(), z, z, z)

    @staticmethod
    def zero_el():
        z = BiPoly.zero()
        return MultiBandElement(z, z, z, z)

    @staticmethod
    def g1():
        """First-band generator."""
        z = BiPoly.zero()
        return MultiBandElement(z, BiPoly.one(), z, z)

    @staticmethod
    def g2():
        """Second-band generator."""
        z = BiPoly.zero()
        return MultiBandElement(z, z, BiPoly.one(), z)

    @staticmethod
    def g1_inv():
        """g₁⁻¹ = s₁ - g₁."""
        z = BiPoly.zero()
        return MultiBandElement(BiPoly.s1(), BiPoly(-1), z, z)

    @staticmethod
    def g2_inv():
        """g₂⁻¹ = s₂ - g₂."""
        z = BiPoly.zero()
        return MultiBandElement(BiPoly.s2(), z, BiPoly(-1), z)

    @staticmethod
    def from_int(n):
        z = BiPoly.zero()
        return MultiBandElement(BiPoly(n), z, z, z)

    # --- Arithmetic ---

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            other = MultiBandElement.from_int(other) if isinstance(other, int) else MultiBandElement(BiPoly.from_fraction(other))
        return MultiBandElement(self.a + other.a, self.b + other.b,
                                self.c + other.c, self.d + other.d)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            other = MultiBandElement.from_int(other) if isinstance(other, int) else MultiBandElement(BiPoly.from_fraction(other))
        return MultiBandElement(self.a - other.a, self.b - other.b,
                                self.c - other.c, self.d - other.d)

    def __neg__(self):
        return MultiBandElement(-self.a, -self.b, -self.c, -self.d)

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(other)
            return MultiBandElement(self.a * c, self.b * c, self.c * c, self.d * c)
        if not isinstance(other, MultiBandElement):
            return NotImplemented

        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d

        _s1 = BiPoly.s1()
        _s2 = BiPoly.s2()

        new_a = a1*a2 - b1*b2 - c1*c2 + d1*d2
        new_b = a1*b2 + b1*a2 + _s1*(b1*b2 - d1*d2) - c1*d2 - d1*c2
        new_c = a1*c2 + c1*a2 - b1*d2 - d1*b2 + _s2*(c1*c2 - d1*d2)
        new_d = a1*d2 + d1*a2 + b1*c2 + c1*b2 + _s1*(b1*d2 + d1*b2) + _s2*(c1*d2 + d1*c2) + _s1*_s2*d1*d2

        return MultiBandElement(new_a, new_b, new_c, new_d)

    def __rmul__(self, other):
        if isinstance(other, (int, Fraction)):
            c = Fraction(other)
            return MultiBandElement(self.a * c, self.b * c, self.c * c, self.d * c)
        return NotImplemented

    # --- Conjugations ---

    def sigma1(self):
        """σ₁: g₁ → g₁⁻¹ = s₁ - g₁, fixes g₂."""
        _s1 = BiPoly.s1()
        return MultiBandElement(
            self.a + _s1 * self.b,
            -self.b,
            self.c + _s1 * self.d,
            -self.d
        )

    def sigma2(self):
        """σ₂: g₂ → g₂⁻¹ = s₂ - g₂, fixes g₁."""
        _s2 = BiPoly.s2()
        return MultiBandElement(
            self.a + _s2 * self.c,
            self.b + _s2 * self.d,
            -self.c,
            -self.d
        )

    # --- Norm ---

    def norm(self):
        """Full norm: N(x) = x · σ₁(x) · σ₂(x) · σ₁σ₂(x).
        Always in Q[s₁, s₂]."""
        # Stage 1: x * σ₁(x) — eliminates g₁
        half = self * self.sigma1()
        # Stage 2: half * σ₂(half) — eliminates g₂
        full = half * half.sigma2()
        return full.a  # should be purely scalar (BiPoly)

    def can_invert(self):
        n = self.norm()
        return n.is_constant() and not n.is_zero()

    def inv(self):
        n = self.norm()
        if n.is_zero():
            raise ZeroDivisionError("Cannot invert the zero element")
        if not n.is_constant():
            raise ValueError(f"Element is not a unit: norm = {n}")
        c_inv = Fraction(1) / n.constant_value()
        num = self.sigma1() * self.sigma2() * self.sigma1().sigma2()
        return MultiBandElement(num.a * c_inv, num.b * c_inv,
                                num.c * c_inv, num.d * c_inv)

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError(f"Exponent must be an integer, got {type(n)}")
        if n == 0:
            return MultiBandElement.one()
        if n < 0:
            return self.inv() ** (-n)
        result = MultiBandElement.one()
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    # --- Queries ---

    def is_zero(self):
        return self.a.is_zero() and self.b.is_zero() and self.c.is_zero() and self.d.is_zero()

    def __eq__(self, other):
        if isinstance(other, (int, Fraction)):
            other = MultiBandElement.from_int(other) if isinstance(other, int) else MultiBandElement(BiPoly.from_fraction(other))
        if not isinstance(other, MultiBandElement):
            return NotImplemented
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    # --- Evaluation ---

    def eval_at(self, s1_val=None, s2_val=None, theta1=None, theta2=None):
        """Evaluate numerically given the two Chebyshev parameters or angles."""
        import cmath, math
        if theta1 is not None:
            s1_num = 2 * math.cos(theta1)
            g1_num = cmath.exp(1j * theta1)
        elif s1_val is not None:
            s1_num = complex(s1_val)
            g1_num = (s1_num + cmath.sqrt(s1_num**2 - 4)) / 2
        else:
            raise ValueError("Provide s1_val or theta1")

        if theta2 is not None:
            s2_num = 2 * math.cos(theta2)
            g2_num = cmath.exp(1j * theta2)
        elif s2_val is not None:
            s2_num = complex(s2_val)
            g2_num = (s2_num + cmath.sqrt(s2_num**2 - 4)) / 2
        else:
            raise ValueError("Provide s2_val or theta2")

        a = self.a.eval_at(s1_num, s2_num)
        b = self.b.eval_at(s1_num, s2_num)
        c = self.c.eval_at(s1_num, s2_num)
        d = self.d.eval_at(s1_num, s2_num)
        return a + b * g1_num + c * g2_num + d * g1_num * g2_num

    # --- Display ---

    def __repr__(self):
        parts = []
        for coeff, basis_name in [(self.a, ''), (self.b, 'g\u2081'), (self.c, 'g\u2082'), (self.d, 'g\u2081g\u2082')]:
            if coeff.is_zero():
                continue
            c_str = repr(coeff)
            if not basis_name:
                parts.append(c_str)
            elif coeff == BiPoly.one():
                parts.append(basis_name)
            elif coeff == BiPoly(-1):
                parts.append(f'-{basis_name}')
            elif coeff.is_constant():
                parts.append(f'{c_str}\u00b7{basis_name}')
            else:
                parts.append(f'({c_str})\u00b7{basis_name}')
        if not parts:
            return '0'
        result = parts[0]
        for p in parts[1:]:
            if p.startswith('-'):
                result += f' - {p[1:]}'
            else:
                result += f' + {p}'
        return result


# ============================================================
# Module-level convenience
# ============================================================

U = Element.u()
V = Element.v()
ONE = Element.one()
ZERO = Element.zero_el()
S = QsPoly.s()

W = TowerElement.w()
W_INV = TowerElement.w_inv()
T = TowerElement.t()
T_ONE = TowerElement.one()
T_ZERO = TowerElement.zero_el()

G1 = MultiBandElement.g1()
G2 = MultiBandElement.g2()
MB_ONE = MultiBandElement.one()
MB_ZERO = MultiBandElement.zero_el()
