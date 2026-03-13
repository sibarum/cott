"""Tests for Traction Theory solver — validates core identities from the reference."""

import pytest
from sympy import S, Pow, Rational, Symbol, Integer, Add, Mul

from traction import (
    Zero, Omega, Null, z, w, null,
    traction_simplify, log0, logw, zpow, wpow
)


# ============================================================
# Section 3.3.1: Base-0 Exponentiation
# ============================================================

class TestBaseZeroPowers:

    def test_zero_pow_zero(self):
        assert z**0 == S.One

    def test_zero_pow_one(self):
        assert isinstance(z**1, Zero)

    def test_zero_pow_neg1(self):
        # 0^(-1) = w
        assert isinstance(z**(-1), Omega)

    def test_zero_pow_neg_n(self):
        # 0^(-n) = w^n
        assert z**(-2) == w**2
        assert z**(-3) == w**3

    def test_zero_pow_omega(self):
        # 0^w = -1
        assert z**w == S.NegativeOne

    def test_zero_pow_zero_pow_n(self):
        # 0^(0^n) = n
        assert z**(z**2) == 2
        assert z**(z**3) == 3
        assert z**(z**5) == 5

    def test_zero_pow_omega_pow_n(self):
        # 0^(w^n) = -n
        assert z**(w**2) == -2
        assert z**(w**3) == -3


# ============================================================
# Section 3.3.2: Base-w Exponentiation
# ============================================================

class TestBaseOmegaPowers:

    def test_omega_pow_zero(self):
        assert w**0 == S.One

    def test_omega_pow_one(self):
        assert isinstance(w**1, Omega)

    def test_omega_pow_neg1(self):
        # w^(-1) = 0
        assert isinstance(w**(-1), Zero)

    def test_omega_pow_neg_n(self):
        # w^(-n) = 0^n
        assert w**(-2) == z**2
        assert w**(-3) == z**3

    def test_omega_pow_omega(self):
        # w^w = -1
        assert w**w == S.NegativeOne

    def test_omega_pow_zero_pow_n(self):
        # w^(0^n) = -n
        assert w**(z**2) == -2
        assert w**(z**3) == -3

    def test_omega_pow_omega_pow_n(self):
        # w^(w^n) = -1/n
        assert w**(w**2) == Rational(-1, 2)
        assert w**(w**3) == Rational(-1, 3)


# ============================================================
# Reciprocals: 1/0 = w, 1/w = 0
# ============================================================

class TestReciprocals:

    def test_reciprocal_of_zero(self):
        assert isinstance(S.One / z, Omega)

    def test_reciprocal_of_omega(self):
        assert isinstance(S.One / w, Zero)

    def test_double_reciprocal_zero(self):
        # 1/(1/0) = 1/w = 0
        assert isinstance(S.One / (S.One / z), Zero)

    def test_double_reciprocal_omega(self):
        # 1/(1/w) = 1/0 = w
        assert isinstance(S.One / (S.One / w), Omega)


# ============================================================
# Section 3.2: Multiplication Table
# ============================================================

class TestMultiplicationTable:

    # Row: 1 x ...
    def test_one_times_zero(self):
        assert isinstance(S.One * z, Zero)

    def test_one_times_omega(self):
        assert isinstance(S.One * w, Omega)

    # Row: 0 x ...
    def test_zero_times_omega(self):
        assert z * w == S.One

    def test_omega_times_zero(self):
        assert w * z == S.One

    def test_zero_times_zero(self):
        # 0*0 = 0^2
        assert z * z == z**2

    def test_neg_one_times_zero(self):
        # (-1)*0 = -0
        result = S.NegativeOne * z
        assert result == -z

    # Row: w x ...
    def test_omega_times_omega(self):
        # w*w = w^2
        assert w * w == w**2

    def test_neg_one_times_omega(self):
        # (-1)*w = -w
        result = S.NegativeOne * w
        assert result == -w


# ============================================================
# Section 3.2: Division Table
# ============================================================

class TestDivisionTable:

    def test_one_div_zero(self):
        # 1/0 = w
        assert isinstance(S.One / z, Omega)

    def test_one_div_omega(self):
        # 1/w = 0
        assert isinstance(S.One / w, Zero)

    def test_zero_div_zero(self):
        # 0/0 = 1 (0 * 0^-1 = 0 * w = 1)
        assert z / z == S.One

    def test_omega_div_omega(self):
        # w/w = 1
        assert w / w == S.One

    def test_zero_div_omega(self):
        # 0/w = 0^2 (0 * w^-1 = 0 * 0 = 0^2)
        assert z / w == z**2

    def test_omega_div_zero(self):
        # w/0 = w^2 (w * 0^-1 = w * w = w^2)
        assert w / z == w**2

    def test_neg_one_div_zero(self):
        # (-1)/0 = -w
        assert S.NegativeOne / z == -w

    def test_neg_one_div_omega(self):
        # (-1)/w = -0
        assert S.NegativeOne / w == -z


# ============================================================
# Section 3.3.1-2: Logarithm Identities
# ============================================================

class TestLog0:

    def test_log0_one(self):
        assert log0(S.One) == S.Zero

    def test_log0_zero(self):
        assert log0(z) == S.One

    def test_log0_omega(self):
        assert log0(w) == S.NegativeOne

    def test_log0_zero_pow_n(self):
        # log_0(0^n) = n
        assert log0(z**2) == 2
        assert log0(z**3) == 3

    def test_log0_omega_pow_n(self):
        # log_0(w^n) = -n
        assert log0(w**2) == -2

    def test_log0_positive_int(self):
        # log_0(n) = 0^n
        assert log0(Integer(3)) == z**3

    def test_log0_negative_int(self):
        # log_0(-n) = w^n
        assert log0(Integer(-3)) == w**3


class TestLogW:

    def test_logw_one(self):
        assert logw(S.One) == S.Zero

    def test_logw_omega(self):
        assert logw(w) == S.One

    def test_logw_zero(self):
        assert logw(z) == S.NegativeOne

    def test_logw_zero_pow_n(self):
        # log_w(0^n) = -n
        assert logw(z**2) == -2

    def test_logw_negative_int(self):
        # log_w(-n) = 0^n
        assert logw(Integer(-3)) == z**3


# ============================================================
# Section 3.1: Addition
# ============================================================

class TestAddition:

    def test_subtraction_erasure_symbol(self):
        # Use evaluate=False to prevent SymPy from eagerly cancelling
        x = Symbol('x')
        expr = Add(x, -x, evaluate=False)
        result = traction_simplify(expr)
        assert isinstance(result, Null)

    def test_subtraction_erasure_integer(self):
        expr = Add(Integer(5), Integer(-5), evaluate=False)
        result = traction_simplify(expr)
        assert isinstance(result, Null)


# ============================================================
# traction_simplify on Mul expressions
# ============================================================

class TestSimplifyMul:

    def test_simplify_zero_times_omega(self):
        expr = traction_simplify(z * w)
        assert expr == S.One

    def test_simplify_scaled(self):
        # 2*0*3*w = 6*(0*w) = 6*1 = 6
        expr = traction_simplify(2 * z * 3 * w)
        assert expr == 6

    def test_simplify_multiple_zeros(self):
        expr = traction_simplify(z * z * z)
        assert expr == z**3


# ============================================================
# Section 5: Derived Identities
# ============================================================

class TestDerived:

    def test_dyadic_imaginary_derivation(self):
        """
        sqrt(-1) = 0^(w/2)
        Derivation: (-1)^(1/2) = 0^(1/2 * log_0(-1)) = 0^(w/2)
        """
        # Step 1: log_0(-1) = w
        assert isinstance(log0(S.NegativeOne), Omega)

        # Step 2: 1/2 * w
        half_w = Rational(1, 2) * w
        assert half_w == w / 2

        # Step 3: 0^(w/2) is the traction representation of i
        i_traction = z**(w / 2)
        assert i_traction == zpow(w / 2)

    def test_dyadic_imaginary_squared(self):
        """(0^(w/2))^2 = 0^w = -1"""
        i_traction = z**(w / 2)
        result = traction_simplify(i_traction**2)
        assert result == S.NegativeOne

    def test_nested_power_collapse_omega_base(self):
        """(w^(1/2))^2 = w via exponent collapsing."""
        result = traction_simplify((w**Rational(1, 2))**2)
        assert isinstance(result, Omega)

    def test_zero_omega_duality(self):
        """0 and w are mutual reciprocals: 0^(-1) = w, w^(-1) = 0."""
        assert isinstance(z**(-1), Omega)
        assert isinstance(w**(-1), Zero)
        # And composing: 0^(-1) * 0 = w * 0 = 1
        assert z**(-1) * z == S.One

    def test_power_chain_identity(self):
        """0^(0^n) = n, so 0^(0^(0^n)) = 0^n."""
        # 0^(0^3) = 3
        assert z**(z**3) == 3
        # 0^(0^(0^3)) = 0^3
        assert z**(z**(z**3)) == z**3
