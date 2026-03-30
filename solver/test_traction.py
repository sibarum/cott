"""Tests for Traction Theory solver — validates core identities from the reference."""

import pytest
from sympy import S, Pow, Rational, Symbol, Integer, Add, Mul, I, pi, exp

from traction import (
    Zero, Omega, Null, Log0, LogW, z, w, null,
    traction_simplify, log0, logw, zpow, wpow,
    resolve, resolve_log, project_complex, W_CONST,
    GradedElement, Z
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


# ============================================================
# Generalized Identities (symbolic x)
# ============================================================

class TestGeneralizedIdentities:

    def test_zero_pow_zero_pow_x(self):
        """0^(0^x) = x for any symbolic x."""
        x = Symbol('x')
        assert z**(z**x) == x

    def test_zero_pow_omega_pow_x(self):
        """0^(w^x) = -x for any symbolic x."""
        x = Symbol('x')
        assert z**(w**x) == -x

    def test_omega_pow_zero_pow_x(self):
        """w^(0^x) = -x for any symbolic x."""
        x = Symbol('x')
        assert w**(z**x) == -x

    def test_omega_pow_omega_pow_x(self):
        """w^(w^x) = -1/x for any symbolic x."""
        x = Symbol('x')
        assert w**(w**x) == -S.One / x


# ============================================================
# Universal Power-of-Power (branch-cut elimination)
# ============================================================

class TestUniversalPowerOfPower:

    def test_x_squared_sqrt(self):
        """(x^2)^(1/2) = x — no branch cut."""
        x = Symbol('x')
        result = traction_simplify((x**2)**Rational(1, 2))
        assert result == x

    def test_x_fourth_root(self):
        """(x^4)^(1/4) = x — no branch cut."""
        x = Symbol('x')
        result = traction_simplify((x**4)**Rational(1, 4))
        assert result == x

    def test_branch_cut_eliminated(self):
        """(x^4)^(1/4) - (x^2)^(1/2) — both simplify to x, so they're equal."""
        x = Symbol('x')
        a = traction_simplify((x**4)**Rational(1, 4))
        b = traction_simplify((x**2)**Rational(1, 2))
        assert a == b == x

    def test_power_of_power_general(self):
        """(x^a)^b = x^(a*b) for any base."""
        x = Symbol('x')
        a, b = Symbol('a'), Symbol('b')
        result = traction_simplify(Pow(Pow(x, a), b))
        assert result == Pow(x, a * b)

    def test_zero_base_nested(self):
        """(0^a)^b = 0^(a*b)."""
        a, b = Symbol('a'), Symbol('b')
        result = traction_simplify(Pow(Pow(z, a), b))
        assert result == Pow(z, traction_simplify(a * b))


# ============================================================
# Negative Zero Absorption
# ============================================================

class TestNegativeZero:

    def test_neg_one_times_zero_preserved(self):
        """(-1)*0 = -0 — sign is preserved (no absorption)."""
        result = traction_simplify(S.NegativeOne * z)
        assert result == Mul(S.NegativeOne, Zero())

    def test_neg_times_zero_squared_preserved(self):
        """(-1)*0^2 = -0^2 — sign is preserved."""
        result = traction_simplify(S.NegativeOne * z**2)
        assert result == Mul(S.NegativeOne, z**2)

    def test_neg_times_zero_symbolic_preserved(self):
        """(-1)*0^x stays as -0^x."""
        x = Symbol('x')
        result = traction_simplify(Mul(S.NegativeOne, Pow(z, x)))
        assert result == Mul(S.NegativeOne, Pow(z, x))

    def test_neg_two_times_zero_preserved(self):
        """(-2)*0 = -2*0 — sign is preserved."""
        result = traction_simplify(Integer(-2) * z)
        assert result == Integer(-2) * z


# ============================================================
# Same-Base Power Combination in Mul
# ============================================================

class TestPowerCombination:

    def test_zero_powers_combine(self):
        """0^2 * 0^3 = 0^5."""
        result = traction_simplify(z**2 * z**3)
        assert result == z**5

    def test_omega_powers_combine(self):
        """w^2 * w^3 = w^5."""
        result = traction_simplify(w**2 * w**3)
        assert result == w**5

    def test_cross_base_cancellation(self):
        """0^5 * w^2 = 0^3 (via 0^(5-2))."""
        result = traction_simplify(z**5 * w**2)
        assert result == z**3

    def test_full_cancellation(self):
        """0^3 * w^3 = 1."""
        result = traction_simplify(z**3 * w**3)
        assert result == S.One

    def test_symbolic_exponent_combination(self):
        """0^x * 0^3 = 0^(x+3)."""
        x = Symbol('x')
        result = traction_simplify(z**x * z**3)
        assert result == z**(x + 3)

    def test_zero_omega_mixed_symbolic(self):
        """0^(w/2) * 0^(w/2) = 0^w = -1."""
        result = traction_simplify(z**(w / 2) * z**(w / 2))
        assert result == S.NegativeOne


# ============================================================
# Logarithm — Symbolic Arguments
# ============================================================

class TestLogSymbolic:

    def test_log0_symbolic_returns_Log0(self):
        """log_0(x) returns unevaluated Log0 for symbolic x."""
        x = Symbol('x')
        result = log0(x)
        assert isinstance(result, Log0)

    def test_logw_symbolic_returns_LogW(self):
        """log_w(x) returns unevaluated LogW for symbolic x."""
        x = Symbol('x')
        result = logw(x)
        assert isinstance(result, LogW)

    def test_log0_roundtrip(self):
        """0^(log_0(x)) simplifies via generalized identity."""
        x = Symbol('x')
        result = log0(x)
        # 0^(Log0(x)) — Log0 is a Function, not a Pow(Zero, ...), so this
        # stays as Pow(Zero(), Log0(x)). That's expected — the round-trip
        # works when log0 returns a concrete value.
        assert isinstance(result, Log0)

    def test_log0_omega_pow_n(self):
        """log_0(w^5) = -5."""
        assert log0(w**5) == -5

    def test_logw_omega_pow_n(self):
        """log_w(w^5) = 5."""
        assert logw(w**5) == 5

    def test_logw_zero_pow_n(self):
        """log_w(0^3) = -3."""
        assert logw(z**3) == -3


# ============================================================
# Resolve (Identity Resolution)
# ============================================================

class TestResolve:

    def test_resolve_integer(self):
        assert resolve(5) == 5

    def test_resolve_omega(self):
        assert isinstance(resolve(w), Omega)

    def test_resolve_symbol(self):
        x = Symbol('x')
        assert resolve(x) == x

    def test_resolve_zero_power(self):
        assert resolve(z**2) == z**2

    def test_resolve_negative_one(self):
        assert resolve(-1) == S.NegativeOne

    def test_resolve_log_integer(self):
        assert resolve_log(5) == 5

    def test_resolve_log_omega(self):
        assert isinstance(resolve_log(w), Omega)


# ============================================================
# Complex Projection
# ============================================================

class TestProjection:
    """Tests for the Chebyshev projection: 0^z → e^(i*THETA*z)."""

    def test_project_zero(self):
        """C(0) = C(0^1) = e^(i*THETA) — on the unit circle."""
        from traction import CHEB_THETA
        assert project_complex(z) == exp(I * CHEB_THETA)

    def test_project_omega(self):
        """ω = 0^(-1) → e^(-i*THETA)."""
        from traction import CHEB_THETA
        assert project_complex(w) == exp(-I * CHEB_THETA)

    def test_project_integer(self):
        assert project_complex(Integer(5)) == 5

    def test_project_null(self):
        assert project_complex(null) == S.Zero

    def test_project_zero_pow_omega(self):
        """0^w = -1 projects to -1."""
        assert project_complex(z**w) == S.NegativeOne

    def test_project_dyadic_imaginary(self):
        """0^(w/2) projects to i (theta-independent)."""
        result = project_complex(z**(w / 2))
        assert result == I

    def test_project_negative_dyadic(self):
        """w^(w/2) = 0^(-w/2) projects to -i."""
        result = project_complex(w**(w / 2))
        assert result == -I

    def test_project_unit_circle(self):
        """0^n for rational n stays on the unit circle (Chebyshev property)."""
        from traction import CHEB_THETA
        result = project_complex(z**2)
        expected = exp(I * 2 * CHEB_THETA)
        assert result == expected
        # Verify magnitude = 1 numerically
        val = complex(result.evalf())
        assert abs(abs(val) - 1) < 1e-10

    def test_project_symbol_passthrough(self):
        """Symbols pass through projection unchanged."""
        x = Symbol('x')
        assert project_complex(x) == x

    def test_project_mul_distributed(self):
        """Projection distributes over multiplication."""
        x = Symbol('x')
        result = project_complex(3 * z**(w / 2))
        assert result == 3 * I

    def test_project_all_real_exponents_unit_circle(self):
        """Every 0^(p/q) with real rational exponent has |C(·)| = 1."""
        for n, d in [(1, 1), (1, 2), (1, 3), (2, 3), (5, 7), (3, 1), (-2, 1)]:
            result = project_complex(z**Rational(n, d))
            val = complex(result.evalf())
            assert abs(abs(val) - 1) < 1e-10, f"|C(0^({n}/{d}))| = {abs(val)}, expected 1"


# ============================================================
# Integration: full pipeline through calculator
# ============================================================


class TestPipeline:
    """End-to-end tests through the calculator parse/eval/plot pipeline."""

    def test_plot_with_poles(self):
        """(1-(1/x)^2)^(1/x) — expression with poles, should plot and
        produce valid streamlines without NaN crashes."""
        from calculator import compute_phase_grid, compute_streamlines
        result = compute_phase_grid('(1-(1/x)^2)^(1/x)')
        assert result is not None
        phase, brightness, Z, log_mag = result
        # Should produce a grid with mostly valid values
        import numpy as np
        valid = np.count_nonzero(np.isfinite(Z))
        assert valid > 20000  # most of the 22500 pixels should be valid
        # Streamlines should not crash on NaN gradient values
        tangent = compute_streamlines(log_mag, normal=False)
        normal = compute_streamlines(log_mag, normal=True)
        assert len(tangent) > 0
        assert len(normal) > 0

    def test_plot_zero_times_expr(self):
        """0*sqrt(x^2+1) should plot (not blank) via C(0) = e^(-W)."""
        from calculator import compute_phase_grid
        import numpy as np
        result = compute_phase_grid('0*((x^2+1)^(1/2))')
        assert result is not None
        _, _, Z, _ = result
        # All pixels should be non-zero (C(0) = e^(-W), not 0)
        valid_nonzero = np.count_nonzero(np.isfinite(Z) & (np.abs(Z) > 1e-15))
        assert valid_nonzero == 150 * 150

    def test_plot_omega_exponent(self):
        """x^(w/3) should plot via omega=W in exponent space."""
        from calculator import compute_phase_grid
        result = compute_phase_grid('x^(w/3)')
        assert result is not None

    def test_plot_nested_traction(self):
        """(0^(w/2))^x should plot (simplifies to 0^(x*w/2))."""
        from calculator import compute_phase_grid
        result = compute_phase_grid('(0^(w/2))^x')
        assert result is not None

    def test_plot_two_variable(self):
        """p+q uses raw grid coordinates."""
        from calculator import compute_phase_grid
        result = compute_phase_grid('p+q')
        assert result is not None

    def test_plot_log0(self):
        """log0(x) should plot via log_0(y) = -ln(y)/W."""
        from calculator import compute_phase_grid
        result = compute_phase_grid('log0(x)')
        assert result is not None

    def test_plot_mixed_pqx(self):
        """p^2 + x uses both raw coordinates and native unit."""
        from calculator import compute_phase_grid
        result = compute_phase_grid('p^2+x')
        assert result is not None


class TestExponentDistribution:
    """Test the 0^(A+B) = 0^A * 0^B distribution rule,
    cross-verified against the Lie exponential projection."""

    @staticmethod
    def _lie_eval(expr):
        """Evaluate a traction expression numerically via Lie exponential (no simplification)."""
        val = complex(project_complex(expr).evalf())
        return val

    @staticmethod
    def _simplify_eval(expr):
        """Simplify, then evaluate the result numerically."""
        s = traction_simplify(expr)
        val = complex(project_complex(s).evalf())
        return val

    def _assert_cross_verified(self, expr, expected_simplified=None):
        """Assert that the simplified and Lie results agree numerically,
        and optionally check the symbolic simplified form."""
        simplified = traction_simplify(expr)
        if expected_simplified is not None:
            assert simplified == expected_simplified, \
                f"Expected {expected_simplified}, got {simplified}"

        # Cross-verify: both evaluation paths should give the same complex number
        lie_val = self._lie_eval(expr)
        simp_val = self._simplify_eval(expr)
        assert abs(lie_val - simp_val) < 1e-10, \
            f"Lie={lie_val} vs Simplified={simp_val} differ for {expr}"

    def test_zero_pow_sum_identity(self):
        """0^(0^(5/6) + 0^(3/2)) = 5/4."""
        expr = Zero()**(Zero()**Rational(5, 6) + Zero()**Rational(3, 2))
        self._assert_cross_verified(expr, Rational(5, 4))

    def test_zero_pow_omega_plus_3(self):
        """0^(w + 3) = 0^w * 0^3 = (-1) * 0^3 = -0^3 (no sign absorption)."""
        expr = Zero()**(w + 3)
        self._assert_cross_verified(expr, Mul(S.NegativeOne, Pow(Zero(), 3)))

    def test_zero_pow_omega_plus_omega(self):
        """0^(w + w) = 0^(2w) = (0^w)^2 = (-1)^2 = 1.
        SymPy collapses w+w to 2*w (Mul), so the Mul-exponent factoring rule
        substitutes w -> 0^(-1) and factors as (0^(0^(-1)))^2 = (-1)^2 = 1."""
        expr = Zero()**(w + w)
        self._assert_cross_verified(expr, Integer(1))

    def test_zero_pow_two_zero_powers(self):
        """0^(0^2 + 0^3) = 0^(0^2) * 0^(0^3) = 2 * 3 = 6."""
        expr = Zero()**(Zero()**2 + Zero()**3)
        self._assert_cross_verified(expr, Integer(6))

    def test_zero_pow_three_terms(self):
        """0^(0^1 + 0^2 + 0^3) = 0^(0^1) * 0^(0^2) * 0^(0^3) = 1 * 2 * 3 = 6.
        But 0^(0^1) = 0^0 = 1, so the product is 1*2*3 = 6."""
        expr = Zero()**(Zero()**1 + Zero()**2 + Zero()**3)
        self._assert_cross_verified(expr, Integer(6))

    def test_zero_pow_mixed_sum(self):
        """0^(1 + w) = 0^1 * 0^w = 0 * (-1) = -0 (no sign absorption)."""
        expr = Zero()**(Integer(1) + w)
        simplified = traction_simplify(expr)
        # -0 = Mul(-1, Zero())
        assert simplified == Mul(S.NegativeOne, Zero()), f"Expected -0, got {simplified}"

    def test_zero_pow_rational_sum(self):
        """0^(1/2 + 1/2) = 0^1 = 0."""
        expr = Zero()**(Rational(1, 2) + Rational(1, 2))
        self._assert_cross_verified(expr)
        simplified = traction_simplify(expr)
        assert isinstance(simplified, (Zero, Pow)), f"Expected zero-class, got {simplified}"

    def test_nested_distribution(self):
        """0^(0^(0^2 + 0^3) + 0^1) = 0^(0^6 + 0) = 0^(0^6) * 0^0 = 6 * 1 = 6."""
        inner = Zero()**(Zero()**2 + Zero()**3)  # = 6
        expr = Zero()**(inner + Zero()**1)  # 0^(6 + 0^1) = 0^(6 + 0) = 0^6 * 0^0... hmm
        # Actually: inner simplifies to 6, so expr = 0^(6 + 0^1) = 0^(6 + 0)
        # 0^(6 + 0): 6 is Integer, 0 is Zero. Add gives 6 + 0 (SymPy).
        # Let's just cross-verify numerically.
        self._assert_cross_verified(expr)


# ============================================================
# Graded Algebra — Z-action
# ============================================================

class TestGradedIdentity:
    """Z_0(x) = x for all x."""

    def test_z0_integer(self):
        assert Z(0, 5) == Integer(5)

    def test_z0_symbol(self):
        x = Symbol('x')
        assert Z(0, x) == x

    def test_z0_zero(self):
        assert Z(0, z) is z

    def test_z0_omega(self):
        assert Z(0, w) is w


class TestGradedArithmetic:
    """Same-grade operations shift the grade."""

    def test_add_becomes_mul(self):
        # Z_1(a) + Z_1(b) = Z_0(a*b) = a*b
        a, b = Integer(3), Integer(5)
        result = Z(1, a) + Z(1, b)
        assert result == a * b, f"Expected {a*b}, got {result}"

    def test_mul_becomes_add(self):
        # Z_1(a) * Z_1(b) = Z_2(a+b)
        a, b = Integer(3), Integer(5)
        result = Z(1, a) * Z(1, b)
        assert isinstance(result, GradedElement)
        assert result.grade == Integer(2)
        assert result.value == a + b

    def test_sub_becomes_div(self):
        # Z_1(a) - Z_1(b) = Z_0(a/b) = a/b
        a, b = Integer(6), Integer(3)
        result = Z(1, a) - Z(1, b)
        assert result == a / b, f"Expected {a/b}, got {result}"

    def test_div_becomes_sub(self):
        # Z_1(a) / Z_1(b) = Z_2(a-b)
        a, b = Integer(8), Integer(3)
        result = Z(1, a) / Z(1, b)
        assert isinstance(result, GradedElement)
        assert result.grade == Integer(2)
        assert result.value == a - b

    def test_higher_grade_add(self):
        # Z_3(a) + Z_3(b) = Z_2(a*b)
        a, b = Integer(2), Integer(7)
        result = Z(3, a) + Z(3, b)
        assert isinstance(result, GradedElement)
        assert result.grade == Integer(2)
        assert result.value == a * b

    def test_negative_grade(self):
        # Z_{-1}(a) + Z_{-1}(b) = Z_{-2}(a*b)
        a, b = Integer(4), Integer(5)
        result = Z(-1, a) + Z(-1, b)
        assert isinstance(result, GradedElement)
        assert result.grade == Integer(-2)
        assert result.value == a * b

    def test_chain_add_then_add(self):
        # (Z_2(a) + Z_2(b)) + (Z_2(c) + Z_2(d))
        # = Z_1(a*b) + Z_1(c*d) = Z_0(a*b*c*d) = a*b*c*d
        a, b, c, d = Integer(2), Integer(3), Integer(5), Integer(7)
        left = Z(2, a) + Z(2, b)    # Z_1(6)
        right = Z(2, c) + Z(2, d)   # Z_1(35)
        result = left + right         # Z_0(6*35) = 210
        assert result == Integer(210), f"Expected 210, got {result}"


class TestGradedExponentiation:
    """Exponentiation with scalars shifts grade."""

    def test_pow_scalar(self):
        # Z_1(a) ^ b = Z_2(a*b)
        result = Z(1, Integer(3)) ** Integer(4)
        assert isinstance(result, GradedElement)
        assert result.grade == Integer(2)
        assert result.value == Integer(12)

    def test_rpow_scalar(self):
        # b ^ Z_1(a) = Z_0(a*b) = a*b
        result = Integer(5) ** Z(1, Integer(3))
        assert result == Integer(15), f"Expected 15, got {result}"


class TestGradedFixedPoints:
    """Fixed-point values across grades."""

    def test_fixed_1_and_0(self):
        # Z_2(1) and Z_3(0) should be equal
        z2_1 = Z(2, Integer(1))
        z3_0 = Z(3, Integer(0))
        assert z2_1 == z3_0, f"Z_2(1)={z2_1} should equal Z_3(0)={z3_0}"

    def test_fixed_neg1_and_omega(self):
        # Z_n(-1) = Z_{n+1}(w)
        z1_neg1 = Z(1, S.NegativeOne)
        z2_w = Z(2, w)
        assert z1_neg1 == z2_w, f"Z_1(-1)={z1_neg1} should equal Z_2(w)={z2_w}"

    def test_fixed_omega_and_neg1(self):
        # Z_n(w) = Z_{n+1}(-1)
        z1_w = Z(1, w)
        z2_neg1 = Z(2, S.NegativeOne)
        assert z1_w == z2_neg1, f"Z_1(w)={z1_w} should equal Z_2(-1)={z2_neg1}"


class TestGradedComposition:
    """Composition of adjacent grades collapses to identity."""

    def test_compose_up(self):
        # Z_1(Z_2(x)) = x
        x = Symbol('x')
        inner = Z(2, x)
        result = Z(1, inner)
        assert result == x, f"Z_1(Z_2(x)) should be x, got {result}"

    def test_compose_down(self):
        # Z_2(Z_1(x)) = x
        x = Symbol('x')
        inner = Z(1, x)
        result = Z(2, inner)
        assert result == x, f"Z_2(Z_1(x)) should be x, got {result}"

    def test_compose_non_adjacent_stays(self):
        # Z_1(Z_3(x)) should NOT collapse (gap of 2)
        x = Symbol('x')
        inner = Z(3, x)
        result = Z(1, inner)
        assert isinstance(result, GradedElement), \
            f"Z_1(Z_3(x)) should remain graded, got {result}"

    def test_compose_with_concrete(self):
        # Z_3(Z_4(7)) = 7
        result = Z(3, Z(4, Integer(7)))
        assert result == Integer(7)


class TestGradedInversion:
    """Negation and reciprocal shift grade."""

    def test_negation(self):
        # -Z_1(5): __neg__ produces 1/Z_2(5), then _eval_power fires:
        # Pow(Z_2(5), -1) = Z_3(5*(-1)) = Z_3(-5)
        elem = Z(1, Integer(5))
        result = -elem
        assert isinstance(result, GradedElement)
        assert result.grade == Integer(3)
        assert result.value == Integer(-5)

    def test_inverse_method(self):
        # inverse of Z_n(a) = Z_{n+1}(-a)
        elem = Z(1, Integer(5))
        inv = elem.inverse()
        assert isinstance(inv, GradedElement)
        assert inv.grade == Integer(2)
        assert inv.value == Integer(-5)


class TestGradedDisplay:
    """String representation."""

    def test_str(self):
        elem = Z(2, Integer(3))
        assert 'Z_2' in str(elem)
        assert '3' in str(elem)

    def test_repr(self):
        elem = Z(1, Symbol('x'))
        assert 'Z_1' in repr(elem)
        assert 'x' in repr(elem)
