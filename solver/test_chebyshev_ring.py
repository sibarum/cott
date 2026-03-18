"""Tests for the Chebyshev ring Q[s][u] / (u² - su + 1)."""

import pytest
import cmath
import math
from fractions import Fraction
from chebyshev_ring import QsPoly, Element, U, V, ONE, ZERO, S


# ============================================================
# QsPoly Tests
# ============================================================

class TestQsPoly:

    def test_construction_from_int(self):
        p = QsPoly(3)
        assert p.coeffs == (Fraction(3),)

    def test_construction_strips_zeros(self):
        p = QsPoly([1, 0, 0])
        assert p.coeffs == (Fraction(1),)

    def test_zero(self):
        z = QsPoly.zero()
        assert z.is_zero()
        assert z.degree() == -1

    def test_one(self):
        o = QsPoly.one()
        assert not o.is_zero()
        assert o.degree() == 0
        assert o.constant_value() == Fraction(1)

    def test_s(self):
        s = QsPoly.s()
        assert s.degree() == 1
        assert s.coeffs == (Fraction(0), Fraction(1))

    def test_add(self):
        a = QsPoly([1, 2])      # 1 + 2s
        b = QsPoly([3, -2, 1])  # 3 - 2s + s²
        c = a + b
        assert c == QsPoly([4, 0, 1])  # 4 + s²

    def test_add_scalar(self):
        a = QsPoly([1, 1])  # 1 + s
        assert a + 2 == QsPoly([3, 1])

    def test_sub(self):
        a = QsPoly([5, 3])
        b = QsPoly([2, 3])
        assert a - b == QsPoly([3])

    def test_neg(self):
        a = QsPoly([1, -2])
        assert -a == QsPoly([-1, 2])

    def test_mul(self):
        s = QsPoly.s()
        assert s * s == QsPoly([0, 0, 1])  # s²

    def test_mul_binomial(self):
        # (1 + s)(1 - s) = 1 - s²
        a = QsPoly([1, 1])
        b = QsPoly([1, -1])
        assert a * b == QsPoly([1, 0, -1])

    def test_mul_scalar(self):
        a = QsPoly([1, 1])  # 1 + s
        assert a * 3 == QsPoly([3, 3])
        assert 3 * a == QsPoly([3, 3])

    def test_scale(self):
        a = QsPoly([2, 4])
        assert a.scale(Fraction(1, 2)) == QsPoly([1, 2])

    def test_shift(self):
        a = QsPoly([1, 1])  # 1 + s
        assert a.shift(2) == QsPoly([0, 0, 1, 1])  # s² + s³

    def test_divmod_exact(self):
        # (s² + 2s + 1) / (s + 1) = (s + 1), remainder 0
        a = QsPoly([1, 2, 1])
        b = QsPoly([1, 1])
        q, r = QsPoly.divmod(a, b)
        assert q == QsPoly([1, 1])
        assert r.is_zero()

    def test_divmod_remainder(self):
        # (s² + 1) / (s + 1) = (s - 1), remainder 2
        a = QsPoly([1, 0, 1])
        b = QsPoly([1, 1])
        q, r = QsPoly.divmod(a, b)
        assert q == QsPoly([-1, 1])
        assert r == QsPoly([2])

    def test_eval(self):
        p = QsPoly([1, 2, 1])  # 1 + 2s + s²
        assert p.eval_at(3.0) == pytest.approx(16.0)  # (1+3)² = 16

    def test_eq(self):
        assert QsPoly([1, 0]) == QsPoly([1])
        assert QsPoly.zero() == QsPoly([0, 0, 0])
        assert QsPoly([1, 2]) != QsPoly([1, 3])

    def test_repr(self):
        assert repr(QsPoly.zero()) == '0'
        assert repr(QsPoly.one()) == '1'
        assert repr(QsPoly.s()) == 's'
        assert 's^2' in repr(QsPoly([0, 0, 1]))


# ============================================================
# Element Tests — Construction
# ============================================================

class TestElementConstruction:

    def test_one(self):
        assert ONE.a == QsPoly.one()
        assert ONE.b == QsPoly.zero()

    def test_zero_element(self):
        assert ZERO.is_zero()
        assert ZERO.a.is_zero()
        assert ZERO.b.is_zero()

    def test_u(self):
        assert U.a == QsPoly.zero()
        assert U.b == QsPoly.one()

    def test_v(self):
        # v = s - u, so a = s, b = -1
        assert V.a == QsPoly.s()
        assert V.b == QsPoly.from_int(-1)

    def test_from_int(self):
        e = Element.from_int(5)
        assert e.a == QsPoly.from_int(5)
        assert e.b.is_zero()


# ============================================================
# Element Tests — Core Arithmetic
# ============================================================

class TestElementArithmetic:

    def test_add(self):
        a = Element(QsPoly([1]), QsPoly([2]))       # 1 + 2u
        b = Element(QsPoly([3]), QsPoly([-1]))       # 3 - u
        c = a + b
        assert c == Element(QsPoly([4]), QsPoly([1]))  # 4 + u

    def test_sub(self):
        a = Element(QsPoly([5]), QsPoly([3]))
        b = Element(QsPoly([2]), QsPoly([3]))
        assert a - b == Element(QsPoly([3]), QsPoly.zero())

    def test_neg(self):
        a = Element(QsPoly([1]), QsPoly([1]))
        assert -a == Element(QsPoly([-1]), QsPoly([-1]))

    def test_mul_u_times_u(self):
        """u * u = u² = -1 + su (the fundamental reduction rule)."""
        result = U * U
        assert result.a == QsPoly.from_int(-1)
        assert result.b == QsPoly.s()

    def test_mul_u_times_v_is_one(self):
        """u * v = 1 (fundamental identity: u and v are inverses)."""
        result = U * V
        assert result == ONE

    def test_mul_v_times_u_is_one(self):
        """v * u = 1 (commutativity check)."""
        result = V * U
        assert result == ONE

    def test_mul_scalar(self):
        e = Element(QsPoly([1]), QsPoly([1]))  # 1 + u
        assert e * 3 == Element(QsPoly([3]), QsPoly([3]))
        assert 3 * e == Element(QsPoly([3]), QsPoly([3]))

    def test_mul_associative(self):
        """(u * u) * u == u * (u * u)"""
        left = (U * U) * U
        right = U * (U * U)
        assert left == right

    def test_mul_distributive(self):
        """u * (1 + u) == u + u²"""
        one_plus_u = ONE + U
        left = U * one_plus_u
        right = U + U * U
        assert left == right


# ============================================================
# Element Tests — Conjugation and Norm
# ============================================================

class TestConjNorm:

    def test_conj_u(self):
        """conj(u) = v = s - u."""
        assert U.conj() == V

    def test_conj_v(self):
        """conj(v) = u."""
        assert V.conj() == U

    def test_conj_involution(self):
        """conj(conj(x)) = x."""
        x = Element(QsPoly([1, 2]), QsPoly([3, -1]))
        assert x.conj().conj() == x

    def test_conj_scalar(self):
        """conj of a scalar is itself."""
        s = Element.from_int(7)
        assert s.conj() == s

    def test_norm_one(self):
        assert ONE.norm() == QsPoly.one()

    def test_norm_u(self):
        """N(u) = 0² + 0·1·s + 1² = 1."""
        assert U.norm() == QsPoly.one()

    def test_norm_v(self):
        """N(v) = s² + s·(-1)·s + (-1)² = s² - s² + 1 = 1."""
        assert V.norm() == QsPoly.one()

    def test_norm_is_product_with_conj(self):
        """x * conj(x) should equal norm(x) as a scalar element."""
        x = Element(QsPoly([2, 1]), QsPoly([1]))  # (2+s) + u
        product = x * x.conj()
        assert product.b.is_zero()  # must be in Q[s]
        assert product.a == x.norm()

    def test_norm_u_power(self):
        """N(u^n) = 1 for all n (u is a unit of norm 1)."""
        for n in range(-5, 6):
            assert Element.u_power(n).norm() == QsPoly.one(), f"N(u^{n}) != 1"


# ============================================================
# Element Tests — Inversion
# ============================================================

class TestInversion:

    def test_inv_u(self):
        """u⁻¹ = v = s - u."""
        assert U.inv() == V

    def test_inv_v(self):
        """v⁻¹ = u."""
        assert V.inv() == U

    def test_u_times_inv_u(self):
        assert U * U.inv() == ONE

    def test_inv_involution(self):
        """inv(inv(x)) = x for units."""
        for n in range(-3, 4):
            x = Element.u_power(n)
            assert x.inv().inv() == x, f"inv(inv(u^{n})) != u^{n}"

    def test_inv_scalar(self):
        e = Element.from_int(3)
        inv_e = e.inv()
        assert e * inv_e == ONE

    def test_inv_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            ZERO.inv()

    def test_can_invert_unit(self):
        assert U.can_invert()
        assert V.can_invert()
        assert ONE.can_invert()

    def test_can_invert_nonunit(self):
        """s + u has a nonconstant norm, so it's not invertible in the ring."""
        x = Element(QsPoly.s(), QsPoly.one())  # s + u
        assert not x.can_invert()

    def test_division(self):
        """(u² + u) / u = u + 1."""
        num = U * U + U
        result = num / U
        assert result == U + ONE


# ============================================================
# Element Tests — Powers
# ============================================================

class TestPowers:

    def test_u_power_0(self):
        assert Element.u_power(0) == ONE

    def test_u_power_1(self):
        assert Element.u_power(1) == U

    def test_u_power_2(self):
        """u² = -1 + su."""
        u2 = Element.u_power(2)
        assert u2.a == QsPoly.from_int(-1)
        assert u2.b == QsPoly.s()

    def test_u_power_neg1(self):
        """u⁻¹ = v = s - u."""
        assert Element.u_power(-1) == V

    def test_u_power_neg2(self):
        """u⁻² = v²."""
        assert Element.u_power(-2) == V * V

    def test_u_power_positive_matches_repeated_mul(self):
        """u^5 via u_power matches u*u*u*u*u."""
        repeated = U
        for _ in range(4):
            repeated = repeated * U
        assert Element.u_power(5) == repeated

    def test_u_power_negative_matches(self):
        """u^{-3} = v^3."""
        assert Element.u_power(-3) == V ** 3

    def test_pow_syntax(self):
        """u ** 4 works."""
        assert U ** 4 == Element.u_power(4)

    def test_large_power(self):
        """u^100 should work without error."""
        e = Element.u_power(100)
        assert e.norm() == QsPoly.one()

    def test_power_inverse_identity(self):
        """u^n * u^{-n} = 1."""
        for n in range(-5, 6):
            assert Element.u_power(n) * Element.u_power(-n) == ONE, f"u^{n} * u^{-n} != 1"


# ============================================================
# Element Tests — Traction Conversion
# ============================================================

class TestTractionConversion:

    def test_traction_zero_pow_0(self):
        """0^0 = u^0 = 1."""
        assert Element.from_traction_exp(0) == ONE

    def test_traction_zero_pow_1(self):
        """0^1 = u^2 = -1 + su (traction zero)."""
        e = Element.from_traction_exp(1)
        assert e == U * U

    def test_traction_zero_pow_neg1(self):
        """0^(-1) = u^(-2) = v² (traction omega)."""
        e = Element.from_traction_exp(-1)
        assert e == V * V

    def test_traction_half(self):
        """0^(1/2) = u."""
        e = Element.from_traction_exp(Fraction(1, 2))
        assert e == U

    def test_traction_neg_half(self):
        """0^(-1/2) = v."""
        e = Element.from_traction_exp(Fraction(-1, 2))
        assert e == V

    def test_traction_three_halves(self):
        """0^(3/2) = u^3."""
        e = Element.from_traction_exp(Fraction(3, 2))
        assert e == U ** 3

    def test_traction_non_half_integer_raises(self):
        """0^(1/3) requires u^(2/3) which isn't an integer power."""
        with pytest.raises(ValueError):
            Element.from_traction_exp(Fraction(1, 3))


# ============================================================
# Element Tests — Chebyshev Identities
# ============================================================

class TestChebyshevIdentities:

    def test_a0(self):
        """a_0 = u^0 + v^0 = 1 + 1 = 2."""
        a0 = Element.u_power(0) + Element.u_power(0)  # trivial
        assert a0 == Element.from_int(2)

    def test_a1_sum(self):
        """a_1 = u + v = s (as a scalar element)."""
        a1 = U + V
        assert a1.b.is_zero()  # must be scalar
        assert a1.a == QsPoly.s()

    def test_a2_sum(self):
        """a_2 = u² + v² = s² - 2."""
        u2 = U ** 2
        v2 = V ** 2
        a2 = u2 + v2
        assert a2.is_scalar()
        assert a2.a == QsPoly([- 2, 0, 1])  # -2 + s²

    def test_half_cycle_sums_are_scalar(self):
        """u^n + v^n should always be in Q[s] (the u-component cancels)."""
        for n in range(8):
            total = Element.u_power(n) + Element.u_power(-n)
            assert total.is_scalar(), f"u^{n} + v^{n} has nonzero u-component"

    def test_chebyshev_recurrence(self):
        """a_n = s * a_{n-1} - a_{n-2} for the half-cycle sums."""
        s_el = Element(QsPoly.s())  # s as an Element
        a = [Element.from_int(2), Element(QsPoly.s())]  # a_0=2, a_1=s
        for n in range(2, 8):
            a_next = s_el * a[n - 1] - a[n - 2]
            a.append(a_next)
            # Verify against direct computation
            direct = Element.u_power(n) + Element.u_power(-n)
            assert a_next == direct, f"Recurrence failed at n={n}"


# ============================================================
# Element Tests — Numeric Evaluation
# ============================================================

class TestEvaluation:

    def test_eval_u_at_pi_over_3(self):
        """At θ=π/3: u = e^{iπ/3}, s = 2cos(π/3) = 1."""
        theta = math.pi / 3
        val = U.eval_at(theta=theta)
        expected = cmath.exp(1j * theta)
        assert abs(val - expected) < 1e-12

    def test_eval_u_squared(self):
        """u² should evaluate to e^{2iθ}."""
        theta = math.pi / 4
        val = (U ** 2).eval_at(theta=theta)
        expected = cmath.exp(2j * theta)
        assert abs(val - expected) < 1e-12

    def test_eval_sum_is_real(self):
        """u^n + v^n = 2cos(nθ) should be real."""
        theta = math.pi / 5
        for n in range(6):
            val = (Element.u_power(n) + Element.u_power(-n)).eval_at(theta=theta)
            assert abs(val.imag) < 1e-12, f"u^{n} + v^{n} is not real"
            assert abs(val.real - 2 * math.cos(n * theta)) < 1e-12

    def test_eval_with_s_val(self):
        """Evaluation via s_val should match theta evaluation."""
        theta = math.pi / 6
        s_val = 2 * math.cos(theta)
        e = U ** 3 + Element.from_int(2)
        val_theta = e.eval_at(theta=theta)
        val_s = e.eval_at(s_val=s_val)
        assert abs(val_theta - val_s) < 1e-12

    def test_eval_product(self):
        """(u + 1)(u - 1) evaluated should match (e^{iθ}+1)(e^{iθ}-1)."""
        theta = 0.7
        u_num = cmath.exp(1j * theta)
        expected = (u_num + 1) * (u_num - 1)
        expr = (U + ONE) * (U - ONE)
        val = expr.eval_at(theta=theta)
        assert abs(val - expected) < 1e-12


# ============================================================
# Element Tests — Display
# ============================================================

class TestDisplay:

    def test_repr_zero(self):
        assert repr(ZERO) == '0'

    def test_repr_one(self):
        assert repr(ONE) == '1'

    def test_repr_u(self):
        assert repr(U) == 'u'

    def test_repr_neg_u(self):
        assert repr(-U) == '-u'

    def test_repr_scalar(self):
        assert repr(Element.from_int(5)) == '5'

    def test_repr_a_plus_bu(self):
        e = Element(QsPoly([3]), QsPoly([2]))  # 3 + 2u
        assert 'u' in repr(e)
