"""
Chebyshev ring decomposition engine.

Converts traction expressions into Chebyshev ring elements,
computes decompositions, reductions, and exact evaluations.
"""

from sympy import S, Integer, Rational, Pow, Mul, Add, Symbol
from traction import Zero, Omega, Null, traction_simplify, project_complex
from formatting import format_result, format_complex


def _extract_zero_exponent(expr):
    """
    If expr is a zero-power (or equivalent), return the exponent.
    Returns None if not a simple zero-power.
    """
    if isinstance(expr, Zero):
        return S.One
    if isinstance(expr, Omega):
        return S.NegativeOne
    if expr is S.One or expr == Integer(1):
        return S.Zero
    if isinstance(expr, Pow):
        if isinstance(expr.base, Zero):
            return expr.exp
        if isinstance(expr.base, Omega):
            return -expr.exp
    return None


def _collect_exponent_denoms(expr):
    """Collect all exponent denominators from zero/omega powers in an expression."""
    denoms = set()
    expr = traction_simplify(expr)

    if isinstance(expr, (Zero, Omega)):
        denoms.add(1)
    elif isinstance(expr, Pow):
        if isinstance(expr.base, (Zero, Omega)) and isinstance(expr.exp, Rational):
            denoms.add(int(expr.exp.q))
        elif isinstance(expr.base, (Zero, Omega)) and isinstance(expr.exp, Integer):
            denoms.add(1)
        # Recurse into non-traction powers
        if not isinstance(expr.base, (Zero, Omega)):
            denoms.update(_collect_exponent_denoms(expr.base))
    elif isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            denoms.update(_collect_exponent_denoms(arg))

    return denoms


def _expr_to_ring(expr):
    """
    Convert a traction expression to a chebyshev_ring.Element with an
    adaptive generator. Returns (element, info_dict) or (None, None).

    Auto-detects the natural base (zero or omega) and band (rational or omega).
    Priority:
    1. Omega band: g = B^(ω/n), θ-independent universal structure
    2. Rational band: g = B^(1/n), θ-dependent
    3. Multi-band: both generators needed
    """
    from chebyshev_ring import Element, ZERO_SPEC, OMEGA_SPEC, ZERO_OMEGA_SPEC, OMEGA_OMEGA_SPEC
    from fractions import Fraction as Frac
    from math import lcm

    expr = traction_simplify(expr)

    # Classify exponents into omega and rational bands, and detect base
    rational_denoms = set()
    omega_denoms = set()
    bases_seen = set()  # 'zero', 'omega'
    _classify_exponents_full(expr, rational_denoms, omega_denoms, bases_seen)

    # Determine the natural base spec
    if bases_seen == {'omega'}:
        rat_spec = OMEGA_SPEC
        omega_spec = OMEGA_OMEGA_SPEC
    elif bases_seen == {'zero'}:
        rat_spec = ZERO_SPEC
        omega_spec = ZERO_OMEGA_SPEC
    else:
        # Mixed bases or scalars — default to zero
        rat_spec = ZERO_SPEC
        omega_spec = ZERO_OMEGA_SPEC

    # --- Priority 1: Pure omega exponents → omega-band ring ---
    if omega_denoms and not rational_denoms:
        omega_lcd = 1
        for d in omega_denoms:
            omega_lcd = lcm(omega_lcd, d)
        el = _convert_omega_band(expr, omega_lcd)
        if el is not None:
            return el, {'band': 'omega', 'omega_lcd': omega_lcd, 'spec': omega_spec}

    # --- Priority 2: Pure rational exponents → rational-band ring ---
    if rational_denoms or not omega_denoms:
        denoms = _collect_exponent_denoms(expr)
        if not denoms:
            denoms = {1}
        base_denom = 1
        for d in denoms:
            base_denom = lcm(base_denom, d)
        base_denom = max(base_denom, 2)

        el = _convert_with_scale(expr, base_denom)
        if el is not None:
            return el, {'band': 'rational', 'base_denom': base_denom, 'spec': rat_spec}

    # --- Priority 3: Mixed → multi-band ---
    mb_result = _try_multiband(expr)
    if mb_result is not None:
        el, rat_scale, omega_scale = mb_result
        return (el, rat_scale, omega_scale), {'band': 'multi', 'rat_spec': rat_spec, 'omega_spec': omega_spec}
    return None, None


def _classify_exponents_full(expr, rational_denoms, omega_denoms, bases_seen):
    """Classify exponent denominators and detect which bases appear."""
    from fractions import Fraction as Frac
    expr = traction_simplify(expr)

    if isinstance(expr, (Integer, Rational, Null)):
        return
    if isinstance(expr, Zero):
        rational_denoms.add(1)
        bases_seen.add('zero')
        return
    if isinstance(expr, Omega):
        rational_denoms.add(1)
        bases_seen.add('omega')
        return
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        base_name = 'zero' if isinstance(expr.base, Zero) else 'omega'
        bases_seen.add(base_name)
        exp = expr.exp
        omega_info = _extract_omega_rational(exp)
        if omega_info is not None:
            coeff, _ = omega_info
            omega_denoms.add(coeff.denominator)
        elif isinstance(exp, (Integer, Rational)):
            rational_denoms.add(int(exp.q) if isinstance(exp, Rational) else 1)
        return
    if isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            _classify_exponents_full(arg, rational_denoms, omega_denoms, bases_seen)
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        _classify_exponents_full(expr.base, rational_denoms, omega_denoms, bases_seen)


def _convert_omega_band(expr, omega_lcd):
    """Convert an expression with omega exponents to Element.
    g = 0^(ω/omega_lcd), scale = omega_lcd.
    0^(ω·p/q) → g^(omega_lcd·p/q)."""
    from chebyshev_ring import Element
    from fractions import Fraction as Frac

    expr = traction_simplify(expr)

    # Scalar atoms
    if isinstance(expr, Integer):
        return Element.from_int(int(expr))
    if isinstance(expr, Rational):
        return Element.from_fraction(Frac(expr.p, expr.q))
    if isinstance(expr, Null):
        return Element.zero_el()

    # Zero-powers with omega exponents
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        exp = expr.exp
        sign = -1 if isinstance(expr.base, Omega) else 1
        omega_info = _extract_omega_rational(exp)
        if omega_info is not None:
            coeff, _ = omega_info
            power = coeff * omega_lcd * sign
            if power.denominator != 1:
                return None
            return Element.u_power(int(power))
        return None

    # Sums
    if isinstance(expr, Add):
        result = None
        for term in Add.make_args(expr):
            t = _convert_omega_band(term, omega_lcd)
            if t is None:
                return None
            result = t if result is None else result + t
        return result

    # Products
    if isinstance(expr, Mul):
        result = None
        for factor in Mul.make_args(expr):
            f = _convert_omega_band(factor, omega_lcd)
            if f is None:
                return None
            result = f if result is None else result * f
        return result

    # Powers with integer exponent
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        base_el = _convert_omega_band(expr.base, omega_lcd)
        if base_el is not None:
            return base_el ** int(expr.exp)

    return None


def _has_omega_exponent(expr):
    """Check if an expression contains zero-powers with omega in the exponent."""
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        return expr.exp.has(Omega) if hasattr(expr.exp, 'has') else False
    if isinstance(expr, (Add, Mul)):
        return any(_has_omega_exponent(arg) for arg in expr.args)
    return False


def _extract_omega_rational(exp):
    """Extract (rational_coeff, omega_power) from an exponent like ω/7 or ω²·3/4.
    Returns (coeff, omega_exp) such that the exponent = coeff * ω^omega_exp,
    or None if it doesn't have this form. For simple ω*c, returns (c, 1)."""
    from fractions import Fraction as Frac

    # ω * rational: Mul(Rational, Omega)
    if isinstance(exp, Mul):
        rational_parts = []
        omega_count = 0
        for f in Mul.make_args(exp):
            if isinstance(f, (Integer, Rational)):
                rational_parts.append(f)
            elif isinstance(f, Omega):
                omega_count += 1
            elif isinstance(f, Pow) and isinstance(f.base, Omega) and isinstance(f.exp, Integer):
                omega_count += int(f.exp)
            else:
                return None
        if omega_count > 0 and rational_parts:
            coeff = Mul(*rational_parts)
            try:
                return Frac(int(coeff.p), int(coeff.q)) if isinstance(coeff, Rational) else Frac(int(coeff)), omega_count
            except (TypeError, ValueError, AttributeError):
                return None
    # Bare omega
    if isinstance(exp, Omega):
        return Frac(1), 1
    return None


def _try_multiband(expr):
    """Try to convert an expression to MultiBandElement.
    Band 1 (g₁): rational exponents. Band 2 (g₂): omega-containing exponents."""
    from chebyshev_ring import MultiBandElement
    from fractions import Fraction as Frac
    from math import lcm

    expr = traction_simplify(expr)

    # Collect LCD for each band
    rational_denoms = set()
    omega_denoms = set()
    _classify_exponents(expr, rational_denoms, omega_denoms)

    if not omega_denoms:
        return None  # no omega exponents, single-band suffices

    # Compute scales for each band
    rat_lcd = 1
    for d in rational_denoms:
        rat_lcd = lcm(rat_lcd, d)
    rat_lcd = max(rat_lcd, 2)

    omega_lcd = 1
    for d in omega_denoms:
        omega_lcd = lcm(omega_lcd, d)

    el = _convert_multiband(expr, rat_lcd, omega_lcd)
    if el is None:
        return None
    return (el, rat_lcd, omega_lcd)


def _classify_exponents(expr, rational_denoms, omega_denoms):
    """Classify exponent denominators into rational and omega-containing bands."""
    from fractions import Fraction as Frac
    expr = traction_simplify(expr)

    if isinstance(expr, (Integer, Rational, Null)):
        return
    if isinstance(expr, (Zero, Omega)):
        rational_denoms.add(1)
        return
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        exp = expr.exp
        omega_info = _extract_omega_rational(exp)
        if omega_info is not None:
            coeff, _ = omega_info
            omega_denoms.add(coeff.denominator)
        elif isinstance(exp, (Integer, Rational)):
            rational_denoms.add(int(exp.q) if isinstance(exp, Rational) else 1)
        return
    if isinstance(expr, (Add, Mul)):
        for arg in expr.args:
            _classify_exponents(arg, rational_denoms, omega_denoms)
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        _classify_exponents(expr.base, rational_denoms, omega_denoms)


def _convert_multiband(expr, rat_scale, omega_scale):
    """Convert a traction expression to MultiBandElement.
    g₁ handles rational exponents (scale=rat_scale), g₂ handles omega exponents (scale=omega_scale)."""
    from chebyshev_ring import MultiBandElement
    from fractions import Fraction as Frac

    expr = traction_simplify(expr)

    # Scalar atoms
    if isinstance(expr, Integer):
        return MultiBandElement.from_int(int(expr))
    if isinstance(expr, Rational):
        return MultiBandElement.from_int(0) + MultiBandElement.from_int(int(expr.p)) * Frac(1, int(expr.q))
    if isinstance(expr, Null):
        return MultiBandElement.zero_el()

    # Zero/Omega atoms → band 1
    if isinstance(expr, Zero):
        return MultiBandElement.g1() ** rat_scale
    if isinstance(expr, Omega):
        return MultiBandElement.g1_inv() ** rat_scale

    # Zero-powers
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        exp = expr.exp
        sign = -1 if isinstance(expr.base, Omega) else 1

        # Check for omega-containing exponent → band 2
        omega_info = _extract_omega_rational(exp)
        if omega_info is not None:
            coeff, _ = omega_info
            power = coeff * omega_scale * sign
            if power.denominator != 1:
                return None
            return MultiBandElement.g2() ** int(power)

        # Rational exponent → band 1
        try:
            r = Frac(int(exp.p), int(exp.q)) if isinstance(exp, Rational) else Frac(int(exp))
        except (TypeError, ValueError, AttributeError):
            return None
        power = r * rat_scale * sign
        if power.denominator != 1:
            return None
        return MultiBandElement.g1() ** int(power)

    # Sums
    if isinstance(expr, Add):
        result = None
        for term in Add.make_args(expr):
            t = _convert_multiband(term, rat_scale, omega_scale)
            if t is None:
                return None
            result = t if result is None else result + t
        return result

    # Products
    if isinstance(expr, Mul):
        result = None
        for factor in Mul.make_args(expr):
            f = _convert_multiband(factor, rat_scale, omega_scale)
            if f is None:
                return None
            result = f if result is None else result * f
        return result

    # Powers with integer exponent
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        base_el = _convert_multiband(expr.base, rat_scale, omega_scale)
        if base_el is not None:
            return base_el ** int(expr.exp)

    return None


def _convert_with_scale(expr, scale):
    """Convert a traction expression to Element using the given scale factor.
    0^(p/q) maps to u^(scale * p/q), which must be an integer."""
    from chebyshev_ring import Element
    from fractions import Fraction as Frac

    expr = traction_simplify(expr)

    # Scalar atoms
    if isinstance(expr, Integer):
        return Element.from_int(int(expr))
    if isinstance(expr, Rational):
        return Element.from_fraction(Frac(expr.p, expr.q))
    if isinstance(expr, Null):
        return Element.zero_el()

    # Zero-powers: 0^(p/q) → u^(scale * p/q)
    if isinstance(expr, Zero):
        return Element.u_power(scale)
    if isinstance(expr, Omega):
        return Element.u_power(-scale)
    if isinstance(expr, Pow) and isinstance(expr.base, (Zero, Omega)):
        exp = expr.exp
        try:
            r = Frac(int(exp.p), int(exp.q)) if isinstance(exp, Rational) else Frac(int(exp))
        except (TypeError, ValueError, AttributeError):
            return None
        if isinstance(expr.base, Omega):
            r = -r
        power = r * scale
        if power.denominator != 1:
            return None
        return Element.u_power(int(power))

    # Sums
    if isinstance(expr, Add):
        result = None
        for term in Add.make_args(expr):
            t = _convert_with_scale(term, scale)
            if t is None:
                return None
            result = t if result is None else result + t
        return result

    # Products
    if isinstance(expr, Mul):
        result = None
        for factor in Mul.make_args(expr):
            f = _convert_with_scale(factor, scale)
            if f is None:
                return None
            result = f if result is None else result * f
        return result

    # Powers with integer exponent
    if isinstance(expr, Pow):
        base_el = _convert_with_scale(expr.base, scale)
        if base_el is not None and isinstance(expr.exp, Integer):
            return base_el ** int(expr.exp)

    return None


def _decompose_ring_element(ring_el, ring_info, traction_str, complex_str, simplified):
    """Build a decomposition result dict from a chebyshev_ring Element."""
    import re
    import math
    from fractions import Fraction as Frac

    def _relabel(s):
        return re.sub(r'\bu\b', 'g', s)

    ring_str = _relabel(repr(ring_el))

    exponent = None
    exp_zero = _extract_zero_exponent(simplified)
    if exp_zero is not None:
        try:
            exponent = Rational(exp_zero)
        except (TypeError, ValueError):
            pass

    conj = ring_el.conj()
    conj_str = _relabel(repr(conj))
    norm = ring_el.norm()
    norm_str = repr(norm)

    band = ring_info.get('band', 'rational')
    base_denom = ring_info.get('base_denom')
    omega_lcd = ring_info.get('omega_lcd')
    spec = ring_info.get('spec')

    ring_label = f'Q[s][g] / (g\u00b2 \u2212 sg + 1)'

    if band == 'omega':
        n = omega_lcd
        sym = spec.symbol if spec else '0'
        g_str = spec.format_generator(n) if spec else f'0^(\u03c9/{n})'

        # s is exact traction: g + g⁻¹
        s_rational = {1: Frac(-2), 2: Frac(0), 3: Frac(1)}.get(n)
        if n == 2:
            gen_note = f'g = {g_str},  s = 0,  g\u00b2 = -1'
        else:
            gen_note = f'g = {g_str},  s = g + g\u207b\u00b9'

        if s_rational is not None:
            a_val = ring_el.a.eval_at(float(s_rational))
            b_val = ring_el.b.eval_at(float(s_rational))
            a_frac = Frac(a_val.real).limit_denominator(10000) if abs(a_val.imag) < 1e-12 else None
            b_frac = Frac(b_val.real).limit_denominator(10000) if abs(b_val.imag) < 1e-12 else None
            components = [
                ('a', str(a_frac) if a_frac is not None else repr(ring_el.a)),
                ('b', str(b_frac) if b_frac is not None else repr(ring_el.b)),
            ]
        else:
            components = [
                ('a', repr(ring_el.a)),
                ('b', repr(ring_el.b)),
            ]
    else:
        sym = spec.symbol if spec else '0'
        g_str = spec.format_generator(base_denom) if spec else f'0^(1/{base_denom})'
        gen_note = f'g = {g_str},  s = g + g\u207b\u00b9'
        components = [
            ('a', repr(ring_el.a)),
            ('b', repr(ring_el.b)),
        ]

    return {
        'traction_str': traction_str,
        'has_decomp': True,
        'ring_form': True,
        'ring_str': ring_str,
        'ring_label': ring_label,
        'gen_note': gen_note,
        'components': components,
        'norm_str': norm_str,
        'conj_str': conj_str,
        'is_unit': ring_el.can_invert(),
        'complex_str': complex_str,
        'exponent': exponent,
        'base_denom': base_denom if band == 'rational' else None,
        'band': band,
        'omega_lcd': omega_lcd,
        '_ring_el': ring_el,
        '_ring_info': ring_info,
    }


def _decompose_multiband(ring_el, rat_scale, omega_scale, traction_str, complex_str, simplified):
    """Build decomposition dict for a MultiBandElement."""
    ring_str = repr(ring_el)

    exponent = None
    exp_zero = _extract_zero_exponent(simplified)
    if exp_zero is not None:
        try:
            exponent = Rational(exp_zero)
        except (TypeError, ValueError):
            pass

    norm = ring_el.norm()
    norm_str = repr(norm)
    conj_str = f'\u03c3\u2081: {repr(ring_el.sigma1())}\n  \u03c3\u2082: {repr(ring_el.sigma2())}'

    if rat_scale == 2:
        g1_str = '0^(1/2)'
    else:
        g1_str = f'0^(1/{rat_scale})'

    if omega_scale == 1:
        g2_str = '0^\u03c9'
    else:
        g2_str = f'0^(\u03c9/{omega_scale})'

    components = [
        ('a', repr(ring_el.a)),
        ('b', repr(ring_el.b)),
        ('c', repr(ring_el.c)),
        ('d', repr(ring_el.d)),
    ]

    return {
        'traction_str': traction_str,
        'has_decomp': True,
        'ring_form': True,
        'ring_str': ring_str,
        'ring_label': 'Q[s\u2081,s\u2082][g\u2081,g\u2082] / (g\u2081\u00b2=s\u2081g\u2081\u22121, g\u2082\u00b2=s\u2082g\u2082\u22121)',
        'gen_note': f'g\u2081 = {g1_str},  g\u2082 = {g2_str},  s\u2081 = g\u2081+g\u2081\u207b\u00b9,  s\u2082 = g\u2082+g\u2082\u207b\u00b9',
        'components': components,
        'norm_str': norm_str,
        'conj_str': conj_str,
        'is_unit': ring_el.can_invert(),
        'complex_str': complex_str,
        'exponent': exponent,
        'base_denom': None,
    }


def _eval_ring_exact(decomp):
    """
    Compute the exact symbolic value of a ring element by substituting
    the traction expressions for g and s, then simplifying.

    Uses Horner's method with traction_simplify at each step to keep
    intermediate expressions manageable.

    Returns a formatted string of the exact traction value.
    """
    from chebyshev_ring import Element, QsPoly
    from sympy import Rational as SRat, Pow

    ring_el = decomp.get('_ring_el')
    ring_info = decomp.get('_ring_info', {})
    band = ring_info.get('band', decomp.get('band', 'rational'))

    if ring_el is None:
        return 'Ring element not available'

    if band == 'omega':
        omega_lcd = ring_info.get('omega_lcd', 2)
        # g = 0^(ω/n), g⁻¹ = 0^(-ω/n), s = g + g⁻¹
        g_expr = Pow(Zero(), Mul(Omega(), SRat(1, omega_lcd)))
        g_inv = Pow(Zero(), Mul(Omega(), SRat(-1, omega_lcd)))
        s_expr = traction_simplify(g_expr + g_inv)
    elif band == 'rational':
        base_denom = ring_info.get('base_denom', decomp.get('base_denom', 2))
        g_expr = Pow(Zero(), SRat(1, base_denom))
        g_inv = Pow(Zero(), SRat(-1, base_denom))
        s_expr = traction_simplify(g_expr + g_inv)
    else:
        return 'Multi-band exact evaluation not yet implemented'

    def eval_poly_horner(poly, x):
        """Evaluate QsPoly at traction expression x using Horner's method.
        Expands and simplifies at each step to keep expressions small."""
        from sympy import expand
        coeffs = poly.coeffs
        if not coeffs:
            return Integer(0)
        result = SRat(coeffs[-1])
        for i in range(len(coeffs) - 2, -1, -1):
            result = traction_simplify(expand(result * x) + SRat(coeffs[i]))
        return result

    a_val = eval_poly_horner(ring_el.a, s_expr)
    b_val = eval_poly_horner(ring_el.b, s_expr)

    if ring_el.b.is_zero():
        exact = a_val
    else:
        from sympy import expand
        exact = traction_simplify(expand(a_val + expand(b_val * g_expr)))

    return format_result(exact)


def _complex_at_pi2(expr):
    """Evaluate a traction expression numerically at θ = π/2.

    Maps:  0 → i,  ω → −i,  0^z → e^{izπ/2}  (with ω → 2 in exponents).
    Returns a Python complex, or None on failure.
    """
    import cmath

    expr = traction_simplify(expr)

    if isinstance(expr, (Integer, Rational)):
        return complex(expr)
    if isinstance(expr, Zero):
        return 1j
    if isinstance(expr, Omega):
        return -1j
    if isinstance(expr, Null):
        return 0.0

    if isinstance(expr, Pow):
        base, exponent = expr.base, expr.exp
        if isinstance(base, (Zero, Omega)):
            sign = 1.0 if isinstance(base, Zero) else -1.0
            try:
                exp_num = complex(exponent.subs(Omega(), Integer(2)).evalf())
            except Exception:
                return None
            return cmath.exp(1j * sign * cmath.pi / 2 * exp_num)
        pb = _complex_at_pi2(base)
        pe = _complex_at_pi2(exponent)
        if pb is None or pe is None:
            return None
        if pb == 0:
            return 0.0
        try:
            return pb ** pe
        except Exception:
            return None

    if isinstance(expr, Mul):
        result = complex(1)
        for arg in Mul.make_args(expr):
            v = _complex_at_pi2(arg)
            if v is None:
                return None
            result *= v
        return result

    if isinstance(expr, Add):
        result = complex(0)
        for arg in Add.make_args(expr):
            v = _complex_at_pi2(arg)
            if v is None:
                return None
            result += v
        return result

    try:
        return complex(expr.evalf())
    except Exception:
        return None


def _chebyshev_relation(n):
    """Compute T_n(s/2) as a polynomial in s (ascending Fraction coefficients).

    Uses the Chebyshev recurrence T_{k+1}(x) = 2x·T_k(x) − T_{k-1}(x)
    with x = s/2, giving T_{k+1}(s/2) = s·T_k(s/2) − T_{k-1}(s/2).

    Returns list of Fraction coefficients [c0, c1, ..., cn].
    """
    from fractions import Fraction as Frac
    if n == 0:
        return [Frac(1)]
    if n == 1:
        return [Frac(0), Frac(1, 2)]
    prev_prev = [Frac(1)]              # T_0(s/2) = 1
    prev = [Frac(0), Frac(1, 2)]      # T_1(s/2) = s/2
    for _ in range(2, n + 1):
        # s · prev  (shift coefficients right by 1)
        s_times = [Frac(0)] + prev[:]
        # subtract prev_prev
        length = max(len(s_times), len(prev_prev))
        cur = []
        for i in range(length):
            a = s_times[i] if i < len(s_times) else Frac(0)
            b = prev_prev[i] if i < len(prev_prev) else Frac(0)
            cur.append(a - b)
        prev_prev = prev
        prev = cur
    return prev


def _poly_mod(p, m):
    """Reduce polynomial p modulo m (both ascending-degree Fraction lists).

    Returns the remainder of p / m with degree < deg(m).
    """
    from fractions import Fraction as Frac
    p = list(p)
    deg_m = len(m) - 1
    lead_m = m[-1]
    while len(p) > deg_m:
        if p[-1] == Frac(0):
            p.pop()
            continue
        factor = p[-1] / lead_m
        offset = len(p) - len(m)
        for i in range(len(m)):
            p[i + offset] -= factor * m[i]
        p.pop()
    while len(p) > 1 and p[-1] == Frac(0):
        p.pop()
    return p


def _eval_frac_poly_horner(coeffs, s_expr):
    """Evaluate a polynomial (ascending-degree Fraction list) at a traction s_expr."""
    from fractions import Fraction as Frac
    from sympy import expand
    # Strip trailing zeros
    while len(coeffs) > 1 and coeffs[-1] == Frac(0):
        coeffs = coeffs[:-1]
    if not coeffs or all(c == Frac(0) for c in coeffs):
        return Integer(0)
    result = Rational(coeffs[-1].numerator, coeffs[-1].denominator)
    for i in range(len(coeffs) - 2, -1, -1):
        c = Rational(coeffs[i].numerator, coeffs[i].denominator)
        result = traction_simplify(expand(result * s_expr) + c)
    return result


def _reduce_ring_form(decomp):
    """
    Reduce the ring element using the Chebyshev identity 0 + ω = ∅
    and substitute g = 0^(1/n) to get a simplified traction expression.

    The identity 0 + ω = ∅ implies T_n(s/2) = 0 where s = g + g⁻¹,
    giving a polynomial relation on s. Reducing the ring polynomials
    mod this relation and substituting back produces a simpler form.

    For n=2: s² = 2, result has ≤ 4 terms (period-8 cycle).
    For n=3: s² = 3, similar structure with 0^(k/3) terms.
    For general n: T_n(s/2) = 0, result has ≤ 2n terms.

    Returns (formatted_string, complex_value) or None if not applicable.
    """
    from fractions import Fraction as Frac

    ring_el = decomp.get('_ring_el')
    ring_info = decomp.get('_ring_info', {})
    band = ring_info.get('band', decomp.get('band', 'rational'))
    base_denom = ring_info.get('base_denom', decomp.get('base_denom', 2))

    if ring_el is None or band != 'rational':
        return None

    n = base_denom

    # Compute the Chebyshev relation T_n(s/2) = 0
    m_poly = _chebyshev_relation(n)

    # Reduce a(s) and b(s) modulo the relation
    a_red = _poly_mod(list(ring_el.a.coeffs), m_poly)
    b_red = _poly_mod(list(ring_el.b.coeffs), m_poly)

    b_is_zero = all(c == Frac(0) for c in b_red)

    # Evaluate reduced polynomials at s = 0^(1/n) + ω^(1/n), g = 0^(1/n)
    s_expr = traction_simplify(
        Pow(Zero(), Rational(1, n)) + Pow(Omega(), Rational(1, n)))
    g_expr = Pow(Zero(), Rational(1, n))

    a_val = _eval_frac_poly_horner(a_red, s_expr)

    if b_is_zero:
        result_expr = a_val
    else:
        from sympy import expand
        b_val = _eval_frac_poly_horner(b_red, s_expr)
        result_expr = traction_simplify(expand(a_val + expand(b_val * g_expr)))

    result_str = format_result(result_expr)
    result_complex = _complex_at_pi2(result_expr)

    return result_str, result_complex


def _step_label(k, denom):
    """Format recurrence step k with denominator denom as a fraction label.
    E.g., k=3, denom=2 → 'a(3/2)'; k=4, denom=2 → 'a(2)'.
    """
    from math import gcd
    g = gcd(k, denom)
    num, den = k // g, denom // g
    if den == 1:
        return f'a({num})'
    return f'a({num}/{den})'


def _cheb_poly_str(n_steps, step_denom=2, var='s'):
    """
    Compute the modified Chebyshev polynomial via recurrence:
    T_0=2, T_1=s, T_n = s*T_{n-1} - T_{n-2}.

    n_steps: number of recurrence iterations.
    step_denom: each step represents 1/step_denom of a full unit.
        (step_denom=2 → half-integer steps, step_denom=16 → 1/16 steps, etc.)
    Returns (formatted_string, trace_list).
    """
    from sympy import Symbol as Sym, expand

    s = Sym(var)

    if n_steps == 0:
        return '2', [(_step_label(0, step_denom), '2')]
    if n_steps == 1:
        return str(s), [(_step_label(0, step_denom), '2'),
                         (_step_label(1, step_denom), str(s))]

    prev2 = Integer(2)
    prev1 = s
    trace = [(_step_label(0, step_denom), '2'),
             (_step_label(1, step_denom), str(s))]

    for k in range(2, n_steps + 1):
        curr = expand(s * prev1 - prev2)
        label = _step_label(k, step_denom)
        poly_str = str(curr).replace('**', '^').replace('*', '\u00b7')
        trace.append((label, poly_str))
        prev2 = prev1
        prev1 = curr

    final_str = str(prev1).replace('**', '^').replace('*', '\u00b7')
    return final_str, trace


def chebyshev_decompose(expr):
    """
    Analyze a traction expression for Chebyshev structure.
    Returns a dict describing the decomposition.
    """
    simplified = traction_simplify(expr)
    traction_str = format_result(simplified)

    # Try to get complex projection (both string and numeric)
    complex_val = None
    try:
        proj = project_complex(simplified)
        complex_str = format_complex(simplified)
        if not complex_str:
            complex_str = str(proj)
        complex_val = complex(proj.evalf())
    except Exception:
        complex_str = ''

    # Check for free symbols (can't decompose symbolically)
    if simplified.free_symbols:
        return {
            'traction_str': traction_str,
            'has_decomp': False,
            'note': 'Contains free variables \u2014 Chebyshev decomposition requires a concrete exponent',
            'complex_str': complex_str,
        }

    # --- Primary path: try to convert to a Chebyshev ring Element ---
    ring_result, ring_info = _expr_to_ring(simplified)
    if ring_result is not None:
        if isinstance(ring_result, tuple):
            # Multi-band: (element, rat_scale, omega_scale)
            el, rat_scale, omega_scale = ring_result
            return _decompose_multiband(el, rat_scale, omega_scale, traction_str, complex_str, simplified)
        else:
            return _decompose_ring_element(ring_result, ring_info, traction_str, complex_str, simplified)

    # --- Fallback: single zero-power analysis ---
    exponent = _extract_zero_exponent(simplified)

    # Handle Mul: scalar * zero-power
    coeff = None
    if exponent is None and isinstance(simplified, Mul):
        scalar_parts = []
        zero_exp = None
        for arg in Mul.make_args(simplified):
            e = _extract_zero_exponent(arg)
            if e is not None and zero_exp is None:
                zero_exp = e
            else:
                scalar_parts.append(arg)
        if zero_exp is not None:
            exponent = zero_exp
            coeff = Mul(*scalar_parts) if scalar_parts else None

    # Handle -1 = 0^2 in the integer cycle
    if exponent is None and simplified == S.NegativeOne:
        exponent = Integer(2)

    if exponent is None:
        return {
            'traction_str': traction_str,
            'has_decomp': False,
            'note': 'Not a zero-power expression',
            'complex_str': complex_str,
        }

    # Check if exponent is numeric
    try:
        exp_val = Rational(exponent)
    except (TypeError, ValueError):
        return {
            'traction_str': traction_str,
            'has_decomp': False,
            'note': f'Symbolic exponent 0^({format_result(exponent)}) \u2014 cannot compute Chebyshev polynomial',
            'complex_str': complex_str,
            'exponent': exponent,
        }

    # Format exponent string
    exp_str = format_result(exp_val)

    # Half-cycle label
    if exp_val.q == 1:
        half_label = str(int(exp_val))
    else:
        half_label = f'{exp_val.p}/{exp_val.q}'

    # Equivalence: what does 0^n simplify to?
    try:
        equiv = traction_simplify(Pow(Zero(), exp_val))
        equiv_str = format_result(equiv)
        if coeff is not None:
            equiv_str = f'{format_result(coeff)}\u00b7{equiv_str}'
    except Exception:
        equiv_str = '?'

    # Chebyshev polynomial: for exponent p/q, we need 2*exponent to be an
    # integer for the standard recurrence (u=0^(1/2), step_denom=2).
    # If not, use a finer grid: u_d = 0^(1/(2q)), step_denom = 2q.
    doubled = exp_val * 2
    if doubled.q == 1:
        # Standard half-integer case: 2*exponent is an integer
        n_steps = abs(int(doubled))
        step_denom = 2
    else:
        # Finer grid needed
        p_num = abs(exp_val.p)
        q_den = exp_val.q
        n_steps = 2 * p_num
        step_denom = 2 * q_den

    cheb_str, trace = _cheb_poly_str(n_steps, step_denom=step_denom)

    # Cap trace display
    if len(trace) > 12:
        trace = trace[:3] + [('...', '...')] + trace[-3:]

    return {
        'traction_str': traction_str,
        'has_decomp': True,
        'exponent': exp_val,
        'exp_str': exp_str,
        'half_label': half_label,
        'equiv_str': equiv_str,
        'cheb_str': cheb_str,
        'trace': trace,
        'complex_str': complex_str,
        'coeff': coeff,
        'step_denom': step_denom,
    }
