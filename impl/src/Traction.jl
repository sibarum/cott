module Traction

export TractionNum, ZeroPow, OmegaPow,
       𝟎, 𝟏, ω, neg1,
       traction, is_real, to_real,
       base0log, base0exp

using Base: show

# =============================================================================
# Core Type: TractionNum
# =============================================================================
#
# Every value in traction algebra is represented as 0^exponent.
# The exponent is a pair (r, w) of rationals, meaning:
#
#   0^(r + w*ω)
#
# where r is the "rational part" and w is the "omega coefficient".
#
# This captures:
#   Integer n     = 0^(0^n)  -- but we store known integers via their log_0
#   0             = 0^1      -- exponent (1, 0)
#   ω             = 0^(-1)   -- exponent (-1, 0)
#   1             = 0^0      -- exponent (0, 0)
#   -1            = 0^ω      -- exponent (0, 1)
#   i = √(-1)    = 0^(ω/2)  -- exponent (0, 1//2)
#   0^n           =          -- exponent (n, 0)
#   ω^n = 0^(-n) =          -- exponent (-n, 0)
#   -n  = 0^(ω^n)           -- exponent needs higher-order representation
#
# For the first milestone we handle the (r, w) plane.
# Higher-order exponents (0^(ω^n) for n>1, 0^(0^n) for n>1) are deferred.

"""
    TractionNum

A value in traction algebra, represented as 0^(r + w*ω) where r and w are rationals.
"""
struct TractionNum
    r::Rational{Int}    # rational part of exponent
    w::Rational{Int}    # ω-coefficient of exponent

    TractionNum(r::Rational{Int}, w::Rational{Int}) = new(r, w)
end

# Convenience constructors
TractionNum(r::Integer, w::Integer) = TractionNum(Rational{Int}(r), Rational{Int}(w))
TractionNum(r::Rational, w::Integer) = TractionNum(Rational{Int}(r), Rational{Int}(w))
TractionNum(r::Integer, w::Rational) = TractionNum(Rational{Int}(r), Rational{Int}(w))
TractionNum(r::Rational, w::Rational) = TractionNum(Rational{Int}(r), Rational{Int}(w))

# Single-argument: pure rational exponent (w=0)
TractionNum(r) = TractionNum(r, 0)

# Exponent accessors
exponent_r(t::TractionNum) = t.r
exponent_w(t::TractionNum) = t.w

# =============================================================================
# Named Constants
# =============================================================================

const 𝟎 = TractionNum(1, 0)       # 0 = 0^1
const 𝟏 = TractionNum(0, 0)       # 1 = 0^0
const ω = TractionNum(-1, 0)       # ω = 0^(-1)
const neg1 = TractionNum(0, 1)     # -1 = 0^ω

# =============================================================================
# Constructors for common forms
# =============================================================================

"""
    ZeroPow(n) -> TractionNum

Construct 0^n.
"""
ZeroPow(n) = TractionNum(n, 0)

"""
    OmegaPow(n) -> TractionNum

Construct ω^n = 0^(-n).
"""
OmegaPow(n) = TractionNum(-n, 0)

"""
    traction(n::Integer) -> TractionNum

Construct a traction representation of an integer.
Uses the identity: n = 0^(0^n), but since 0^n is itself a TractionNum
with exponent (n,0), the integer n has exponent (0^n) which in the
(r,w) plane maps to... a recursive structure.

For practical purposes, small integers are handled by known identities:
  0 = 0^1
  1 = 0^0
  -1 = 0^ω
  ω = 0^(-1)

Other integers require higher-order exponents (0^(0^n)) which are
outside our current (r, w) plane. We mark them specially.
"""
function traction(n::Integer)
    if n == 0
        return 𝟎
    elseif n == 1
        return 𝟏
    elseif n == -1
        return neg1
    else
        # n = 0^(0^n). The exponent 0^n has representation (n, 0) in our system,
        # but that makes the *exponent* a TractionNum, not a rational.
        # For now, we can't represent arbitrary integers in the (r,w) plane.
        # This is an honest limitation we'll address with nested exponents.
        error("Integer $n requires higher-order exponents (0^(0^$n)), not yet implemented")
    end
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, t::TractionNum)
    r, w = t.r, t.w

    # Check for known constants first
    if w == 0
        if r == 0
            print(io, "1")
        elseif r == 1
            print(io, "0")
        elseif r == -1
            print(io, "ω")
        else
            # 0^r for arbitrary rational r
            print(io, "0^")
            _print_exponent_part(io, r)
        end
    elseif r == 0
        if w == 1
            print(io, "-1")
        elseif w == -1
            print(io, "0^(-ω)")
        else
            print(io, "0^(")
            _print_w_exponent(io, w)
            print(io, ")")
        end
    else
        # General case: 0^(r + w*ω)
        print(io, "0^(")
        _print_exponent_part(io, r)
        if w > 0
            print(io, "+")
        end
        _print_w_exponent(io, w)
        print(io, ")")
    end
end

function _print_exponent_part(io::IO, r::Rational)
    if denominator(r) == 1
        print(io, numerator(r))
    else
        print(io, r)
    end
end

function _print_w_exponent(io::IO, w::Rational)
    if w == 1
        print(io, "ω")
    elseif w == -1
        print(io, "-ω")
    elseif denominator(w) == 1
        print(io, numerator(w), "ω")
    else
        # e.g. ω/2 or 3ω/4
        n, d = numerator(w), denominator(w)
        if n == 1
            print(io, "ω/", d)
        elseif n == -1
            print(io, "-ω/", d)
        else
            print(io, n, "ω/", d)
        end
    end
end

# =============================================================================
# Equality
# =============================================================================

Base.:(==)(a::TractionNum, b::TractionNum) = (a.r == b.r) && (a.w == b.w)
Base.hash(t::TractionNum, h::UInt) = hash(t.w, hash(t.r, h))

# =============================================================================
# base-0 logarithm and exponentiation (the core involution)
# =============================================================================

"""
    base0log(t::TractionNum) -> TractionNum

Compute log_0(t). Since t = 0^(r + w*ω), we have log_0(t) = r + w*ω,
which is itself a traction exponent. But representing "r + w*ω" as a
TractionNum means finding e such that 0^e = r + w*ω.

For values where the exponent is purely in {0, ω} basis:
  log_0(0^n) = n  (when n is representable)
  log_0(1) = 0
  log_0(0) = 1
  log_0(ω) = -1
  log_0(-1) = ω
"""
function base0log(t::TractionNum)
    # log_0(0^(r + w*ω)) = r + w*ω as a value
    # We need to represent the rational r + w*ω as 0^something.
    #
    # Special cases we can handle:
    if t.w == 0
        # t = 0^r, so log_0(t) = r
        # r as a traction value: if r is an integer we need 0^(0^r),
        # but for r=0 -> log is 0 (= 0^1), r=1 -> log is 1 (= 0^0), etc.
        r = t.r
        if r == 0
            # log_0(1) = 0
            return 𝟎
        elseif r == 1
            # log_0(0) = 1
            return 𝟏
        elseif r == -1
            # log_0(ω) = -1
            return neg1
        else
            # log_0(0^r) = r, but r as a TractionNum requires 0^(0^r)
            # which is higher-order. For now, return the exponent as ZeroPow.
            # This is valid: r = 0^(0^r) for integer r, and for rational r
            # it's the generalized form.
            return ZeroPow(r)
        end
    elseif t.r == 0 && t.w == 1
        # t = -1 = 0^ω, so log_0(-1) = ω
        return ω
    end

    error("base0log not yet implemented for general exponent ($(t.r), $(t.w))")
end

"""
    base0exp(t::TractionNum) -> TractionNum

Compute 0^t. Since t is itself 0^(r + w*ω), this gives 0^(0^(r+w*ω)).

For simple cases:
  0^1 = 0, 0^0 = 1, 0^(-1) = ω, 0^ω = -1
"""
function base0exp(t::TractionNum)
    # 0^t where t = 0^(r + w*ω)
    # We use the identity 0^(0^n) = n for the involution.
    r, w = t.r, t.w

    if w == 0
        # t = 0^r (a pure zero-power), so 0^t = 0^(0^r)
        # By the involution: 0^(0^n) = n for integer n
        # For rational r, this generalizes.
        if r == 0
            # t = 1, so 0^1 = 0
            return 𝟎
        elseif r == 1
            # t = 0, so 0^0 = 1
            return 𝟏
        elseif r == -1
            # t = ω, so 0^ω = -1
            return neg1
        end
    elseif r == 0 && w == 1
        # t = -1, so 0^(-1) = ω
        return ω
    end

    error("base0exp not yet implemented for $(t)")
end

# =============================================================================
# Multiplication: 0^a * 0^b = 0^(a+b)
# =============================================================================
#
# This is the fundamental rule: multiplication in traction algebra corresponds
# to addition of exponents.
#
# 0^(r1 + w1*ω) * 0^(r2 + w2*ω) = 0^((r1+r2) + (w1+w2)*ω)

function Base.:*(a::TractionNum, b::TractionNum)
    TractionNum(a.r + b.r, a.w + b.w)
end

# =============================================================================
# Division: 0^a / 0^b = 0^(a-b)
# =============================================================================

function Base.:/(a::TractionNum, b::TractionNum)
    TractionNum(a.r - b.r, a.w - b.w)
end

# =============================================================================
# Multiplicative Inverse: (0^a)^(-1) = 0^(-a)
# =============================================================================

Base.inv(t::TractionNum) = TractionNum(-t.r, -t.w)

# =============================================================================
# Exponentiation: (0^a)^b
# =============================================================================
#
# The rule is (0^a)^b = 0^(a*b).
# When b is a TractionNum 0^(rb + wb*ω), we need to compute:
#   a * b  where a = (ra + wa*ω) and b is the VALUE represented by 0^(rb + wb*ω)
#
# This is where it gets interesting. We can't just multiply exponents naively
# because b is a traction value, not a plain number.
#
# For the cases we can handle:
#   - b is a known integer or simple rational
#   - b is a zero/omega power we recognize

function Base.:^(base::TractionNum, exp::TractionNum)
    # (0^a)^b = 0^(a * b_value)
    # where a = (base.r, base.w) and b_value is what exp represents as a number.
    #
    # Strategy: if we can extract a "scalar value" from exp, multiply the exponent.
    # Otherwise, build a symbolic result.

    rb, wb = exp.r, exp.w

    # Case: exp is a pure zero-power with integer exponent -> exp represents 0^rb
    # The VALUE of 0^rb depends on rb:
    #   0^0 = 1 (value 1), 0^1 = 0 (value 0), etc.
    # But we want to raise base to the POWER of exp's value.
    #
    # For the key use case: base^n where n is an integer-like value.
    # If exp = 0^0 = 1, then base^1 = base
    # If exp = 0^(0,0) = 1, result is base
    #
    # Actually, let's think about this differently.
    # We want: (0^a)^(0^e) = 0^(a * 0^e)
    # The exponent of the result is a * 0^e, which means we need to
    # multiply the exponent pair (ra, wa) by the VALUE that 0^e represents.
    #
    # For exp where we know the scalar value:

    val = _try_scalar_value(exp)
    if val !== nothing
        return TractionNum(base.r * val, base.w * val)
    end

    # Special case: base is 0 (= 0^1), so 0^exp stays as-is but with
    # exponent = exp's exponent coordinates. Wait, no:
    # 0^exp = 0^(0^(rb + wb*ω))
    # If base = 0 = 0^1, then (0^1)^exp = 0^(1 * exp_value) = 0^exp_value
    if base == 𝟎
        # 0 raised to a power: the result's exponent IS the value of exp
        # For exp = 0^(rb + wb*ω), the value is whatever 0^(rb + wb*ω) represents
        # In the (r,w) plane, this is just exp itself, so result is 0^exp
        # which means the result's exponent is (rb, wb)... no wait.
        # 0^exp means the result has exponent = exp_value.
        # exp = 0^(rb + wb*ω), and we're computing 0^(that value).
        # The result is 0^(0^(rb + wb*ω)) and by involution for certain cases
        # 0^(0^n) = n.
        if wb == 0 && denominator(rb) == 1
            # exp = 0^n for integer n, and 0^(0^n) = n
            n = numerator(rb)
            return traction(n)
        end
    end

    # General power: encode as 0^(a * exp_value)
    # where exp_value in the (r,w) plane is the exponent pair of exp
    error("Exponentiation ($(base))^($(exp)) not yet reducible in the (r,w) plane")
end

"""
Try to extract a plain rational scalar value from a TractionNum,
for cases where we know the mapping.
"""
function _try_scalar_value(t::TractionNum)::Union{Rational{Int}, Nothing}
    r, w = t.r, t.w
    if r == 0 && w == 0
        return 1 // 1      # 0^0 = 1
    end
    # For other values, we can't easily extract a scalar without
    # the full higher-order exponent machinery.
    # But we DO know some:
    #   0^1 = 0 -> scalar 0... but multiplying by 0 in exponent space
    #   would collapse everything. That's actually correct for (0^a)^0 = 0^0 = 1.
    #   Wait: x^0 = 1 for all x. So if exp represents value 0, result is 1.
    if r == 1 && w == 0
        return 0 // 1      # 0^1 = 0 (the value zero)
    end

    # -1 = 0^ω
    if r == 0 && w == 1
        return -1 // 1
    end

    # Rational p/q where we can determine it from the exponent
    # This is tricky because most traction values aren't simple rationals.
    # For now, only handle ±1 and 0.
    nothing
end

# =============================================================================
# Exponentiation with rational exponents (convenience)
# =============================================================================

function Base.:^(base::TractionNum, n::Integer)
    TractionNum(base.r * n, base.w * n)
end

function Base.:^(base::TractionNum, q::Rational)
    TractionNum(base.r * q, base.w * q)
end

# =============================================================================
# Addition: requires matching exponent structure
# =============================================================================
#
# Addition in traction algebra does NOT correspond to a simple operation on
# exponents. 0^a + 0^b has no closed form in general.
# However, specific cases work:
#   - 0^a + 0^a = 2 * 0^a (but 2 is itself 0^(0^2), higher-order)
#   - Additive identity: t + 0 preserves magnitude
#   - Additive inverse: t + (-t) = null (erasure)
#
# For now, we implement what we can and leave the rest symbolic.

struct TractionSum
    terms::Vector{TractionNum}
end

function Base.:+(a::TractionNum, b::TractionNum)
    # Additive cancellation: a + (-a) = null
    # -a has exponent (a.r, a.w) with sign flip via ω:
    # Negation: -t. If t = 0^(r + w*ω), then -t = (-1)*t = 0^ω * 0^(r+wω) = 0^(r + (w+1)ω)
    neg_a = TractionNum(a.r, a.w + 1)
    if neg_a == b
        return 𝟎  # erasure to additive identity (simplified; should be null)
    end

    neg_b = TractionNum(b.r, b.w + 1)
    if neg_b == a
        return 𝟎
    end

    # Adding zero (magnitude preservation)
    if b == 𝟎
        return a
    end
    if a == 𝟎
        return b
    end

    # General case: return symbolic sum
    return TractionSum([a, b])
end

# Negation: -t = (-1) * t = 0^ω * 0^(r + wω) = 0^(r + (w+1)ω)
Base.:-(t::TractionNum) = TractionNum(t.r, t.w + 1)

function Base.:-(a::TractionNum, b::TractionNum)
    a + (-b)
end

function Base.show(io::IO, s::TractionSum)
    for (i, term) in enumerate(s.terms)
        if i > 1
            print(io, " + ")
        end
        print(io, term)
    end
end

# =============================================================================
# Square Root: √(t) = t^(1/2)
# =============================================================================

"""
    sqrt(t::TractionNum)

Compute √t = t^(1/2) in traction algebra.
Key derivation: √(-1) = (-1)^(1/2) = (0^ω)^(1/2) = 0^(ω/2)
"""
function Base.sqrt(t::TractionNum)
    return t ^ (1//2)
end

# =============================================================================
# Projection to ℝ (or ℂ) -- lossy, for comparison with classical arithmetic
# =============================================================================

"""
    is_real(t::TractionNum) -> Bool

Check if this traction value has a real numeric projection.
"""
function is_real(t::TractionNum)
    t.w == 0 && t.r == 0 ||          # 1
    t.r == 1 && t.w == 0 ||          # 0
    t.r == 0 && t.w == 1 ||          # -1
    t.r == -1 && t.w == 0            # ω (projects to Inf)
end

"""
    to_real(t::TractionNum) -> Float64

Project a traction value to a real number. Lossy.
"""
function to_real(t::TractionNum)
    r, w = t.r, t.w
    if w == 0
        if r == 0;  return 1.0; end
        if r == 1;  return 0.0; end
        if r == -1; return Inf; end
    end
    if r == 0 && w == 1
        return -1.0
    end
    error("No real projection for $(t)")
end

# =============================================================================
# Predicate helpers
# =============================================================================

is_zero(t::TractionNum) = t == 𝟎
is_one(t::TractionNum) = t == 𝟏
is_omega(t::TractionNum) = t == ω
is_neg_one(t::TractionNum) = t == neg1

"""True if this is a pure zero-power 0^n with no ω component."""
is_zero_power(t::TractionNum) = t.w == 0

"""True if this has an ω component in its exponent."""
has_omega(t::TractionNum) = t.w != 0

"""Return the exponent as a human-readable string."""
function exponent_str(t::TractionNum)
    r, w = t.r, t.w
    if w == 0
        return string(r)
    elseif r == 0
        return w == 1 ? "ω" : "$(w)ω"
    else
        return "$(r) + $(w)ω"
    end
end

end # module Traction
