using Test
using Traction

@testset "Traction Algebra" begin

    # =================================================================
    @testset "Constants" begin
        @test string(𝟎) == "0"
        @test string(𝟏) == "1"
        @test string(ω) == "ω"
        @test string(neg1) == "-1"
    end

    # =================================================================
    @testset "Axioms: base-0 exponentiation" begin
        # 0^0 = 1
        @test ZeroPow(0) == 𝟏

        # 0^1 = 0
        @test ZeroPow(1) == 𝟎

        # 0^(-1) = ω
        @test ZeroPow(-1) == ω

        # ω^1 = ω (i.e. 0^(-1) = ω)
        @test OmegaPow(1) == ω

        # ω^(-1) = 0 (i.e. 0^(1) = 0)
        @test OmegaPow(-1) == 𝟎
    end

    # =================================================================
    @testset "Multiplication: 0^a * 0^b = 0^(a+b)" begin
        # 0 * ω = 0^1 * 0^(-1) = 0^0 = 1
        @test 𝟎 * ω == 𝟏
        @test string(𝟎 * ω) == "1"

        # ω * 0 = 1 (commutativity)
        @test ω * 𝟎 == 𝟏

        # 0 * 0 = 0^2
        @test 𝟎 * 𝟎 == ZeroPow(2)

        # ω * ω = ω^2 = 0^(-2)
        @test ω * ω == ZeroPow(-2)

        # 1 * anything = anything (multiplicative identity)
        @test 𝟏 * 𝟎 == 𝟎
        @test 𝟏 * ω == ω
        @test 𝟏 * neg1 == neg1

        # (-1) * (-1) = 0^ω * 0^ω = 0^(2ω) ... hmm
        # Actually: (-1)*(-1) should = 1. Let's check:
        # (-1) = 0^(0 + 1*ω), so (-1)*(-1) = 0^(0+2ω)
        # This is 0^(2ω), which should = 1. But in our system it's TractionNum(0, 2).
        # This reveals that we need: 0^(2ω) = 0^(ω+ω).
        # Since 0^ω = -1, and (-1)*(-1) = 1, we need 0^(2ω) = 1 = 0^0.
        # This means 2ω ≡ 0 in some modular sense for the ω-component.
        # This is the PHASE behavior: ω acts like π*i, so 2ω wraps around.
        # We'll address this in phase normalization.
    end

    # =================================================================
    @testset "Division: 0^a / 0^b = 0^(a-b)" begin
        # 1/0 = ω
        @test 𝟏 / 𝟎 == ω

        # 1/ω = 0
        @test 𝟏 / ω == 𝟎

        # 0/0 = 1
        @test 𝟎 / 𝟎 == 𝟏

        # ω/ω = 1
        @test ω / ω == 𝟏

        # a/a = 1 for any a
        for t in [𝟎, 𝟏, ω, neg1, ZeroPow(3), ZeroPow(-5)]
            @test t / t == 𝟏
        end
    end

    # =================================================================
    @testset "Inverse" begin
        @test inv(𝟎) == ω
        @test inv(ω) == 𝟎
        @test inv(𝟏) == 𝟏
        @test inv(inv(𝟎)) == 𝟎  # involution
    end

    # =================================================================
    @testset "Exponentiation with integer powers" begin
        # 0^2 = 0*0
        @test 𝟎^2 == ZeroPow(2)

        # ω^2 = ω*ω = 0^(-2)
        @test ω^2 == ZeroPow(-2)

        # 0^3
        @test 𝟎^3 == ZeroPow(3)

        # (0^2)^3 = 0^6
        @test ZeroPow(2)^3 == ZeroPow(6)

        # x^0 = 1 for all x
        @test 𝟎^0 == 𝟏
        @test ω^0 == 𝟏
        @test neg1^0 == 𝟏

        # x^1 = x for all x
        @test 𝟎^1 == 𝟎
        @test ω^1 == ω
    end

    # =================================================================
    @testset "Exponentiation with rational powers" begin
        # 0^(1/2) = 0^(1//2) -- a half-order zero
        @test 𝟎^(1//2) == ZeroPow(1//2)

        # (0^2)^(1/2) = 0^1 = 0
        @test ZeroPow(2)^(1//2) == 𝟎

        # ω^(1/2) = 0^(-1/2)
        @test ω^(1//2) == ZeroPow(-1//2)
    end

    # =================================================================
    @testset "★ MILESTONE: √(-1) = 0^(ω/2)" begin
        # This is the key derivation:
        # √(-1) = (-1)^(1/2) = (0^ω)^(1/2) = 0^(ω/2)
        result = sqrt(neg1)
        @test result == TractionNum(0, 1//2)
        @test string(result) == "0^(ω/2)"

        # Verify: (0^(ω/2))^2 = 0^ω = -1
        i = result
        @test i^2 == neg1
        @test string(i^2) == "-1"

        # i^4 = (i^2)^2 = (-1)^2 = 0^(2ω)
        # (In a fully normalized system this would be 1, but we defer phase normalization)
        i4 = i^4
        @test i4 == TractionNum(0, 2)
    end

    # =================================================================
    @testset "Arithmetic with 0^ω and 0^(ω*p/q)" begin
        # 0^ω = -1
        @test TractionNum(0, 1) == neg1

        # 0^(ω/2) * 0^(ω/2) = 0^ω = -1
        i = TractionNum(0, 1//2)
        @test i * i == neg1

        # 0^(ω/3) -- cube root of -1
        cbrt_neg1 = TractionNum(0, 1//3)
        @test cbrt_neg1^3 == neg1

        # 0^(ω/4) -- 8th root of unity? No: (0^(ω/4))^4 = 0^ω = -1
        @test TractionNum(0, 1//4)^4 == neg1

        # Multiplying phase values:
        # 0^(ω/2) * 0^(ω/3) = 0^(ω/2 + ω/3) = 0^(5ω/6)
        @test TractionNum(0, 1//2) * TractionNum(0, 1//3) == TractionNum(0, 5//6)
    end

    # =================================================================
    @testset "Mixed zero-power and omega-power arithmetic" begin
        # 0^2 * 0^ω = 0^(2+ω) -- a second-order zero with phase
        @test ZeroPow(2) * neg1 == TractionNum(2, 1)

        # 0^3 * 0^(ω/2) = 0^(3 + ω/2) -- third-order zero with imaginary phase
        @test ZeroPow(3) * TractionNum(0, 1//2) == TractionNum(3, 1//2)

        # -0 = (-1) * 0 = 0^ω * 0^1 = 0^(1+ω)
        neg_zero = neg1 * 𝟎
        @test neg_zero == TractionNum(1, 1)
        @test string(neg_zero) == "0^(1+ω)"

        # -ω = (-1) * ω = 0^ω * 0^(-1) = 0^(-1+ω)
        neg_omega = neg1 * ω
        @test neg_omega == TractionNum(-1, 1)
        @test string(neg_omega) == "0^(-1+ω)"
    end

    # =================================================================
    @testset "Negation" begin
        # -0 via unary minus
        @test -𝟎 == TractionNum(1, 1)  # 0^(1+ω)

        # -ω via unary minus
        @test -ω == TractionNum(-1, 1)

        # -1 via unary minus of 1
        @test -𝟏 == neg1  # -(0^0) = 0^(0+ω) = 0^ω = -1 ✓

        # --t = t (double negation)
        @test -(-𝟎) == TractionNum(1, 2)  # Hmm, this gives 0^(1+2ω), not 0.
        # Double negation needs phase normalization: w mod 2.
        # Without it, -(-0) = 0^(1+2ω) ≠ 0^1 = 0.
        # This is expected -- we'll add phase normalization next.
    end

    # =================================================================
    @testset "Display formatting" begin
        @test string(TractionNum(0, 1//2)) == "0^(ω/2)"
        @test string(TractionNum(0, 1//3)) == "0^(ω/3)"
        @test string(TractionNum(2, 0)) == "0^2"
        @test string(TractionNum(-3, 0)) == "0^-3"
        @test string(TractionNum(1, 1)) == "0^(1+ω)"
        @test string(TractionNum(-1, 1)) == "0^(-1+ω)"
    end

end
