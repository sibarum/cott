 using Traction
using Plots

# =============================================================================
# Exponent Plane Visualization
# =============================================================================
#
# Every traction number 0^(r + w*ω) maps to a point (r, w) on the
# "exponent plane." This is a log-polar space where:
#
#   - r (horizontal) controls magnitude: 0^r
#   - w (vertical)   controls phase:     w*ω acts like w*π*i
#
# Multiplication becomes vector addition of exponent pairs.
# Division becomes vector subtraction.
# Exponentiation becomes scalar scaling.


"""
    plot_exponent_plane(; kwargs...) -> Plot

Draw the exponent plane with named constants and grid.
"""
function plot_exponent_plane(; xlims=(-3, 4), ylims=(-1, 2.5), size=(700, 550))
    p = plot(;
        xlims, ylims, size,
        xlabel = "r  (magnitude exponent)",
        ylabel = "w  (phase coefficient)",
        title = "Traction Exponent Plane",
        aspect_ratio = 1,
        legend = :topright,
        grid = true,
        gridalpha = 0.3,
        framestyle = :origin,
        fontfamily = "serif",
        titlefontsize = 14,
        guidefontsize = 11,
        background_color = :white,
    )

    # Phase reference bands
    hspan!(p, [0.98, 1.02]; fillalpha=0.07, fillcolor=:red, label=nothing)
    annotate!(p, xlims[2] - 0.3, 1.07, text("negative", 8, :right, :red, "serif"))

    hspan!(p, [0.48, 0.52]; fillalpha=0.07, fillcolor=:purple, label=nothing)
    annotate!(p, xlims[2] - 0.3, 0.57, text("imaginary", 8, :right, :purple, "serif"))

    # Named constants
    constants = [
        (1,    0,    "0",  :below),
        (0,    0,    "1",  :below),
        (-1,   0,    "ω",  :below),
        (0,    1,    "-1", :above),
        (0,    1//2, "i = 0^(ω/2)", :right),
        (2,    0,    "0²", :below),
        (-2,   0,    "ω²", :below),
        (1,    1,    "-0 = 0^(1+ω)", :right),
        (-1,   1,    "-ω = 0^(-1+ω)", :left),
        (3,    0,    "0³", :below),
    ]

    xs = [Float64(c[1]) for c in constants]
    ys = [Float64(c[2]) for c in constants]
    scatter!(p, xs, ys;
        markersize = 7,
        markercolor = :crimson,
        markerstrokewidth = 1.5,
        markerstrokecolor = :darkred,
        label = "constants",
    )
    for (x, y, lbl, pos) in constants
        offset = pos == :above ? (0, 10) :
                 pos == :below ? (0, -12) :
                 pos == :right ? (10, 0) :
                 (-10, 0)
        annotate!(p, Float64(x) + offset[1]/60, Float64(y) + offset[2]/60,
            text(lbl, 9, pos == :left ? :right : :left, :darkred, "serif"))
    end

    return p
end


"""
    plot_multiply!(p, a::TractionNum, b::TractionNum; kwargs...)

Show multiplication a * b = a+b as vector addition on the exponent plane.
Draws vectors from origin to a, origin to b, and the parallelogram to a*b.
"""
function plot_multiply!(p, a::TractionNum, b::TractionNum;
        color_a = :seagreen, color_b = :royalblue, color_result = :darkorange)

    ar, aw = Float64(a.r), Float64(a.w)
    br, bw = Float64(b.r), Float64(b.w)
    result = a * b
    cr, cw = Float64(result.r), Float64(result.w)

    # Vectors from origin
    quiver!(p, [0], [0]; quiver=([ar], [aw]),
        linewidth=2.5, color=color_a, label="a = $(a)")
    quiver!(p, [0], [0]; quiver=([br], [bw]),
        linewidth=2.5, color=color_b, label="b = $(b)")

    # Parallelogram: show b shifted to tip of a
    plot!(p, [ar, cr], [aw, cw];
        linewidth=1.5, linestyle=:dash, color=color_b, label=nothing)
    # And a shifted to tip of b
    plot!(p, [br, cr], [bw, cw];
        linewidth=1.5, linestyle=:dash, color=color_a, label=nothing)

    # Result vector
    quiver!(p, [0], [0]; quiver=([cr], [cw]),
        linewidth=3, color=color_result, label="a×b = $(result)")

    # Result point
    scatter!(p, [cr], [cw];
        markersize=9, markercolor=color_result,
        markerstrokewidth=2, markerstrokecolor=:black, label=nothing)

    return p
end


"""
    plot_divide!(p, a::TractionNum, b::TractionNum; kwargs...)

Show division a / b as vector subtraction on the exponent plane.
"""
function plot_divide!(p, a::TractionNum, b::TractionNum;
        color_a = :seagreen, color_b = :royalblue, color_result = :darkorange)

    ar, aw = Float64(a.r), Float64(a.w)
    br, bw = Float64(b.r), Float64(b.w)
    result = a / b
    cr, cw = Float64(result.r), Float64(result.w)

    # Vector a
    quiver!(p, [0], [0]; quiver=([ar], [aw]),
        linewidth=2.5, color=color_a, label="a = $(a)")

    # Vector b (shown dashed, being subtracted)
    quiver!(p, [0], [0]; quiver=([br], [bw]),
        linewidth=2, linestyle=:dash, color=color_b, label="b = $(b)")

    # Subtraction: -b from a
    plot!(p, [ar, cr], [aw, cw];
        linewidth=1.5, linestyle=:dash, color=color_b, label=nothing,
        arrow=true)

    # Result
    quiver!(p, [0], [0]; quiver=([cr], [cw]),
        linewidth=3, color=color_result, label="a÷b = $(result)")

    scatter!(p, [cr], [cw];
        markersize=9, markercolor=color_result,
        markerstrokewidth=2, markerstrokecolor=:black, label=nothing)

    return p
end


"""
    plot_power!(p, base::TractionNum, n; kwargs...)

Show exponentiation base^n as scalar scaling of the exponent vector.
"""
function plot_power!(p, base::TractionNum, n;
        color_base = :seagreen, color_result = :darkorange)

    br, bw = Float64(base.r), Float64(base.w)
    result = base ^ n
    cr, cw = Float64(result.r), Float64(result.w)

    # Original vector
    quiver!(p, [0], [0]; quiver=([br], [bw]),
        linewidth=2.5, color=color_base, label="base = $(base)")

    # Scaling line through origin
    t_min, t_max = min(0.0, Float64(n), 1.0) - 0.3, max(0.0, Float64(n), 1.0) + 0.3
    plot!(p, [t_min * br, t_max * br], [t_min * bw, t_max * bw];
        linewidth=1, linestyle=:dot, color=:gray, label=nothing)

    # Result vector
    quiver!(p, [0], [0]; quiver=([cr], [cw]),
        linewidth=3, color=color_result, label="base^$(n) = $(result)")

    scatter!(p, [cr], [cw];
        markersize=9, markercolor=color_result,
        markerstrokewidth=2, markerstrokecolor=:black, label=nothing)

    return p
end


"""
    plot_phase_circle(; kwargs...) -> Plot

Show the phase circle: the w-axis wraps with period 2,
so w=0 → 1, w=1/2 → i, w=1 → -1, w=3/2 → -i.
"""
function plot_phase_circle(; size=(500, 500))
    θ = range(0, 2π; length=200)

    p = plot(cos.(θ), sin.(θ);
        linewidth=2, color=:gray, label=nothing,
        aspect_ratio=1, size,
        title="Phase Circle (w-axis)",
        xlabel="real part", ylabel="imaginary part",
        framestyle=:origin,
        xlims=(-1.6, 1.6), ylims=(-1.6, 1.6),
        fontfamily="serif",
        titlefontsize=14,
        gridalpha=0.2,
    )

    # Key phase points: w maps to angle w*π
    phases = [
        (0,    "1\nw=0"),
        (1//2, "i\nw=½"),
        (1,    "-1\nw=1"),
        (3//2, "-i\nw=3/2"),
    ]

    for (w, lbl) in phases
        angle = Float64(w) * π
        x, y = cos(angle), sin(angle)
        scatter!(p, [x], [y];
            markersize=8, markercolor=:crimson,
            markerstrokewidth=2, markerstrokecolor=:darkred,
            label=nothing)

        # Label offset
        ox = x * 0.25
        oy = y * 0.25
        annotate!(p, x + ox, y + oy, text(lbl, 10, :center, :darkred, "serif"))
    end

    # Draw some intermediate phase points
    for w in [1//4, 1//3, 2//3, 3//4, 5//4, 4//3, 5//3, 7//4]
        angle = Float64(w) * π
        x, y = cos(angle), sin(angle)
        scatter!(p, [x], [y];
            markersize=4, markercolor=:steelblue,
            markerstrokewidth=1, label=nothing)
    end

    # Arrow showing w increasing (counterclockwise)
    annotate!(p, 0.0, 0.0, text("w increases\ncounterclockwise", 9, :center, :gray, "serif"))

    return p
end


"""
    demo_multiplication() -> Plot

Demonstrate that 0 × ω = 1 via vector addition of exponents.
"""
function demo_multiplication()
    p = plot_exponent_plane()
    plot_multiply!(p, 𝟎, ω)
    title!(p, "Multiplication: 0 × ω = 1\n(exponent addition: (1,0) + (-1,0) = (0,0))")
    return p
end


"""
    demo_sqrt_neg1() -> Plot

Show that √(-1) = 0^(ω/2) via exponent scaling.
"""
function demo_sqrt_neg1()
    p = plot_exponent_plane()
    plot_power!(p, neg1, 1//2)
    title!(p, "Square Root: √(-1) = (-1)^(1/2) = 0^(ω/2)\n(exponent scaling: (0,1) × ½ = (0,½))")
    return p
end


"""
    demo_division_by_zero() -> Plot

Show that 1 ÷ 0 = ω, totally defined via vector subtraction.
"""
function demo_division_by_zero()
    p = plot_exponent_plane()
    plot_divide!(p, 𝟏, 𝟎)
    title!(p, "Division by Zero: 1 ÷ 0 = ω\n(exponent subtraction: (0,0) - (1,0) = (-1,0))")
    return p
end


"""
    demo_all()

Generate and display all demo plots.
"""
function demo_all()
    p1 = plot_exponent_plane()
    p2 = demo_multiplication()
    p3 = demo_division_by_zero()
    p4 = demo_sqrt_neg1()
    p5 = plot_phase_circle()

    combined = plot(p1, p2, p3, p4, p5, layout=(3, 2), size=(1400, 1200))
    display(combined)

    return combined
end


# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    demo_all()
    println("\nPress Enter to exit...")
    readline()
end
