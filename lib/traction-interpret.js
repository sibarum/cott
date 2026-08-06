INTERPRET = {};


/**
 * ------------------------------
 * ---------- Traction ----------
 * ------------------------------
 */

/**
 * This class for representing a magnitude. This is just a plain rational number.
 */
INTERPRET.TractionMagnitude = class {
    constructor(m) {
        if (typeof m === 'string') {
            m = Number.parseInt(m);
        }
        this.m = m
    }

    equals(other) {
        if (other instanceof INTERPRET.TractionMagnitude) {
            return other.m === this.m;
        }
    }

    plus(other) {
        if (other instanceof INTERPRET.TractionMagnitude) {
            return new TractionMagnitude(other.m + this.m);
        }
    }

    times(other) {
        if (other instanceof INTERPRET.TractionMagnitude) {
            return new TractionMagnitude(other.m * this.m);
        }
    }
}

INTERPRET.TractionResidual = class {
    constructor(r) {
        if (r === undefined || r === null || isNaN(r)) {
            r = 1;
        }
        this.r = r;
    }

    equals(other) {
        if (other instanceof INTERPRET.TractionResidual) {
            return this.r === other.r;
        }
    }

    plus(other) {
        if (other instanceof INTERPRET.TractionResidual) {
            return new TractionResidual(other.r + this.r)
        }
    }

    times(other) {
        if (other instanceof INTERPRET.TractionResidual) {
            return new TractionResidual(other.r * this.r);
        }
    }
}

INTERPRET.TractionInfinity = class {
    constructor(w) {
        if (w === undefined || w === null || isNaN(w)) {
            w = -1;
        }
        this.w = w;
    }

    equals(other) {
        if (other instanceof INTERPRET.TractionInfinity) {
            return this.w === other.w;
        }
    }

    plus(other) {
        if (other instanceof INTERPRET.TractionInfinity) {
            return new TractionInfinity(other.w + this.w);
        }
    }

    plus(other) {
        if (other instanceof INTERPRET.TractionInfinity) {
            return new TractionInfinity(other.w * this.w);
        }
    }
}

INTERPRET.UnaryOperation = class {
    constructor(op, operand) {
        this.op = op;
        this.operand = operand;
    }

    eval() {
        return this;

    }

}

INTERPRET.BinaryOperation = class {
    constructor(left, op, right) {
        this.left = left;
        this.op = op;
        this.right = right;
    }

    isBoundaryLift(node) {
        return node instanceof INTERPRET.BinaryOperation &&
            node.op === "^" &&
            node.left instanceof INTERPRET.TractionResidual;
    }

    eval() {
        if (this.isBoundaryLift(this.left) && this.isBoundaryLift(this.right)) {
            return new INTERPRET.TractionTerm(

            )
        }
        if (this.op === "+") {
            if (this.left.op === "+" === this.right.op) {
                return new INTERPRET.TractionTerm(
                    "+",
                    this.left.m.plus(this.right.m),
                    this.left.r.plus(this.right.r),
                    this.left.w.plus(this.right.w)
                )
            } else {
                return this;
            }
        } else if (this.op === "*") {
            if (this.left.op === "*" === this.right.op) {
                return new INTERPRET.TractionTerm(
                    "*",
                    this.left.m.times(this.right.m),
                    this.left.r.times(this.right.r),
                    this.left.w.times(this.right.w)
                )
            } else {
                return this;
            }
        }
    }
}


/**
 * The class for representing values in Traction arithmetic.
 */
INTERPRET.TractionTerm = class {
    /**
     * m op 0^r op 0^(-w)
     * m * 0^r * 0^(-w)
     * m + 0^r + 0^(-w)
     *
     *
     * @param op The operator on this term. May only be + or *.
     * @param m The magnitude (rational) part of this term.
     * @param r The residual (zero displacement) 0-lift of this term.
     * @param w The infinite (ordinal) w-lift of this term.
     */
    constructor(op, m, r, w) {
        this.op = op;
        this.m = new TractionMagnitude(m);
        this.r = new TractionResidual(r);
        this.w = new TractionInfinity(w);
    }

    equals(other) {
        return other.op === this.op &&
            other.m === this.m &&
            other.r === this.r &&
            other.w === this.w
    }

    plus(other) {
        return new INTERPRET.BinaryOperation(this, "+", other).eval();
    }

    times(other) {
        return new INTERPRET.BinaryOperation(this, "*", other).eval();
    }

}


/**
 * Parser Grammar
 */

;(() => {
    const grammar = ohm.grammar(`
Expression {

  Exp
    = AssignmentExp
    
  AssignmentExp
    = AssignmentExp operation AddExp  -- assignment
    | AddExp                          -- cascade

  AddExp
    = AddExp "+" MulExp        -- add
    | AddExp "-" MulExp        -- subtract
    | AddExp "(+/-)" MulExp    -- plusminus
    | AddExp "(-/+)" MulExp    -- minusplus
    | AddExp "(+)" MulExp      -- oplus
    | MulExp                   -- cascade

  MulExp
    = MulExp "*" PowExp        -- multiply
    | MulExp "/" PowExp        -- divide
    | MulExp "(*)" PowExp      -- dotmultiply
    | MulExp "(><)" PowExp     -- cross
    | Term                     -- term
    | PowExp                   -- cascade
    
  Term
    = integer ident         -- variable
    | integer Group         -- group

  PowExp
    = PriExp "^" PowExp     -- power
    | PriExp                -- cascade

  PriExp
    = "-" PriExp            -- inversion
    | "+-" PriExp           -- signed
    | Group                 -- group
    | FuncCall              -- function
    | number                -- number
    | QualifiedIdent        -- identity

  Group
    = "(" Exp ")"           -- paren
    | "[" ArgList "]"       -- bracket
    | "{" ArgList "}"       -- brace
    | "|" Exp "|"           -- magnitude
    | escape                -- escape

  FuncCall
    = QualifiedIdent "(" ArgList ")"
    | QualifiedIdent "{" ArgList "}"

  QualifiedIdent
    = ident "_{" (~"}" any)+ "}"    -- normal
    | ident "_" Group               -- qualified
    | ident "^" integer             -- super
    | ident                         -- simple

  ArgList
    = ListOf<Exp, ",">
    
  operation
    = "(<=>)"
    | "(=>)"
    | "(<=)"
    | ":="
    | "=->"
    | "<-="
    | "<=>"
    | "<>"
    | ">="
    | "=<"
    | "~="
    | "!="
    | "=>"
    | "<="
    | "="
    | ">"
    | "<"

  escape
    = "'" (~"'" any)* "'"

  number
    = float                 -- float
    | integer               -- integer

  float
    = digit+ "." digit+     -- float
    | "." digit+            -- decimal
    | digit+ "."            -- whole

  integer
    = digit+

  ident
    = letter+

}
`);

    /**
     * ----------------------------
     * ---------- toHTML ----------
     * ----------------------------
     */

    const semantics = grammar.createSemantics();
    semantics.addOperation('toHTML', {

        AssignmentExp_assignment(ident, _eq, value) {
            return ident.toHTML() +
                `<wbr><span class="symbol">${_eq.toHTML()}</span>` +
                value.toHTML();
        },

        AddExp_add(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol">+</span><wbr>` +
                right.toHTML();
        },

        AddExp_subtract(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol op-minus"><span class="copyonly">-</span></span><wbr>` +
                right.toHTML();
        },

        MulExp_multiply(left, _op, right) {
            // Implicit multiplication: 2*x*y^2 displays as 2xy² (no dot) when both
            // factors are "juxtaposable" — numbers, variables, or powers of them.
            // Function calls, groups, etc. keep the · for clarity. Either way a
            // copyonly "*" rides along so the clipboard still gets 2*x*y^2.
            const juxtapose = left.isJuxtaposable() && right.isJuxtaposable();
            const op = juxtapose
                ? `<span class="copyonly">*</span>`
                : `<span class="symbol op-times"><span class="copyonly">*</span></span>`;
            return left.toHTML() + op + right.toHTML();
        },

        MulExp_divide(left, _op, right) {
            // The fraction bar already delimits each slot, so a paren group that
            // wraps an ENTIRE operand is redundant on screen — render its inner
            // content instead. The copyonly parens below still restore grouping
            // in the clipboard, so the pasted text re-parses:  (num)/(den).
            const num = (left.parenInner()  || left ).toHTML();
            const den = (right.parenInner() || right).toHTML();
            return `<span class="fraction">` +
                `<span class="copyonly">(</span><span class="numerator">` +
                num +
                `</span><span class="copyonly">)</span>` +
                `<span class="h-divider symbol">/</span>` +
                `<span class="copyonly">(</span><span class="denominator">` +
                den +
                `</span><span class="copyonly">)</span></span>`;
        },

        // Explicit (parenthesised) operators: fixed glyph on screen, verbatim
        // "(TOK)" in the copy stream so they always re-parse to themselves.
        AddExp_plusminus(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol op-pm"><span class="copyonly">(+/-)</span></span><wbr>` +
                right.toHTML();
        },
        AddExp_minusplus(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol op-mp"><span class="copyonly">(-/+)</span></span><wbr>` +
                right.toHTML();
        },
        AddExp_oplus(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol op-oplus"><span class="copyonly">(+)</span></span><wbr>` +
                right.toHTML();
        },
        MulExp_dotmultiply(left, _op, right) {
            // forces the dot even where implicit multiplication would drop it
            return left.toHTML() +
                `<span class="symbol op-times"><span class="copyonly">(*)</span></span>` +
                right.toHTML();
        },
        MulExp_cross(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol op-cross"><span class="copyonly">(&gt;&lt;)</span></span>` +
                right.toHTML();
        },

        Term_variable(_int, _ident) {
            return `<span class="number">${_int.toHTML()}</span><span class="variable">${_ident.toHTML()}</span>`
        },

        Term_group(_int, _group) {
            return `<span class="number">${_int.toHTML()}</span>${_group.toHTML()}`
        },

        PowExp_power(base, _op, exponent) {
            // wrap exponent in <sup>, no caret
            let exp = `<span class="copyonly">^</span>${exponent.toHTML()}`;
            return base.toHTML() + `<sup class="smaller">${exp}</sup>`;
        },

        PowExp_cascade(expr) {
            return expr.toHTML();
        },

        PriExp_inversion(_neg, inner) {
            return `<span class="symbol op-minus"><span class="copyonly">-</span></span>` + inner.toHTML();
        },

        PriExp_signed(_neg, inner) {
            return `<span class="symbol op-pm"><span class="copyonly">+-</span></span>` + inner.toHTML();
        },

        Group_paren(_open, expr, _close) {
            return `<span class="symbol">(</span>${expr.toHTML()}<span class="symbol">)</span>`;
        },
        Group_bracket(_open, expr, _close) {
            return `<span class="symbol">[</span>${expr.toHTML()}<span class="symbol">]</span>`;
        },
        Group_brace(_open, expr, _close) {
            return `<span class="copyonly">(</span>${expr.toHTML()}<span class="copyonly">)</span>`;
        },
        Group_magnitude(_open, expr, _close) {
            return `<span class="symbol">|</span>${expr.toHTML()}<span class="symbol">|</span>`;
        },

        number(_chars) {
            return `<span class="number">${this.sourceString}</span>`;
        },

        ident(_chars) {
            let css = 'variable';
            let name = this.sourceString;
            if (name === 'w') {
                css = 'symbol';
                name = '&omega;'
            } else if (name === 'i') {
                css = 'symbol';
                name = 'i'
            } else if (name === 'C') {
                css = 'symbol';
                name = '&complexes;'
            } else if (name === 'N') {
                css = 'symbol';
                name = '&naturals;'
            } else if (name === 'R') {
                css = 'symbol';
                name = '&reals;'
            } else if (name === 'P') {
                css = 'symbol';
                name = '&Popf;'
            } else if (name === 'Q') {
                css = 'symbol';
                name = '&rationals;'
            } else if (name === 'pi') {
                css = 'symbol';
                name = '&pi;'
            } else if (name === 'null') {
                css = 'symbol';
                name = '&empty;'
            }
            return `<span class="${css}">${name}</span>`;
        },

        QualifiedIdent_normal(name, _before, qualifier, _after) {
            return `${name.toHTML()}<span class="copyonly">${_before.sourceString}</span><sub class="smaller">${qualifier.toHTML()}</sub><span class="copyonly">${_after.sourceString}</span>`;
        },
        QualifiedIdent_qualified(name, _underscore, qualifier) {
            return `${name.toHTML()}<span class="copyonly">_</span><sub class="smaller">${qualifier.toHTML()}</sub>`;
        },
        QualifiedIdent_super(name, _caret, supe) {
            return `${name.toHTML()}<span class="copyonly">^</span><sup class="smaller">${supe.toHTML()}</sup>`;
        },
        QualifiedIdent_simple(name) {
            return name.toHTML();
        },

        FuncCall(name, _open, args, _close) {
            let elements = args.asIteration().children;
            if (name.sourceString === 'sqrt') {
                // screen: √‾overline‾ ;  clipboard: sqrt( … )
                // the overbar delimits the radicand, so a fully-wrapping paren
                // group is redundant on screen (kept in copy via sqrt(…)).
                const rad = (elements[0].parenInner() || elements[0]).toHTML();
                let html = `<span class="sqrt">`;
                html += `<span class="copyonly">sqrt(</span>`
                html += `<span class="symbol op-radic"></span>`
                html += `<span class="number radicand">${rad}</span>`
                html += `<span class="copyonly">)</span>`
                html += `</span>`
                return html;
            }
            let argHTML = elements.map(a => a.toHTML()).join(',');
            if (_open.sourceString === "(") {
                argHTML = `<span class="symbol">(</span>${argHTML}<span class="symbol">)</span>`;
            } else {
                argHTML = `<span class="copyonly">(</span>${argHTML}<span class="copyonly">)</span>`;
            }
            return name.toHTML() + argHTML;
        },

        ArgList(list) {
            return list.asIteration().children.map(a => a.toHTML()).join(',&nbsp;');
        },
        operation(op) {
            const s = op.sourceString;
            // Explicit (parenthesised) logic arrows — glyph on screen, "(TOK)" in copy.
            if (s === '(=>)')  return ` <span class="symbol op-dimplies"><span class="copyonly">(=&gt;)</span></span> `;
            if (s === '(<=)')  return ` <span class="symbol op-dimpliedby"><span class="copyonly">(&lt;=)</span></span> `;
            if (s === '(<=>)') return ` <span class="symbol op-iff"><span class="copyonly">(&lt;=&gt;)</span></span> `;
            // definition — renders ≔, copies ":=" so it round-trips
            if (s === ':=')    return ` <span class="symbol op-def"><span class="copyonly">:=</span></span> `;
            // Comparisons: bare <= =< >= now render ≤ / ≥ (round-trip via copyonly).
            if (s === '<=' || s === '=<')
                return ` <span class="symbol op-le"><span class="copyonly">${s === '=<' ? '=&lt;' : '&lt;='}</span></span> `;
            if (s === '>=')
                return ` <span class="symbol op-ge"><span class="copyonly">&gt;=</span></span> `;
            // Existing relations (unchanged).
            if (s === '~=')  return ' &approx; ';
            if (s === '=->') return ' &rarr; ';
            if (s === '<-=') return ' &larr; ';
            if (s === '=>')  return ' &rArr; ';
            if (s === '<=>') return ' &hArr; ';
            if (s === '<>')  return ' &loz; ';
            if (s === '!=')  return '&ne;';
            return `${s}`;
        },

        _terminal() { return this.sourceString; },
        _nonterminal(...children) {
            return children.map(c => c.toHTML()).join('');
        },
        _iter(...children) {
            return children.map(c => c.toHTML());
        }
    });

    /**
     * ---------------------------------
     * ---------- parenInner -----------
     * ---------------------------------
     * If this node is (ultimately, through cascade rules) a single paren group
     * "(Exp)", return the inner Exp node; otherwise null. Used to suppress the
     * parentheses that merely delimit a slot (numerator, denominator, radicand)
     * whose bar/overbar already provides the grouping. Anything that is NOT one
     * fully-wrapping group — e.g. "(x+1)(x-2)" or "2(x+1)" — returns null, so its
     * parens are kept.
     */
    semantics.addOperation('parenInner', {
        Group_paren(_open, expr, _close) { return expr; },
        _nonterminal(...children) {
            return children.length === 1 ? children[0].parenInner() : null;
        },
        _iter(...children) {
            return children.length === 1 ? children[0].parenInner() : null;
        },
        _terminal() { return null; }
    });

    /**
     * -------------------------------------
     * ---------- isJuxtaposable -----------
     * -------------------------------------
     * True when a node renders as a bare multiplicative atom that may sit next to
     * another with no · between them: a number, a variable/constant (with any
     * sub/superscript), a power whose base is such an atom, or a product of these.
     * Function calls, parenthesised groups, sums, etc. return false so they keep
     * the explicit dot. The explicit true/false handlers stop the single-child
     * passthrough at the leaves that matter.
     */
    semantics.addOperation('isJuxtaposable', {
        number(_chars)                          { return true; },
        ident(_chars)                           { return true; },
        QualifiedIdent_simple(_n)               { return true; },
        QualifiedIdent_qualified(_n, _u, _q)    { return true; },
        QualifiedIdent_super(_n, _c, _s)        { return true; },
        QualifiedIdent_normal(_n, _b, _q, _a)   { return true; },
        Term_variable(_int, _ident)             { return true; },
        PowExp_power(base, _op, _exp)           { return base.isJuxtaposable(); },
        MulExp_multiply(l, _op, r)              { return l.isJuxtaposable() && r.isJuxtaposable(); },
        _nonterminal(...children) {
            return children.length === 1 ? children[0].isJuxtaposable() : false;
        },
        _iter(...children) {
            return children.length === 1 ? children[0].isJuxtaposable() : false;
        },
        _terminal() { return false; }
    });


    /**
     * --------------------------
     * ---------- eval ----------
     * --------------------------
     */
    semantics.addOperation('eval', {
        AddExp_add(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "+", right.eval());
        },

        AddExp_subtract(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "-", right.eval());
        },

        AddExp_plusminus(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "+-", right.eval());
        },

        AddExp_minusplus(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "-+", right.eval());
        },

        MulExp_multiply(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "*", right.eval());
        },

        MulExp_divide(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "/", right.eval());
        },

        AddExp_oplus(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "(+)", right.eval());
        },

        MulExp_dotmultiply(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "*", right.eval());
        },

        MulExp_cross(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "><", right.eval());
        },

        PowExp_power(base, _op, exponent) {
            return new INTERPRET.BinaryOperation(base.eval(), "^", exponent.eval());
        },

        PriExp_inversion(_neg, inner) {
            let value = inner.eval();
            if (supportsNativeArithmetic(value)) {
                return -value;
            } else {
                return {
                    operation: "n",
                    value
                }
            }

        },

        PowExp_cascade(expr) {
            return expr.eval();
        },

        Group_paren(_open, expr, _close) {
            return expr.eval();
        },
        Group_bracket(_open, expr, _close) {
            return expr.eval();
        },
        Group_brace(_open, expr, _close) {
            return expr.eval();
        },

        number_float(_chars) {
            let m = Number.parseFloat(this.sourceString);
            if (m === 0) {
                return new INTERPRET.TractionResidual(1);
            } else {
                return new INTERPRET.TractionMagnitude(m);
            }
        },

        number_integer(_chars) {
            let m = Number.parseInt(this.sourceString);
            if (m === 0) {
                return new INTERPRET.TractionResidual(1);
            } else {
                return new INTERPRET.TractionMagnitude(m);
            }
        },

        ident(_chars) {
            if (this.sourceString === "w") {
                return new INTERPRET.TractionInfinity(1);
            } else {
                return {
                    value: this.sourceString
                }
            }
        },

        QualifiedIdent_normal(name, _before, qualifier, _after) {
            return {
                name: name.sourceString,
                qualifier: qualifier.sourceString
            };
        },

        QualifiedIdent_qualified(name, _underscore, qualifier) {
            return {
                name: name.sourceString,
                qualifier: qualifier.sourceString
            };
        },
        QualifiedIdent_simple(name) {
            return {
                value: this.sourceString
            }
        },

        FuncCall(name, _open, args, _close) {
            return {
                operation: 'function'
            }
        },

        ArgList(list) {
            return {
                operation: 'arguments'
            }
        },

        _terminal() { return this.sourceString; },
        _nonterminal(...children) {
            if (children.length === 1) {
                return children[0].eval();
            } else {
                return children.map(c => c.eval());
            }
        },
        _iter(...children) {
            if (children.length === 1) {
                return children[0].eval();
            } else {
                return children.map(c => c.eval());
            }
        }
    });


    /**
     * ------------------------------------
     * ---------- PUBLIC METHODS ----------
     * ------------------------------------
     */


    INTERPRET.parseExpression = function(exprString) {
        const result = grammar.match(exprString);

        if (result.succeeded()) {
            return semantics(result);
        } else {
            return {
                error: true,
                message: result.message,
            };
        }
    }

    INTERPRET.printToHtml = function(parseResult, target) {
        if (parseResult.error) {
            target.innerHTML = `<pre class="tiny-error">${parseResult.message}</pre>`;
        } else {
            target.innerHTML = parseResult.toHTML();
        }
    }

})();

document.addEventListener("DOMContentLoaded", function() {
    // Select all elements with the target class
    const elements = document.querySelectorAll(".dynamicexpr");

    elements.forEach(el => {
        INTERPRET.printToHtml(INTERPRET.parseExpression(el.textContent), el);
    });
});
