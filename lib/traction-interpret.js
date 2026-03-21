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
    = AddExp "+" MulExp     -- add
    | AddExp "-" MulExp     -- subtract
    | MulExp                -- cascade

  MulExp
    = MulExp "*" PowExp     -- multiply
    | MulExp "/" PowExp     -- divide
    | Term                  -- term
    | PowExp                -- cascade
    
  Term
    = integer ident         -- variable
    | integer Group         -- group

  PowExp
    = PriExp "^" PowExp     -- power
    | PriExp                -- cascade

  PriExp
    = "-" PriExp            -- inversion
    | Group                 -- group
    | FuncCall              -- function
    | number                -- number
    | QualifiedIdent        -- identity

  Group
    = "(" Exp ")"           -- paren
    | "[" ArgList "]"       -- bracket
    | "{" Exp "}"           -- brace
    | "|" Exp "|"           -- magnitude
    | escape                -- escape

  FuncCall
    = QualifiedIdent "(" ArgList ")"
    | QualifiedIdent "{" ArgList "}"

  QualifiedIdent
    = ident "_{" (~"}" any)+ "}"    -- normal
    | ident "_" Group               -- qualified
    | ident                         -- simple

  ArgList
    = ListOf<Exp, ",">
    
  operation
    = "=->"
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
                `<span class="symbol">&ndash;</span><wbr>` +
                right.toHTML();
        },

        MulExp_multiply(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol">&middot;</span>` +
                right.toHTML();
        },

        MulExp_divide(left, _op, right) {
            return `<span class="fraction"><span class="numerator">` +
                left.toHTML() +
                `</span><span class="h-divider symbol">/</span><span class="denominator">` +
                right.toHTML() +
                `</span></span>`;
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
            return `<span class="symbol">&ndash;</span>` + inner.toHTML();
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
        QualifiedIdent_simple(name) {
            return name.toHTML();
        },

        FuncCall(name, _open, args, _close) {
            if (name.sourceString === 'sqrt') {
                let html = `<span class="sqrt">`;
                html += `<span class="symbol">&radic;</span>`
                html += `<span class="number radicand">${args.children[0].toHTML()}</span>`
                html += `</span>`
                return html;
            }
            let argHTML = args.children.map(a => a.toHTML()).join(',&nbsp;');
            if (_open.sourceString === "(") {
                argHTML = `<span class="symbol">(</span>${argHTML}<span class="symbol">)</span>`;
            } else {
                argHTML = `<span class="copyonly">(</span>${argHTML}<span class="copyonly">)</span>`;
            }
            return name.toHTML() + argHTML;
        },

        ArgList(list) {
            return list.children.map(a => a.toHTML()).join(',&nbsp;');
        },
        operation(op) {
            if (op.sourceString === '~=') {
                return ' &approx; ';
            } else if (op.sourceString === '=->') {
                return ' &rarr; ';
            } else if (op.sourceString === '<-=') {
                return ' &larr; ';
            } else if (op.sourceString === '=>') {
                return ' &rArr; ';
            } else if (op.sourceString === '<=>') {
                return ' &hArr; ';
            } else if (op.sourceString === '<>') {
                return ' &loz; ';
            } else {
                return `${op.sourceString}`;
            }
        },

        _terminal() { return this.sourceString; },
        _nonterminal(...children) {
            return children.map(c => c.toHTML()).join('&nbsp;');
        },
        _iter(...children) {
            return children.map(c => c.toHTML());
        }
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

        MulExp_multiply(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "*", right.eval());
        },

        MulExp_divide(left, _op, right) {
            return new INTERPRET.BinaryOperation(left.eval(), "/", right.eval());
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
