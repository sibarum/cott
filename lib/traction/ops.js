OPS = {};

OPS.One = class {
    /**
     * =1*displacement
     * @param displacement
     */
    constructor(displacement) {
        this.displacement = displacement;
    }

    eval() {
        return this.displacement;
    }
    toString() {
        return `${this.displacement}`;
    }
    times(other) {
        if (typeof other === 'number') {
            other = new OPS.One(other);
        }
        if (other instanceof OPS.One) {
            return new OPS.One(this.displacement + other.displacement);
        }
    }
}

OPS.Zero = class {
    /**
     * =0^phase
     * @param phase
     */
    constructor(phase) {
        this.phase = phase;
    }

    eval() {
        let phase = this.phase;
        if (typeof phase !== 'number') {
            phase = this.phase.eval();
        }
        if (typeof phase === 'number') {
            if (phase === 1) {
                return 0;
            }
            if (phase === 0) {
                return 1;
            }
            if (phase === Infinity) {
                return -1;
            }
        }
        return new OPS.Zero(phase);
    }
    toString() {
        if (this.phase === 0) {
            return "1";
        } else if (this.phase === -1) {
            return "w";
        } else if (this.phase === 1) {
            return "0";
        } else if (typeof this.phase === "number") {
            return `0^{${this.phase}}`;
        } else {
            return `0^{${this.phase.toString()}}`;
        }
    }
    times(other) {
        if (typeof other === 'number') {
            other = new OPS.Zero(new OPS.Zero(other))
        }
        if (other instanceof OPS.Zero) {
            return new OPS.Zero(
                new OPS.Addition([
                    this.phase, other.phase
                ])
            ).eval();
        }
    }
}

OPS.Log = class {
    /**
     * log_{base}(residual)
     * @param base
     * @param residual
     */
    constructor(base, residual) {
        this.base = base;
        this.residual = residual;
    }

    eval() {
        if (this.base === 0) {
            if (this.residual === 0) {
                return 1;
            } else if (this.residual === 1) {
                return 0;
            }
        } else {
            return this;
        }
    }
    toString() {
        return `log_{${this.base.toString()}}(${this.residual.toString()})`;
    }
}

OPS.Inverse = class {
    /**
     * -(term)
     * @param term
     */
    constructor(term) {
        this.term = term;
    }

    eval() {
        if (typeof this.term === 'number') {
            return -this.term;
        } else {
            const t = this.term.eval();
            if (typeof t === 'number') {
                return -t;
            } else {
                return new OPS.Inverse(t);
            }
        }
    }
    toString() {
        return this.term.toString();
    }
}

OPS.Reciprocal = class {
    /**
     * 1/term
     * @param term
     */
    constructor(term) {
        this.term = term;
    }

    eval() {
        if (typeof this.term === 'number') {
            return 1/this.term;
        } else {
            const t = this.term.eval();
            if (typeof t === 'number') {
                return 1/t;
            } else {
                return new OPS.Reciprocal(t);
            }
        }
    }
    toString() {
        return `1/(${this.term.toString()})`
    }
}

OPS.Addition = class {
    /**
     * a + b
     * @param terms
     */
    constructor(terms) {
        this.terms = terms;
    }

    eval() {
        const evaluated = this.terms.map((term) => {
            if (typeof term === 'number') {
                return term;
            } else {
                return term.eval();
            }
        });
        let total = 0;
        let terms = [];
        for (let t of evaluated) {
            if (typeof t === 'number') {
                total += t;
            } else {
                terms.push(t);
            }
        }
        if (total !== 0) {
            terms.push(total);
        }
        if (terms.length > 1) {
            return new OPS.Addition(terms);
        } else if (terms.length === 1) {
            return terms[0];
        } else {
            return new OPS.Zero(1);
        }
    }
    toString() {
        return '('+this.terms.map(term => term.toString()).join(")+(") + ')';
    }
    times(other) {
        if (other instanceof OPS.Addition) {
            let newTerms = [];
            for (let t1 of this.terms) {
                for (let t2 of other.terms) {
                    if (t1.times) {
                        newTerms.push(t1.times(t2));
                    } else if (t2.times) {
                        newTerms.push(t2.times(t1));
                    } else {
                        newTerms.push(new OPS.Multiplication([t1, t2]).eval());
                    }
                }
            }
            return new OPS.Addition(newTerms).eval();
        }
        return new OPS.Multiplication([this, other]);
    }
}

OPS.Multiplication = class {
    /**
     * a * b
     * @param terms
     */
    constructor(terms) {
        this.terms = terms;
    }

    eval() {
        const evaluated = this.terms.map((term) => {
            if (typeof term === 'number') {
                return term;
            } else {
                return term.eval();
            }
        });
        if (evaluated.length > 1) {
            const reducedTerms = [];
            while (evaluated.length > 0) {
                let term = evaluated.pop();
                if (term.times) {
                    for (let i=0; i<evaluated.length; i++) {
                        let t = evaluated[i];
                        let newTerm = term.times(t);
                        if (newTerm) {
                            reducedTerms.push(newTerm);
                            evaluated.splice(i, 1);
                            break;
                        }
                    }
                } else {
                    reducedTerms.push(term);
                }
            }
            if (reducedTerms.length === 1) {
                return reducedTerms[0];
            } else {
                return new OPS.Multiplication(reducedTerms);
            }
        } else if (evaluated.length === 1) {
            return evaluated[0];
        }
        throw new Error("Invalid state")
    }
    toString() {
        return '('+this.terms.map(term => term.toString()).join(")*(") + ')';
    }
}

OPS.Exponentiation = class {
    /**
     * base^{exponent}
     * @param base
     * @param exponent
     */
    constructor(base, exponent) {
        this.base = base;
        this.exponent = exponent;
    }

    eval() {
        const base = this.base.eval();
        const exponent = this.exponent.eval();
        if (typeof base === 'number') {
            if (base === 0) {
                return new OPS.Zero(exponent);
            } else if (base === Infinity) {
                return new OPS.Zero(new OPS.Inverse(exponent))
            }
            if (typeof exponent === 'number') {
                return Math.pow(base, exponent);
            }
        } else if (typeof exponent === 'number') {
            if (base instanceof OPS.Zero) {
                return new OPS.Zero(new OPS.Multiplication([base.phase, exponent])).eval();
            } else if (exponent%1 === 0) {
                let terms = [];
                for (let i=0; i<exponent; i++) {
                    terms.push(base);
                }
                return new OPS.Multiplication(terms).eval();
            }
        }
        return new OPS.Exponentiation(base, exponent);
    }

    toString() {
        return `(${this.base.toString()})^(${this.exponent.toString()})`;
    }
}

OPS.Expr = class {
    constructor(root) {
        this.root = root;
    }

    eval() {
        return new OPS.Expr(this.root.eval());
    }

    toString() {
        return this.root.toString();
    }
}

OPS.ExprBuilder = class {
    constructor() {}

    expr(root) {
        return new OPS.Expr(root);
    }

    number(n) {
        if (n === 0) {
            return new OPS.Zero(1);
        } else if (n === Infinity) {
            return new OPS.Zero(-1);
        } else if (n === -Infinity) {
            return new OPS.Inverse(new OPS.Zero(-1));
        } else {
            return new OPS.One(n);
        }
    }

    add(terms) {
        return new OPS.Addition(terms);
    }

    multiply(terms) {
        return new OPS.Multiplication(terms);
    }

    exp(base, exponent) {
        return new OPS.Exponentiation(base, exponent);
    }

    sqrt(operand) {
        return this.exp(operand, 0.5);
    }

    log(base, operand) {
        if (base === 0) {
            return new OPS.Log(base, operand);
        }
    }

    phaseLift(phase) {
        return new OPS.Zero(new OPS.Multiplication([new OPS.One(phase), new OPS.Zero(-1)]))
    }


}