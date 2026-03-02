INTERPRET = {};

;(() => {
    const grammar = ohm.grammar(`
Expression {

  Exp
    = AddExp

  AddExp
    = AddExp "+" MulExp     -- add
    | AddExp "-" MulExp     -- subtract
    | MulExp                -- cascade

  MulExp
    = MulExp "*" PowExp     -- multiply
    | MulExp "/" PowExp     -- divide
    | PowExp                -- cascade

  PowExp
    = PriExp "^" PowExp     -- power
    | PriExp                -- cascade

  PriExp
    = "-" PriExp            -- inversion
    | Group                 -- group
    | FuncCall              -- function
    | number                -- number
    | ident                 -- identity

  Group
    = "(" Exp ")"           -- paren
    | "[" Exp "]"           -- bracket
    | "{" Exp "}"           -- brace

  FuncCall
    = QualifiedIdent "(" ArgList? ")"

  QualifiedIdent
    = ident "_" ident       -- qualified
    | ident                 -- simple

  ArgList
    = ListOf<Exp, ",">

  number
    = float
    | integer

  float
    = digit+ "." digit+     -- float
    | "." digit+            -- decimal
    | digit+ "."            -- whole

  integer
    = digit+

  ident
    = letter (letter | digit | "_")*

}
`);

    const semantics = grammar.createSemantics().addOperation('toHTML', {
        AddExp_add(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol">+</span>` +
                right.toHTML();
        },

        AddExp_subtract(left, _op, right) {
            return left.toHTML() +
                `<span class="symbol">-</span>` +
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
                `</span><span class="h-divider">/</span><span class="denominator">` +
                right.toHTML() +
                `</span></span>`;
        },

        PowExp_power(base, _op, exponent) {
            // wrap exponent in <sup>, no caret
            return base.toHTML() + `<sup>${exponent.toHTML()}</sup>`;
        },

        PowExp_cascade(expr) {
            return expr.toHTML();
        },

        PriExp_inversion(_neg, inner) {
            return `<span class="symbol">-</span>` + inner.toHTML();
        },

        Group_paren(_open, expr, _close) {
            return `(${expr.toHTML()})`;
        },
        Group_bracket(_open, expr, _close) {
            return `[${expr.toHTML()}]`;
        },
        Group_brace(_open, expr, _close) {
            return `{${expr.toHTML()}}`;
        },

        number(_chars) {
            return `<span class="number">${this.sourceString}</span>`;
        },

        ident(_chars,qual) {
            let css = 'variable';
            let name = this.sourceString;
            if (name === 'w') {
                css = 'symbol';
                name = '&omega;'
            }
            return `<span class="${css}">${name}</span>`;
        },

        QualifiedIdent_qualified(name, _underscore, qualifier) {
            return `<span class="variable">${name.sourceString}</span><sub>${qualifier.sourceString}</sub>`;
        },
        QualifiedIdent_simple(name) {
            return `<span class="variable">${name.sourceString}</span>`;
        },

        FuncCall(name, _open, args, _close) {
            let argHTML = args.children.map(a => a.toHTML()).join(', ');
            return name.toHTML() + `(${argHTML})`;
        },

        ArgList(list) {
            return list.children.map(a => a.toHTML()).join(', ');
        },

        _terminal() { return this.sourceString; },
        _nonterminal(...children) {
            return children.map(c => c.toHTML()).join('');
        }
    });


    INTERPRET.parseExpression = function(exprString) {
        const result = grammar.match(exprString);

        if (result.succeeded()) {
            return result;
        } else {
            console.log(result);
            return {
                error: result.message,
                errorIndex: result.getRightmostFailurePosition()
            };
        }
    }

    INTERPRET.printToHtml = function(parseResult) {
        return '<div class="expression">' + semantics(parseResult).toHTML() + '</div>';
    }

})();

