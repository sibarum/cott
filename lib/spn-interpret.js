
;(() => {
    const symbolsGrammar = ohm.grammar(`
SPN {

Root
    = Declaration+

Declaration
    = NumericDeclaration                    -- numeric
    | ValueDeclaration                      -- value
    | ClassDeclaration                      -- class

ClassDeclaration
    = "class" IdentifierTuple? identifier "{" ClassMember* "}"

ClassMember
    = ClassProperty                         -- property
    | ClassModifier                         -- modifier

ClassModifier
    = UnionModifier                         -- union
    | ProductModifier                       -- product
    | RuleModifier                          -- rule
    | NextModifier                          -- next

UnionModifier
    = "union" "of" IdentifierTuple ";"

ProductModifier
    = "product" "of" IdentifierTuple ";"

RuleModifier
    = "rule" Assignment ";"

ClassProperty
    = "property" IdentifierTuple? identifier ";"

NextModifier
    = "next" IdentifierTuple?";"

ValueDeclaration
    = "value" identifier ";"

NumericDeclaration
    = "numeric" identifier ("," ListOf<DeclarationConstraint, ",">)? ";"

DeclarationConstraint
    = "constraint" Assignment

identifier
    = letter+
    | "?"

keyword
    = ("class" | "property" | "rule" | "union" | "of"
    |  "value" | "numeric" | "constraint") ~letter

IdentifierTuple
    = "(" ListOf<identifier, ","> ")"

Exp
    = Assignment                            -- assignment
    | AddExp                                -- cascade

AddExp
    = AddExp "+" MulExp                     -- add
    | AddExp "-" MulExp                     -- subtract
    | MulExp                                -- cascade

MulExp
    = MulExp "*" PowExp                     -- multiply
    | MulExp "/" PowExp                     -- divide
    | PowExp                                -- cascade

PowExp
    = PriExp "^" PowExp                     -- power
    | PriExp                                -- cascade

PriExp
    = "~"? "(" identifier ")"               -- paren
    | number                                -- numeric
    | identifier                            -- identifier

Assignment
    = Exp assignmentSymbol AddExp

assignmentSymbol
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
    | "%"
    | "_"
    | "."

number
    = float                                 -- float
    | integer                               -- integer

float
    = digit+ "." digit+                     -- float
    | "." digit+                            -- decimal
    | digit+ "."                            -- whole

integer
    = digit+

}
`);
    const semantics = symbolsGrammar.createSemantics();

    // Helper to wrap text in a span with a CSS class
    const span = (cls, text) => `<span class="spn-${cls}">${text}</span>`;

    semantics.addOperation('toHTML', {

        // --- Top-level ---

        Root(decls) {
            return decls.children.map(c => c.toHTML()).join('\n');
        },

        // --- Declarations ---

        Declaration_numeric(d) { return d.toHTML(); },
        Declaration_value(d)   { return d.toHTML(); },
        Declaration_class(d)   { return d.toHTML(); },

        NumericDeclaration(_numeric, id, _comma, constraints, _semi) {
            const chtml = constraints.toHTML();
            return span('keyword', 'numeric') + ' ' + id.toHTML()
                + (chtml ? ' ' + chtml : '')
                + span('punct', ';<br>');
        },

        DeclarationConstraint(_constraint, assignment) {
            return '<br>\t' + span('keyword', 'constraint') + ' ' + assignment.toHTML();
        },

        ValueDeclaration(_value, id, _semi) {
            return span('keyword', 'value') + ' ' + id.toHTML() + span('punct', ';<br>');
        },

        ClassDeclaration(_class, tuple, id, _lbrace, members, _rbrace) {
            return '<br>' + span('keyword', 'class') + ' '
                + tuple.children.map(t => t.toHTML()).join('')
                + (tuple.children.length ? ' ' : '')
                + id.toHTML() + ' '
                + span('punct', '{') + '<br><br>'
                + members.children.map(m => '  ' + m.toHTML()).join('<br>')
                + (members.children.length ? '<br><br>' : '')
                + span('punct', '}<br>');
        },

        // --- Class members ---

        ClassMember_property(p) { return p.toHTML(); },
        ClassMember_modifier(m) { return m.toHTML(); },

        ClassModifier_union(u)  { return u.toHTML(); },
        ClassModifier_product(p)  { return p.toHTML(); },
        ClassModifier_rule(r)   { return r.toHTML(); },
        ClassModifier_next(n)   { return n.toHTML(); },

        UnionModifier(_union, _of, tuple, _semi) {
            return span('keyword', 'union') + ' '
                + span('keyword', 'of') + ' '
                + tuple.toHTML() + span('punct', ';');
        },

        ProductModifier(_union, _of, tuple, _semi) {
            return span('keyword', 'product') + ' '
                + span('keyword', 'of') + ' '
                + tuple.toHTML() + span('punct', ';');
        },

        RuleModifier(_rule, assignment, _semi) {
            return span('keyword', 'rule') + ' ' + assignment.toHTML() + span('punct', ';');
        },

        ClassProperty(_property, tuple, id, _semi) {
            return span('keyword', 'property') + ' '
                + tuple.children.map(t => t.toHTML()).join('')
                + (tuple.children.length ? ' ' : '')
                + id.toHTML() + span('punct', ';');
        },

        NextModifier(_property, tuple, _semi) {
            return span('keyword', 'next') + ' '
                + tuple.children.map(t => t.toHTML()).join('')
                + span('punct', ';');
        },

        // --- Identifiers & tuples ---

        identifier(letters) {
            return span('ident', this.sourceString);
        },

        IdentifierTuple(_lparen, list, _rparen) {
            return span('punct', '(')
                + list.asIteration().children.map(c => c.toHTML()).join(span('punct', ',') + ' ')
                + span('punct', ')');
        },

        // --- Expressions ---

        Exp_assignment(a)  { return a.toHTML(); },
        Exp_cascade(a)     { return a.toHTML(); },

        AddExp_add(left, _op, right) {
            return left.toHTML() + ' ' + span('op', '+') + ' ' + right.toHTML();
        },
        AddExp_subtract(left, _op, right) {
            return left.toHTML() + ' ' + span('op', '-') + ' ' + right.toHTML();
        },
        AddExp_cascade(e) { return e.toHTML(); },

        MulExp_multiply(left, _op, right) {
            return left.toHTML() + ' ' + span('op', '*') + ' ' + right.toHTML();
        },
        MulExp_divide(left, _op, right) {
            return left.toHTML() + ' ' + span('op', '/') + ' ' + right.toHTML();
        },
        MulExp_cascade(e) { return e.toHTML(); },

        PowExp_power(base, _op, exp) {
            return base.toHTML() + span('op', '^') + exp.toHTML();
        },
        PowExp_cascade(e) { return e.toHTML(); },

        PriExp_paren(_mod, _lp, exp, _rp) {
            let modifier = _mod.toHTML();
            const negate = modifier === "~" ? span("not", modifier) : "";
            return negate + span('punct', '(') + exp.toHTML() + span('punct', ')');
        },
        PriExp_numeric(n)    { return n.toHTML(); },
        PriExp_identifier(id) { return id.toHTML(); },

        // --- Assignment ---

        Assignment(left, sym, right) {
            return left.toHTML() + ' ' + sym.toHTML() + ' ' + right.toHTML();
        },

        assignmentSymbol(_) {
            return span('assign', this.sourceString);
        },

        // --- Numbers ---

        number_float(f)   { return f.toHTML(); },
        number_integer(i) { return i.toHTML(); },

        float_float(_int, _dot, _frac)  { return span('number', this.sourceString); },
        float_decimal(_dot, _frac)       { return span('number', this.sourceString); },
        float_whole(_int, _dot)          { return span('number', this.sourceString); },

        integer(_digits) {
            return span('number', this.sourceString);
        },

        // --- Built-in list support ---

        NonemptyListOf(first, _sep, rest) {
            return [first.toHTML(), ...rest.children.map(c => c.toHTML())].join(span('punct', ',') + ' ');
        },

        EmptyListOf() {
            return '';
        },

        // --- Fallback ---

        _terminal() {
            return this.sourceString;
        },

        _iter(...children) {
            return children.map(c => c.toHTML()).join('');
        },
    });

    // --- DOM integration: highlight all .code elements ---

    function highlightCodeBlocks() {
        document.querySelectorAll('.code').forEach(el => {
            const src = el.textContent;
            const match = symbolsGrammar.match(src);
            if (match.succeeded()) {
                el.innerHTML = semantics(match).toHTML();
            } else {
                // Show parse failure with the source preserved
                el.innerHTML = span('error', 'Parse error: ' + match.message)
                    + '\n' + src.replace(/</g, '&lt;');
            }
        });
    }

    // Run on DOMContentLoaded or immediately if already loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', highlightCodeBlocks);
    } else {
        highlightCodeBlocks();
    }

    // Expose for programmatic use
    window.SPN = { grammar: symbolsGrammar, semantics, highlightCodeBlocks };

})();
