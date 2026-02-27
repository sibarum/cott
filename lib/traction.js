/**
 * Traction Algebra, Constructive Operational Type Theory (COTT)
 * An implementation of the discrete euler (zero lift), the totalized exp&log involution,
 * branch-free complex arithmetic.
 * Written by hand
 * James Watkins, @DoozerDiffuser
 *
 * For more information see:
 * sibarum.github.io/cott
 */

TRACTION = {};

/**
 * Reusing Infinity as the constant w, handy since it's already zero's reciprocal.
 */
TRACTION.W = Infinity;

/*
 * ##### TractionNumber #####
 */

/**
 * A single number. No operations attached.
 * @param value
 * @constructor
 */
TRACTION.TractionNumber = function(value) {
    if ((typeof value !== 'number' && value !== TRACTION.W) || isNaN(value)) {
        throw new TypeError("Not a number: " + value);
    }
    this.value = value;
}
TRACTION.TractionNumber.prototype.isZero = function() {
    return this.value === 0;
}
TRACTION.TractionNumber.prototype.isW = function() {
    return this.value === TRACTION.W || this.value === -TRACTION.W;
}
TRACTION.TractionNumber.prototype.isNegative = function() {
    return this.value < 0;
}
TRACTION.TractionNumber.prototype.toMarkup = function() {
    const element = document.createElement("span");
    if (this.isW()) {
        element.classList.add("symbol");
        element.innerHTML = "&omega;";
    } else {
        element.classList.add("number");
        element.textContent = this.value;
    }
    return element;
}

/*
 * ##### ReciprocalTerm #####
 */

TRACTION.ReciprocalTerm = function(term) {
    this.term = term;
    this.isReciprocal = true;
}
TRACTION.ReciprocalTerm.prototype.toMarkup = function() {
    return this.term.toMarkup();
}

TRACTION.InverseTerm = function(term) {
    this.term = term;
    this.isInverse = true;
}
TRACTION.InverseTerm.prototype.toMarkup = function() {
    return this.term.toMarkup();
}


/*
 * ##### TractionExponential #####
 */
TRACTION.TractionExponential = function(base, exponent) {
    this.base = base;
    this.exponent = exponent;
}
TRACTION.TractionExponential.prototype.isBoundaryLift = function() {
    let baseIsNumber = Object.getPrototypeOf(this.base) === TRACTION.TractionNumber;
    if (baseIsNumber) {
        return this.base.isW() || this.base.isZero();
    }
    return false;
}
TRACTION.TractionExponential.prototype.toMarkup = function() {
    const container = document.createElement("span");
    const base = this.base.toMarkup();
    container.appendChild(base);

    const exponent = document.createElement("sup");

    const hiddenCopyText1 = document.createElement("span");
    hiddenCopyText1.classList.add("copyonly");
    hiddenCopyText1.textContent = "^(";
    exponent.appendChild(hiddenCopyText1);

    exponent.appendChild(this.exponent.toMarkup());

    const hiddenCopyText2 = document.createElement("span");
    hiddenCopyText2.classList.add("copyonly");
    hiddenCopyText2.textContent = ")";
    exponent.appendChild(hiddenCopyText2);

    container.appendChild(exponent);
    return container;
}

/*
 * ##### TractionExpression #####
 */
/**
 * A Traction Expression is a group of values with additive association.
 * @param terms
 * @constructor
 */
TRACTION.TractionExpression = function(terms) {
    if (!Array.isArray(terms)) {
        throw new TypeError("Not an array: " + terms);
    }
    this.terms = terms;
}
TRACTION.TractionExpression.prototype.toMarkup = function() {
    const element = document.createElement("span");
    for (let i=0; i<this.terms.length; i++) {
        let term = this.terms[i];
        if (i > 0) {
            const operatorElement = document.createElement("span");
            operatorElement.classList.add("symbol")
            if (term.isInverse) {
                operatorElement.textContent = "-";
            } else {
                operatorElement.textContent = "+";
            }
            element.appendChild(operatorElement);
        }
        element.appendChild(term.toMarkup());
    }
    return element;
}

/**
 * A Traction Term is a group of values with multiplicative association.
 * @constructor
 */
TRACTION.TractionTerm = function(terms) {
    if (!Array.isArray(terms)) {
        throw new TypeError("Not an array: " + terms);
    }
    this.terms = terms;
};
TRACTION.TractionTerm.prototype.toMarkup = function() {
    const element = document.createElement("span");
    for (let i=0; i<this.terms.length; i++) {
        let term = this.terms[i];
        if (i > 0) {
            const operatorElement = document.createElement("span");
            operatorElement.classList.add("symbol")
            if (term.isReciprocal) {
                operatorElement.textContent = "/";
            } else {
                operatorElement.textContent = "*";
            }
            element.appendChild(operatorElement);
        }
        element.appendChild(term.toMarkup());
    }
    return element;
}

/*
 * ##### Tokens #####
 */
;(() => {
    const numberExpr = /^\s*(-?[0-9]+)\s*$/

    function parseNumber(str) {
        const numberMatcher = str.match(numberExpr);
        if (numberMatcher) {
            if (numberMatcher[1].length > 0) {
                let parsed = Number.parseInt(numberMatcher[1]);
                if (isNaN(parsed)) {
                    throw "Can't parse "+numberMatcher[1];
                }
                return parsed;
            }
        }
        return null;
    }

    TRACTION.TractionToken = function (value) {
        let number = parseNumber(value);
        if (number !== null) {
            this.value = number;
        } else if (value === 'w') {
            this.value = TRACTION.W;
        } else {
            this.value = value;
        }
    }

})();

/*
 * ##### TractionTokenizer #####
 */
;(() => {
    const parensGroupExpr = /^(.*)[(]([^)]*?)[)](.*)/s
    const numberExpr = /^\s*([-\s]*[0-9]+)(.*)/s
    const wExpr = /^\s*([-\s]*?[0-9]*[\s*]?w)(.*)/s
    const operationExpr = /^\s*([+\-/*^])(.*)/s
    const logExpr = /^\s*(log_?[0-9w]*)(.*)/s
    const whitespaceExpr = /^\s+/s

    TRACTION.TractionTokenizer = function({} = {}) {
        // Add tokenizer options here
    }

    /**
     * Produces a recursive array of token strings (grouped by parenthesis)
     * @param exprString
     */
    TRACTION.TractionTokenizer.prototype.parseExpression = function(exprString) {
        const parensMatcher = exprString.match(parensGroupExpr);
        if (parensMatcher) {
            const before = this.parseExpression(parensMatcher[1]);
            const inside = this.parseExpression(parensMatcher[2]);
            const after = this.parseExpression(parensMatcher[3]);
            return before.concat([inside], after);
        }
        const operationMatcher = exprString.match(operationExpr);
        if (operationMatcher) {
            const operation = new TRACTION.TractionToken(operationMatcher[1]);
            const after = this.parseExpression(operationMatcher[2]);
            return [operation].concat(after);
        }
        const wMatcher = exprString.match(wExpr);
        if (wMatcher) {
            const wNumber = new TRACTION.TractionToken(wMatcher[1]);
            const after = this.parseExpression(wMatcher[2]);
            return [wNumber].concat(after);
        }
        const numberMatcher = exprString.match(numberExpr);
        if (numberMatcher) {
            const number = new TRACTION.TractionToken(numberMatcher[1]);
            const after = this.parseExpression(numberMatcher[2]);
            return [number].concat(after);
        }
        const logMatcher = exprString.match(logExpr);
        if (logMatcher) {
            const log = new TRACTION.TractionToken(logMatcher[1]);
            const after = this.parseExpression(logMatcher[2]);
            return [log].concat(after);
        }
        const whitespaceMatcher = exprString.match(whitespaceExpr);
        if (whitespaceMatcher || exprString.length === 0) {
            return [];
        }

        throw new TypeError("Unable to parse substring: "+exprString);
    }
})();


/*
 * ##### Traction AST #####
 */

;(() => {

    const findTokenValue = function (array, value) {
        for (let i=0; i<array.length; i++) {
            if (array[i].value === value) {
                return i;
            }
        }
        return -1;
    }

    const tokenArrayToASTRecursion = function(tokenArray, middleElement, index) {
        let previous = [];
        if (index-1 >= 0) {
            previous = tokenArray.slice(0, index - 1);
        }
        let next = [];
        if (index+2 <= tokenArray.length) {
            next = tokenArray.slice(index+2, tokenArray.length);
        }
        return TRACTION.tokenArrayToAST([].concat(
            previous,
            [middleElement],
            next
        ));
    }

    const recurseIfArray = function(token) {
        if (Array.isArray(token) && token.length > 0) {
            let tokenArrayToAST = TRACTION.tokenArrayToAST(token);
            if (tokenArrayToAST.length === 1) {
                return tokenArrayToAST[0];
            } else {
                return tokenArrayToAST;
            }
        } else {
            return token;
        }
    }

    const parseBinaryOperation = function(tokenArray, symbol, factory) {
        const symbolIndex = findTokenValue(tokenArray, symbol);
        if (symbolIndex > 0) {
            const previousToken = recurseIfArray(tokenArray[symbolIndex-1]);
            const nextToken = recurseIfArray(tokenArray[symbolIndex+1]);
            return tokenArrayToASTRecursion(
                tokenArray,
                factory(convertTokenToValue(previousToken), convertTokenToValue(nextToken)),
                symbolIndex
            );
        }
        return null;
    }

    const convertTokenToValue = function(token) {
        if(typeof token.value === 'number') {
            return new TRACTION.TractionNumber(token.value);
        }
        return token;
    }

    TRACTION.tokenArrayToAST = function(tokenArray) {
        let op = null;
        if (tokenArray.length === 0) {
            return [];
        }

        op = parseBinaryOperation(
            tokenArray,
            '^',
            (base, exp) => new TRACTION.TractionExponential(base, exp)
        );
        if (op) return op;

        op = parseBinaryOperation(
            tokenArray,
            '*',
            (left, right) => new TRACTION.TractionTerm([left, right])
        );
        if (op) return op;

        op = parseBinaryOperation(
            tokenArray,
            '/',
            (left, right) => new TRACTION.TractionTerm([left, new TRACTION.ReciprocalTerm(right)])
        );
        if (op) return op;

        op = parseBinaryOperation(
            tokenArray,
            '+',
            (left, right) => new TRACTION.TractionExpression([left, right])
        );
        if (op) return op;

        op = parseBinaryOperation(
            tokenArray,
            '-',
            (left, right) => new TRACTION.TractionExpression([left, new TRACTION.InverseTerm(right)])
        );
        if (op) return op;

        if (tokenArray.length === 1) {
            const val = convertTokenToValue(tokenArray[0]);
            if (val) {
                return val;
            }
        }

        return tokenArray;
    }

    TRACTION.parseExpressionToAST = function(expressionString) {
        const tokenizer = new TRACTION.TractionTokenizer();
        const tokens = tokenizer.parseExpression(expressionString);
        return TRACTION.tokenArrayToAST(tokens);
    }

    const generateMarkupForAST = function(parentElement, ast) {
        if (Array.isArray(ast)) {
            for (let i=0; i<ast.length; i++) {
                const node = ast[i];
                generateMarkupForAST(parentElement, node);
            }
        } else {
            parentElement.appendChild(ast.toMarkup());
        }
    }

    TRACTION.generateMarkupForAST = function(ast) {
        const expression = document.createElement("div");
        expression.classList.add("expression");
        expression.classList.add("font-lg");
        generateMarkupForAST(expression, ast);
        return expression;
    }
})();

