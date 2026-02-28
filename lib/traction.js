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
    if (!base.toMarkup) {
        console.log(base);
        throw new Error("Invalid argument base");
    }
    if (!exponent.toMarkup) {
        console.log(exponent);
        throw new Error("Invalid argument exponent");
    }
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
 * ##### TractionLogarithm #####
 */
TRACTION.TractionLogarithm = function(base, operand) {
    if (Array.isArray(operand) && operand.length === 1) {
        operand = operand[0];
    }
    this.base = base;
    this.operand = operand;
}
TRACTION.TractionLogarithm.prototype.isBoundaryLift = function() {
    let baseIsNumber = Object.getPrototypeOf(this.base) === TRACTION.TractionNumber;
    if (baseIsNumber) {
        return this.base.isW() || this.base.isZero();
    }
    return false;
}
TRACTION.TractionLogarithm.prototype.toMarkup = function() {
    const container = document.createElement("span");

    const label = document.createElement("span");
    label.classList.add("variable");
    label.textContent = "log";
    container.appendChild(label);

    const hiddenCopyText1 = document.createElement("span");
    hiddenCopyText1.classList.add("copyonly");
    hiddenCopyText1.textContent = "_";
    container.appendChild(hiddenCopyText1)

    const subscript = document.createElement("sub");
    const base = this.base.toMarkup();
    subscript.appendChild(base);
    container.appendChild(subscript);

    const operand = document.createElement("span");
    const openParens = document.createElement("span");
    openParens.classList.add("symbol");
    openParens.textContent = "(";
    operand.appendChild(openParens);

    operand.appendChild(this.operand.toMarkup());

    const closeParens = document.createElement("span");
    closeParens.classList.add("symbol");
    closeParens.textContent = ")";
    operand.appendChild(closeParens);

    container.appendChild(operand);
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
            let ast = TRACTION.tokenArrayToAST(token);
            if (ast.length === 1) {
                return ast[0];
            } else if (ast.length === 2) {
                return new TRACTION.TractionExpression(ast[0].terms.concat(ast[1].terms));
            } else {
                return ast;
            }
        } else {
            return token;
        }
    }

    const parseBinaryOperation = function(tokenArray, symbol, factory) {
        const symbolIndex = findTokenValue(tokenArray, symbol);
        if (symbolIndex > 0) {
            const previousToken = tokenArray[symbolIndex-1];
            const nextToken = tokenArray[symbolIndex+1];
            return tokenArrayToASTRecursion(
                tokenArray,
                factory(TRACTION.tokenArrayToAST(previousToken), TRACTION.tokenArrayToAST(nextToken)),
                symbolIndex
            );
        }
        return null;
    }

    const convertTokenToValue = function(token) {
        if (token) {
            if(typeof token.value === 'number') {
                return new TRACTION.TractionNumber(token.value);
            }
        }
        return token;
    }

    const findSpecialToken = function(array, type) {
        for (let i=0; i<array.length; i++) {
            if (array[i].type && array[i].type === type) {
                return i;
            }
        }
        return -1;
    }

    const parseLogarithm = function(tokenArray) {
        const logTokenIndex = findSpecialToken(tokenArray, 'log');
        if (logTokenIndex >= 0) {
            const logBase = convertTokenToValue(tokenArray[logTokenIndex]);

            const nextToken = TRACTION.tokenArrayToAST(tokenArray[logTokenIndex+1]);

            let previous = [];
            if (logTokenIndex >= 0) {
                previous = tokenArray.slice(0, logTokenIndex);
            }
            let next = [];
            if (logTokenIndex + 2 <= tokenArray.length) {
                next = tokenArray.slice(logTokenIndex+2, tokenArray.length);
            }

            return TRACTION.tokenArrayToAST([].concat(
                previous,
                [new TRACTION.TractionLogarithm(logBase, nextToken)],
                next,
            ));
        }
        return null;
    }

    TRACTION.tokenArrayToAST = function(tokenArray) {
        if (!Array.isArray(tokenArray)) {
            tokenArray = [tokenArray];
        }
        let op = null;
        if (tokenArray.length === 0) {
            return [];
        }

        op = parseLogarithm(
            tokenArray
        );
        if (op) return op;

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
        if (op) return op

        const terms = [];
        for (let i=0; i<tokenArray.length; i++) {
            let tokenArrayElement = tokenArray[i];
            if (tokenArrayElement.toMarkup) {
                terms.push(tokenArrayElement);
            } else if (Object.getPrototypeOf(tokenArrayElement) === TRACTION.TractionToken.prototype) {
                terms.push(new TRACTION.TractionNumber(tokenArrayElement.value))
            } else {
                console.log(tokenArrayElement);
                throw new Error("token element not recognized.");
            }
        }
        if (terms.length === 1) {
            return terms[0];
        }
        return terms;

    }

    TRACTION.parseExpressionToAST = function(expressionString) {
        const tokenizer = new TRACTION.TractionTokenizer();
        const tokens = tokenizer.parseExpression(expressionString);
        console.log(tokens);
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

