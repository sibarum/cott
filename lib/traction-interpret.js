INTERPRET = {};

INTERPRET.NUMBER = 'n';
INTERPRET.NAME = 'm';
INTERPRET.GROUP = 'g';
INTERPRET.OPERATION = 'o';
INTERPRET.UNKNOWN = '?';


;(() => {

    INTERPRET.AstVisitor = function({visitAll = ()=>{}, visitUnknown = ()=>{}}) {
        this.visitAll = visitAll;
        this.visitUnknown = visitUnknown;
    }

    INTERPRET.AstVisitor.prototype.walk = function(astRoot) {
        for (let node of astRoot) {
            if (node.type === INTERPRET.GROUP) {
                this.walk(node.value);
            }else if (node.type === INTERPRET.UNKNOWN) {
                this.visitUnknown(node);
            }else if (node.type === INTERPRET.OPERATION) {
                this.walk(node.left);
                this.walk(node.right);
            }

            this.visitAll(node);
        }
    }

})();

;(()=>{
    const groupOpenExpr = /^(.*?)([{\[(])(.*)$/
    const groupCloseExpr = /^(.*?)([)}\]])(.*)$/

    const operationExprs = [
        /(.*)([+-])(.*)/,
        /(.*)([*/])(.*)/,
        /(.*)(\^)(.*)/,
        /(.*)(_)(.*)/,
    ]

    const whitespaceExpr = /^(\s+)(.*)$/
    const numberExpr = /^([0-9]+)(.*)$/
    const nameExpr = /^([a-zA-Z]+)(.*)$/

    const splitByParenthesis = function(tokens, str) {
        if (!str) {
            return null;
        }

        const groupOpenMatch = str.match(groupOpenExpr);
        if (groupOpenMatch) {
            let before = groupOpenMatch[1];
            splitByOperator(tokens, before);
            let afterTokens = [];
            let after = splitByParenthesis(afterTokens, groupOpenMatch[3]);
            tokens.push({
                value: afterTokens,
                type: INTERPRET.GROUP,
            });
            return splitByParenthesis(tokens, after);
        }

        const groupCloseMatch = str.match(groupCloseExpr);
        if (groupCloseMatch) {
            splitByOperator(tokens, groupCloseMatch[1]);
            return splitByParenthesis(tokens, groupCloseMatch[3]);
        }

        return splitByOperator(tokens, str);
    }

    const splitByOperator = function(tokens, str) {
        if (!str){
            return null;
        }

        for (let opRegex of operationExprs) {
            const operationMatch = str.match(opRegex);
            if (operationMatch) {
                let beforeTokens = [];
                splitByOperator(beforeTokens, operationMatch[1]);
                let afterTokens = [];
                let remainder = splitByOperator(afterTokens, operationMatch[3]);
                tokens.push({
                    value: operationMatch[2],
                    type: INTERPRET.OPERATION,
                    left: beforeTokens,
                    right: afterTokens
                });
                return remainder;
            }
        }
        tokens.push({
            value: str,
            type: INTERPRET.UNKNOWN,
        });
        return str;
    }

    const tokenScan = function(tokens, str) {
        const whitespaceMatch = str.match(whitespaceExpr);
        if (whitespaceMatch) {
            return tokenScan(tokens, whitespaceMatch[2]);
        }

        const numberMatch = str.match(numberExpr);
        if (numberMatch) {
            const num = Number.parseInt(numberMatch[1]);
            tokens.push({value: num, type: INTERPRET.NUMBER});
            return tokenScan(tokens, numberMatch[2]);
        }

        const nameMatch = str.match(nameExpr);
        if (nameMatch) {
            let name = nameMatch[1];
            if (name === 'w') {
                tokens.push({value: Infinity, type: INTERPRET.NUMBER});
            } else {
                tokens.push({value: name, type: INTERPRET.NAME});
            }
            return tokenScan(tokens, nameMatch[2]);
        }

    }

    /**
     * Tokenize one complete expression.
     * Returns an array of tokens.
     */
    INTERPRET.tokenize = function(expressionString) {
        const tokens = [];
        let remainder = splitByParenthesis(tokens, expressionString);
        console.log(remainder);
        return tokens;
    }
})();

;(()=>{

    const LeafFirstMutatingVisitor = function({visit=()=>{}}) {
        this.visit = visit;
    }

    LeafFirstMutatingVisitor.prototype.mutate = function(tokens) {
        for (let i=0; i<tokens.length; i++) {
            let token = tokens[i];
            if (token.type === INTERPRET.GROUP) {
                token.ast = this.mutate(token.value);
            }
        }
        return this.visit(tokens);
    }

    const astBuildingVisitor = function(tokens) {
        console.log(tokens);
        return null;
    }

    INTERPRET.tokensToAst = function(tokens) {
        const mutatingVisitor = new LeafFirstMutatingVisitor({visit: astBuildingVisitor});
        mutatingVisitor.mutate(tokens);
        return tokens;
    }

})();