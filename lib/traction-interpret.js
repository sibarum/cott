INTERPRET = {};

;(() => {
    const whitespaceExpr = /^(\s+)(.*)$/
    const numberExpr = /^([0-9]+)(.*)$/
    const groupOpenExpr = /^([{\[(])(.*)$/
    const nameExpr = /^([a-zA-Z]+)(.*)$/
    const groupCloseExpr = /^([)}\]])(.*)$/
    const operationExpr = /^([-/+*^_])(.*)$/

    const tokenScan = function(tokens, str) {
        const whitespaceMatch = str.match(whitespaceExpr);
        if (whitespaceMatch) {
            return tokenScan(tokens, whitespaceMatch[2]);
        }

        const numberMatch = str.match(numberExpr);
        if (numberMatch) {
            const num = Number.parseInt(numberMatch[1]);
            tokens.push(num);
            return tokenScan(tokens, numberMatch[2]);
        }

        const groupOpenMatch = str.match(groupOpenExpr);
        if (groupOpenMatch) {
            const innerTokens = []
            innerTokens.push(groupOpenMatch[1]);
            tokens.push(innerTokens);
            const remainder = tokenScan(innerTokens, groupOpenMatch[2]);
            return tokenScan(tokens, remainder);
        }

        const nameMatch = str.match(nameExpr);
        if (nameMatch) {
            let name = nameMatch[1];
            if (name === 'w') {
                tokens.push(Infinity);
            } else {
                tokens.push(name);
            }
            return tokenScan(tokens, nameMatch[2]);
        }

        const groupCloseMatch = str.match(groupCloseExpr);
        if (groupCloseMatch) {
            tokens.push(groupCloseMatch[1]);
            return groupCloseMatch[2];
        }

        const operationMatch = str.match(operationExpr);
        if (operationMatch) {
            tokens.push(operationMatch[1]);
            return tokenScan(tokens, operationMatch[2]);
        }

    }

    /**
     * Tokenize one complete expression.
     * Returns an array of tokens.
     */
    INTERPRET.tokenize = function(expressionString) {
        const tokens = [];
        tokenScan(tokens, expressionString);
        return tokens;
    }
})();