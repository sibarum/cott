INTERPRET = {};

INTERPRET.NUMBER = 'n';
INTERPRET.NAME = 'm';
INTERPRET.GROUP = 'g';
INTERPRET.OPERATION = 'o';
INTERPRET.UNKNOWN = '?';


;(() => {

    /**
     * Handy utility object for walking syntax trees, or really anything else.
     * Can slate objects for removal from arrays,
     * then can be triggered to execute removal later,
     * respecting any index changes that may occur in the process.
     * @constructor
     */
    INTERPRET.VisitorActions = function() {
        this.slatedForRemoval = [];
        this.root = null;
        this.currentIndex = null;
    }
    INTERPRET.VisitorActions.prototype._sync = function(root, currentIndex) {
        this.root = root;
        this.currentIndex = currentIndex;
    }
    INTERPRET.VisitorActions.prototype.getCurrentState = function() {
        return {
            root: this.root,
            index: this.currentIndex,
            node: this.root[this.currentIndex]
        };
    }
    INTERPRET.VisitorActions.prototype.remove = function(state) {
        this.slatedForRemoval.push(state || this.getCurrentState());
    }

    INTERPRET.VisitorActions.prototype.execute = function() {
        const indexAdjustments = {};
        for (let removal of this.slatedForRemoval) {
            if (!indexAdjustments[removal.root]) {
                indexAdjustments[removal.root] = 0;
            }
            removal.root.splice(removal.index, 1 + indexAdjustments[removal.root]);
            indexAdjustments[removal.root] = indexAdjustments[removal.root] - 1;
        }
    }

    /**
     * Visitor object to trivialize walking and mutating the AST.
     * @param visitAllBefore
     * @param visitAllAfter
     * @param visitUnknown
     * @param visitOperationsBefore
     * @param visitOperationsAfter
     * @param visitGroupsBefore
     * @constructor
     */
    INTERPRET.AstVisitor = function({
                                        visitAllBefore = ()=>{},
                                        visitAllAfter = ()=>{},
                                        visitUnknown = ()=>{},
                                        visitOperationsBefore = ()=>{},
                                        visitOperationsAfter = ()=>{},
                                        visitGroupsBefore = ()=>{},
                                    }) {
        this.visitAllBefore = visitAllBefore;
        this.visitAllAfter = visitAllAfter;
        this.visitUnknown = visitUnknown;
        this.visitOperationsBefore = visitOperationsBefore;
        this.visitOperationsAfter = visitOperationsAfter;
        this.visitGroupsBefore = visitGroupsBefore;

        this.actions = new INTERPRET.VisitorActions();
    }

    INTERPRET.AstVisitor.prototype.walk = function(astRoot) {
        for (let [i, node] of astRoot.entries()) {
            this.actions._sync(astRoot, i);
            this.visitAllBefore(node, this.actions);
            if (node.type === INTERPRET.GROUP) {
                this.visitGroupsBefore(node, this.actions);
                this.walk(node.value);
            }else if (node.type === INTERPRET.UNKNOWN) {
                this.visitUnknown(node, this.actions);
            }else if (node.type === INTERPRET.OPERATION) {
                this.visitOperationsBefore(node, this.actions);
                this.walk(node.left);
                this.walk(node.right);
                this.actions._sync(astRoot, i);
                this.visitOperationsAfter(node, this.actions);
            }
            this.actions._sync(astRoot, i);
            this.visitAllAfter(node, this.actions);

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

    /**
     * Recursively splits out all groups first.
     * This is done first because it's the most difficult to do reliably,
     * as groups can be placed practically anywhere,
     * and require a start and end token which can be ambiguous.
     * After the grouping is done, it then splits by operators,
     * because it's convenient to do so.
     * @param tokens
     * @param str
     * @returns {*|null}
     */
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

    /**
     * Splits by operator and fills in left/right operands
     */
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

    /**
     * Matches up groups with adjacent operations
     * @param tokens
     */
    const normalizeOperationGraph = function(tokens) {
        let hold = null;
        let nodeVisitor;
        let nextIndex = 0;
        let lastState = null;

        const visitAllBefore = (token) => {
            token.index = nextIndex++;
            if (hold) {
                hold.right = [token];
                nodeVisitor.actions.remove();
                hold = null;
            }
        }

        const visitAllAfter = (token) => {
            lastState = nodeVisitor.actions.getCurrentState();
        }

        const visitOps = (token) => {
            if (token.left.length === 0) {
                token.left = [lastState.node];
                nodeVisitor.actions.remove(lastState);

            }
            if (token.right.length === 0) {
                hold = token;
            }
        }

        nodeVisitor = new INTERPRET.AstVisitor({
            visitOperationsAfter: visitOps,
            visitAllBefore: visitAllBefore,
            visitAllAfter: visitAllAfter,
        });

        nodeVisitor.walk(tokens);
        nodeVisitor.actions.execute();
    }

    /**
     * This is THE method for parsing a single expression.
     * Returns a normalized AST.
     */
    INTERPRET.tokenize = function(expressionString) {
        const tokens = [];
        splitByParenthesis(tokens, expressionString);
        normalizeOperationGraph(tokens);
        return tokens;
    }
})();
