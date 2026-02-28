INTERPRET = {};

INTERPRET.INTEGER = 'i';
INTERPRET.FLOAT = 'f';
INTERPRET.NAME = 'm';
INTERPRET.GROUP = 'g';
INTERPRET.OPERATION = 'o';
INTERPRET.UNARY_OP = 'u';
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
    /**
     * Walking the tree has its ups and downs.
     * This function resets the state of the visitor whenever the node changes.
     * @param root
     * @param currentIndex
     * @private
     */
    INTERPRET.VisitorActions.prototype._sync = function(root, currentIndex) {
        this.root = root;
        this.currentIndex = currentIndex;
    }
    /**
     * The current state can be remembered just in case the node is later marked for removal.
     * @returns {{root: null, index: null, node: *}}
     */
    INTERPRET.VisitorActions.prototype.getCurrentState = function() {
        return {
            root: this.root,
            index: this.currentIndex,
            node: this.root[this.currentIndex]
        };
    }
    /**
     * Marks the array element at the given state for removal,
     * or at the current state if one isn't provided.
     */
    INTERPRET.VisitorActions.prototype.remove = function(state) {
        this.slatedForRemoval.push(state || this.getCurrentState());
    }

    /**
     * Performs the actual removal of elements from their respective arrays.
     */
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
                                        visitOperationsMiddle = ()=>{},
                                        visitOperationsAfter = ()=>{},
                                        visitGroupsBefore = ()=>{},
                                        visitGroupsAfter = ()=>{},
                                    }) {
        this.visitAllBefore = visitAllBefore;
        this.visitAllAfter = visitAllAfter;
        this.visitUnknown = visitUnknown;
        this.visitOperationsBefore = visitOperationsBefore;
        this.visitOperationsMiddle = visitOperationsMiddle;
        this.visitOperationsAfter = visitOperationsAfter;
        this.visitGroupsBefore = visitGroupsBefore;
        this.visitGroupsAfter = visitGroupsAfter;

        this.actions = new INTERPRET.VisitorActions();
    }

    /**
     * Performs the AST walk, using all visit callbacks defined in the constructor.
     * @param astRoot
     */
    INTERPRET.AstVisitor.prototype.walk = function(astRoot) {
        for (let [i, node] of astRoot.entries()) {
            this.actions._sync(astRoot, i);
            this.visitAllBefore(node, this.actions);
            if (node.type === INTERPRET.GROUP) {
                this.visitGroupsBefore(node, this.actions);
                this.walk(node.value);
                this.actions._sync(astRoot, i);
                this.visitGroupsAfter(node, this.actions);
            }else if (node.type === INTERPRET.UNKNOWN) {
                this.visitUnknown(node, this.actions);
            }else if (node.type === INTERPRET.OPERATION) {
                this.visitOperationsBefore(node, this.actions);
                this.walk(node.left);
                this.actions._sync(astRoot, i);
                this.visitOperationsMiddle(node, this.actions);
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

    /**
     * AST Parsing Step 1
     * Recursively splits out all groups first.
     * This is done first because it's the most difficult to do reliably,
     * as groups can be placed practically anywhere,
     * and require a start and end token which can be ambiguous.
     * After the grouping is done, it then splits by operators,
     * because it's convenient to do so.
     * Expectation: All groups are separated, all operations identified.
     * Operations may not be associated with both of their operands.
     * Value nodes will not be converted or identified.
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
     * AST Parsing Step 1.1
     * Splits by operator and fills in left/right operands.
     * Not intended to be used directly, called from splitByParenthesis.
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
     * AST Parsing Step 2
     * Matches up operation operands with adjacent groups.
     * Expectation: All operations should have both operands assigned.
     * @param tokens
     */
    const normalizeOperationGraph = function(tokens) {
        let hold = null; // Should only be held until the next token
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

    const intExpr = /^\s*([0-9]+)\s*$/
    const floatExpr = /^\s*([0-9]+[.][0-9]*)\s*$/
    const nameExpr = /^\s*([a-zA-Z]+)\s*$/

    /**
     * AST Parsing Step 3
     * Walks the tree, assigning types, parsing values, and combining nodes.
     * @param ast
     */
    const annotateValueNodes = function(ast) {

        const populateValueNode = function(token) {
            let matchInteger = token.value.match(intExpr);
            let matchFloat = token.value.match(floatExpr);
            let matchName = token.value.match(nameExpr);
            if (matchInteger) {
                token.value = Number.parseInt(matchInteger[1]);
                token.type = INTERPRET.INTEGER;
            } else if (matchFloat) {
                token.value = Number.parseFloat(matchInteger[1]);
                token.type = INTERPRET.FLOAT;
            } else if (matchName) {
                token.value = matchName[1];
                token.type = INTERPRET.NAME;
            } else {
                throw new Error("Value not recognized: " + token.value)
            }
        }

        let hold = null;
        const combineFunctions = function(token) {
            if (token.value === '_') {
                token.type = INTERPRET.UNARY_OP;
                token.value = token.left[0].value;
                token.base = token.right[0].value;
                delete token.left;
                hold = token;
            }
        }

        const removeFunctionGroup = function(token) {
            if (hold) {
                hold.right = [token]
                nodeVisitor.actions.remove();
            }
        }

        const nodeVisitor = new INTERPRET.AstVisitor({
            visitUnknown: populateValueNode,
            visitOperationsAfter: combineFunctions,
            visitGroupsBefore: removeFunctionGroup,
        });
        nodeVisitor.walk(ast);
        nodeVisitor.actions.execute();
    }

    /**
     * Parses a single expression.
     * Returns a normalized AST, or tokens arranged in a graph.
     * Arranging it in a graph here reduces logic duplication.
     */
    INTERPRET.parseExpression = function(expressionString) {
        const ast = [];
        splitByParenthesis(ast, expressionString);
        normalizeOperationGraph(ast);
        annotateValueNodes(ast);
        return ast;
    }

    INTERPRET.printUniversalFormat = function(ast) {
        let stringified = "";

        const visitOperationsMiddle = (token) => {
            stringified += token.value;
        }
        const visitGroupsBefore = (token) => {
            stringified += "(";
        }
        const visitGroupsAfter = (token) => {
            stringified += ")";
        }
        const visitOthers = (token) => {
            stringified += token.value;
        }
        const nodeVisitor = new INTERPRET.AstVisitor({
            visitUnknown: visitOthers,
            visitOperationsMiddle,
            visitGroupsBefore,
            visitGroupsAfter,
        });
        nodeVisitor.walk(ast);

        return stringified;
    }

    INTERPRET.printMarkup = function(ast) {
        let stringified = "";

        const visitOperationsMiddle = (token) => {
            stringified += token.value;
        }
        const visitGroupsBefore = (token) => {
            stringified += "(";
        }
        const visitGroupsAfter = (token) => {
            stringified += ")";
        }
        const visitOthers = (token) => {
            stringified += token.value;
        }
        const nodeVisitor = new INTERPRET.AstVisitor({
            visitUnknown: visitOthers,
            visitOperationsMiddle,
            visitGroupsBefore,
            visitGroupsAfter,
        });
        nodeVisitor.walk(ast);

        return stringified;
    }
})();
