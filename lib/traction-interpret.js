/**
 * The code for parsing and printing COTT math notation.
 */

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
        this.error = false;
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
     * TODO: sort by index
     */
    INTERPRET.VisitorActions.prototype.execute = function() {
        const indexAdjustments = {};
        for (let removal of this.slatedForRemoval) {
            if (!indexAdjustments[removal.root]) {
                indexAdjustments[removal.root] = 0;
            }
            removal.root.splice(removal.index + indexAdjustments[removal.root], 1);
            indexAdjustments[removal.root] = indexAdjustments[removal.root] - 1;
        }
    }

    INTERPRET.VisitorActions.prototype.setError = function(error) {
        console.log("error at VisitorActions");
        this.error = error;
    }

    /**
     * Visitor object to trivialize walking and mutating the AST.
     * @param visitAllBefore
     * @param visitAllAfter
     * @param visitUnknown
     * @param visitNumbers
     * @param visitNames
     * @param visitOperationsBefore
     * @param visitOperationsMiddle
     * @param visitOperationsAfter
     * @param visitGroupsBefore
     * @param visitGroupsAfter
     * @constructor
     */
    INTERPRET.AstVisitor = function({
                                        visitAllBefore = ()=>{},
                                        visitAllAfter = ()=>{},
                                        visitUnknown = ()=>{},
                                        visitNumbers = ()=>{},
                                        visitNames = ()=>{},
                                        visitOperationsBefore = ()=>{},
                                        visitOperationsMiddle = ()=>{},
                                        visitOperationsAfter = ()=>{},
                                        visitGroupsBefore = ()=>{},
                                        visitGroupsAfter = ()=>{},
                                    }) {
        this.visitAllBefore = visitAllBefore;
        this.visitAllAfter = visitAllAfter;
        this.visitUnknown = visitUnknown;
        this.visitNumbers = visitNumbers;
        this.visitNames = visitNames;
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
            } else if (node.type === INTERPRET.UNARY_OP) {
                this.visitOperationsBefore(node, this.actions);
                this.walk(node.right);
                this.actions._sync(astRoot, i);
                this.visitOperationsAfter(node, this.actions);
            }else if (node.type === INTERPRET.INTEGER || node.type === INTERPRET.FLOAT) {
                this.visitNumbers(node, this.actions);
            } else if (node.type === INTERPRET.NAME) {
                this.visitNames(node, this.actions);
            }
            if (this.actions.error) {
                console.log("error at AstVisitor.walk");
                return;
            }
            this.actions._sync(astRoot, i);
            this.visitAllAfter(node, this.actions);
        }
    }

})();

;(()=>{
    const groupSymbolExpr = /^(.*?)([{\[()}\]])(.*)$/
    const groupOpenExpr = /^(.*?)([{\[(])(.*)$/
    const groupCloseExpr = /^(.*?)([)}\]])(.*)$/

    /**
     * AST Parsing Step 1
     * Take the string and split it into a rough outline of the AST tree.
     * @param tokens
     * @param str
     * @param context
     * @returns {*|null}
     */
    const splitByParenthesis = function(tokens, str, context={groupCount:0, groupMatchStack:[]}) {
        if (!str) {
            return null;
        }

        let groupSymbolMatch = str.match(groupSymbolExpr);
        if (groupSymbolMatch) {
            let groupSymbol = groupSymbolMatch[2];
            let groupOpenMatch = groupSymbol.match(groupOpenExpr);
            let groupCloseMatch = groupSymbol.match(groupCloseExpr);
            if (groupOpenMatch) {
                if (groupSymbolMatch[1].length > 0) {
                    tokens.push({
                        value: groupSymbolMatch[1],
                        type: INTERPRET.UNKNOWN,
                    });
                }
                if (groupSymbolMatch[3].length > 0) {
                    context.groupCount = context.groupCount + 1;
                    context.groupMatchStack.push({tokens});
                    const nextTokens = [];
                    tokens.push({
                        value: nextTokens,
                        type: INTERPRET.GROUP,
                    })
                    splitByParenthesis(nextTokens, groupSymbolMatch[3], context);
                } else {
                    throw new Error("Unexpected end of group")
                }
            } else if (groupCloseMatch) {
                if (groupSymbolMatch[1].length > 0) {
                    tokens.push({
                        value: groupSymbolMatch[1],
                        type: INTERPRET.UNKNOWN,
                    });
                }
                context.groupCount = context.groupCount - 1;
                tokens = context.groupMatchStack.pop().tokens;

                if (groupSymbolMatch[3].length > 0) {
                    const nextTokens = [];
                    tokens.push({
                        value: nextTokens,
                        type: INTERPRET.UNKNOWN,
                    });
                    splitByParenthesis(nextTokens, groupSymbolMatch[3], context);
                }
            }

        }
        if (context.groupCount > 0) {
            console.log("Group match error");
            throw new Error("Parse Error");
        }
    }

    const operationExprs = [
        /(.*?)([+-])(.*)/,
        /(.*?)([*/])(.*)/,
        /(.*?)(\^)(.*)/,
      //  /(.*?)(_)(.*)/,
    ]

    /**
     * AST Parsing Step 2
     */
    const splitByOperator = function(ast) {
        let previousNodeState = null;
        let hold = null;

        const visitAllBefore = function(node, actions) {
            if (hold) {
                hold.node.right.push(node);
                hold = null;
                actions.remove();
            }
            if (node.type === INTERPRET.UNKNOWN) {
                for (let opExpr of operationExprs) {
                    if (typeof node.value === 'string') {
                        let match = node.value.match(opExpr);
                        if (match) {
                            node.value = match[2];
                            node.left = [];
                            node.right = [];
                            let before = match[1];
                            if (before.length > 0) {
                                let nextNode = {
                                    type: INTERPRET.UNKNOWN,
                                    value: before
                                };
                                visitAllBefore(nextNode, actions);
                                node.left.push(nextNode);
                            } else {
                                if (previousNodeState) {
                                    node.left.push(previousNodeState.node);
                                    actions.remove(previousNodeState);
                                } else {
                                    throw new Error("Missing operand")
                                }
                            }
                            let after = match[3];
                            if (after.length > 0) {
                                let nextNode = {
                                    type: INTERPRET.UNKNOWN,
                                    value: after
                                };
                                visitAllBefore(nextNode, actions);
                                node.right.push(nextNode);
                            } else {
                                hold = actions.getCurrentState();
                            }
                            break;
                        } else if (node.value === "log_0") {
                            node.type = INTERPRET.UNARY_OP;
                            node.right = [];
                            hold = actions.getCurrentState();
                        }

                    }
                }
            }
        }

        const visitAllAfter = function(node, actions) {
            previousNodeState = actions.getCurrentState();
        }

        const visitor = new INTERPRET.AstVisitor({
            visitAllBefore,
            visitAllAfter
        });
        visitor.walk(ast);
        visitor.actions.execute();
    }


    const intExpr = /^\s*([0-9]+)\s*$/
    const floatExpr = /^\s*([0-9]+[.][0-9]*)\s*$/
    const nameExpr = /^\s*([a-zA-Z]+)\s*$/
    const operationExpr = /^\s*([-*+/^])\s*$/

    /**
     * AST Parsing Step 3
     * Walks the tree, assigning types, parsing values, and combining nodes.
     * @param ast
     */
    const annotateValueNodes = function(ast) {

        const populateValueNode = function(token) {
            if (token.type === INTERPRET.UNKNOWN && typeof token.value === 'string') {
                let matchInteger = token.value.match(intExpr);
                let matchFloat = token.value.match(floatExpr);
                let matchName = token.value.match(nameExpr);
                let matchOperation = token.value.match(operationExpr)
                if (matchInteger) {
                    token.value = Number.parseInt(matchInteger[1]);
                    token.type = INTERPRET.INTEGER;
                } else if (matchFloat) {
                    token.value = Number.parseFloat(matchInteger[1]);
                    token.type = INTERPRET.FLOAT;
                } else if (matchName) {
                    token.value = matchName[1];
                    token.type = INTERPRET.NAME;
                } else if (matchOperation) {
                    token.value = matchOperation[1];
                    token.type = INTERPRET.OPERATION;
                } else {
                    token.error = true;
                    nodeVisitor.actions.error = true;
                    console.log("annotateValueNodes error");
                    console.log("Unexpected value " + token.value)
                }
            } else if (token.type === INTERPRET.UNARY_OP) {
                if (token.value === 'log_0') {
                    token.value = 'log';
                    token.base = 0;
                }
            }
        }

        const nodeVisitor = new INTERPRET.AstVisitor({
            visitAllBefore: populateValueNode,
        });
        nodeVisitor.walk(ast);
        nodeVisitor.actions.execute();
    }

    const markParseError = function(e, ast) {
        console.log(e);
        if (ast.length > 0) {
            ast[ast.length-1].error = true;
        }
    }

    /**
     * Parses a single expression.
     * Returns a normalized AST, or tokens arranged in a graph.
     * Arranging it in a graph here reduces logic duplication.
     */
    INTERPRET.parseExpression = function(expressionString) {
        const ast = [];
        try {
            splitByParenthesis(ast, expressionString);
            splitByOperator(ast);
            annotateValueNodes(ast);
            console.log(ast);


        } catch (e) {
            markParseError(e, ast);
            return ast;
        }
        // try {
        //     normalizeOperationGraph(ast);
        // } catch (e) {
        //     markParseError(e, ast);
        //     return ast;
        // }
        // try {
        //     annotateValueNodes(ast);
        // } catch (e) {
        //     markParseError(e, ast);
        //     return ast;
        // }
        return ast;
    }

    INTERPRET.printUniversalFormat = function(ast) {
        let nodeVisitor;
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
        const visitAllBefore = (token) => {
            if (token.error) {
                console.log("printUniversalFormat visitAllBefore error");
                stringified += "?"
                nodeVisitor.actions.setError(true);
            }
        }
        nodeVisitor = new INTERPRET.AstVisitor({
            visitAllBefore,
            visitUnknown: visitOthers,
            visitOperationsMiddle,
            visitGroupsBefore,
            visitGroupsAfter,
        });
        try {
            nodeVisitor.walk(ast);
        } catch (e) {
            console.log(e);
            stringified += "?"
        }

        return stringified;
    }

    INTERPRET.printMarkup = function(ast) {
        let stringified = "";
        let nodeVisitor;

        const visitOperationsBefore = (token) => {
            if (token.type === INTERPRET.UNARY_OP) {
                if (token.base !== null) {
                    stringified += "<span class='variable'>" + token.value + "</span>"
                        + "<sub>" + token.base + "</sub>"
                        + "<span class='symbol'>(</span>";
                } else {
                    stringified += token.value;
                }
            }
        }
        const visitOperationsMiddle = (token) => {
            if (token.value === '^') {
                stringified += "<sup>"
            } else if (token.value === '*') {
                stringified += "<span class='symbol'>&middot;</span>";
            } else {
                stringified += token.value;
            }
        }
        const visitOperationsAfter = (token) => {
            if (token.value === '^') {
                stringified += "</sup>"
            } else if (token.type === INTERPRET.UNARY_OP) {
                if (token.base !== null) {
                    stringified += "<span class='symbol'>)</span>";
                }
            }
        }
        const visitGroupsBefore = (token) => {
            stringified += "<span class='symbol'>"
            stringified += "(";
            stringified += "</span>"
        }
        const visitGroupsAfter = (token) => {
            stringified += "<span class='symbol'>"
            stringified += ")";
            stringified += "</span>"
        }
        const visitNumbers = (token) => {
            stringified += "<span class='number'>"
            stringified += token.value;
            stringified += "</span>";
        }
        const visitNames = (token) => {
            if (token.value === 'w') {
                stringified += "<span class='symbol'>"
                stringified += "&omega;";
            } else {
                stringified += "<span class='variable'>"
                stringified += token.value;
            }
            stringified += "</span>";
        }
        const visitAllBefore = (token) => {
            if (token.error) {
                console.log("INTERPRET.printMarkup visitAllBefore error");
                stringified += "?";
                nodeVisitor.actions.setError(true);
            }
        }
        nodeVisitor = new INTERPRET.AstVisitor({
            visitAllBefore,
            visitNumbers,
            visitNames,
            visitOperationsBefore,
            visitOperationsMiddle,
            visitOperationsAfter,
            visitGroupsBefore,
            visitGroupsAfter,
        });
        try {
            nodeVisitor.walk(ast);
        } catch (e) {
            console.log(e);
            console.log("INTERPRET.printMarkup visitor.walk error");
            stringified += "<span class='error'>?</span>"
        }

        stringified = "<div class='expression font-lg'>" + stringified + "</div>";

        return stringified;
    }
})();
