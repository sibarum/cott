
TN = {};

(()=>{
    TN.PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

    TN.Scalar = class {
        constructor(magnitude) {
            this.magnitude = magnitude;
        }

        type_hash() {
            return TN.Scalar.TYPE_HASH;
        }

        eval_hash() {
            return this.type_hash();
        }

        eval_with(f) {
            throw new Error("Can't eval a value directly.")
        }

        walk(callbacks) {
            callbacks.before(this);
            callbacks.after(this);
            return callbacks.mutate(this);
        }
    }
    TN.Scalar.TYPE_HASH = TN.PRIMES[0];

    TN.Residual = class {
        constructor(magnitude) {
            this.magnitude = magnitude;
        }

        type_hash() {
            return TN.Residual.TYPE_HASH;
        }

        eval_hash() {
            return this.type_hash();
        }

        eval_with(f) {
            throw new Error("Can't eval a value directly.")
        }

        walk(callbacks) {
            callbacks.before(this);
            callbacks.after(this);
            return callbacks.mutate(this);
        }
    }
    TN.Residual.TYPE_HASH = TN.PRIMES[1];

    TN.BinaryOp = class {
        constructor(left, right, op_code) {
            this.left = left;
            this.right = right;
            this.op_code = op_code;
        }

        type_hash() {
            return TN.BinaryOp.TYPE_HASH;
        }

        eval_hash() {
            return hash([this.left.type_hash(), this.left.type_hash(), this.op_code]);
        }

        eval_with(f) {
            return f(this.left, this.right);
        }

        walk(callbacks) {
            callbacks.before(this);
            this.left = this.left.walk(callbacks);
            this.right = this.right.walk(callbacks);
            callbacks.after(this);
            return callbacks.mutate(this);
        }
    }
    TN.BinaryOp.TYPE_HASH = TN.PRIMES[2];

    TN.UnaryOp = class {
        constructor(operand, op_code) {
            this.operand = operand;
            this.op_code = op_code;
        }

        type_hash() {
            return TN.UnaryOp.TYPE_HASH;
        }

        eval_hash() {
            return hash([this.operand.type_hash(), this.op_code]);
        }

        eval_with(f) {
            return f(this.operand);
        }

        walk(callbacks) {
            callbacks.before(this);
            this.operand = this.operand.walk(callbacks);
            callbacks.after(this);
            return callbacks.mutate(this);
        }
    }
    TN.UnaryOp.TYPE_HASH = TN.PRIMES[8];


    const hash = arr => arr.reduce((num, x) => num * x, 1);

    TN.ADD = TN.PRIMES[3];
    TN.MULTIPLY = TN.PRIMES[4];
    TN.NEGATE = TN.PRIMES[5];
    TN.INVERT = TN.PRIMES[6];
    TN.EXP = TN.PRIMES[7];

    TN.Variable = class {
        constructor(name, value=null) {
            this.name = name;
            this.value = value;
        }

        type_hash() {
            return TN.Variable.TYPE_HASH;
        }

        eval_hash() {
            return this.type_hash();
        }

        eval_with(f) {
            return f(this);
        }

        walk(callbacks) {
            callbacks.before(this);
            callbacks.after(this);
            return callbacks.mutate(this);
        }
    }
    TN.Variable.TYPE_HASH = TN.PRIMES[9];


    // Unary Ops

    function scalar_negation(x) {
        return new TN.Scalar(-x.magnitude);
    }
    scalar_negation.hash = hash([TN.NEGATE, TN.Scalar.TYPE_HASH])

    function scalar_inversion(x) {
        return new TN.Scalar(1/x.magnitude);
    }
    scalar_inversion.hash = hash([TN.INVERT, TN.Scalar.TYPE_HASH])

    function residual_negation(x) {
        return new TN.Residual(-x.magnitude);
    }
    residual_negation.hash = hash([TN.NEGATE, TN.Residual.TYPE_HASH])

    function residual_inversion(x) {
        return new TN.Residual(1/x.magnitude);
    }
    residual_inversion.hash = hash([TN.INVERT, TN.Residual.TYPE_HASH])

    // Variable

    function eval_variable(v) {
        if (v.value === null) {
            throw new Error(`Variable ${v.name} was not defined.`)
        }
        return v.value;
    }
    eval_variable.hash = hash([TN.Variable.TYPE_HASH])

    // Same Type Addition

    function add_scalar_scalar(a,b) {
        return new TN.Scalar(a.magnitude + b.magnitude);
    }
    add_scalar_scalar.hash = hash([TN.Scalar.TYPE_HASH, TN.Scalar.TYPE_HASH, TN.ADD])

    function add_residual_residual(a,b) {
        return new TN.Residual(a.magnitude + b.magnitude);
    }
    add_residual_residual.hash = hash([TN.Residual.TYPE_HASH, TN.Residual.TYPE_HASH, TN.ADD])

    function add_binop_binop(a,b) {
        return new TN.BinaryOp(a, b, TN.ADD);
    }
    add_binop_binop.hash = hash([TN.BinaryOp.TYPE_HASH, TN.BinaryOp.TYPE_HASH, TN.ADD])

    function add_unnop_unnop(a,b) {
        return new TN.UnaryOp(a, b, TN.ADD);
    }
    add_unnop_unnop.hash = hash([TN.UnaryOp.TYPE_HASH, TN.UnaryOp.TYPE_HASH, TN.ADD])

    // Same Type Multiplication

    function multiply_scalar_scalar(a,b) {
        return new TN.Scalar(a.magnitude * b.magnitude);
    }
    multiply_scalar_scalar.hash = hash([TN.Scalar.TYPE_HASH, TN.Scalar.TYPE_HASH, TN.MULTIPLY])

    function multiply_residual_residual(a,b) {
        return new TN.Residual(a.magnitude * b.magnitude);
    }
    multiply_residual_residual.hash = hash([TN.Residual.TYPE_HASH, TN.Residual.TYPE_HASH, TN.MULTIPLY])

    function multiply_binop_binop(a,b) {
        return new TN.BinaryOp(a, b, TN.MULTIPLY);
    }
    multiply_binop_binop.hash = hash([TN.BinaryOp.TYPE_HASH, TN.BinaryOp.TYPE_HASH, TN.MULTIPLY])

    function multiply_unnop_unnop(a,b) {
        return new TN.UnaryOp(a, b, TN.MULTIPLY);
    }
    multiply_unnop_unnop.hash = hash([TN.UnaryOp.TYPE_HASH, TN.UnaryOp.TYPE_HASH, TN.MULTIPLY])

    // Same Type EXP

    function exp_scalar_scalar(a,b) {
        return new TN.Scalar(Math.pow(a.magnitude, b.magnitude));
    }
    exp_scalar_scalar.hash = hash([TN.Scalar.TYPE_HASH, TN.Scalar.TYPE_HASH, TN.EXP])

    function exp_residual_residual(a,b) {
        return new TN.Residual(Math.pow(a.magnitude, b.magnitude));
    }
    exp_residual_residual.hash = hash([TN.Residual.TYPE_HASH, TN.Residual.TYPE_HASH, TN.EXP])

    function exp_binop_binop(a,b) {
        return new TN.BinaryOp(a, b, TN.EXP);
    }
    exp_binop_binop.hash = hash([TN.BinaryOp.TYPE_HASH, TN.BinaryOp.TYPE_HASH, TN.EXP])

    function exp_unnop_unnop(a,b) {
        return new TN.UnaryOp(a, b, TN.EXP);
    }
    exp_unnop_unnop.hash = hash([TN.UnaryOp.TYPE_HASH, TN.UnaryOp.TYPE_HASH, TN.EXP])


    const ops_array = [
        scalar_negation,
        scalar_inversion,
        residual_negation,
        residual_inversion,
        eval_variable,
        add_scalar_scalar,
        add_residual_residual,
        add_binop_binop,
        add_unnop_unnop,
        multiply_scalar_scalar,
        multiply_residual_residual,
        multiply_binop_binop,
        multiply_unnop_unnop,
    ];

    const ops_map = {};
    for (let op of ops_array) {
        ops_map[op.hash] = op;
    }

    function evaluate_node(node) {
        const eval_hash = node.eval_hash();
        const strategy = ops_map[eval_hash];
        if (strategy) {
            node.eval_with(strategy);
        } else {
            throw new Error("No strategy for "+node);
        }
    }

    function id(_) {
        return _;
    }

    function walk_ast(rootNode, callbacks = {before: id, after: id, mutate: id}) {
        rootNode.walk(callbacks);
    }

    TN.eval_ast = function(rootNode) {
        walk_ast(rootNode, {before: id, after: id, mutate: evaluate_node})
    }

})()