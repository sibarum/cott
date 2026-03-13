
SAMPLES = {};

SAMPLES.buildExponentSample1 = function(x=0,y=1) {
    const eb = new OPS.ExprBuilder();
    return eb.expr(
        eb.exp(
            eb.add([
                eb.number(x),
                eb.multiply([
                    eb.number(y),
                    eb.phaseLift(1/2)
                ])
            ]),
            eb.number(2)
        )
    );
}