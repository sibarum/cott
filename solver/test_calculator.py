"""Quick smoke tests for the calculator parser/evaluator."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from calculator import parse_and_eval, format_result

tests = [
    ('0^0', '1'),
    ('0^w', '-1'),
    ('w^w', '-1'),
    ('0^(-1)', '\u03c9'),
    ('w^(-1)', '0'),
    ('0^(-3)', '\u03c9^3'),
    ('w^(-2)', '0^2'),
    ('0*w', '1'),
    ('0^(0^3)', '3'),
    ('w^(w^2)', '-1/2'),
    ('2+3', '5'),
    ('2*3', '6'),
    ('0^(w^2)', '-2'),
    ('(0^2)*w', '0'),
    ('1/0', '\u03c9'),
    ('1/w', '0'),
    ('5^0', '1'),
    ('3*0*w', '3'),
    ('x+x', '2\u00b7x'),
    ('x*0', '0\u00b7x'),
    ('log0(1)', '0'),
    ('log0(0)', '1'),
    ('logw(1)', '0'),
    ('logw(\u03c9)', '1'),
    ('log0(0^3)', '3'),
    # Negative zero absorption
    ('(-0)*w', '1'),
    ('(-1)*0', '0'),
    ('(-0)^(-1)', '\u03c9'),
    ('(-2)*0', '2\u00b70'),
    # Power combination
    ('(0^2)*w', '0'),
    ('0^5*w^2', '0^3'),
    # Universal power-of-power
    ('(x^2)^(1/2)', 'x'),
    ('(x^4)^(1/4)', 'x'),
    # Symbolic log
    ('log0(x)', 'log\u2080(x)'),
    ('logw(x)', 'log\u03c9(x)'),
    # Variables
    ('x+y', 'x+y'),
    ('y^2', 'y^2'),
    # Zero/omega interactions
    ('0^(w/2)', '0^(1/2\u00b7\u03c9)'),
    ('0/0', '1'),
    ('w/w', '1'),
]

passed = 0
failed = 0
for expr, expected in tests:
    result = format_result(parse_and_eval(expr))
    ok = result == expected
    status = 'PASS' if ok else f'FAIL (got {result})'
    print(f'  {expr:20s} = {result:10s}  {status}')
    if ok:
        passed += 1
    else:
        failed += 1

print(f'\n{passed}/{passed + failed} passed')
