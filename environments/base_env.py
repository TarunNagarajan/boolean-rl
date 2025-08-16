import sympy
from sympy.logic.boolalg import BooleanFunction, And, Or, Not, Equivalent, Implies, Xor
import numpy as np
import random
from multiprocessing import Process, Queue

def _apply_rule_wrapper(rule, expr, queue, *args):
    try:
        if args:
            result = rule(expr, *args)
        else:
            result = rule(expr)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def _simplify_logic_default(expr):
    return sympy.simplify_logic(expr)

class BaseBooleanEnv:
    def __init__(self, max_expression_depth, max_literals, max_steps):
        self.max_expression_depth = max_expression_depth
        self.max_literals = max_literals
        self.max_steps = max_steps

        self.literals = [sympy.Symbol(chr(ord('A') + i)) for i in range(max_literals)]
        self.current_expression = None
        self.initial_complexity = 0
        self.known_best_complexity = 0
        self.steps_taken = 0
        self.history = []

    def _generate_random_expr(self, depth):
        if depth == 0 or random.random() < 0.3:
            return random.choice(self.literals)
        else:
            op_types = [And, Or, Not, Equivalent, Implies, Xor]
            op_select = random.choice(op_types)

            if op_select == Not:
                return op_select(self._generate_random_expr(depth - 1))
            else:
                arg1 = self._generate_random_expr(depth - 1)
                arg2 = self._generate_random_expr(depth - 1)
                return op_select(arg1, arg2)

    def _get_complexity(self, expr):
        if isinstance(expr, sympy.Symbol):
            return 1
        elif isinstance(expr, BooleanFunction):
            complexity = 1
            for arg in expr.args:
                complexity += self._get_complexity(arg)
            return complexity
        else:
            return 0

    def _get_state(self):
        raise NotImplementedError

    def get_action_size(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self, max_retries=10):
        # sympy operations can occasionally hang on complex, pathological expressions.
        # to ensure robust episode generation, the simplification call to establish a
        # complexity baseline is run in a separate process with a timeout. if it
        # hangs, the process is terminated and a new expression is generated.
        # this prevents the entire training run from freezing on a single bad case.
        for _ in range(max_retries):
            self.current_expression = self._generate_random_expr(self.max_expression_depth)
            if isinstance(self.current_expression, sympy.Symbol) or not self.current_expression.args:
                continue

            self.initial_complexity = self._get_complexity(self.current_expression)

            q = Queue()
            p = Process(target=_apply_rule_wrapper, args=(_simplify_logic_default, self.current_expression, q))
            p.start()
            p.join(5)

            if p.is_alive():
                p.terminate()
                p.join()
                continue

            result = q.get()
            if isinstance(result, Exception):
                continue

            self.known_best_complexity = self._get_complexity(result)

            if self.initial_complexity > self.known_best_complexity:
                self.steps_taken = 0
                self.history = [self.current_expression]
                return self._get_state()

        self.current_expression = self.literals[0] & self.literals[1]
        self.initial_complexity = self._get_complexity(self.current_expression)
        self.known_best_complexity = self._get_complexity(sympy.simplify_logic(self.current_expression))
        self.steps_taken = 0
        self.history = [self.current_expression]
        return self._get_state()