import torch
import sympy
from sympy.logic.boolalg import BooleanFunction, And, Or, Not, Equivalent, Implies, Xor
import numpy as np
import random

class BooleanSimplificationEnv:
    def __init__(self, max_expression_depth, max_literals, max_steps):
        self.max_expression_depth = max_expression_depth
        self.max_literals = max_literals
        self.max_steps = max_steps

        self.literals = [sympy.Symbol(chr(ord('A') + i )) for i in range(max_literals)]
        self.current_expression = None
        self.initial_complexity = 0
        self.steps_taken = 0
        self.action_space_size = 10

    def _generate_random_expr(self, depth):
        if depth == 0 or random.random() < 0.3:
            return random.choice(self.literals)
        
        else:
            op_types = [And, Or, Not, Equivalent, Implies, Xor]
            op_select = random.choice(op_types)

            if op_select == Not:
                # unary operators
                return op_select(self._generate_random_expr(depth - 1))
            
            else:
                arg1 = self._generate_random_expr(depth - 1)
                arg2 = self._generate_random_expr(depth - 1)
                return op_select(arg1, arg2)
            
    def _get_complexity(self, expr):
        # base case: if it is a literal (A, B) complexity is 1
        if (isinstance(expr, sympy.Symbol)):
            return 1
        
        elif (isinstance(expr, BooleanFunction)):
            # complexity of the operator itself
            complexity = 1
            for arg in expr.args:
                complexity += self._get_complexity(arg)
            return complexity

        # fallback
        else: 
            return 0

    def reset(self):
        # fix: generate a random boolean expression
        self.current_expression = self._generate_random_expr(self.max_expression_depth)
        # compute complexity
        self.initial_complexity = self._get_complexity(self.current_expression)
        # reset steps_taken to zero
        self.steps_taken = 0        

        return self._get_state()
    
    def _get_state(self):
        # convert self.current_expression into a numerical feature vector for the neural net to work on
        # [count_literals, count_and, count_or, count_not, count_equivalent, count_implies, count_xor, max_expr_depth, current_complexity, is_simplified]

        count_literals = len(self.current_expression.atoms(sympy.Symbol))
        count_and = len(self.current_expression.atoms(And))
        count_or = len(self.current_expression.atoms(Or))
        count_not = len(self.current_expression.atoms(Not))
        count_equivalent = len(self.current_expression.atoms(Equivalent))
        count_implies = len(self.current_expression.atoms(Implies))
        count_xor = len(self.current_expression.atoms(Xor))

        def get_depth(expr):
            if not isinstance(expr, BooleanFunction):
                # for relative depth, consider the depth of literal as zero
                return 0
            
            if not expr.args:
                return 1
            
            return 1 + max(get_depth(arg) for arg in expr.args)
        
        depth = get_depth(self.current_expression)
        current_complexity = self._get_complexity(self.current_expression)

        # fix: check if current_expression is logically equivalent with sympy simplified form, with canonical expression
        is_simplified = int(sympy.Equivalent(sympy.simplify_logic(self.current_expression), self.current_expression))
        # numerical feature vector
        state = np.array([count_literals, count_and, count_or, count_not, count_equivalent, count_implies, count_xor, depth, current_complexity, is_simplified])
        return state
    
    def 
    

