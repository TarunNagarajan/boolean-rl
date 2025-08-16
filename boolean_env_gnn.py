import torch
import sympy
from sympy.logic.boolalg import BooleanFunction, And, Or, Not, Equivalent, Implies, Xor
import numpy as np
import random
from multiprocessing import Process, Queue
from torch_geometric.data import Data

from base_env import BaseBooleanEnv, _apply_rule_wrapper

def _apply_sympy_replace(expr, pattern, replacement):
    result, _ = expr.replace(pattern, replacement, map=True, simultaneous=True)
    return result

class BooleanSimplificationEnvGNN(BaseBooleanEnv):
    def __init__(self, max_expression_depth, max_literals, max_steps):
        super().__init__(max_expression_depth, max_literals, max_steps)

        self.NODE_TYPES = [And, Or, Not, Equivalent, Implies, Xor, type(sympy.true), type(sympy.false)] + self.literals
        self.NODE_TYPE_MAP = {nt: i for i, nt in enumerate(self.NODE_TYPES)}

        self.local_rules = [
            (sympy.sympify("A & A"), sympy.sympify("A"), "Idempotence (AND)"),
            (sympy.sympify("A | A"), sympy.sympify("A"), "Idempotence (OR)"),
            (sympy.sympify("A & True"), sympy.sympify("A"), "Identity (AND)"),
            (sympy.sympify("A | False"), sympy.sympify("A"), "Identity (OR)"),
            (sympy.sympify("A & False"), sympy.sympify("False"), "Domination (AND)"),
            (sympy.sympify("A | True"), sympy.sympify("True"), "Domination (OR)"),
            (sympy.sympify("~~A"), sympy.sympify("A"), "Double Negation"),
            (sympy.sympify("~(A & B)"), sympy.sympify("~A | ~B"), "De Morgan's (AND)"),
            (sympy.sympify("~(A | B)"), sympy.sympify("~A & ~B"), "De Morgan's (OR)"),
            (sympy.sympify("A & ~A"), sympy.sympify("False"), "Negation (AND)"),
            (sympy.sympify("A | ~A"), sympy.sympify("True"), "Negation (OR)"),
            (sympy.sympify("A & (A | B)"), sympy.sympify("A"), "Absorption (AND)"),
            (sympy.sympify("A | (A & B)"), sympy.sympify("A"), "Absorption (OR)"),
            (sympy.sympify("A ^ A"), sympy.sympify("False"), "XOR Self-Inverse"),
            (sympy.sympify("A >> A"), sympy.sympify("True"), "Implication Self"),
        ]
        self.action_space_size = len(self.local_rules)

        self.GLOBAL_FEATURE_MIN_VALUES = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.GLOBAL_FEATURE_MAX_VALUES = np.array([self.max_literals, 50, 50, 50, 50, 50, 50, self.max_expression_depth, 300], dtype=np.float32)
        
        self.reset()

    def _sympy_to_graph(self, expr):
        # this function translates a sympy expression into a graph representation
        # suitable for torch_geometric. the goal is to move from a flat, feature-based
        # state (like in the mlp) to a structural one. each node in the expression's
        # abstract syntax tree becomes a graph node, and its type is one-hot encoded.
        # this structural embedding allows the gnn to learn relationships between
        # operators and literals, a more fundamental approach than simple feature counts.
        nodes, edges, node_map = [], [], {}

        def _add_node(sub_expr, depth):
            if sub_expr in node_map:
                return node_map[sub_expr]

            node_idx = len(nodes)
            node_map[sub_expr] = node_idx

            node_type_idx = self.NODE_TYPE_MAP.get(type(sub_expr), -1) if not isinstance(sub_expr, sympy.Symbol) else self.NODE_TYPE_MAP.get(sub_expr, -1)
            one_hot = np.zeros(len(self.NODE_TYPES))
            if node_type_idx != -1:
                one_hot[node_type_idx] = 1

            is_literal = 1 if isinstance(sub_expr, sympy.Symbol) else 0
            is_operator = 1 if isinstance(sub_expr, BooleanFunction) else 0

            features = np.concatenate([one_hot, [is_literal, is_operator, depth]])
            nodes.append(features)

            if isinstance(sub_expr, BooleanFunction):
                for child_expr in sub_expr.args:
                    child_idx = _add_node(child_expr, depth + 1)
                    edges.extend([[node_idx, child_idx], [child_idx, node_idx]])
            
            return node_idx

        _add_node(expr, 0)

        x = torch.tensor(np.array(nodes), dtype=torch.float)
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, num_nodes=len(nodes))

    def _scale_global_features(self, features):
        clipped_features = np.clip(features, self.GLOBAL_FEATURE_MIN_VALUES, self.GLOBAL_FEATURE_MAX_VALUES)
        scaled_features = (clipped_features - self.GLOBAL_FEATURE_MIN_VALUES) / (self.GLOBAL_FEATURE_MAX_VALUES - self.GLOBAL_FEATURE_MIN_VALUES + 1e-8)
        return scaled_features

    def _get_state(self):
        atoms = self.current_expression.atoms()
        count_literals = len(atoms)
        count_and = str(self.current_expression).count('&')
        count_or = str(self.current_expression).count('|')
        count_not = str(self.current_expression).count('~')
        count_equivalent = str(self.current_expression).count('Equivalent')
        count_implies = str(self.current_expression).count('>>')
        count_xor = str(self.current_expression).count('^')
        
        current_complexity = self._get_complexity(self.current_expression)
        
        depth = max(len(str(a)) for a in atoms) if atoms else 0

        global_features = np.array([count_literals, count_and, count_or, count_not, count_equivalent, count_implies, count_xor, depth, current_complexity])
        scaled_global_features = self._scale_global_features(global_features)
        
        data = self._sympy_to_graph(self.current_expression)
        data.global_features = torch.tensor(scaled_global_features, dtype=torch.float).unsqueeze(0)

        return data

    def step(self, action):
        self.steps_taken += 1

        rules = self._get_available_rules()
        if not (0 <= action < len(rules)):
            return self._get_state(), -10.0, True, {'applied_rule': 'Invalid Action'}

        old_complexity = self._get_complexity(self.current_expression)

        rule_func, rule_name, pattern, replacement = rules[action]
        
        q = Queue()
        p = Process(target=_apply_rule_wrapper, args=(rule_func, self.current_expression, q, pattern, replacement))
        p.start()
        
        p.join(5)

        if p.is_alive():
            p.terminate()
            p.join()
            return self._get_state(), -10.0, True, {'applied_rule': f'{rule_name} (Timeout)'}

        result = q.get()
        if isinstance(result, Exception):
            return self._get_state(), -10.0, True, {'applied_rule': f'{rule_name} (Error)'}

        self.current_expression = result
        new_complexity = self._get_complexity(self.current_expression)

        # the reward function is the philosophical core of the agent. early versions
        # suffered from reward hacking, where the agent would increase complexity to
        # farm rewards from later, trivial simplifications. this final design is
        # direct and robust: it rewards any reduction in complexity, gives a large
        # bonus for reaching the known optimum, and penalizes inefficiency or
        # complexity increases. this avoids loopholes and directly incentivizes the
        # desired emergent behavior of efficient simplification.
        reward = 0.0
        done = False

        complexity_reduction = old_complexity - new_complexity

        if new_complexity < self.known_best_complexity:
            reward += 100.0
            self.known_best_complexity = new_complexity
            done = True
        elif new_complexity == self.known_best_complexity:
            reward += 50.0
            done = True
        elif complexity_reduction > 0:
            reward += 10.0 * complexity_reduction
        else:
            reward -= 2.0

        reward -= 1.0

        if self.steps_taken >= self.max_steps:
            done = True
            if new_complexity > self.known_best_complexity:
                reward -= 20.0

        return self._get_state(), reward, done, {'applied_rule': rule_name}

    def get_gnn_input_size(self):
        return len(self.NODE_TYPES) + 3
        
    def get_global_feature_size(self):
        return len(self.GLOBAL_FEATURE_MIN_VALUES)

    def _get_available_rules(self):
        return [(_apply_sympy_replace, name, pattern, replacement) for pattern, replacement, name in self.local_rules]

    def get_action_size(self):
        return self.action_space_size