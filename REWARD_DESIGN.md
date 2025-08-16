# Reward Function Design Evolution

This document outlines the iterative design process of the reward function for the Boolean Simplification RL environment.

## Version 1: The "One True Target" Fallacy

The initial reward function was based on a flawed premise: that there exists a single, canonical "simplest" form for any given boolean expression.

### Logic:
1.  At the start of each episode, the environment would generate a random expression.
2.  It would immediately use `sympy.simplify_logic` to generate a `target_simplified_expression`.
3.  The agent's reward was heavily based on whether its `current_expression` matched the exact string representation of the `target_simplified_expression`.
4.  Small rewards were given for reducing complexity, but the primary driver was matching the target.

### The Flaw:
Boolean algebra does not guarantee a unique simplest form. For example, `(A | B) & (A | C)` is logically equivalent to `A | (B & C)`. The `sympy` library would choose one, and the agent would be punished for finding the other, equally valid, simplified form. This created a brittle and often misleading goal.

---

## Version 2: Principled Reward Shaping

To address the "One True Target" fallacy, the reward system was redesigned to focus on the *property* of simplicity (i.e., low complexity) rather than a specific string representation.

### Logic:
1.  The concept of a `target_simplified_expression` was abolished.
2.  Instead, `sympy.simplify_logic` was used at the start of an episode to calculate a `known_best_complexity`—a benchmark for the agent to beat.
3.  The reward function was "shaped" to provide dense, informative signals to the agent:
    *   **Goal Bonus:** A large, one-time reward (`+50.0`) for achieving a complexity less than or equal to the `known_best_complexity`.
    *   **Shaped Progress Reward:** A continuous reward scaled by how much progress the agent made towards the goal (`(complexity_reduction / total_possible_reduction) * 10.0`).
    *   **Penalties:** Explicit penalties for increasing complexity or for taking actions that resulted in no change.
    *   **Efficiency Penalty:** A small, constant step penalty to encourage finding the shortest simplification path.

### The Flaw (The Reward Hacking Exploit):
This version introduced a subtle but critical exploit. The "Shaped Progress Reward" was based on the *total possible reduction from the start of the episode*. If the agent took an action that dramatically *increased* the complexity (e.g., from 17 to 62), it created a massive new "problem space." It could then make many small, easy simplifications on this new, larger expression. The sum of the small rewards for "cleaning up its own mess" far outweighed the penalty for creating it, leading to absurdly high scores for counterproductive behavior.

---

## Version 3: Robust, Direct Reward

The final version of the reward function was simplified to remove the scaling that enabled the reward hacking exploit. The reward is now based on the direct, immediate consequences of the agent's actions.

### Logic:
1.  **Direct Complexity Reward:** The base reward is simply `old_complexity - new_complexity`. This directly rewards simplification and penalizes increasing complexity.
2.  **Goal Bonus:** The large, one-time bonus (`+50.0`) for achieving the `known_best_complexity` is retained.
3.  **Penalties:**
    *   A penalty (`-1.0`) is applied if the complexity does not change, discouraging no-op actions.
    *   A small step penalty (`-0.1`) remains to encourage efficiency.
    *   A large penalty (`-10.0`) is applied if the agent fails to solve the problem within the time limit.

This final design is simpler, more robust, and directly incentivizes the desired behavior (reducing complexity) without creating loopholes for the agent to exploit.
