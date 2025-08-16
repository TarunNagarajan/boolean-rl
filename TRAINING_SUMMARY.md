# Boolean Simplification RL Agent: Training Summary and Future Directions

This document summarizes the current status of the reinforcement learning agent's training for boolean expression simplification, key observations, and discussions regarding potential architectural improvements.

## Current Training Status (as of Episode ~431)

The agent is currently in the "Emerging Competence" phase of its training.

*   **Average Score:** The average score has transitioned from heavily negative (e.g., -140) to consistently positive (e.g., +24.14 at Episode 404). This indicates that the agent's successful episodes are now outweighing its failures, demonstrating a clear learning trend.
*   **Epsilon Decay:** Epsilon has decayed significantly (e.g., 0.37 at Episode 404), meaning the agent is now primarily exploiting its learned policy rather than relying on random exploration. This confirms that the observed performance is due to learned intelligence.
*   **Training Speed:** Recent observations indicate a significant speedup in training, with approximately 40 episodes completing in 15 minutes (22.5 seconds/episode). This suggests the agent is finding solutions more quickly or encountering fewer time-consuming operations.
*   **Remaining Episodes:** Based on the current speed, approximately 9.5 to 10 hours remain to complete the full 2000-episode run.

## Key Trends Noticed in the Runs

1.  **Initial Chaos (Episodes 1-100):** Characterized by high epsilon (random actions) and wildly fluctuating, often heavily negative, scores. This phase was crucial for populating the replay buffer with diverse experiences.
2.  **Reward Hacking (Early Episodes):** An initial flaw in the reward function led to the agent exploiting the system by increasing expression complexity to gain higher rewards from subsequent "cleanup" steps. This was identified and fixed by simplifying the reward function to directly reflect complexity change per step.
3.  **Hesitant Shift (Episodes 100-300):** Epsilon began decaying, and the agent started to attempt policy-driven actions. This phase saw continued negative scores as the agent's nascent policy made mistakes, but these failures provided valuable learning signals.
4.  **Emerging Competence (Episodes 300+):** The agent learned to avoid catastrophic actions, leading to a significant increase in the average score. This indicates the agent has learned a foundational "defense" strategy.

## Discussion: GNNs for Structural Understanding

A significant limitation of the current agent is its state representation and action space, which struggle with the inherent structural complexity of boolean expressions.

*   **Current Limitations:**
    *   **"Illiteracy":** The agent's state vector (counts of operators, depth) provides only a "bag-of-features" view, ignoring the actual structure (e.g., `A & (B | C)` vs. `(A & B) | C`). This makes it difficult for the agent to generalize effectively to new, complex expressions.
    *   **"Blunt Instrument" Actions:** Actions apply globally to the entire expression, preventing surgical, localized simplifications.
    *   **"Curse of Complexity":** The agent struggles with very large expressions because its exploration becomes inefficient, and its global actions are less effective.

*   **Potential of Graph Neural Networks (GNNs):**
    *   **Structural Understanding:** GNNs are designed to process graph-structured data. Representing boolean expressions as Abstract Syntax Trees (ASTs) would allow the agent to "see" and learn from the actual relationships between operators and literals.
    *   **Generalization:** GNNs learn how node types and their connections interact, enabling them to generalize knowledge about local simplification patterns across expressions of varying sizes.
    *   **Surgical Actions:** A GNN-based agent could potentially be designed to apply simplification rules to specific nodes or sub-expressions within the AST, allowing for much more precise and efficient actions.

*   **Trade-offs:** Implementing a GNN would significantly increase the complexity of the project, requiring specialized libraries and more intricate data handling.

## Conclusion

The current training run is demonstrating clear learning, validating the refined reward function and training setup. While the agent is showing competence within its current architectural limitations, a GNN represents the next major leap in capability, directly addressing the challenges posed by the structural nature and vast state space of boolean expressions. We are continuing the current run to establish a strong baseline and observe the full extent of learning possible with the current setup.
