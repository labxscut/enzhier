# enzhier
## Summary
EnzHier is a novel machine-learning model integrating contrastive learning with hierarchical triplet loss to utilize the functional relationships encoded in EC numbers.

![image](https://github.com/user-attachments/assets/2fc199a2-4c74-426d-851d-6fa83e767506)

EnzHier consists of four modules in order:
1. Sequence Encoding: A protein language model encodes amino acid sequences into embedding vectors.
2. Sample Selection: The process identifies anchor (Anchor), positive (Positive), and negative (Negative) samples for contrastive learning, where positive samples share the same EC number as the anchor, while negative samples differ.
3. Margin Adjustment: Margins are adjusted based on the hierarchical structure of EC numbers, with higher-level EC numbers receiving larger margins for effective functional distinction.
4. Loss Calculation: The loss function, derived from geometric distances between triplets and adjusted margins, aims to minimize the distance between anchor and positive samples while maximizing the distance between anchor and negative samples, with margins dynamically adjusted during training to enhance the model's ability to discern finer functional distinctions.

The network architecture for contrastive learning is inspired by the work presented in [Enzyme function prediction using contrastive learning](https://www.science.org/doi/full/10.1126/science.adf2465), which outlines key methodologies and findings relevant to our approach.
