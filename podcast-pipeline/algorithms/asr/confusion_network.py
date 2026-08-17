from typing import List, Tuple

def build_confusion_network(aligned_sequences: List[List[Tuple]]) -> List[List[str]]:
    """
    Build a confusion network from aligned sequences.
    
    Args:
        aligned_sequences: List of aligned (token, token, ...) tuples
        
    Returns:
        List of candidate tokens for each position
    """
    if not aligned_sequences:
        return []

    max_len = max(len(seq) for seq in aligned_sequences)

    confusion_network = []
    for pos in range(max_len):
        candidates = []
        for seq in aligned_sequences:
            if pos < len(seq):
                token = seq[pos]
                if token is not None and token != "":
                    candidates.append(token)
        confusion_network.append(candidates)

    return confusion_network
