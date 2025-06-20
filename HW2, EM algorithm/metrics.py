from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    intersection = 0
    total_predicted = 0
    for i, target in enumerate(reference):
        P = target.sure + target.possible
        S = target.sure
        A = predicted[i]
        A_P = list(set(A) & set(P))
        intersection += len(A_P)
        total_predicted += len(A)

    return intersection, total_predicted

def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    intersection = 0
    total_predicted = 0
    for i, target in enumerate(reference):
        S = target.sure
        A = predicted[i]
        A_S = list(set(A) & set(S))
        intersection += len(A_S)
        total_predicted += len(S)

    return intersection, total_predicted


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    intersection_precision, total_predicted_precision = compute_precision(reference, predicted)
    intersection_recall, total_predicted_recall = compute_recall(reference, predicted)
    aer = 1 - (intersection_precision + intersection_recall)/(total_predicted_precision + total_predicted_recall)
    return aer
