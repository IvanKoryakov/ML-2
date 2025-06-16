from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]

def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.replace('&', '&amp;')
    root = ET.fromstring(content)

    sentence_pairs = []
    alignments = []

    for s in root.findall('s'):
        english = s.find('english')
        czech = s.find('czech')
        sure = s.find('sure')
        possible = s.find('possible')

        english_text = english.text.strip().split() if english is not None else []
        czech_text = czech.text.strip().split() if czech is not None else []

        sure_alignments = []
        possible_alignments = []

        if sure is not None and sure.text:
            sure_alignments = [tuple(map(int, pair.split('-'))) for pair in sure.text.strip().split()]
        if possible is not None and possible.text:
            possible_alignments = [tuple(map(int, pair.split('-'))) for pair in possible.text.strip().split()]

        sentence_pair = SentencePair(source=english_text, target=czech_text)
        alignment = LabeledAlignment(sure=sure_alignments, possible=possible_alignments)

        sentence_pairs.append(sentence_pair)
        alignments.append(alignment)

    return sentence_pairs, alignments

def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_tokens = []
    target_tokens = []

    for pair in sentence_pairs:
        source_tokens.extend(pair.source)
        target_tokens.extend(pair.target)

    source_counted = Counter(source_tokens)
    target_counted = Counter(target_tokens)


    sorted_source_tokens = sorted(source_counted.keys(), key=lambda x: source_counted[x], reverse=True)
    sorted_target_tokens = sorted(target_counted.keys(), key=lambda x: target_counted[x], reverse=True)

    if freq_cutoff is not None:
        sorted_source_tokens = sorted_source_tokens[:freq_cutoff]
        sorted_target_tokens = sorted_target_tokens[:freq_cutoff]

    source_dict = {token: i for i, token in enumerate(sorted_source_tokens)}
    target_dict = {token: i for i, token in enumerate(sorted_target_tokens)}

    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.

    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    for pair in sentence_pairs:
        source_indices = [source_dict[token] for token in pair.source if token in source_dict]
        target_indices = [target_dict[token] for token in pair.target if token in target_dict]

        if len(source_indices) == 0 or len(target_indices) == 0:
            continue

        tokenized_sentence_pair = TokenizedSentencePair(source_tokens=np.array(source_indices), target_tokens=np.array(target_indices))

        tokenized_sentence_pairs.append(tokenized_sentence_pair)
    return tokenized_sentence_pairs

