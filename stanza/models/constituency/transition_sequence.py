"""
Build a transition sequence from parse trees.

Supports multiple transition schemes - TOP_DOWN and variants, IN_ORDER
"""

import logging

from stanza.models.common import utils
#from stanza.models.constituency.parse_transitions import Shift, CompoundUnary, OpenConstituent, CloseConstituent, TransitionScheme, Finalize
# from stanza.models.constituency.parse_transitions import Shift, LeftArc, RightArc, TransitionScheme
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.tree_reader import read_trees
from stanza.utils.get_tqdm import get_tqdm
from stanza.utils.conll import CoNLL
from collections import defaultdict

tqdm = get_tqdm()

logger = logging.getLogger('stanza.constituency.trainer')



def Shift(buffer, stack, steps, done): 
    steps.append([buffer[:], stack[:], "SHIFT"])
    stack.append(buffer.pop(0))

def LeftArc(buffer, stack, steps, done):
    steps.append([buffer[:], stack[:], "LEFTARC"])
    done.add(stack.pop(-2))
    
def RightArc(buffer, stack, steps, done):
    steps.append([buffer[:], stack[:], "RIGHTARC"])
    done.add(stack.pop(-1))

def is_done(dependent, dependents, done):
    return all(d in done for d in dependents[dependent])

def UD_to_oracle(sentence):
    """
    Convert a projective UD tree to arc-standard oracle steps using the Shift/LeftArc/RightArc functions.

    Args:
        sentence (List[Dict]): Each token has 'id' and 'head' fields.

    Returns:
        List[List]: Each step as [buffer, stack, action]
    """
    # Add ROOT node
    root = {'id': 0, 'form': 'ROOT', 'head': None}
    tokens = [root] + [t for _, t in sentence.items() if '-' not in t['id'] and '.' not in t['id']]
    tokens = [{**t, 'id': int(t['id']), 'head': int(t['head']) if t['head'] is not None else -1} for t in tokens]

    heads = {tok['id']: tok['head'] for tok in tokens if tok['id'] != 0}
    dependents = defaultdict(list)
    for tok in tokens:
        if tok['head'] not in (None, -1):
            dependents[tok['head']].append(tok['id'])

    buffer = [tok['id'] for tok in tokens if tok['id'] != 0]  # exclude ROOT from buffer
    stack = [0]  # start with ROOT on stack
    done = set()
    steps = []

    def can_LeftArc():
        if len(stack) < 2:
            return False
        s1, s0 = stack[-2], stack[-1]
        return heads.get(s1) == s0 and is_done(s1, dependents, done)

    def can_RightArc():
        if len(stack) < 2:
            return False
        s1, s0 = stack[-2], stack[-1]
        return heads.get(s0) == s1 and is_done(s0, dependents, done)

    while buffer or len(stack) > 1:
        if can_LeftArc():
            LeftArc(buffer, stack, steps, done)
        elif can_RightArc():
            RightArc(buffer, stack, steps, done)
        elif buffer:
            Shift(buffer, stack, steps, done)
        else:
            raise ValueError("Non-projective tree or stuck parser.")

    return steps

def oracle_to_UD(tokens, actions):
    """
    Reconstruct UD heads from a list of arc-standard parser actions.

    Args:
        tokens (List[Dict]): List of token dicts with at least 'id' and 'form'.
                             Should NOT include ROOT; we add ROOT internally.
        actions (List[str]): List of parser actions: 'SHIFT', 'LEFTARC', 'RIGHTARC'

    Returns:
        List[Dict]: Tokens with reconstructed 'head' fields.
    """

    # Add ROOT
    root = {'id': 0, 'form': 'ROOT'}
    tokens = [root] + [dict(t) for t in tokens]  # deep copy to avoid mutation
    for tok in tokens:
        tok['id'] = int(tok['id'])

    buffer = [tok['id'] for tok in tokens if tok['id'] != 0]
    stack = [0]  # ROOT
    arcs = []

    for action in actions:
        if action == "SHIFT":
            stack.append(buffer.pop(0))
        elif action == "LEFTARC":
            head = stack[-1]
            dep = stack[-2]
            arcs.append((head, dep))  # head ← dep
            stack.pop(-2)
        elif action == "RIGHTARC":
            head = stack[-2]
            dep = stack[-1]
            arcs.append((head, dep))  # head ← dep
            stack.pop()
        else:
            raise ValueError(f"Invalid action: {action}")

    # Assign heads
    heads = {dep: head for head, dep in arcs}
    for tok in tokens:
        if tok['id'] == 0:
            continue
        tok['head'] = str(heads.get(tok['id'], 0))  # default to ROOT if not found
        tok['id'] = str(tok['id'])  # convert back to string for consistency

    return tokens[1:]  # exclude the added ROOT token




def build_sequence(tree, transition_scheme=TransitionScheme.IN_ORDER):
    """
    Turn a single tree into a list of transitions based on the TransitionScheme
    """
    return UD_to_oracle(tree)

def build_treebank(trees, transition_scheme=TransitionScheme.IN_ORDER, reverse=False):
    """
    Turn each of the trees in the treebank into a list of transitions based on the TransitionScheme
    """
    if reverse:
        return [build_sequence(tree.reverse(), transition_scheme) for tree in trees]
    else:
        return [build_sequence(tree, transition_scheme) for tree in trees]

def all_transitions(transition_lists):
    """
    Given a list of transition lists, combine them all into a list of unique transitions.
    """
    transitions = set()
    for trans_list in transition_lists:
        transitions.update(trans_list)
    return sorted(transitions)

def convert_trees_to_sequences(trees, treebank_name, transition_scheme, reverse=False):
    """
    Wrap both build_treebank and all_transitions, possibly with a tqdm

    Converts trees to a list of sequences, then returns the list of known transitions
    """
    if len(trees) == 0:
        return [], []
    logger.info("Building %s transition sequences", treebank_name)
    if logger.getEffectiveLevel() <= logging.INFO:
        trees = tqdm(trees)
    sequences = build_treebank(trees, transition_scheme, reverse)
    transitions = all_transitions(sequences)
    return sequences, transitions

def main():
    """
    Convert a sample tree and print its transitions
    """
    text = """
    # sent_id = test-s1
    # text = 然而，這樣的處理也衍生了一些問題。
    1	然而	然而	SCONJ	RB	_	7	mark	_	SpaceAfter=No|Translit=rán'ér|LTranslit=rán'ér
    2	，	，	PUNCT	,	_	1	punct	_	SpaceAfter=No|Translit=,|LTranslit=,
    3	這樣	這樣	PRON	PRD	_	5	det	_	SpaceAfter=No|Translit=zhèyàng|LTranslit=zhèyàng
    4	的	的	PART	DEC	Case=Gen	3	case	_	SpaceAfter=No|Translit=de|LTranslit=de
    5	處理	處理	NOUN	NN	_	7	nsubj	_	SpaceAfter=No|Translit=chùlǐ|LTranslit=chùlǐ
    6	也	也	SCONJ	RB	_	7	mark	_	SpaceAfter=No|Translit=yě|LTranslit=yě
    7	衍生	衍生	VERB	VV	_	0	root	_	SpaceAfter=No|Translit=yǎnshēng|LTranslit=yǎnshēng
    8	了	了	AUX	AS	Aspect=Perf	7	aux	_	SpaceAfter=No|Translit=le|LTranslit=le
    9	一些	一些	ADJ	JJ	_	10	amod	_	SpaceAfter=No|Translit=yīxiē|LTranslit=yīxiē
    10	問題	問題	NOUN	NN	_	7	obj	_	SpaceAfter=No|Translit=wèntí|LTranslit=wèntí
    11	。	。	PUNCT	.	_	7	punct	_	SpaceAfter=No|Translit=.|LTranslit=.

    """

    tree = CoNLL.conll2dict(input_str=text)

    print(tree)
    transitions = build_sequence(tree)
    print(transitions)

if __name__ == '__main__':
    main()
