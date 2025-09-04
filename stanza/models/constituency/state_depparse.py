from collections import namedtuple

class State(namedtuple('State', ['word_queue', 'transitions', 'stack', 'created_arcs', 
                                 'gold_arcs', 'gold_sequence',
                                 'sentence_length', 'word_position', 'score'])):
    
    """
    Represents a partially completed transition parse

    - word_queue: list of all words in the sentence, will not be modified
        ?(The word_queue should have both a start and an end word.)
    - transitions: list of transitions taken to reach this state
    - stack: list of indices
    - created_arcs: set of (head, dependent) tuples representing arcs created so far
    - gold_arcs: set of (head, dependent) tuples representing the gold arcs for this sentence, might be None (None in runtime)
    - gold_sequence: the original transition sequence, might be None (None in runtime)
    - sentence_length: length of the sentence
    - word_position: current position of the buffer in the word queue

    """

    def empty_word_queue(self):
        # the first element of each stack is a sentinel with no value
        # and no parent
        return self.word_position == self.sentence_length

    def empty_transitions(self):
        # the first element of each stack is a sentinel with no value
        # and no parent
        return self.transitions.parent is None