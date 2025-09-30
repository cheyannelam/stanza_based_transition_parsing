from collections import namedtuple

class State(namedtuple('State', ['word_queue', 'transitions', 'constituents', 'stacks', 'created_arcs', 
                                 'gold_tree', 'gold_sequence',
                                 'sentence_length', 'word_position', 'score'])):
    
    """
    Represents a partially completed transition parse

    - word_queue: list of all words in the sentence, will not be modified
        ?(The word_queue should have both a start and an end word.)
    - transitions: list of transitions taken to reach this state
    - constituents: list of word forms in the stack (naming is for compatibility with constituency parser)
    - stack: list of indices in the stack
    - created_arcs: set of (head, dependent) tuples representing arcs created so far
    - gold_tree: set of (head, dependent) tuples representing the gold arcs for this sentence, might be None (None in runtime)
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

    def has_one_constituent(self):
        # a length of 1 represents no constituents
        return self.constituents.length == 2

    @property
    def empty_constituents(self):
        return self.constituents.length == 1

    def num_constituents(self):
        return self.constituents.length - 1
    
    @property
    def empty_stacks(self):
        return self.stack.length == 0

    @property
    def num_transitions(self):
        # -1 for the sentinel value
        return self.transitions.length - 1
    
    @property
    def num_stacks(self):
        return self.stacks.length - 1

    @property
    def get_word(self, pos):
        # +1 to handle the initial sentinel value
        # (which you can actually get with pos=-1)
        return self.word_queue[pos+1]

    def finished(self, model):
        return self.empty_word_queue() and self.empty_stacks


    def all_transitions(self, model):
        # TODO: rewrite this to be nicer / faster?  or just refactor?
        all_transitions = []
        transitions = self.transitions
        while transitions.parent is not None:
            all_transitions.append(model.get_top_transition(transitions))
            transitions = transitions.parent
        return list(reversed(all_transitions))
    
    def all_constituents(self, model):
        # TODO: rewrite this to be nicer / faster?
        all_constituents = []
        constituents = self.constituents
        while constituents.parent is not None:
            all_constituents.append(model.get_top_constituent(constituents))
            constituents = constituents.parent
        return list(reversed(all_constituents))
    
    def all_stacks(self, model):
        all_stacks = []
        stacks = self.stacks
        words = self.word_queue
        for id in stacks:
            all_stacks.append(words[id])
        return all_stacks
    
    def all_buffers(self, model):
        return self.word_queue[self.word_position:]
    
    def all_created_arcs(self, model):
        created_arcs = self.created_arcs
        all_created_arcs = []
        for (h, d) in created_arcs:
            all_created_arcs.append((self.word_queue[h], self.word_queue[d]))
        return all_created_arcs

    def all_words(self, model):
        return [model.get_word(x) for x in self.word_queue]

    def to_string(self, model):
        return "State(\n  stacks:%s\n  buffers:%s\  transitions:%s\n)" % (str(self.all_words(model)), str(self.all_transitions(model)), str(self.all_constituents(model)), self.word_position, self.num_opens)

    def __str__(self):
        """'word_queue', 'transitions', 'stacks', 'created_arcs', 
                                 'gold_arcs', 'gold_sequence',
                                 'sentence_length', 'word_position', 'score'"""
        
        print("------------------------------")
        print(f"word_queue: {self.word_queue}")
        print(f"transitions: {self.transitions}")
        print(f"stacks: {self.stacks}")
        print(f"created_arcs: {self.created_arcs}")
        print(f"gold_arcs: {self.gold_arcs}")
        print(f"gold_sequence: {self.gold_sequence}")
        print(f"sentence_length: {self.sentence_length}")
        print(f"word_position: {self.word_position}")
        print(f"score: {self.score}")

        return None