"""
Defines a series of transitions for the dependency parser: shift, leftarc, rightarc, (swap?)

Also defines a State which holds the various data needed to build
a parse tree out of tagged words.
"""

from abc import ABC, abstractmethod
import ast
from collections import defaultdict
from enum import Enum
import functools
import logging

from stanza.models.constituency.parse_tree import Tree

logger = logging.getLogger('stanza')

class TransitionScheme(Enum):
    def __new__(cls, value, short_name):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.short_name = short_name
        return obj


    # top down, so the open transition comes before any constituents
    # score on vi_vlsp22 with 5 different sizes of bert layers,
    # bert tagger, no silver dataset:
    #   0.8171
    TOP_DOWN           = 1, "top"
    # unary transitions are modeled as one entire transition
    # version that uses one transform per item,
    # score on experiment described above:
    #   0.8157
    # score using one combination step for an entire transition:
    #   0.8178
    TOP_DOWN_COMPOUND  = 2, "topc"
    # unary is a separate transition.  doesn't help
    # score on experiment described above:
    #   0.8128
    TOP_DOWN_UNARY     = 3, "topu"

    # open transition comes after the first constituent it cares about
    # score on experiment described above:
    #   0.8205
    # note that this is with an oracle, whereas IN_ORDER_COMPOUND does
    # not have a dynamic oracle, so there may be room for improvement
    IN_ORDER           = 4, "in"

    # in order, with unaries after preterminals represented as a single
    # transition after the preterminal
    # and unaries elsewhere tied to the rest of the constituent
    # score: 0.8186
    IN_ORDER_COMPOUND  = 5, "inc"

    # in order, with CompoundUnary on both preterminals and internal nodes
    # score: 0.8166
    IN_ORDER_UNARY     = 6, "inu"

@functools.total_ordering
class Transition(ABC):
    """
    model is passed in as a dependency injection
    for example, an LSTM model can update hidden & output vectors when transitioning
    """
    @abstractmethod
    def update_state(self, state, model):
        """
        update the word queue position, possibly remove old pieces from the constituents state, and return the new constituent

        the return value should be a tuple:
          updated word_position
          updated constituents (without the top item)
          top constituent
          updated stacks (without the top item)
          top stack
          updated created_arcs
        
        the design might allow the lstm to do batch operations

        """

    def delta_opens(self):
        return 0

    def apply(self, state, model):
        """
        return a new State transformed via this transition

        convenience method to call bulk_apply, which is significantly
        faster than single operations for an NN based model
        """
        update = model.bulk_apply([state], [self])
        return update[0]

    @abstractmethod
    def is_legal(self, state, model):
        """
        assess whether or not this transition is legal in this state

        at parse time, the parser might choose a transition which cannot be made
        """

    def components(self):
        """
        Return a list of transitions which could theoretically make up this transition

        For example, an Open transition with multiple labels would
        return a list of Opens with those labels
        """
        return [self]

    @abstractmethod
    def short_name(self):
        """
        A short name to identify this transition
        """

    def short_label(self):
        if not hasattr(self, "label"):
            return self.short_name()

        if isinstance(self.label, str):
            label = self.label
        elif len(self.label) == 1:
            label = self.label[0]
        else:
            label = self.label
        return "{}({})".format(self.short_name(), label)

    def __lt__(self, other):
        # put the Shift at the front of a list, and otherwise sort alphabetically
        if self == other:
            return False
        if isinstance(self, Shift):
            return True
        if isinstance(other, Shift):
            return False
        return str(self) < str(other)


    @staticmethod
    def from_repr(desc):
        """
        This method is to avoid using eval() or otherwise trying to
        deserialize strings in a possibly untrusted manner when
        loading from a checkpoint
        """
        # if desc == 'Shift':
        #     return Shift()
        # if desc == 'CloseConstituent':
        #     return CloseConstituent()
        # labels = desc.split("(", maxsplit=1)
        # if labels[0] not in ('CompoundUnary', 'OpenConstituent', 'Finalize'):
        #     raise ValueError("Unknown Transition %s" % desc)
        # if len(labels) == 1:
        #     raise ValueError("Unexpected Transition repr, %s needs labels" % labels[0])
        # if labels[1][-1] != ')':
        #     raise ValueError("Expected Transition repr for %s: %s(labels)" % (labels[0], labels[0]))
        # trans_type = labels[0]
        # labels = labels[1][:-1]
        # labels = ast.literal_eval(labels)
        # if trans_type == 'CompoundUnary':
        #     return CompoundUnary(*labels)
        # if trans_type == 'OpenConstituent':
        #     return OpenConstituent(*labels)
        # if trans_type == 'Finalize':
        #     return Finalize(*labels)
        # raise ValueError("Unexpected Transition %s" % desc)
        if desc == 'Shift':
            return Shift()
        if desc == 'LeftArc':
            return LeftArc()
        if desc == 'RightArc':
            return RightArc()


class Shift(Transition):
    def update_state(self, state, model):
        """
        This will handle all aspects of a shift transition

        - push the top element of the word queue onto constituents
        - pop the top element of the word queue

        """
        new_constituent = state.word_queue[state.word_position]

        return state.word_position+1, state.constituents, new_constituent, state.stacks, state.word_position, state.created_arcs

    def is_legal(self, state, model):
        """
        Disallow shifting when the word queue is empty or there are no opens to eventually eat this word
        """
        
        if state.is_empty_buffer():
            return False
        print(state)
        return True

    def short_name(self):
        return "Shift"

    def __repr__(self):
        return "Shift"

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Shift):
            return True
        return False

    def __hash__(self):
        return hash(37)
    
class LeftArc(Transition):
    def update_state(self, state, model):
        """
        This will handel all aspects of a left arc transition

        - create a new arc in the stack by using the last two word, the left one being the head
        - pop the last word from the stack

        """
        constituents = state.constituents[2:]
        stacks = state.stacks[2:]
        new_constituent = state.constituents[1]
        new_stacks = state.stacks[1]
        # add new arc
        new_arc = (constituents[1].value, constituents[0].value)

        return state.word_position, constituents, new_constituent, stacks, new_stacks, state.created_arcs.push(new_arc)


    def is_legal(self, state, model):
        """
        Disallow left arc when there are less than two words in the stack
        """
        print(state)
        if state.num_stacks >= 2:
            return True

    def short_name(self):
        return "LeftArc"

    def __repr__(self):
        return "LeftArc"

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, LeftArc):
            return True
        return False

    def __hash__(self):
        return hash(17)
        

class RightArc(Transition):
    def update_state(self, state, model):
        """
        This will handel all aspects of a right arc transition

        - create a new arc in the stack by using the last two word, the right one being the head
        - pop the last word from the stack
        """
        constituents = state.constituents[2:]
        stacks = state.stacks[2:]
        new_constituent = state.constituents[0]
        new_stacks = state.stacks[0]
        # add new arc
        new_arc = (constituents[0].value, constituents[1].value)

        return state.word_position, constituents, new_constituent, stacks, new_stacks, state.created_arcs.push(new_arc)


    def is_legal(self, state, model):
        """
        Disallow left arc when there are less than two words in the stack
        """
        print(state)
        if state.num_stacks >= 2:
            return True

    def short_name(self):
        return "RightArc"

    def __repr__(self):
        return "RightArc"

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, RightArc):
            return True
        return False

    def __hash__(self):
        return hash(71)
    
def check_transitions(train_transitions, other_transitions, treebank_name):
    """
    Check that all the transitions in the other dataset are known in the train set

    Weird nested unaries are warned rather than failed as long as the
    components are all known

    There is a tree in VLSP, for example, with three (!) nested NP nodes
    If this is an unknown compound transition, we won't possibly get it
    right when parsing, but at least we don't need to fail
    """
    unknown_transitions = set()
    for trans in other_transitions:
        if trans not in train_transitions:
            for component in trans.components():
                if component not in train_transitions:
                    raise RuntimeError("Found transition {} in the {} set which don't exist in the train set".format(trans, treebank_name))
            unknown_transitions.add(trans)
    if len(unknown_transitions) > 0:
        logger.warning("Found transitions where the components are all valid transitions, but the complete transition is unknown: %s", sorted(unknown_transitions))