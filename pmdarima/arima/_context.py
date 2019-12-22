# -*- coding: utf-8 -*-
#
# Author: Krishna Sunkara (kpsunkara)
#
# Re-entrant, reusable context manager to store execution context. Introduced
# in pmdarima 1.5.0 (see #221), redesigned not to use thread locals in #273
# (see #275 for context).

from abc import ABC, abstractmethod
from enum import Enum
import collections

__all__ = ['AbstractContext', 'ContextStore', 'ContextType']


class _CtxSingleton:
    """Singleton class to store context information"""
    store = {}


_ctx = _CtxSingleton()


class ContextType(Enum):
    """Context Type Enumeration

    An enumeration of Context Types known to :class:`ContextStore`
    """
    EMPTY = 0
    STEPWISE = 1


class AbstractContext(ABC):
    """An abstract context manager to store execution context.

    A generic, re-entrant, reusable context manager to store
    execution context. Has helper methods to iterate over the context info
    and provide a string representation of the context info.
    """
    def __init__(self, **kwargs):
        # remove None valued entries,
        # since __getattr__ returns None if an attr is not present
        self.props = {k: v for k, v in kwargs.items() if v is not None} \
            if kwargs else {}

    def __enter__(self):
        ContextStore._add_context(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        ContextStore._remove_context(self)

    def __getattr__(self, item):
        """Lets us access, e.g., ``ctx.max_steps`` even if not in a context"""
        return self.props[item] if item in self.props else None

    def __contains__(self, item):
        return item in self.props

    def __getitem__(self, item):
        return self.props[item] if item in self.props else None

    def __iter__(self):
        return iter(self.props)

    def keys(self):
        return self.props.keys()

    def values(self):
        return self.props.values()

    def items(self):
        return self.props.items()

    def update(self, other):
        parent_props = dict(other)
        parent_props.update(self.props)
        self.props = parent_props

    def __repr__(self):
        return self.props.__repr__()

    @abstractmethod
    def get_type(self):
        """Get the ContextType"""


class _emptyContext(AbstractContext):
    """An empty context for convenience use"""

    def __init__(self):
        super(_emptyContext, self).__init__()

    def get_type(self):
        """Indicates we are not in a context manager"""
        return ContextType.EMPTY


class ContextStore:
    """A class to wrap access to the global context store

    This class hosts static methods to wrap access to and encapsulate the
    singleton content store instance
    """
    @staticmethod
    def get_context(context_type):
        """Returns most recently added instance of given Context Type

        Parameters
        ----------
        context_type : ContextType
            Context type to retrieve from the store

        Returns
        -------
        res : AbstractContext
            An instance of AbstractContext subclass or None
        """
        if not isinstance(context_type, ContextType):
            raise ValueError('context_type must be an instance of ContextType')

        if context_type in _ctx.store and len(_ctx.store[context_type]) > 0:
            return _ctx.store[context_type][-1]

        # If not present
        return None

    @staticmethod
    def get_or_default(context_type, default):
        """Returns most recent instance of given Context Type or default

        Parameters
        ----------
        context_type : ContextType
            Context type to retrieve from the store

        default : AbstractContext
            Value to return in case given context does not exist

        Returns
        -------
        ctx : AbstractContext
            An instance of AbstractContext subclass or default
        """
        ctx = ContextStore.get_context(context_type)
        return ctx if ctx else default

    @staticmethod
    def get_or_empty(context_type):
        """Returns recent instance of given Context Type or an empty context

        Parameters
        ----------
        context_type : ContextType
            Context type to retrieve from the store

        Returns
        -------
        res : AbstractContext
            An instance of AbstractContext subclass
        """
        return ContextStore.get_or_default(context_type, _emptyContext())

    @staticmethod
    def _add_context(ctx):
        """Add given instance of AbstractContext subclass to context store

        This private member is only called by ``AbstractContext.__init__()``

        if the given ctx is nested, merge parent context, to support
        following usage:

        Examples
        --------
        >>> from pmdarima.arima import StepwiseContext, auto_arima
        >>> with StepwiseContext(max_steps=10):
        ...     with StepwiseContext(max_dur=30):
        ...         auto_arima(samp,...)

        This is identical to:
        >>> from contextlib import ExitStack
        ... stack = ExitStack()
        ... outer_ctx = StepwiseContext(max_steps=10)
        ... inner_ctx = StepwiseContext(max_dur=30)
        ... stack.enter_context(outer_ctx)
        ... stack.enter_context(inner_ctx)
        ... with stack:
        ...     auto_arima(samp, ...)


        However, the nested context can override parent context. In the
        example below, the effective context for inner most call to
        ``auto_arima(...)`` is: ``max_steps=15, max_dur=30``. The effective
        context for the second call to ``auto_arima(..)`` is: ``max_steps=10``

        >>> with StepwiseContext(max_steps=10):
        ...     with StepwiseContext(max_steps=15, max_dur=30):
        ...         auto_arima(samp,...)
        ...
        ...     auto_arima(samp,...)
        """
        if not isinstance(ctx, AbstractContext):
            raise ValueError('ctx must be be an instance of AbstractContext')

        # if given Context Type is not present into store, make an entry
        context_type = ctx.get_type()
        if context_type not in _ctx.store:
            _ctx.store[context_type] = collections.deque()

        # if the context is nested, merge with parent's context
        if len(_ctx.store[context_type]) > 0:
            parent = _ctx.store[context_type][-1]
            ctx.update(parent)

        _ctx.store[context_type].append(ctx)

    @staticmethod
    def _remove_context(ctx):
        """Removes the most recently added context of given Context Type

        This private member is only used by ``AbstractContext``
        :param ctx:
        :return: None
        """
        if not isinstance(ctx, AbstractContext):
            raise ValueError('ctx must be be an instance of AbstractContext')

        context_type = ctx.get_type()

        if context_type not in _ctx.store or \
                len(_ctx.store[context_type]) == 0:
            return

        _ctx.store[context_type].pop()
