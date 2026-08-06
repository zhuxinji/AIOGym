"""Numeric operation adapters shared by process-model implementations."""
from __future__ import annotations

import math


def _maxv(a, b):
    return a if a > b else b


class _NumericOps:
    symbolic = False

    @staticmethod
    def sqrt(v):
        return math.sqrt(v)

    @staticmethod
    def exp(v):
        return math.exp(v)

    @staticmethod
    def sin(v):
        return math.sin(v)

    @staticmethod
    def cos(v):
        return math.cos(v)

    @staticmethod
    def tan(v):
        return math.tan(v)

    @staticmethod
    def log(v):
        return math.log(v)

    @staticmethod
    def max(a, b):
        return _maxv(a, b)

    @staticmethod
    def smooth_max(a, b, eps=1e-6):
        return _maxv(a, b)

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def if_else(condition, when_true, when_false):
        return when_true if condition else when_false

    @staticmethod
    def abs(v):
        return abs(v)

    @staticmethod
    def vector(values):
        return list(values)


def _casadi_ops(ca):
    class _CasadiOps:
        symbolic = True

        @staticmethod
        def sqrt(v):
            return ca.sqrt(v)

        @staticmethod
        def exp(v):
            return ca.exp(v)

        @staticmethod
        def sin(v):
            return ca.sin(v)

        @staticmethod
        def cos(v):
            return ca.cos(v)

        @staticmethod
        def tan(v):
            return ca.tan(v)

        @staticmethod
        def log(v):
            return ca.log(v)

        @staticmethod
        def max(a, b):
            return ca.fmax(a, b)

        @staticmethod
        def smooth_max(a, b, eps=1e-6):
            return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps ** 2))

        @staticmethod
        def min(a, b):
            return ca.fmin(a, b)

        @staticmethod
        def if_else(condition, when_true, when_false):
            return ca.if_else(condition, when_true, when_false)

        @staticmethod
        def abs(v):
            return ca.fabs(v)

        @staticmethod
        def vector(values):
            return ca.vertcat(*values)

    return _CasadiOps


_NUMERIC_OPS = _NumericOps()
