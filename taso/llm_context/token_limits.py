"""Explicit output caps or omission of the application's cap; no API calls."""
import argparse


def normalize_max_tokens(value):
    """None/provider means omit max_tokens, not unlimited provider generation."""
    if value is None or value == 'provider':
        return None
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise ValueError('max_tokens must be a positive integer <= INT_MAX or null (provider default)')
    return value


def parse_max_tokens(value):
    # Keep a sentinel distinct from argparse's None (= no CLI override).
    if value == 'provider':
        return 'provider'
    try:
        return normalize_max_tokens(int(value))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('use a positive integer or provider') from None


def limit_mode(value):
    return 'provider_default' if normalize_max_tokens(value) is None else 'explicit'
