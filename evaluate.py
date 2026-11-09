#!/usr/bin/env python3
"""Portable entry point to the original full-volume evaluation implementation."""
from stress_mamba.launch import configure

if __name__ == '__main__':
    configure('evaluate')
    from stress_mamba.evaluation import main
    main()
