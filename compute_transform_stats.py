#!/usr/bin/env python3
"""Fit normalization on the configured training subset only."""
from stress_mamba.launch import configure

if __name__ == '__main__':
    configure('train')
    from stress_mamba.statistics import main
    main()
