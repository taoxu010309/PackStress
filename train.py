#!/usr/bin/env python3
"""Portable entry point to the original training implementation."""
from stress_mamba.launch import configure

if __name__ == '__main__':
    configure('train')
    from stress_mamba.training import main
    main()
