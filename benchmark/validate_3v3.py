"""Validate dataset.jsonl against props.yaml (v0.2).
Usage: python validate.py [dataset.jsonl] [props.yaml]
"""
import json, re, sys
from collections import defaultdict
import yaml