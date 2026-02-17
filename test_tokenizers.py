#!/usr/bin/env python
"""
Test script to compare AutoTokenizer vs ProkBERTTokenizer for all three ProkBERT models.
"""

import sys

MODELS = [
    "neuralbioinfo/prokbert-mini",
    "neuralbioinfo/prokbert-mini-long",
    "neuralbioinfo/prokbert-mini-c",
]

TEST_SEQUENCE = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

print("=" * 70)
print("TEST 1: AutoTokenizer.from_pretrained() for each model")
print("=" * 70)

from transformers import AutoTokenizer

for model_name in MODELS:
    print(f"\n--- {model_name} ---")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"  Loaded OK: {type(tok).__name__}")
        encoded = tok(TEST_SEQUENCE, return_tensors=None)
        print(f"  input_ids ({len(encoded['input_ids'])} tokens): {encoded['input_ids'][:10]}...")
        print(f"  attention_mask: {encoded['attention_mask'][:10]}...")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

print("\n")
print("=" * 70)
print("TEST 2: ProkBERTTokenizer for each model")
print("=" * 70)

from prokbert.prokbert_tokenizer import ProkBERTTokenizer

PARAMS = {
    "neuralbioinfo/prokbert-mini":      {"kmer": 6, "shift": 1},
    "neuralbioinfo/prokbert-mini-long": {"kmer": 6, "shift": 2},
    "neuralbioinfo/prokbert-mini-c":    {"kmer": 1, "shift": 1},
}

for model_name in MODELS:
    print(f"\n--- {model_name} ---")
    try:
        params = PARAMS[model_name]
        tok = ProkBERTTokenizer(tokenization_params=params, operation_space='sequence')
        print(f"  Loaded OK: {type(tok).__name__} (kmer={params['kmer']}, shift={params['shift']})")
        encoded = tok.encode_plus(TEST_SEQUENCE)
        input_ids = list(encoded['input_ids'])
        attention_mask = list(encoded['attention_mask'])
        print(f"  input_ids ({len(input_ids)} tokens): {input_ids[:10]}...")
        print(f"  attention_mask: {attention_mask[:10]}...")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

print("\n")
print("=" * 70)
print("TEST 3: Compare outputs where both work")
print("=" * 70)

for model_name in MODELS:
    print(f"\n--- {model_name} ---")
    try:
        auto_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        auto_enc = auto_tok(TEST_SEQUENCE, return_tensors=None)
        auto_ids = auto_enc['input_ids']
    except Exception as e:
        print(f"  AutoTokenizer failed: {e}")
        auto_ids = None

    params = PARAMS[model_name]
    prokbert_tok = ProkBERTTokenizer(tokenization_params=params, operation_space='sequence')
    prokbert_enc = prokbert_tok.encode_plus(TEST_SEQUENCE)
    prokbert_ids = list(prokbert_enc['input_ids'])

    if auto_ids is not None:
        match = auto_ids == prokbert_ids
        print(f"  AutoTokenizer ids:    {auto_ids[:10]}... (len={len(auto_ids)})")
        print(f"  ProkBERTTokenizer ids: {prokbert_ids[:10]}... (len={len(prokbert_ids)})")
        print(f"  Match: {match}")
    else:
        print(f"  ProkBERTTokenizer ids: {prokbert_ids[:10]}... (len={len(prokbert_ids)})")
        print(f"  (Cannot compare - AutoTokenizer failed)")
