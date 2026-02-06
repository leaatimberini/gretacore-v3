#!/usr/bin/env python3
"""
analyze_b3_60_attention_block.py
Attention Block Input/Output Bisect Analyzer (B3.60)

Detects the FIRST_FAIL point in the attention block pipeline:
embedding_out → attn_block_in → attn_rms_in → attn_norm_out → 
q_pre_rope → q_post_rope → kv_cache_fp → attn_out → wo_out → residual_out
"""

import json
import os
import argparse
import pandas as pd
import numpy as np

# ROOT_CAUSE enums
ROOT_CAUSE = {
    'PASS': 'PASS',
    'EMBED_TO_ATTN': 'ROUTING_EMBED_TO_ATTN',
    'RMS_INPUT': 'RMS_INPUT_SELECTION',
    'RMS_KERNEL': 'RMS_KERNEL',
    'QKV_PROJ': 'QKV_PROJ_INPUT',
    'ROPE_APPLY': 'ROPE_APPLY',
    'KV_CACHE': 'KV_CACHE_COHERENCE',
    'ATTN_CORE': 'ATTN_CORE',
    'RESIDUAL': 'RESIDUAL_ADD',
}

TRACE_POINTS = [
    'embedding_out',
    'attn_block_in',
    'attn_rms_in',
    'attn_norm_out',
    'q_pre_rope',
    'q_post_rope',
    'k_cache_fp_pos0',
    'k_cache_fp_mid',
    'k_cache_fp_last',
    'v_cache_fp_pos0',
    'v_cache_fp_mid',
    'v_cache_fp_last',
    'attn_out',
    'wo_out',
    'residual_out',
]

def load_trace(file_path):
    """Load JSONL trace file."""
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(events)

def get_event_hash(df, point, phase, token_id):
    """Get hash for a specific trace point."""
    filtered = df[(df['point'] == point) & 
                  (df['phase'] == phase) & 
                  (df['token_id'] == token_id)]
    if filtered.empty:
        return None
    return filtered.iloc[0].get('hash', None)

def get_event_nz(df, point, phase, token_id):
    """Get nz_count for a specific trace point."""
    filtered = df[(df['point'] == point) & 
                  (df['phase'] == phase) & 
                  (df['token_id'] == token_id)]
    if filtered.empty:
        return None
    return filtered.iloc[0].get('nz_count', None)

def detect_first_fail(row):
    """
    Detect FIRST_FAIL by checking hash matches in priority order.
    Returns (first_fail_point, root_cause)
    """
    points = TRACE_POINTS
    
    for i, point in enumerate(points):
        p_val = row.get(f'{point}_p_hash')
        d_val = row.get(f'{point}_d_hash')
        
        if p_val is None and d_val is None:
            continue  # Skip if both missing
        if p_val is None or d_val is None:
            # One is missing, this is the first fail
            return point, map_point_to_root_cause(point)
        if p_val != d_val:
            return point, map_point_to_root_cause(point)
    
    return None, 'PASS'

def map_point_to_root_cause(point):
    """Map trace point to root cause enum."""
    mapping = {
        'embedding_out': 'PASS',
        'attn_block_in': 'ROUTING_EMBED_TO_ATTN',
        'attn_rms_in': 'RMS_INPUT_SELECTION',
        'attn_norm_out': 'RMS_KERNEL',
        'q_pre_rope': 'QKV_PROJ_INPUT',
        'q_post_rope': 'ROPE_APPLY',
        'k_cache_fp_pos0': 'KV_CACHE_COHERENCE',
        'k_cache_fp_mid': 'KV_CACHE_COHERENCE',
        'k_cache_fp_last': 'KV_CACHE_COHERENCE',
        'v_cache_fp_pos0': 'KV_CACHE_COHERENCE',
        'v_cache_fp_mid': 'KV_CACHE_COHERENCE',
        'v_cache_fp_last': 'KV_CACHE_COHERENCE',
        'attn_out': 'ATTN_CORE',
        'wo_out': 'ATTN_CORE',
        'residual_out': 'RESIDUAL_ADD',
    }
    return mapping.get(point, 'UNKNOWN')

def analyze_pair(prefill_df, decode_df, token_id):
    """Analyze a single token_id pair."""
    row = {'token_id': int(token_id)}
    
    for point in TRACE_POINTS:
        # Prefill values
        p_hash = get_event_hash(prefill_df, point, 'prefill_last', token_id)
        p_nz = get_event_nz(prefill_df, point, 'prefill_last', token_id)
        
        # Decode values
        d_hash = get_event_hash(decode_df, point, 'decode0', token_id)
        d_nz = get_event_nz(decode_df, point, 'decode0', token_id)
        
        row[f'{point}_p_hash'] = p_hash
        row[f'{point}_d_hash'] = d_hash
        row[f'{point}_p_nz'] = p_nz
        row[f'{point}_d_nz'] = d_nz
        row[f'{point}_match'] = (p_hash == d_hash) if p_hash and d_hash else None
    
    # Detect first fail
    first_fail, root_cause = detect_first_fail(row)
    row['first_fail'] = first_fail
    row['root_cause'] = root_cause
    row['status'] = 'OK' if root_cause == 'PASS' else 'FAIL'
    
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with trace JSONL files")
    parser.add_argument("--output", required=True, help="Output analysis report path")
    args = parser.parse_args()
    
    all_rows = []
    
    for f in os.listdir(args.input_dir):
        if not f.endswith(".jsonl"):
            continue
            
        file_path = os.path.join(args.input_dir, f)
        print(f"Processing: {f}")
        
        df = load_trace(file_path)
        if df.empty:
            print(f"  Warning: Empty file {f}")
            continue
        
        # Separate prefill and decode
        prefill = df[df['phase'] == 'prefill_last']
        decode = df[df['phase'] == 'decode0']
        
        if decode.empty:
            print(f"  Warning: No decode events in {f}")
            continue
        
        # Analyze each token (skip token_id=0 and NaN for decode)
        for token_id in decode['token_id'].unique():
            if token_id == 0 or pd.isna(token_id):
                continue
            
            result = analyze_pair(prefill, decode, token_id)
            result['file'] = f
            all_rows.append(result)
    
    if not all_rows:
        print("No data found")
        with open(args.output, 'w') as out:
            out.write("# B3.60 Attention Block Analysis\n\nNo data found.\n")
        return
    
    results_df = pd.DataFrame(all_rows)
    
    # Generate report
    with open(args.output, 'w') as f:
        f.write("# B3.60 Attention Block Bisect Audit Report\n\n")
        
        # Summary
        f.write("## Summary\n")
        f.write(f"- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Tokens Analyzed**: {len(results_df)}\n")
        f.write(f"- **OK**: {len(results_df[results_df['status'] == 'OK'])}\n")
        f.write(f"- **FAIL**: {len(results_df[results_df['status'] == 'FAIL'])}\n\n")
        
        # Root cause breakdown
        f.write("## Root Cause Breakdown\n")
        rc_counts = results_df[results_df['status'] == 'FAIL']['root_cause'].value_counts()
        for rc, count in rc_counts.items():
            f.write(f"- **{rc}**: {count}\n")
        f.write("\n")
        
        # First fail breakdown
        f.write("## First Fail Point Breakdown\n")
        ff_counts = results_df[results_df['status'] == 'FAIL']['first_fail'].value_counts()
        for point, count in ff_counts.items():
            f.write(f"- **{point}**: {count}\n")
        f.write("\n")
        
        # Per-prompt summary
        f.write("## Per-Prompt Summary\n")
        for fname in results_df['file'].unique():
            pdf = results_df[results_df['file'] == fname]
            f.write(f"\n### {fname}\n")
            f.write(f"- Total: {len(pdf)}\n")
            f.write(f"- OK: {len(pdf[pdf['status'] == 'OK'])}\n")
            f.write(f"- FAIL: {len(pdf[pdf['status'] == 'FAIL'])}\n")
        
        # Detailed table
        f.write("\n## Detailed Results\n")
        # Select key columns for display
        display_cols = ['token_id', 'file', 'first_fail', 'root_cause', 'status']
        # Add match columns for key points
        key_points = ['embedding_out', 'attn_block_in', 'residual_out']
        for pt in key_points:
            display_cols.append(f'{pt}_match')
        
        display_df = results_df[display_cols].copy()
        display_df['token_id'] = display_df['token_id'].astype(int)
        f.write(display_df.to_markdown(index=False))
        
        # Recommendations
        f.write("\n## Recommendations for B3.61\n")
        top_rc = rc_counts.index[0] if not rc_counts.empty else 'PASS'
        f.write(f"- **Primary Root Cause**: {top_rc}\n")
        f.write(f"- **Recommended Focus**: ")
        
        if top_rc == 'ROUTING_EMBED_TO_ATTN':
            f.write("Investigate the routing between embedding output and attention block input.\n")
        elif top_rc == 'RMS_INPUT_SELECTION':
            f.write("Check RMSNorm input selection logic in the attention block.\n")
        elif top_rc == 'KV_CACHE_COHERENCE':
            f.write("Audit KV-cache page mapping and stride handling.\n")
        elif top_rc == 'RESIDUAL_ADD':
            f.write("Debug residual addition before FFN normalization.\n")
        else:
            f.write("Continue bisecting from the identified first failure point.\n")
    
    print(f"Report written to: {args.output}")
    print(f"Total tokens analyzed: {len(results_df)}")

if __name__ == "__main__":
    main()
