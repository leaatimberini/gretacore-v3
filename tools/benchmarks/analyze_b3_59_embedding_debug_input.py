import json
import os
import argparse
import pandas as pd
import numpy as np

def analyze_trace(file_path):
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except:
                continue
                
    df = pd.DataFrame(events)
    if df.empty:
        return None
        
    report = []
    
    # Filter relevant events
    rel = df[df['point'].isin(['embd_w_hash', 'embed_out', 'x_in'])]
    
    # Identify prefill_last vs decode0
    prefill = rel[rel['phase'] == 'prefill_last']
    decode = rel[rel['phase'] == 'decode0']
    
    if decode.empty:
        return "ERROR: Missing decode events"

    # Pair them by token_id
    for tid in decode['token_id'].unique():
        if tid == 0: continue
        
        p_ev = prefill[prefill['token_id'] == tid]
        d_ev = decode[decode['token_id'] == tid]
        
        if d_ev.empty: continue
            
        # Comparison logic
        d_embed = d_ev[d_ev['point'] == 'embed_out']
        d_xin = d_ev[d_ev['point'] == 'x_in']
        
        if d_embed.empty or d_xin.empty: continue
            
        d_embed = d_embed.iloc[0]
        d_xin = d_xin.iloc[0]
        
        # Prefill comparison (if available)
        p_embed = p_ev[p_ev['point'] == 'embed_out']
        p_emb_hash = p_embed.iloc[0]['hash'] if not p_embed.empty else "N/A"
        p_emb_nz = p_embed.iloc[0].get('nz_count', 0) if not p_embed.empty else 0
        
        row = {
            'token_id': int(tid),
            'p_emb_hash': p_emb_hash,
            'd_emb_hash': d_embed['hash'],
            'd_xin_hash': d_xin['hash'],
            'p_emb_nz': p_emb_nz,
            'd_emb_nz': d_embed.get('nz_count', 0),
            'd_xin_nz': d_xin.get('nz_count', 0),
            'd_xin_abs_sum': d_xin.get('abs_sum', 0.0),
            'route': d_embed.get('route', 'UNKNOWN'),
            'status': 'OK'
        }
        
        if p_emb_hash != "N/A" and p_emb_hash != row['d_emb_hash']:
            row['status'] = 'EMBED_MISMATCH'
        elif row['d_emb_hash'] != row['d_xin_hash']:
            row['status'] = 'XIN_FORWARD_MISMATCH'
            
        if row['d_emb_nz'] == 0:
            row['status'] = 'EMBED_ZEROED'
        elif row['d_xin_nz'] == 0:
            row['status'] = 'XIN_ZEROED'
            
        report.append(row)
        
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    all_rows = []
    for f in os.listdir(args.input_dir):
        if f.endswith(".jsonl"):
            res = analyze_trace(os.path.join(args.input_dir, f))
            if isinstance(res, list):
                for r in res:
                    r['file'] = f
                    all_rows.append(r)
                    
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data found")
        return
        
    with open(args.output, 'w') as f:
        f.write("# B3.59 Embedding + StageDebugInput Audit Report\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary\n")
        f.write(f"- Total Tokens Analyzed: {len(df)}\n")
        f.write(f"- OK: {len(df[df['status'] == 'OK'])}\n")
        f.write(f"- Errors: {len(df[df['status'] != 'OK'])}\n")

if __name__ == "__main__":
    main()
