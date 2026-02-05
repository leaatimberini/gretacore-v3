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
        
    # Group by point, phase, layer
    # We care about:
    # embed_out @ prefill_last
    # embed_out @ decode0
    # x_in @ decode0 (layer 0)
    
    report = []
    
    # Filter relevant events
    rel = df[df['point'].isin(['embed_out', 'x_in'])]
    
    # Identify prefill_last vs decode0
    prefill = rel[rel['phase'] == 'prefill_last']
    decode = rel[rel['phase'] == 'decode0']
    
    if prefill.empty or decode.empty:
        return "ERROR: Missing prefill or decode events"

    # Pair them by token_id
    # Note: prefill_last might have multiple tokens if S > 1, but we usually look at the LAST one.
    # Our instrumentation sets token_id to tokens[stage_token_index] where index=S-1.
    
    for tid in decode['token_id'].unique():
        if tid == 0: continue
        
        p_ev = prefill[prefill['token_id'] == tid]
        d_ev = decode[decode['token_id'] == tid]
        
        if p_ev.empty or d_ev.empty:
            continue
            
        # Comparison
        p_embed = p_ev[p_ev['point'] == 'embed_out'].iloc[0]
        d_embed = d_ev[d_ev['point'] == 'embed_out'].iloc[0]
        d_xin = d_ev[d_ev['point'] == 'x_in'].iloc[0]
        
        row = {
            'token_id': int(tid),
            'prefill_embed_hash': p_embed['hash'],
            'decode0_embed_hash': d_embed['hash'],
            'decode0_xin_hash': d_xin['hash'],
            'prefill_nz': p_embed.get('nz_count', 0),
            'decode0_nz': d_embed.get('nz_count', 0),
            'xin_nz': d_xin.get('nz_count', 0),
            'route': d_embed.get('route', 'UNKNOWN'),
            'status': 'OK'
        }
        
        if row['prefill_embed_hash'] != row['decode0_embed_hash']:
            row['status'] = 'EMBED_MISMATCH'
        elif row['decode0_embed_hash'] != row['decode0_xin_hash']:
            row['status'] = 'X_IN_MISMATCH'
            
        if row['decode0_nz'] == 0:
            row['status'] += '_ZEROED'
            
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
        f.write(f"- Issues: {len(df[df['status'] != 'OK'])}\n")

if __name__ == "__main__":
    main()
