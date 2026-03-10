import pandas as pd
import numpy as np

# Read with error handling for the extra column issue
df = pd.read_csv('logs/signal_outcomes.csv', on_bad_lines='warn')

# If that fails, read manually
try:
    lines = open('logs/signal_outcomes.csv').readlines()
    header = lines[0].strip().split(',')
    print(f"Header columns ({len(header)}): {header}")
    
    # Check which lines have extra columns
    extras = 0
    for i, line in enumerate(lines[1:], 1):
        cols = line.strip().split(',')
        if len(cols) != len(header):
            extras += 1
    print(f"Lines with mismatched columns: {extras}/{len(lines)-1}")
except:
    pass

# Robust read
df = pd.read_csv('logs/signal_outcomes.csv', on_bad_lines='skip')
# Filter out header rows that snuck in
df = df[df['predicted'].isin(['UP', 'DOWN'])]
df['correct'] = df['correct'].map({'True': True, 'False': False, True: True, False: False})
df['green_prob'] = pd.to_numeric(df['green_prob'], errors='coerce')
df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
df['kelly'] = pd.to_numeric(df['kelly'], errors='coerce')

print(f"\n=== TOTAL VALID OUTCOMES: {len(df)}")
print(f"\n=== DIRECTION DISTRIBUTION ===")
print(df['predicted'].value_counts())

total_correct = df['correct'].sum()
total = len(df)
print(f"\n=== OVERALL WIN RATE: {total_correct}/{total} = {100*total_correct/total:.1f}%")

# Win rate by direction
for d in ['UP', 'DOWN']:
    sub = df[df['predicted'] == d]
    w = sub['correct'].sum()
    print(f"  {d}: {w}/{len(sub)} = {100*w/len(sub):.1f}%")

# Win rate by symbol
print(f"\n=== WIN RATE BY SYMBOL ===")
for sym in df['symbol'].unique():
    sub = df[df['symbol'] == sym]
    w = sub['correct'].sum()
    print(f"  {sym}: {w}/{len(sub)} = {100*w/len(sub):.1f}%")

# Win rate by regime
print(f"\n=== WIN RATE BY REGIME ===")
for reg in df['regime'].unique():
    sub = df[df['regime'] == reg]
    w = sub['correct'].sum()
    print(f"  {reg}: {w}/{len(sub)} = {100*w/len(sub):.1f}%")

# Parse timestamps
df['ts'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
df['hour'] = df['ts'].dt.hour

# Win rate by hour
print(f"\n=== WIN RATE BY HOUR (UTC) ===")
for h in sorted(df['hour'].dropna().unique()):
    sub = df[df['hour'] == h]
    w = sub['correct'].sum()
    if len(sub) >= 3:
        print(f"  Hour {int(h):02d}: {w}/{len(sub)} = {100*w/len(sub):.1f}%")

# Session analysis if available
if 'session' in df.columns:
    print(f"\n=== WIN RATE BY SESSION ===")
    for s in df['session'].dropna().unique():
        sub = df[df['session'] == s]
        w = sub['correct'].sum()
        print(f"  {s}: {w}/{len(sub)} = {100*w/len(sub):.1f}%")

# CRITICAL: green_prob vs confidence comparison
print(f"\n=== GREEN_PROB vs CONFIDENCE ===")
same = (df['green_prob'] == df['confidence']).sum()
print(f"  Identical green_prob/confidence: {same}/{len(df)} ({100*same/len(df):.1f}%)")
print(f"  green_prob stats: mean={df['green_prob'].mean():.2f}, std={df['green_prob'].std():.2f}")
print(f"  confidence stats: mean={df['confidence'].mean():.2f}, std={df['confidence'].std():.2f}")
gp_diff = (df['confidence'] - df['green_prob']).abs()
print(f"  Avg |conf - green_prob|: {gp_diff.mean():.4f}")

# Most common green_prob values (detecting stale predictions)
print(f"\n=== MOST COMMON GREEN_PROB VALUES ===")
top_gp = df['green_prob'].round(2).value_counts().head(15)
for val, cnt in top_gp.items():
    print(f"  {val}: {cnt} times")

# CRITICAL: Consecutive same-direction streaks
print(f"\n=== CONSECUTIVE DIRECTION STREAKS ===")
df_sorted = df.sort_values('ts').reset_index(drop=True)
streaks = []
curr_dir = None
curr_len = 0
for _, row in df_sorted.iterrows():
    if row['predicted'] == curr_dir:
        curr_len += 1
    else:
        if curr_len > 0:
            streaks.append((curr_dir, curr_len))
        curr_dir = row['predicted']
        curr_len = 1
if curr_len > 0:
    streaks.append((curr_dir, curr_len))

streak_lens = [s[1] for s in streaks]
print(f"  Total streaks: {len(streaks)}")
print(f"  Avg streak length: {np.mean(streak_lens):.1f}")
print(f"  Max streak length: {max(streak_lens)}")
print(f"  Streaks of 5+: {sum(1 for s in streak_lens if s >= 5)}")
print(f"  Streaks of 10+: {sum(1 for s in streak_lens if s >= 10)}")
print(f"  Streaks of 20+: {sum(1 for s in streak_lens if s >= 20)}")

# Distribution of streak lengths
from collections import Counter
sc = Counter(streak_lens)
print(f"  Streak length distribution:")
for k in sorted(sc.keys()):
    print(f"    Length {k}: {sc[k]} times")

# Win rate during long streaks vs short streaks
print(f"\n=== WIN RATE: LONG STREAKS vs SHORT ===")
df_sorted['streak_id'] = 0
sid = 0
prev = None
for i, row in df_sorted.iterrows():
    if row['predicted'] != prev:
        sid += 1
        prev = row['predicted']
    df_sorted.at[i, 'streak_id'] = sid

for sid_val in df_sorted['streak_id'].unique():
    grp = df_sorted[df_sorted['streak_id'] == sid_val]
    df_sorted.loc[grp.index, 'streak_len'] = len(grp)

short = df_sorted[df_sorted['streak_len'] <= 3]
medium = df_sorted[(df_sorted['streak_len'] > 3) & (df_sorted['streak_len'] <= 10)]
long = df_sorted[df_sorted['streak_len'] > 10]

if len(short) > 0:
    w = short['correct'].sum()
    print(f"  Short streaks (<=3): {w}/{len(short)} = {100*w/len(short):.1f}%")
if len(medium) > 0:
    w = medium['correct'].sum()
    print(f"  Medium streaks (4-10): {w}/{len(medium)} = {100*w/len(medium):.1f}%")
if len(long) > 0:
    w = long['correct'].sum()
    print(f"  Long streaks (>10): {w}/{len(long)} = {100*w/len(long):.1f}%")

# Win rate by position within streak
print(f"\n=== WIN RATE BY POSITION IN STREAK ===")
df_sorted['pos_in_streak'] = 0
for sid_val in df_sorted['streak_id'].unique():
    idx = df_sorted[df_sorted['streak_id'] == sid_val].index
    for pos, i in enumerate(idx, 1):
        df_sorted.at[i, 'pos_in_streak'] = pos

for pos in range(1, 8):
    sub = df_sorted[df_sorted['pos_in_streak'] == pos]
    if len(sub) >= 5:
        w = sub['correct'].sum()
        print(f"  Position {pos}: {w}/{len(sub)} = {100*w/len(sub):.1f}%")

# CRITICAL: Last 50 signals analysis
print(f"\n=== LAST 50 SIGNALS ===")
last50 = df_sorted.tail(50)
w = last50['correct'].sum()
print(f"  Win rate: {w}/{len(last50)} = {100*w/len(last50):.1f}%")
print(f"  Direction: {last50['predicted'].value_counts().to_dict()}")
print(f"  Symbols: {last50['symbol'].value_counts().to_dict()}")
print(f"  green_prob==confidence: {(last50['green_prob']==last50['confidence']).sum()}/50")

# Last 100 signals  
print(f"\n=== LAST 100 SIGNALS ===")
last100 = df_sorted.tail(100)
w = last100['correct'].sum()
print(f"  Win rate: {w}/{len(last100)} = {100*w/len(last100):.1f}%")
print(f"  Direction: {last100['predicted'].value_counts().to_dict()}")

# Win rate evolution over time (50-signal rolling windows)
print(f"\n=== WIN RATE EVOLUTION (50-signal windows) ===")
step = 50
for start in range(0, len(df_sorted) - step + 1, step):
    window = df_sorted.iloc[start:start+step]
    w = window['correct'].sum()
    ts_start = window['ts'].iloc[0]
    ts_end = window['ts'].iloc[-1]
    dirs = window['predicted'].value_counts().to_dict()
    print(f"  [{start}:{start+step}] {100*w/step:.0f}% | {dirs} | {str(ts_start)[:16]} to {str(ts_end)[:16]}")

# CRITICAL: Same green_prob in same timestamp (batch predictions)
print(f"\n=== SAME-TIMESTAMP BATCH ANALYSIS ===")
df_sorted['ts_rounded'] = df_sorted['ts'].dt.floor('1min')
batches = df_sorted.groupby('ts_rounded')
multi_batches = [(ts, grp) for ts, grp in batches if len(grp) > 1]
print(f"  Total multi-signal batches: {len(multi_batches)}")
if multi_batches:
    same_gp_batches = 0
    for ts, grp in multi_batches:
        if grp['green_prob'].nunique() == 1:
            same_gp_batches += 1
    print(f"  Batches with identical green_prob: {same_gp_batches}/{len(multi_batches)}")
    
    # Show a few examples
    print(f"  Example batches:")
    for ts, grp in multi_batches[-5:]:
        print(f"    {str(ts)[:16]}: {list(zip(grp['symbol'], grp['green_prob'].round(2)))}")

# Kelly distribution
print(f"\n=== KELLY DISTRIBUTION ===")
print(f"  Mean: {df['kelly'].mean():.2f}")
print(f"  Median: {df['kelly'].median():.2f}")
print(f"  Std: {df['kelly'].std():.2f}")
print(f"  <0: {(df['kelly'] < 0).sum()}")
print(f"  0-5: {((df['kelly'] >= 0) & (df['kelly'] < 5)).sum()}")
print(f"  5-10: {((df['kelly'] >= 5) & (df['kelly'] < 10)).sum()}")
print(f"  10-20: {((df['kelly'] >= 10) & (df['kelly'] < 20)).sum()}")
print(f"  20+: {(df['kelly'] >= 20).sum()}")

print(f"\n=== ANALYSIS COMPLETE ===")
