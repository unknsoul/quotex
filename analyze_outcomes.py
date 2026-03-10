"""Analyze signal outcomes for v11 strategy design."""
import pandas as pd
import numpy as np

df = pd.read_csv('logs/signal_outcomes.csv', header=None,
    names=['timestamp','symbol','predicted','actual','correct','green_prob','confidence','kelly','regime','session','latency'])
print('=== TOTAL OUTCOMES:', len(df))

# Direction distribution
print('\n=== DIRECTION DISTRIBUTION ===')
print(df['predicted'].value_counts())

# Win rate overall
correct = df['correct'].astype(str).str.strip()
wins = (correct == 'True').sum()
losses = (correct == 'False').sum()
total = wins + losses
print(f'\n=== WIN RATE: {wins}/{total} = {wins/total*100:.1f}%')

# Win rate by direction
for d in df['predicted'].unique():
    sub = df[df['predicted'] == d]
    c = sub['correct'].astype(str).str.strip()
    w = (c == 'True').sum()
    t = len(sub)
    print(f'  {d}: {w}/{t} = {w/t*100:.1f}%')

# Win rate by symbol
print('\n=== WIN RATE BY SYMBOL ===')
for sym in df['symbol'].unique():
    sub = df[df['symbol'] == sym]
    c = sub['correct'].astype(str).str.strip()
    w = (c == 'True').sum()
    t = len(sub)
    print(f'  {sym}: {w}/{t} = {w/t*100:.1f}%')

# Win rate by regime
print('\n=== WIN RATE BY REGIME ===')
for reg in df['regime'].dropna().unique():
    sub = df[df['regime'] == reg]
    c = sub['correct'].astype(str).str.strip()
    w = (c == 'True').sum()
    t = len(sub)
    if t > 5:
        print(f'  {reg}: {w}/{t} = {w/t*100:.1f}%')

# Win rate by hour
print('\n=== WIN RATE BY HOUR (UTC) ===')
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
for h in sorted(df['hour'].unique()):
    sub = df[df['hour'] == h]
    c = sub['correct'].astype(str).str.strip()
    w = (c == 'True').sum()
    t = len(sub)
    if t >= 3:
        pct = w/t*100
        marker = " ***" if pct >= 60 else " !!!" if pct < 45 else ""
        print(f'  {h:02d}:00 — {w}/{t} = {pct:.1f}%{marker}')

# Consecutive same direction
print('\n=== LAST 50 SIGNAL DIRECTIONS ===')
last50 = df.tail(50)
vc = last50['predicted'].value_counts().to_dict()
print(f'  Distribution: {vc}')
last50_correct = last50['correct'].astype(str).str.strip()
l50w = (last50_correct == 'True').sum()
print(f'  Win rate (last 50): {l50w}/50 = {l50w/50*100:.1f}%')

# Confidence distribution
df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
print('\n=== CONFIDENCE DISTRIBUTION ===')
print(df['confidence'].describe())

# Are green_prob and confidence identical?
df['green_prob'] = pd.to_numeric(df['green_prob'], errors='coerce')
exact_match = (df['green_prob'].round(2) == df['confidence'].round(2)).sum()
print(f'\ngreen_prob == confidence: {exact_match}/{len(df)} ({exact_match/len(df)*100:.1f}%)')

# Repeated green_prob values
print('\n=== MOST COMMON green_prob VALUES ===')
print(df['green_prob'].round(2).value_counts().head(15))

# Consecutive same candle analysis
print('\n=== CONSECUTIVE SAME-DIRECTION CANDLE STREAKS ===')
df['actual_dir'] = df['actual'].astype(str).str.strip()
streaks = []
current_streak = 1
for i in range(1, len(df)):
    if df['actual_dir'].iloc[i] == df['actual_dir'].iloc[i-1] and df['symbol'].iloc[i] == df['symbol'].iloc[i-1]:
        current_streak += 1
    else:
        if current_streak >= 3:
            streaks.append((df['symbol'].iloc[i-1], df['actual_dir'].iloc[i-1], current_streak, df['timestamp'].iloc[i-1]))
        current_streak = 1
print(f'  Long streaks (3+): {len(streaks)}')
for s in streaks[-10:]:
    print(f'    {s[0]} {s[1]} x{s[2]} at {s[3]}')

# Win rate when predicting in direction of streak vs against
print('\n=== WIN RATE: STREAK CONTEXT ===')
df['prev_actual'] = df['actual'].shift(1)
df['prev_symbol'] = df['symbol'].shift(1)
same_sym = df['symbol'] == df['prev_symbol']
with_streak = df[same_sym & (df['predicted'] == df['prev_actual'])]
against_streak = df[same_sym & (df['predicted'] != df['prev_actual'])]
c1 = with_streak['correct'].astype(str).str.strip()
c2 = against_streak['correct'].astype(str).str.strip()
w1 = (c1 == 'True').sum()
w2 = (c2 == 'True').sum()
t1, t2 = len(with_streak), len(against_streak)
if t1 > 0:
    print(f'  With prev direction:    {w1}/{t1} = {w1/t1*100:.1f}%')
if t2 > 0:
    print(f'  Against prev direction: {w2}/{t2} = {w2/t2*100:.1f}%')

# Kelly histogram
df['kelly'] = pd.to_numeric(df['kelly'], errors='coerce')
print('\n=== KELLY DISTRIBUTION ===')
print(df['kelly'].describe())

# Session analysis
print('\n=== WIN RATE BY SESSION ===')
for sess in df['session'].dropna().unique():
    sub = df[df['session'] == sess]
    c = sub['correct'].astype(str).str.strip()
    w = (c == 'True').sum()
    t = len(sub)
    if t > 3:
        print(f'  {sess}: {w}/{t} = {w/t*100:.1f}%')

# Recent trend: win rate per 50-signal window
print('\n=== WIN RATE TREND (50-signal windows) ===')
for i in range(0, len(df), 50):
    chunk = df.iloc[i:i+50]
    c = chunk['correct'].astype(str).str.strip()
    w = (c == 'True').sum()
    t = len(chunk)
    ts = chunk['timestamp'].iloc[0][:10]
    te = chunk['timestamp'].iloc[-1][:10]
    print(f'  Signals {i}-{i+t}: {w}/{t} = {w/t*100:.1f}%  ({ts} to {te})')
