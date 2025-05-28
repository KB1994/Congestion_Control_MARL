#!/usr/bin/env python3
import sys
import re
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime

# === Configuration: set your fixed parameters here ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#INPUT_PATH = "ryu_metrics_log_20250527_161251.log"  # e.g., "/var/log/app.log"
INPUT_PATH = "ryu_metrics_log_20250528_122318.log"
START_PATTERN = r"^\[Metrics @ \d{2}:\d{2}:\d{2}\]"  # lines like "[Metrics @ 12:29:38]..."
END_PATTERN = START_PATTERN  # same pattern for delimiting end of one block and start of next
OUTPUT_PATH = None  # e.g., "extracted.txt" or None to print to stdout
OUTPUT_CSV = f'Result_stats_{timestamp}.csv'
OUTPUT_CSV_CLEAR = f'Result_stats_Clear_OffLine_{timestamp}.csv'
#OUTPUT_CSV = f'Result_stats_Online_{timestamp}.csv'
#OUTPUT_CSV_CLEAR = f'Result_stats_Clear_Online_{timestamp}.csv'
# ====================================================

def extract_packetsin_stats(block):
    FLOW_PATTERN = re.compile(r"^\[Metrics @ (?P<time>\d{2}:\d{2}:\d{2})\] Packet-ins: (?P<pkts>\d+), Flow-adds: (?P<flows>\d+), Total Bytes: (?P<bytes>\d+)")
    flow_ = FLOW_PATTERN.search(block)
    dict_flow = {}
    if flow_:
        dict_flow['time'] = flow_.group('time')
        dict_flow['pkts'] = int(flow_.group('pkts'))
        dict_flow['flows'] = int(flow_.group('flows'))
        dict_flow['bytes_'] = int(flow_.group('bytes'))
    return dict_flow

def extract_ports_stats(block):
    PORT_PATTERN = re.compile(r"""
        ^(?P<date>\d{4}-\d{2}-\d{2})\s+
        (?P<time>\d{2}:\d{2}:\d{2},\d{3})\s+
        \[INFO\]\s+\[Port\ Stats\]\s+
        Switch\ (?P<switch_num>\d+)\s*-\s*
        Port\ (?P<port_num>\d+):\s+
        TX=(?P<tx_pkts>\d+)\s+pkts\s*/\s*(?P<tx_bytes>\d+)\s+bytes,\s+
        RX=(?P<rx_pkts>\d+)\s+pkts\s*/\s*(?P<rx_bytes>\d+)\s+bytes,\s+
        ERR=(?P<err>\d+),\s+
        DURATION=(?P<duration>\d+\.\d+)s,\s+
        SPEED=(?P<speed>\d+\.\d+)\s+Mbps,\s+
        FREE_BW=(?P<free_bw>\d+\.\d+)\s+Mbps$
    """, re.VERBOSE)

    rows = []
    for line in block.splitlines():
        m = PORT_PATTERN.match(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append({
            'switch_num': int(d['switch_num']),
            'port_num':   int(d['port_num']),
            'tx_pkts':    int(d['tx_pkts']),
            'tx_bytes':   int(d['tx_bytes']),
            'rx_pkts':    int(d['rx_pkts']),
            'rx_bytes':   int(d['rx_bytes']),
            'error':      int(d['err']),
            'duration':   float(d['duration']),
            'speed':      float(d['speed']),
            'free_bw':    float(d['free_bw']),
        })

    numeric_cols = ['tx_pkts','tx_bytes','rx_pkts','rx_bytes','error','duration','speed','free_bw']
    if rows:
        df = pd.DataFrame(rows)
        agg = df.groupby(['switch_num','port_num'], as_index=False)[numeric_cols].mean()
        agg['switch_num'] = agg['switch_num'].astype(int)
        agg['port_num']   = agg['port_num'].astype(int)
        return agg
    else:
        empty = pd.DataFrame(columns=['switch_num','port_num'] + numeric_cols)
        empty = empty.astype({
            'switch_num': 'Int64',
            'port_num':   'Int64',
            **{c: 'float64' for c in numeric_cols}
        })
        return empty

def extract_delay_stats(block):
    DELAY_PATTERN = re.compile(
        r"^(?P<date>\d{4}-\d{2}-\d{2})\s+"
        r"(?P<time>\d{2}:\d{2}:\d{2},\d{3})\s+"
        r"\[INFO\]\s+Delay\s+for\s+dpid\s+(?P<dpid>\d+):\s+"
        r"(?P<delay>[-+]?\d*\.\d+)\s+ms$"
    )
    cols = ['switch_num','delay']
    rows = []
    for line in block.splitlines():
        m = DELAY_PATTERN.match(line)
        if not m: continue
        rows.append({
            'switch_num': int(m.group('dpid')),
            'delay':      float(m.group('delay'))
        })
    return pd.DataFrame(rows, columns=cols)

def aggregate_flow_stats(block):
    FLOW_RE = re.compile(
        r"^.*DPID=(?P<dpid>\d+),.*PACKETS=(?P<packets>\d+),\s*BYTES=(?P<bytes>\d+)$"
    )
    cols = ['switch_num','flow_count','total_packets','total_bytes']
    rows = []
    for line in block.splitlines():
        m = FLOW_RE.match(line)
        if not m: continue
        rows.append({
            'switch_num':     int(m.group('dpid')),
            'flow_count':     1,
            'total_packets':  int(m.group('packets')),
            'total_bytes':    int(m.group('bytes'))
        })

    if rows:
        df = pd.DataFrame(rows)
        return df.groupby('switch_num').agg(
            flow_count=('flow_count','sum'),
            total_packets=('total_packets','sum'),
            total_bytes=('total_bytes','sum')
        ).reset_index()[cols]
    else:
        return pd.DataFrame(columns=cols)

def aggregate_pkt_delta(block):
    DELTA_RE = re.compile(
        r"^.*DPID=(?P<dpid>\d+),.*Rate=(?P<rate>[-+]?\d*\.\d+),\s*Loss\s+Est=(?P<loss>[-+]?\d*\.\d+)%$"
    )
    cols = ['switch_num','avg_rate','avg_loss_pct']
    rows = []
    for line in block.splitlines():
        m = DELTA_RE.match(line)
        if not m: continue
        rows.append({
            'switch_num':  int(m.group('dpid')),
            'rate':        float(m.group('rate')),
            'loss_est':    float(m.group('loss'))
        })

    if rows:
        df = pd.DataFrame(rows)
        return df.groupby('switch_num').agg(
            avg_rate=('rate','mean'),
            avg_loss_pct=('loss_est','mean')
        ).reset_index()[cols]
    else:
        return pd.DataFrame(columns=cols)

def safe_df(df, key):
    if df is None:
        return pd.DataFrame({key: pd.Series(dtype='int64')})
    df[key] = pd.to_numeric(df[key], errors='coerce').astype('Int64')
    return df

def extract_blocks(input_path: str, start_pattern: str, end_pattern: str):
    try:
        start_re = re.compile(start_pattern)
        end_re = re.compile(end_pattern)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            capturing = False
            block_lines = []
            for line in f:
                if start_re.search(line):
                    if capturing:
                        yield ''.join(block_lines)
                        block_lines = []
                    capturing = True
                    block_lines.append(line)
                    continue
                if capturing:
                    if end_pattern != start_pattern and end_re.search(line):
                        block_lines.append(line)
                        yield ''.join(block_lines)
                        capturing = False
                        block_lines = []
                    else:
                        block_lines.append(line)
            if capturing and block_lines:
                yield ''.join(block_lines)
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}", file=sys.stderr)
    except re.error as e:
        print(f"Error: Invalid regex pattern: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)

def plot_free_bw(combined, time_col='time', switch_col='switch_num', port_col='port_num', bw_col='free_bw', cols=3):
    combined = combined.copy()
    combined['timestamp'] = pd.to_datetime(combined[time_col], format='mixed')
    pairs = combined[[switch_col, port_col]].drop_duplicates().reset_index(drop=True)
    n_pairs = len(pairs)
    rows = math.ceil(n_pairs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)
    for idx, (sw, pt) in pairs.iterrows():
        ax = axes[idx // cols][idx % cols]
        df_pair = combined[(combined[switch_col] == sw) & (combined[port_col] == pt)]
        df_pair = df_pair.sort_values('timestamp')
        ax.plot(df_pair['timestamp'], df_pair[bw_col], marker='o')
        ax.set_title(f"Switch {sw} – Port {pt}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Free BW (%)")
        ax.grid(True)
    for ax in axes.flatten()[n_pairs:]:
        fig.delaxes(ax)
    plt.tight_layout()
    plt.show()

def main():
    out_stream = open(OUTPUT_PATH, 'w', encoding='utf-8') if OUTPUT_PATH else sys.stdout
    all_dfs = []
    for block in extract_blocks(INPUT_PATH, START_PATTERN, END_PATTERN):
        packet_ins  = extract_packetsin_stats(block)
        ports_stats = safe_df(extract_ports_stats(block), key='port_num')
        delay_stats = safe_df(extract_delay_stats(block), key='switch_num')
        flow_stats  = safe_df(aggregate_flow_stats(block), key='switch_num')
        loss_stats  = safe_df(aggregate_pkt_delta(block), key='switch_num')
        df = ports_stats.merge(delay_stats, on='switch_num', how='outer')
        df = df.merge(flow_stats,  on='switch_num', how='outer')
        df = df.merge(loss_stats,  on='switch_num', how='outer')
        if packet_ins:
            for col in ('time','pkts','flows','bytes_'):
                df[col] = packet_ins.get(col)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    front = ['time', 'switch_num', 'port_num']
    rest  = [c for c in combined.columns if c not in front]
    combined = combined[front + rest]
    print(combined, file=out_stream)
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved aggregated results to {OUTPUT_CSV}", file=sys.stderr)

    df_cleaned = combined[combined['port_num'] != 4294967295]

    # Remove rows with no traffic
    traffic_cols = ['tx_pkts', 'tx_bytes', 'rx_pkts', 'rx_bytes']
    df_cleaned = df_cleaned[~((df_cleaned[traffic_cols] == 0).all(axis=1))]
    df_cleaned.to_csv(OUTPUT_CSV_CLEAR, index=False)

    plot_free_bw(combined)
    if out_stream is not sys.stdout:
        out_stream.close()

if __name__ == '__main__':
    main()
