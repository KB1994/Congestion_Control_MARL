#!/usr/bin/env python3
import json
import time
import logging
import sys
import numpy as np
from datetime import datetime
import os

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.lib import hub

# RL imports: make sure marl_agent is on PYTHONPATH
from marl_agent.Enhanced_Agent import QLearningClassifier, DQNAgent

# --------------------------------------------------------------------------------
# Logger Setup (identical to your original RL controller)
# --------------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./ryu_metrics_log_DQNonly_{timestamp}.log"
logger = logging.getLogger("ryu.metrics")
logger.setLevel(logging.INFO)

file_handler    = logging.FileHandler(log_filename)
console_handler = logging.StreamHandler(sys.stdout)
formatter       = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --------------------------------------------------------------------------------
# Reward function (must exactly match what you used during training)
# --------------------------------------------------------------------------------
def compute_reward(delay, packet_loss, jitter, utilization, action_name):
    if delay < 70 and packet_loss < 0.01 and jitter < 20 and utilization < 0.8:
        r = 3
    elif delay < 120 and packet_loss < 0.02 and jitter < 30 and utilization < 0.9:
        r = 2
    elif delay < 150 and packet_loss < 0.05 and jitter < 50 and utilization < 0.95:
        r = 1
    else:
        r = -2

    congested = (delay > 120 or packet_loss > 0.02 or jitter > 30 or utilization > 0.9)
    if congested:
        if "reroute" in action_name:
            r += 2
        elif "decrease_bw" in action_name:
            r += 1
        elif action_name == "hold_bw_keep_path":
            r -= 3
    else:
        if action_name == "hold_bw_keep_path":
            r += 2
        elif "reroute" in action_name or "decrease_bw" in action_name:
            r -= 2

    return r

# --------------------------------------------------------------------------------
# RyuApp that uses “DQN-only” (no classifier)
# --------------------------------------------------------------------------------
class UnifiedController_DQNonly(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(UnifiedController_DQNonly, self).__init__(*args, **kwargs)

        # — Basic L2 learning state + counters —
        self.mac_to_port      = {}    # {dpid_hex: {mac_src: port}}
        self.datapaths        = {}    # {dpid_int: datapath_obj}
        self.packet_in_count  = 0
        self.flow_add_count   = 0
        self.total_bytes      = 0
        self.last_log_time    = time.time()

        # — Delay (OFPEcho) —
        self.echo_latency     = {}    # {dpid_int: last_ms}
        self.delay_thread     = hub.spawn(self._delay_detector)

        # — Port stats (to compute free_bw, etc.) —
        self.port_stats       = {}    # {dpid: { port_no: {rx_pkts, tx_pkts, rx_bytes, tx_bytes, rx_dropped, tx_dropped} }}
        self.delta_port_stats = {}    # same structure, keep last two samples
        self.port_thread      = hub.spawn(self._monitor_ports)

        # — Flow stats (for packet-loss estimate) —
        self.flow_stats       = {}    # {dpid: { flow_key: (pkt_count, byte_count) }}
        self.delta_flow_stats = {}
        self.flow_thread      = hub.spawn(self._monitor_flow_stats)

        # — Packet‐loss thread (periodic logging) —
        self.loss_thread      = hub.spawn(self._monitor_packet_loss)

        logger.info("DQN-only Ryu controller started (logging to %s)", log_filename)

        # — DQN Agent initialization (input_dim = 5) —
        input_dim = 5
        self.dqn_agent = DQNAgent(input_dim=input_dim)

        # — Load pretrained .npz weights —
        npz_path = "./dqn_only_weights.npz"  # ← must match what you saved in train_dqn_only.py
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"DQN weights file not found: {npz_path}")

        arr = np.load(npz_path)
        ordered = [arr[k] for k in sorted(arr.files, key=lambda x: int(x.strip("arr_")))]
        self.dqn_agent.model.set_weights(ordered)

        # — ε-greedy parameters for online operation —
        self.epsilon   = 0.05   # you can decay this more if you want
        self.eps_end   = 0.01
        self.eps_decay = self.dqn_agent.eps_decay

        # — Spawn RL decision loop every 10s —
        self.rl_thread = hub.spawn(self._rl_decision_loop)

    # ------------------------------------------------------------------------
    # Standard RYU event handlers (SwitchFeatures, PacketIn, etc.)
    # ------------------------------------------------------------------------

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)
        self.flow_add_count += 1

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        self.packet_in_count += 1
        now = time.time()
        if now - self.last_log_time >= 5:
            logger.info(f"\n[Metrics @ {time.strftime('%H:%M:%S')}] Packet-ins: {self.packet_in_count}, Flow-adds: {self.flow_add_count}, Total Bytes: {self.total_bytes}")
            #self.logger.info(f"\n[Metrics @ {time.strftime('%H:%M:%S')}] Packet-ins: {self.packet_in_count}, Flow-adds: {self.flow_add_count}, Total Bytes: {self.total_bytes}")
            self.packet_in_count = 0
            self.flow_add_count = 0
            self.total_bytes = 0
            self.last_log_time = now

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src

        dpid = format(datapath.id, "d").zfill(16)
        self.mac_to_port.setdefault(dpid, {})

        #self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        self.total_bytes += len(msg.data)

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def _delay_detector(self):
        while True:
            for dpid, dp in self.datapaths.items():
                parser = dp.ofproto_parser
                data_time = "%.12f" % time.time()
                byte_arr = bytearray(data_time.encode())
                echo_req = parser.OFPEchoRequest(dp, data=byte_arr)
                dp.send_msg(echo_req)
            hub.sleep(5)

    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def _echo_reply_handler(self, ev):
        now = time.time()
        try:
            latency = now - eval(ev.msg.data)
            self.echo_latency[ev.msg.datapath.id] = latency * 1000
            logger.info(f"Delay for dpid {ev.msg.datapath.id}: {latency * 1000:.3f} ms")
            #self.logger.info(f"Delay for dpid {ev.msg.datapath.id}: {latency * 1000:.3f} ms")
        except:
            pass

    def _monitor_ports(self):
        while True:
            for dp in self.datapaths.values():
                parser = dp.ofproto_parser
                req = parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY)
                dp.send_msg(req)
            hub.sleep(10)


    def _calculate_port_loss(self, dpid):
        for port_no, hist in self.delta_port_stats.get(dpid, {}).items():
            if len(hist) < 2:
                continue
            prev, curr = hist[-2], hist[-1]

            # compute deltas, handle rollover (assume 32-bit counters)
            def delta(a, b):
                d = curr[b] - prev[b]
                if d < 0:
                    d += (1 << 32)
                return d

            d_rx = delta(prev,  'rx_pkts')
            d_tx = delta(prev,  'tx_pkts')
            d_dr = delta(prev,  'rx_dropped') + delta(prev, 'tx_dropped')
            total = d_rx + d_tx + d_dr

            if total > 0:
                loss = d_dr / total
                logger.info(
                    f"[Port Loss] DPID={dpid} Port={port_no}: "
                    f"delta_RX={d_rx}, delta_TX={d_tx}, delta_DROP={d_dr}, "
                    f"Loss={loss:.2%}"
                )

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):

        dpid = ev.msg.datapath.id
        self.port_stats.setdefault(dpid, {})
        self.delta_port_stats.setdefault(dpid, {})
        for stat in ev.msg.body:
            tx_bytes = stat.tx_bytes
            rx_bytes = stat.rx_bytes
            tx_packets = stat.tx_packets
            rx_packets = stat.rx_packets
            rx_errors = stat.rx_errors
            duration = stat.duration_sec + stat.duration_nsec / 1e9
            port_no = stat.port_no
            speed = (tx_bytes + rx_bytes) * 8 / duration / 1e6 if duration > 0 else 0
            capacity = 100  # Assume 100 Mbps if not available
            free_bw = max(capacity - speed, 0)

            logger.info(f"[Port Stats] Switch {ev.msg.datapath.id} - Port {port_no}: TX={tx_packets} pkts / {tx_bytes} bytes, RX={rx_packets} pkts / {rx_bytes} bytes, ERR={rx_errors}, DURATION={duration:.2f}s, SPEED={speed:.2f} Mbps, FREE_BW={free_bw:.2f} Mbps")
            #self.logger.info(f"[Port Stats] Switch {ev.msg.datapath.id} - Port {port_no}: TX={tx_packets} pkts / {tx_bytes} bytes, RX={rx_packets} pkts / {rx_bytes} bytes, ERR={rx_errors}, DURATION={duration:.2f}s, SPEED={speed:.2f} Mbps, FREE_BW={free_bw:.2f} Mbps")

            key = stat.port_no
            cur = {
                'rx_pkts':   stat.rx_packets,
                'tx_pkts':   stat.tx_packets,
                'rx_dropped': stat.rx_dropped,
                'tx_dropped': stat.tx_dropped
            }
            # store latest
            self.port_stats[dpid][key] = cur

            # keep history
            hist = self.delta_port_stats[dpid].setdefault(key, [])
            hist.append(cur)
            if len(hist) > 2:
                hist.pop(0)

        # after updating all ports, run loss calc
        self._calculate_port_loss(dpid)
            

    def _monitor_flow_stats(self):
        while True:
            for dp in self.datapaths.values():
                parser = dp.ofproto_parser
                req = parser.OFPFlowStatsRequest(dp)
                dp.send_msg(req)
            hub.sleep(10)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        self.flow_stats.setdefault(dpid, {})
        self.delta_flow_stats.setdefault(dpid, {})

        for stat in ev.msg.body:
            if stat.priority == 1:
                key = (stat.match.get('in_port'), stat.instructions[0].actions[0].port,
                    stat.match.get('eth_src'), stat.match.get('eth_dst'))
                value = (stat.packet_count, stat.byte_count, stat.duration_sec, stat.duration_nsec)

                # Store the latest flow stats
                self.flow_stats[dpid][key] = value

                # Maintain a history for delta computation
                self.delta_flow_stats[dpid].setdefault(key, [])
                self.delta_flow_stats[dpid][key].append(value)
                if len(self.delta_flow_stats[dpid][key]) > 5:
                    self.delta_flow_stats[dpid][key].pop(0)

                logger.info(
                    f"[Flow Stats] DPID={dpid}, IN={key[0]}, OUT={key[1]}, SRC={key[2]}, DST={key[3]}, "
                    f"PACKETS={value[0]}, BYTES={value[1]}"
                )

                #self.logger.info(
                #    f"[Flow Stats] DPID={dpid}, IN={key[0]}, OUT={key[1]}, SRC={key[2]}, DST={key[3]}, "
                #    f"PACKETS={value[0]}, BYTES={value[1]}"
                #)



    def _calculate_packet_loss(self):
        for dpid, flows in self.delta_flow_stats.items():
            for key, history in flows.items():
                if len(history) >= 2:
                    prev_pkts = history[-2][0]
                    curr_pkts = history[-1][0]
                    delta = curr_pkts - prev_pkts

                    if prev_pkts > 0:
                        rate = delta / prev_pkts
                        loss_estimate = 1.0 - rate
                        loss_estimate = max(0.0, min(loss_estimate, 1.0))
                        logger.info(f"[Pkt Delta] DPID={dpid}, Flow={key}, Rate={rate:.4f}, Loss Est={loss_estimate:.4%}")
                        #self.logger.info(f"[Pkt Delta] DPID={dpid}, Flow={key}, Rate={rate:.4f}, Loss Est={loss_estimate:.4%}")
                else:
                    logger.debug(f"[Pkt Delta] DPID={dpid}, Flow={key}: prev_pkts=0, need one more sample")

        
    def _monitor_packet_loss(self):
        while True:
            self._calculate_packet_loss()
            logger.info("Calculating packet loss (placeholder)")
            #self.logger.info("Calculating packet loss (placeholder)")
            hub.sleep(15)



    # ————————————————————————————————
    # (7) RL Decision Loop: every 10 s, feed raw 5-dim state into DQN
    # ————————————————————————————————
    def _rl_decision_loop(self):
        INTERVAL = 10  # seconds between decisions
        step = 0

        while True:
            for dpid, dp in self.datapaths.items():
                # a) Estimate packet‐loss across flows
                loss_vals = []
                for hist in self.delta_flow_stats.get(dpid, {}).values():
                    if len(hist) == 2 and hist[-2][0] > 0:
                        prev = hist[-2][0]
                        curr = hist[-1][0]
                        ratio = float(curr - prev) / float(prev)
                        loss_vals.append(max(0.0, min(1.0, 1.0 - ratio)))
                loss = float(np.mean(loss_vals)) if loss_vals else 0.0

                # b) Last measured delay (ms)
                delay = self.echo_latency.get(dpid, 0.0)

                # c) Throughput = total bytes across all ports since last port-stats interval
                stats = list(self.port_stats.get(dpid, {}).values())
                tx_bytes = sum(s["tx_bytes"] for s in stats) if stats else 0
                rx_bytes = sum(s["rx_bytes"] for s in stats) if stats else 0
                throughput = float(tx_bytes + rx_bytes)

                # d) Jitter = |rx_bytes − tx_bytes|
                jitter = float(abs(rx_bytes - tx_bytes))

                # e) Free_bw = average across ports: ∼ 100 − ((tx_bytes+rx_bytes)/10s→Mbps)
                free_bw = 0.0
                for s in self.port_stats.get(dpid, {}).values():
                    used_mbps = float(s["tx_bytes"] + s["rx_bytes"]) * 8.0 / 10.0 / 1e6
                    free_bw += max(0.0, 100.0 - used_mbps)
                free_bw = free_bw / float(len(self.port_stats.get(dpid, {}))) if self.port_stats.get(dpid, {}) else 0.0
                util = free_bw / 100.0  # fraction

                # — Build state vector of length=5 (no classification one-hot) —
                state = np.array([loss, delay, throughput, jitter, util], dtype=np.float32)

                # — ε-greedy action (ε decays each “round”) —
                action = self.dqn_agent.choose_action(state, epsilon=self.epsilon)
                logger.info(
                    "[RL Decision] DPID=%d LOSS=%.4f DELAY=%.2fms THR=%.0fB JIT=%.0fB FREE_BW=%.2fMbps → ACTION=%s",
                    dpid, loss, delay, throughput, jitter, free_bw, action
                )

                # — Apply chosen action —
                self._apply_action(dp, action)

            # Decay ε after each sweep over all switches
            step += 1
            new_eps = max(self.eps_end, self.epsilon * self.eps_decay)
            self.epsilon = new_eps

            hub.sleep(INTERVAL)

    # ————————————————————————————————
    # (8) Meter modification or reroute
    # ————————————————————————————————
    def _apply_action(self, datapath, action):
        ofp    = datapath.ofproto
        parser = datapath.ofproto_parser

        if action == "hold_bw_keep_path":
            return

        if action == "decrease_bw_keep_path":
            band = parser.OFPMeterBandDrop(rate=10000, burst_size=0)
            req  = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                                     flags=ofp.OFPMF_KBPS,
                                     meter_id=1,
                                     bands=[band])
            datapath.send_msg(req)
            return

        if action == "increase_bw_keep_path":
            band = parser.OFPMeterBandDrop(rate=50000, burst_size=0)
            req  = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                                     flags=ofp.OFPMF_KBPS,
                                     meter_id=1,
                                     bands=[band])
            datapath.send_msg(req)
            return

        if action == "decrease_bw_reroute":
            # 1) decrease meter
            band = parser.OFPMeterBandDrop(rate=10000, burst_size=0)
            req  = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                                     flags=ofp.OFPMF_KBPS,
                                     meter_id=1,
                                     bands=[band])
            datapath.send_msg(req)
            # 2) reroute all flows to port=2
            match = parser.OFPMatch()
            inst  = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                   [parser.OFPActionOutput(2)])]
            fm = parser.OFPFlowMod(datapath, priority=10, match=match, instructions=inst)
            datapath.send_msg(fm)
            return

        if action == "increase_bw_reroute":
            band = parser.OFPMeterBandDrop(rate=50000, burst_size=0)
            req  = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                                     flags=ofp.OFPMF_KBPS,
                                     meter_id=1,
                                     bands=[band])
            datapath.send_msg(req)
            # reroute to port=3 (example)
            match = parser.OFPMatch()
            inst  = [parser.OFPInstructionActions(ofp.OFPMFIT_APPLY_ACTIONS,
                                                   [parser.OFPActionOutput(3)])]
            fm = parser.OFPFlowMod(datapath, priority=10, match=match, instructions=inst)
            datapath.send_msg(fm)
            return

        if action == "hold_bw_reroute":
            # keep same meter, but reroute to port=4
            match = parser.OFPMatch()
            inst  = [parser.OFPInstructionActions(ofp.OFPMIT_APPLY_ACTIONS,
                                                   [parser.OFPActionOutput(4)])]
            fm = parser.OFPFlowMod(datapath, priority=10, match=match, instructions=inst)
            datapath.send_msg(fm)
            return

        # you can add more composite actions here…

if __name__ == "__main__":
    app_manager.run_apps(["__main__.UnifiedController_DQNonly"])
