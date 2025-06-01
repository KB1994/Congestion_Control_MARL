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
from marl_agent.Enhanced_Agent import QLearningClassifier, DDQNAgent

# --- Logger Setup --------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./ryu_metrics_log_Online_DDQN_{timestamp}.log"
logger = logging.getLogger("ryu.metrics")
logger.setLevel(logging.INFO)

file_handler    = logging.FileHandler(log_filename)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Helper: derive_label for Q‐learning discretization ------------------
def derive_label(loss, delay):
    """
    Convert (loss fraction ∈ [0,1], delay in ms) → one of four labels.
    """
    if loss < 0.5 and delay < 70:
        return "Efficient"
    elif loss >= 0.5 and delay < 70:
        return "Loss-Degraded"
    elif loss < 0.5 and delay >= 70:
        return "Delay-Degraded"
    else:
        return "Congested"

# --- UnifiedController Definition ---------------------------------------
class UnifiedController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(UnifiedController, self).__init__(*args, **kwargs)

        # Basic controller state
        self.mac_to_port      = {}    # { dpid_hex: {mac_src: port_no} }
        self.datapaths        = {}    # { dpid_int: datapath_obj }
        self.packet_in_count  = 0
        self.flow_add_count   = 0
        self.total_bytes      = 0
        self.last_log_time    = time.time()

        # Delay measurement via OFPEcho
        self.echo_latency = {}  # { dpid_int: last_latency_ms }
        self.delay_thread  = hub.spawn(self._delay_detector)

        # Port stats: { dpid: { port_no: {rx_pkts, tx_pkts, rx_dropped, tx_dropped} } }
        self.port_stats       = {}
        self.delta_port_stats = {}
        self.port_thread      = hub.spawn(self._monitor_ports)

        # Flow stats: { dpid: { (in_port, out_port, src_mac, dst_mac): (pkt_count, byte_count, sec, nsec) } }
        self.flow_stats       = {}
        self.delta_flow_stats = {}
        self.flow_thread      = hub.spawn(self._monitor_flow_stats)
        self.loss_thread      = hub.spawn(self._monitor_packet_loss)

        logger.info("Ryu controller started. Logging initialized.")

        # --- RL Initialization (load offline‐trained models) ----------------
        # 1) Q‐Learning classifier
        self.qagent = QLearningClassifier(fp_penalty=1.0, lr=0.1, gamma=0.9)
        with open("qtable.json", "r") as f:
            # q_table was saved as nested JSON; we need to transform it back into QLearningClassifier.q_table
            nested = json.load(f)
            # QLearningClassifier expects a flat dict {(loss_bin, delay_bin, action): Q_value}
            # We assume QLearningClassifier has a helper to load from nested JSON if you wrote one. 
            # If not, you must invert this nested dict. For simplicity, assume you stored it directly
            # as QLearningClassifier.q_table = nested in training code. 
            self.qagent.q_table = nested

        # 2) DDQN agent
        #    Input dimension = 5 raw metrics  +  (one‐hot length = number of Q classifier classes)
        num_labels = len(self.qagent.classes)
        input_dim  = 5 + num_labels
        self.ddqn_agent = DDQNAgent(input_dim=input_dim)

        npz_path = "/home/kboussaoud/mawi_data/New_test/ddqn_model_weights.npz"
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"DDQN weights file not found: {npz_path}")

        # 3) Load via the new numpy method
        self.ddqn_agent.load_weights_npz(npz_path)

       

        # ε‐greedy parameters for online decision
        self.epsilon   = 1.00
        self.eps_end   = 0.01
        self.eps_decay = self.ddqn_agent.eps_decay

        # Start the RL‐based decision loop
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


    # ------------------------------------------------------------------------
    #  RL Decision Loop: uses DDQNAgent to pick an action every 10s per switch
    # ------------------------------------------------------------------------

    def _rl_decision_loop(self):
        INTERVAL = 10  # seconds between RL decisions
        step = 0

        while True:
            for dpid, dp in self.datapaths.items():
                # 1) Gather metrics for this switch
                # a) Estimate packet‐loss (flow‐based). Use average across all flows from this switch
                loss_vals = []
                for hist in self.delta_flow_stats.get(dpid, {}).values():
                    if len(hist) == 2 and hist[-2][0] > 0:
                        prev = hist[-2][0]
                        curr = hist[-1][0]
                        ratio = float(curr - prev) / float(prev)
                        loss_vals.append(max(0.0, min(1.0, 1.0 - ratio)))
                loss = float(np.mean(loss_vals)) if loss_vals else 0.0

                # b) Delay (ms) from last OFPEcho reply
                delay = self.echo_latency.get(dpid, 0.0)

                # c) Throughput = total bytes across all ports in last interval
                stats = list(self.port_stats.get(dpid, {}).values())
                tx_bytes = sum(s['tx_bytes'] for s in stats) if stats else 0
                rx_bytes = sum(s['rx_bytes'] for s in stats) if stats else 0
                throughput = float(tx_bytes + rx_bytes)

                # d) Jitter = |rx_bytes - tx_bytes|
                jitter = float(abs(rx_bytes - tx_bytes))

                # e) Free_bw = average(free_bw across all ports), we compute free_bw = capacity (100Mbps) - measured_speed
                #    But we only store rx_bytes/tx_bytes above, so approximate free_bw by 100 - ( (tx+rx)*8 / 10s / 1e6 )
                #    because _monitor_ports() runs every 10s. If you want more accuracy, store port speeds directly.
                free_bw = 0.0
                for port_no, s in self.port_stats.get(dpid, {}).items():
                    # approximate speed_mbps = (tx_bytes + rx_bytes)*8 / 10s / 1e6
                    spd = float(s['tx_bytes'] + s['rx_bytes']) * 8.0 / 10.0 / 1e6
                    free_bw += max(0.0, (100.0 - spd))
                free_bw = free_bw / float(len(self.port_stats.get(dpid, {}))) if self.port_stats.get(dpid, {}) else 0.0

                util = free_bw / 100.0  # fraction of capacity remaining

                # 2) Q‐learning classifier → one‐hot label
                label = derive_label(loss, delay)
                onehot = [1.0 if label == c else 0.0 for c in self.qagent.classes]

                # 3) Build RL state vector
                state = [loss, delay, throughput, jitter, util] + onehot

                # 4) ε‐greedy action from DDQN
                action = self.ddqn_agent.choose_action(state, epsilon=self.epsilon)
                logger.info(f"[RL Decision] DPID={dpid}  LOSS={loss:.4f}  DELAY={delay:.2f}ms  THR={throughput:.0f}B  JIT={jitter:.0f}B  FREE_BW={free_bw:.2f}Mbps  LABEL={label}  ACTION={action}")

                # 5) Apply the chosen action
                self._apply_action(dp, action)

            # 6) Decay ε after each “round” over all switches
            step += 1
            new_eps = max(self.eps_end, self.epsilon * self.eps_decay)
            self.epsilon = new_eps

            # Sleep until next decision interval
            hub.sleep(INTERVAL)

    # ------------------------------------------------------------------------
    # _apply_action: meter modification or reroute
    # ------------------------------------------------------------------------
    def _apply_action(self, datapath, action):
        ofp    = datapath.ofproto
        parser = datapath.ofproto_parser

        if action == 'hold_bw_keep_path':
            # do nothing
            return

        if action == 'decrease_bw_keep_path':
            # Example: modify meter_id=1 to drop cold at 10 Mbps
            band = parser.OFPMeterBandDrop(rate=10000, burst_size=0)
            req  = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                                     flags=ofp.OFPMF_KBPS,
                                     meter_id=1,
                                     bands=[band])
            datapath.send_msg(req)
            return

        if action == 'increase_bw_keep_path':
            # Raise meter threshold from 10→50 Mbps (no reroute)
            band = parser.OFPMeterBandDrop(rate=50000, burst_size=0)
            req  = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                                     flags=ofp.OFPMF_KBPS,
                                     meter_id=1,
                                     bands=[band])
            datapath.send_msg(req)
            return

        if action == 'decrease_bw_reroute':
            # 1) Decrease meter: drop above 10 Mbps
            band = parser.OFPMeterBandDrop(rate=10000, burst_size=0)
            req  = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                                     flags=ofp.OFPMF_KBPS,
                                     meter_id=1,
                                     bands=[band])
            datapath.send_msg(req)
            # 2) Reroute example: send all flows to port=2
            match = parser.OFPMatch()
            inst  = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                   [parser.OFPActionOutput(2)])]
            fm = parser.OFPFlowMod(datapath, priority=10, match=match, instructions=inst)
            datapath.send_msg(fm)
            return

        if action == 'increase_bw_reroute':
            # 1) Increase meter threshold: 50 Mbps
            band = parser.OFPMeterBandDrop(rate=50000, burst_size=0)
            req  = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                                     flags=ofp.OFPMF_KBPS,
                                     meter_id=1,
                                     bands=[band])
            datapath.send_msg(req)
            # 2) Reroute to port=3
            match = parser.OFPMatch()
            inst  = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                   [parser.OFPActionOutput(3)])]
            fm = parser.OFPFlowMod(datapath, priority=10, match=match, instructions=inst)
            datapath.send_msg(fm)
            return

        if action == 'hold_bw_reroute':
            # Keep current meter, but reroute to port=4
            match = parser.OFPMatch()
            inst  = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,
                                                   [parser.OFPActionOutput(4)])]
            fm = parser.OFPFlowMod(datapath, priority=10, match=match, instructions=inst)
            datapath.send_msg(fm)
            return

        # If you add more actions, handle them here...

if __name__ == '__main__':
    app_manager.run_apps(['__main__.UnifiedController'])
