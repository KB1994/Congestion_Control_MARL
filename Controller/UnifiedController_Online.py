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

# RL imports
from marl_agent.Enhanced_Agent import QLearningClassifier, DQNAgent

# --- Logger Setup --------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./ryu_metrics_log_Online_{timestamp}.log"
logger = logging.getLogger("ryu.metrics")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filename)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Helper: derive_label for Q-learning discretization ------------------
def derive_label(loss, delay):
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
        self.mac_to_port = {}
        self.datapaths = {}
        self.packet_in_count = 0
        self.flow_add_count = 0
        self.total_bytes = 0
        self.last_log_time = time.time()

        self.echo_latency = {}
        self.sending_echo_request_interval = 0.05
        self.delay_thread = hub.spawn(self._delay_detector)

        self.port_stats = {}
        self.delta_port_stats = {}
        self.port_features = {}
        self.free_bandwidth = {}
        self.port_thread = hub.spawn(self._monitor_ports)

        self.flow_stats = {}
        self.delta_flow_stats = {}
        self.flow_thread = hub.spawn(self._monitor_flow_stats)
        self.packet_loss_thread = hub.spawn(self._monitor_packet_loss)
        logger.info("Ryu controller started. Logging initialized.")

        # --- RL Initialization (load offline-trained models) ----------------
        # Q-learning classifier
        self.qagent = QLearningClassifier(fp_penalty=1.0, lr=0.1, gamma=0.9)
        with open("qtable.json") as f:
            self.qagent.q_table = json.load(f)
        # DQN agent
        input_dim = 5 + len(self.qagent.classes)
        self.dqn_agent = DQNAgent(input_dim=input_dim)
        #weights_file = "/home/kboussaoud/mawi_data/New_test/dqn_model.weights.h5"
        #if not os.path.exists(weights_file):
        #    raise FileNotFoundError(f"DQN weights not found: {weights_file}")
        #if os.path.getsize(weights_file) == 0:
        #    raise ValueError(f"Weight file is empty: {weights_file}")
        #self.dqn_agent.load_model(weights_file)
        npz_path = "/home/kboussaoud/mawi_data/New_test/dqn_model_weights.npz"
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"DQN weights file not found: {npz_path}")

        # 3) Load via the new numpy method
        self.dqn_agent.load_weights_npz(npz_path)
        # Epsilon for ε-greedy
        self.epsilon = 1.0
        self.eps_end = 0.01
        # Start RL decision loop
        self.rl_thread = hub.spawn(self._rl_decision_loop)

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

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
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

        
    def _monitor_packet_loss(self):
        while True:
            self._calculate_packet_loss()
            logger.info("Calculating packet loss (placeholder)")
            #self.logger.info("Calculating packet loss (placeholder)")
            hub.sleep(15)


    # --- 8) RL Decision Loop --------------------------------------------
    def _rl_decision_loop(self):
        INTERVAL = 10
        while True:
            for dpid, dp in self.datapaths.items():
                loss = self._estimate_loss(dpid)
                delay = self.echo_latency.get(dpid, 0)
                stats = self.port_stats.get(dpid, {}).values()
                tx = sum(s['tx_bytes'] for s in stats)
                rx = sum(s['rx_bytes'] for s in stats)
                throughput = tx + rx
                jitter = abs(rx - tx)
                free_bw = np.mean([s['free_bw'] for s in stats]) if stats else 0
                util = free_bw / 100.0
                label = derive_label(loss, delay)
                onehot = [1 if label==c else 0 for c in self.qagent.classes]
                state = [loss, delay, throughput, jitter, util] + onehot
                action = self.dqn_agent.choose_action(state, epsilon=self.epsilon)
                logger.info(f"DPID {dpid}: action={action}")
                self._apply_action(dp, action)
            self.epsilon = max(self.eps_end, self.epsilon * self.dqn_agent.eps_decay)
            hub.sleep(INTERVAL)

    def _estimate_loss(self, dpid):
        vals = []
        for hist in self.delta_flow_stats.get(dpid, {}).values():
            if len(hist)==2 and hist[0]>0:
                rate=(hist[1]-hist[0])/hist[0]
                vals.append(max(0.0, min(1.0,1-rate)))
        return float(np.mean(vals)) if vals else 0.0

    def _apply_action(self, datapath, action):
        ofp = datapath.ofproto; parser = datapath.ofproto_parser
        if action == 'hold_bw_keep_path':
            return
        elif action == 'decrease_bw_keep_path':
            mod = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                flags=ofp.OFPMF_KBPS, meter_id=1,
                bands=[parser.OFPMeterBandDrop(rate=10000, burst_size=0)])
            datapath.send_msg(mod)
        elif action == 'increase_bw_reroute':
            mod = parser.OFPMeterMod(datapath, ofp.OFPMC_MODIFY,
                flags=ofp.OFPMF_KBPS, meter_id=1,
                bands=[parser.OFPMeterBandDrop(rate=50000, burst_size=0)])
            datapath.send_msg(mod)
            # e.g., reroute flow to port 2
            match = parser.OFPMatch()
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, [parser.OFPActionOutput(2)])]
            fm = parser.OFPFlowMod(datapath, priority=10, match=match, instructions=inst)
            datapath.send_msg(fm)
        # add decrease_bw_reroute, hold_bw_reroute similarly

if __name__ == '__main__':
    app_manager.run_apps(['__main__.UnifiedController'])
