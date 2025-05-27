from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.base.app_manager import lookup_service_brick
from ryu.lib import hub
import time
import operator
import logging
import sys
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./ryu_metrics_log_{timestamp}.log"  

# Create a logger manually
logger = logging.getLogger("ryu.metrics")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)



class UnifiedController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(UnifiedController, self).__init__(*args, **kwargs)
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




