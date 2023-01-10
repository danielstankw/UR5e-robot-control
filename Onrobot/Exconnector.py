# import urx  # oren wrote this but didnt used it
import time
import numpy as np
import sys
import socket
import struct


class FT_sensor(object):

    def __init__(self, ip_address="192.168.1.1", port=49151):
        self.tcp_ip = ip_address
        self.tcp_port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # UDP
        self.sock.connect((self.tcp_ip, self.tcp_port))

    def force_moment_feedback(self):
        result = bytearray.fromhex('00000000000000000000')
        self.sock.send(result)
        data = self.sock.recv(24)
        _, _, Fx, Fy, Fz, Tx, Ty, Tz = struct.unpack('>hhhhhhhh', data)

        result = bytearray.fromhex('01000000000000000000')
        self.sock.send(result)
        data = self.sock.recv(24)
        _, _, _, CPF, CPT, sfx, sfy, sfz, sTx, sTy, sTz = struct.unpack('>hbbllhhhhhh', data)

        Force_vector = np.array(
            [Fx * sfx / CPF, Fy * sfy / CPF, Fz * sfz / CPF, Tx * sTx / CPT, Ty * sTy / CPT, Tz * sTz / CPT])

        return Force_vector
