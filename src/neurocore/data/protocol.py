"""Pure FTDI sample-frame decoding for the verified N256 profile; no device I/O."""
import struct
import numpy as np


def crc16(data):
    """Non-reflected 0x8005, init=0, no final XOR (not Modbus)."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x8005 if crc & 0x8000 else 0)) & 0xffff
    return crc


def decode_frame(frame):
    """Decode exactly 1032 bytes into two copied uint16 arrays after CRC checks."""
    if len(frame) != 1032 or frame[0] != 0x71 or struct.unpack_from('<H', frame, 1)[0] != 256:
        raise ValueError('Expected verified q/N256 complete frame')
    if frame[1027] != 0x0d or frame[-2:] != b'\x00\x00' or crc16(frame[3:-2]) != 0:
        raise ValueError('Invalid CR, reserved bytes or CRC')
    samples = np.frombuffer(frame[3:1027], dtype='<u2').reshape(256, 2)
    return samples[:, 0].copy(), samples[:, 1].copy()
