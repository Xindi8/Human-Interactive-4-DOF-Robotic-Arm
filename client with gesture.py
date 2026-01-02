#!/usr/bin/env python3
#This file is assited with ChatGPT5 11/19/2025
import socket, sys, time
from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedDPS

gear_ratio_A = 56.0/24.0
gear_ratio_B = 3.0
gear_ratio_C = 3.0
gear_ratio_D = 8.0

dir_A = -1.0
dir_B = -1.0
dir_C = -1.0
dir_D = -1.0

zero_offset_A = 0.0
zero_offset_B = 120.0
zero_offset_C = 40.0
zero_offset_D = 0.0

DEFAULT_SPEED_DPS = 100
WAIT_SEC = 0.2
SETTLE_SEC = 0.05

def joint_to_motor_deg(joint_deg, zero_offset, gear_ratio, direction):
    return zero_offset + joint_deg * (direction * gear_ratio)

def motor_to_joint_deg(motor_deg, zero_offset, gear_ratio, direction):
    denom = (direction * gear_ratio)
    if abs(denom) < 1e-9:
        return 0.0
    return (motor_deg - zero_offset) / denom

class Arm4DOF:
    def __init__(self):
        self.mA = MediumMotor(OUTPUT_A)
        self.mB = LargeMotor(OUTPUT_B)
        self.mC = LargeMotor(OUTPUT_C)
        self.mD = MediumMotor(OUTPUT_D)
        for m in (self.mA, self.mB, self.mC, self.mD):
            m.stop_action = "brake"
            m.position = 0

    def _set_stop_action(self, action: str):
        for m in (self.mA, self.mB, self.mC, self.mD):
            m.stop_action = action

    def relax(self):
        self._set_stop_action("coast")
        self.stop()

    def engage(self):
        self._set_stop_action("brake")

    def stop(self):
        for m in (self.mA, self.mB, self.mC, self.mD):
            m.stop()

    def read_joint_angles(self):
        mA = float(self.mA.position)
        mB = float(self.mB.position)
        mC = float(self.mC.position)
        mD = float(self.mD.position)

        qA = motor_to_joint_deg(mA, zero_offset_A, gear_ratio_A, dir_A)
        relB = motor_to_joint_deg(mB, zero_offset_B, gear_ratio_B, dir_B)
        relC = motor_to_joint_deg(mC, zero_offset_C, gear_ratio_C, dir_C)
        qD = motor_to_joint_deg(mD, zero_offset_D, gear_ratio_D, dir_D)

        qB = relB + zero_offset_B
        qC = relC + zero_offset_C
        return (qA, qB, qC, qD)

    def move_to_joint_angles(self, qA, qB, qC, qD, speed_dps=DEFAULT_SPEED_DPS):
        relB = qB - zero_offset_B
        relC = qC - zero_offset_C
        mA = joint_to_motor_deg(qA, zero_offset_A, gear_ratio_A, dir_A)
        mB = joint_to_motor_deg(relB, zero_offset_B, gear_ratio_B, dir_B)
        mC = joint_to_motor_deg(relC, zero_offset_C, gear_ratio_C, dir_C)
        mD = joint_to_motor_deg(qD, zero_offset_D, gear_ratio_D, dir_D)

        spd = SpeedDPS(speed_dps)

        self.mB.on_to_position(spd, mB, block=False)
        self.mC.on_to_position(spd, mC, block=True)
        time.sleep(WAIT_SEC)
        self.mA.on_to_position(spd, mA, block=False)
        self.mD.on_to_position(spd, mD, block=True)
        time.sleep(SETTLE_SEC)

        return self.read_joint_angles()

class Client:
    def __init__(self, host: str, port: int):
        print("Setting up client\\nAddress: {}\\nPort: {}".format(host, port))
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))

    def poll_data(self) -> str:
        data = self.s.recv(256).decode("utf-8")
        return data.strip() if data else ""

    def send(self, payload: str):
        self.s.send(payload.encode("utf-8"))

    def send_done(self, qA=None, qB=None, qC=None, qD=None):
        if None in (qA, qB, qC, qD):
            self.send("DONE")
        else:
            self.send("DONE,{:.3f},{:.3f},{:.3f},{:.3f}".format(qA, qB, qC, qD))

    def send_reset(self): self.send("RESET")
    def close(self):
        try: self.s.close()
        except OSError: pass

def main(host="169.254.56.169", port=9999):
    arm = Arm4DOF()
    cli = Client(host, port)
    try:
        while True:
            cmd = cli.poll_data()
            if not cmd:
                continue
            if cmd == "EXIT":
                break
            try:
                if cmd == "SAFETY_ON":
                    arm.relax()
                    qA,qB,qC,qD = arm.read_joint_angles()
                    cli.send_done(qA,qB,qC,qD); continue
                if cmd == "SAFETY_OFF":
                    arm.engage()
                    qA,qB,qC,qD = arm.read_joint_angles()
                    cli.send_done(qA,qB,qC,qD); continue
                parts = cmd.split(",")
                if len(parts) == 4:
                    qA = float(parts[0]); qB = float(parts[1])
                    qC = float(parts[2]); qD = float(parts[3])
                    qA,qB,qC,qD = arm.move_to_joint_angles(qA,qB,qC,qD)
                    cli.send_done(qA,qB,qC,qD); continue
                print("Unknown command:", cmd)
                cli.send_reset()
            except Exception as exc:
                print("Error executing '{}': {}".format(cmd, exc))
                cli.send_reset()
    finally:
        arm.relax()
        cli.close()

if __name__ == "__main__":
    try:
        if len(sys.argv)>=3: main(sys.argv[1], int(sys.argv[2]))
        else: main()
    except KeyboardInterrupt:
        print("Client interrupted.")
