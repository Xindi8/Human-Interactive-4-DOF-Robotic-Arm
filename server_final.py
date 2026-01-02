#!/usr/bin/env python3
#This file is assited with ChatGPT5 12/1/2025
import math, socket, sys, time, threading
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

yellowLowMask = (20, 100, 100)
yellowHighMask = (35, 255, 255)
blueLowMask   = (90, 80, 80)
blueHighMask  = (110, 255, 255)
greenLowMask  = (40, 80, 60)
greenHighMask = (85, 255, 255)

left = False

class DualTracker:
    def __init__(self, cam0=1, cam1=2, show=True, width=None, height=None):
        self.cam0 = cam0
        self.cam1 = cam1
        self.show = show
        self.width = width
        self.height = height
        self.tip0 = (0, 0)
        self.goal0 = (0, 0)
        self.tip1 = (0, 0)
        self.goal1 = (0, 0)
        self.ok0 = False
        self.ok1 = False
        self.ok0_tip  = False
        self.ok0_goal = False
        self.ok1_tip  = False
        self.ok1_goal = False
        self.has_goal0_once = False
        self.has_goal1_once = False
        self.stop_flag = False
        self.t0 = threading.Thread(target=self._loop, args=(0,), daemon=True)
        self.t1 = threading.Thread(target=self._loop, args=(1,), daemon=True)
        self.t0.start()
        self.t1.start()

    def stop(self):
        self.stop_flag = True
        try:
            cv2.destroyAllWindows()
        except:
            pass

    def _open_capture(self, cam_index: int):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_index)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _loop(self, which: int):
        cam_index = self.cam0 if which == 0 else self.cam1
        cap = self._open_capture(cam_index)
        if not cap.isOpened():
            return
        win = f"Cam{which}"
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        global left

        while not self.stop_flag:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            blurred = cv2.medianBlur(frame, 11)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            GoalLowmask = blueLowMask
            GoalHighmask = blueHighMask
            if left:
                GoalLowmask = yellowLowMask
                GoalHighmask = yellowHighMask

            mTip  = cv2.inRange(hsv, greenLowMask, greenHighMask)
            mGoal = cv2.inRange(hsv, GoalLowmask,  GoalHighmask)
            mTip  = cv2.erode(mTip,  kernel, iterations=2)
            mTip  = cv2.dilate(mTip, kernel, iterations=5)
            mGoal = cv2.erode(mGoal, kernel, iterations=2)
            mGoal = cv2.dilate(mGoal, kernel, iterations=5)
            gTip  = cv2.cvtColor(cv2.bitwise_and(blurred, blurred, mask=mTip),  cv2.COLOR_BGR2GRAY)
            gGoal = cv2.cvtColor(cv2.bitwise_and(blurred, blurred, mask=mGoal), cv2.COLOR_BGR2GRAY)

            cTip  = cv2.HoughCircles(gTip,  cv2.HOUGH_GRADIENT, 1.5, 300, param1=100, param2=20, minRadius=8, maxRadius=200)
            cGoal = cv2.HoughCircles(gGoal, cv2.HOUGH_GRADIENT, 1.5, 300, param1=100, param2=20, minRadius=8, maxRadius=200)

            tip_found  = False
            goal_found = False

            if cTip is not None:
                x = int(round(float(cTip[0, 0, 0])))
                y = int(round(float(cTip[0, 0, 1])))
                r = int(round(float(cTip[0, 0, 2])))
                if which == 0:
                    self.tip0 = (x, y)
                else:
                    self.tip1 = (x, y)
                tip_found = True
                if self.show:
                    cv2.circle(frame, (x, y), r, (0, 255, 255), 4)

            if cGoal is not None:
                x = int(round(float(cGoal[0, 0, 0])))
                y = int(round(float(cGoal[0, 0, 1])))
                r = int(round(float(cGoal[0, 0, 2])))
                if which == 0:
                    self.goal0 = (x, y)
                    self.has_goal0_once = True
                else:
                    self.goal1 = (x, y)
                    self.has_goal1_once = True
                goal_found = True
                if self.show:
                    cv2.circle(frame, (x, y), r, (255, 0, 0), 4)

            if which == 0:
                self.ok0_tip  = tip_found
                self.ok0_goal = goal_found
                self.ok0      = tip_found or goal_found
            else:
                self.ok1_tip  = tip_found
                self.ok1_goal = goal_found
                self.ok1      = tip_found or goal_found

            if self.show:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_flag = True
                    break
        cap.release()

    def read_pair(self, timeout=4.0, sleep=0.02, require_both=False):
        t0 = time.time()
        while time.time() - t0 < timeout and not self.stop_flag:
            if require_both:
                if (self.ok0_tip and self.ok0_goal and self.ok1_tip and self.ok1_goal):
                    return (self.tip0, self.goal0, self.tip1, self.goal1, True, True, True, True)
            else:
                if self.ok0 or self.ok1:
                    return (self.tip0, self.goal0, self.tip1, self.goal1, self.ok0, self.ok1, self.ok0_tip, self.ok1_tip)
            time.sleep(sleep)
        return None

JOINT_LIMITS = {"A": (-180, 180), "B": (20, 175), "C": (-80, 90), "D": (-90, 90)}
def clamp(x, lo, hi): return max(lo, min(hi, x))

class TCPRobotClient4:
    def __init__(self, host="169.254.56.169", port=9999):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        self.cs, addr = srv.accept()
        self.last_angles = (0, 90, 0, 0)

    def _ask(self, payload):
        self.cs.send(payload.encode("utf-8"))
        return self.cs.recv(256).decode("utf-8").strip()

    def request_angles(self):
        rep = self._ask("SAFETY_OFF")
        ps = rep.split(",")
        if len(ps) >= 5:
            self.last_angles = (float(ps[1]), float(ps[2]), float(ps[3]), float(ps[4]))
        return self.last_angles

    def move_to(self, a, b, c, d):
        a = clamp(a, *JOINT_LIMITS["A"])
        b = clamp(b, *JOINT_LIMITS["B"])
        c = clamp(c, *JOINT_LIMITS["C"])
        d = clamp(d, *JOINT_LIMITS["D"])
        rep = self._ask(f"{a:.3f},{b:.3f},{c:.3f},{d:.3f}")
        ps = rep.split(",")
        if len(ps) >= 5:
            self.last_angles = (float(ps[1]), float(ps[2]), float(ps[3]), float(ps[4]))
        return self.last_angles

    def shutdown(self):
        try:
            self.cs.send(b"EXIT")
        except:
            pass
        try:
            self.cs.close()
        except:
            pass

def matT(A):
    return tuple(tuple(A[j][i] for j in range(len(A))) for i in range(len(A[0])))

def matmul(A, B):
    r = len(A)
    k = len(A[0])
    c = len(B[0])
    out = [[0.0] * c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            out[i][j] = s
    return tuple(tuple(row) for row in out)

def mat_add_lambdaI(A, lam):
    n = len(A)
    return tuple(tuple(A[i][j] + (lam if i == j else 0.0) for j in range(n)) for i in range(n))

def mat_inv(A):
    n = len(A)
    M = [list(row) + [0.0] * n for row in A]
    for i in range(n):
        M[i][n + i] = 1.0
    for col in range(n):
        piv = col
        piv_abs = abs(M[piv][col])
        for r in range(col + 1, n):
            v = abs(M[r][col])
            if v > piv_abs:
                piv = r
                piv_abs = v
        if piv_abs == 0.0:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        inv_pv = 1.0 / pv
        for j in range(2 * n):
            M[col][j] *= inv_pv
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor != 0.0:
                for j in range(2 * n):
                    M[r][j] -= factor * M[col][j]
    inv = [tuple(M[i][n:]) for i in range(n)]
    return tuple(inv)

def dls(J, e, lam):
    JT = matT(J)
    H  = mat_add_lambdaI(matmul(JT, J), lam)
    Hinv = mat_inv(H)
    if Hinv is None:
        return None
    JTe = matmul(JT, tuple((-float(x),) for x in e))
    dth = matmul(Hinv, JTe)
    return tuple(dth[i][0] for i in range(4))

WAIT_SEC    = 0.35
PERTURB_DEG = 10.0
LAMBDA      = 20.0
STEP_CAP    = 2.0
XY_TOL      = 180.0
Z_TOL       = 100.0
FLIP_X1 = False
FLIP_Y1 = True

def measure_pair(trk, timeout=3, require_both=False):
    return trk.read_pair(timeout, require_both=require_both)

def err4_with_flags(t0, g0, t1, g1, ok0_pair, ok1_pair):
    if ok1_pair:
        dx_xy = float(g1[0] - t1[0])
        dy_xy = float(g1[1] - t1[1])
    else:
        dx_xy = 0.0
        dy_xy = 0.0

    if ok0_pair:
        dx_z = float(g0[0] - t0[0])
        dy_z = float(g0[1] - t0[1])
    else:
        dx_z = 0.0
        dy_z = 0.0

    if FLIP_X1:
        dx_z = -dx_z
    if FLIP_Y1:
        dy_z = -dy_z

    return (dx_xy, dy_xy, dx_z, dy_z)

def norm_active(e, ok0_pair, ok1_pair):
    acc = 0.0
    cnt = 0
    if ok1_pair:
        acc += e[0]*e[0] + e[1]*e[1]
        cnt += 2
    if ok0_pair:
        acc += e[2]*e[2] + e[3]*e[3]
        cnt += 2
    if cnt == 0:
        return float("inf")
    return math.sqrt(acc)

def estimate_J(robot, trk):
    base = measure_pair(trk, 8, require_both=False)
    if base is None:
        raise RuntimeError("no vision for J estimation")

    qA, qB, qC, qD = robot.request_angles()
    ddeg = PERTURB_DEG
    drad = math.radians(ddeg)

    def sense():
        m = measure_pair(trk, 3, require_both=False)
        if m is None:
            raise RuntimeError("vision lost during J estimation")
        t0, g0, t1, g1, ok0, ok1, ok0_tip, ok1_tip = m
        ok0_pair = ok0_tip and (trk.ok0_goal or trk.has_goal0_once)
        ok1_pair = ok1_tip and trk.ok1_goal
        e = err4_with_flags(t0, g0, t1, g1, ok0_pair, ok1_pair)
        return e

    cols = []
    for j in range(4):
        q = [qA, qB, qC, qD]
        q[j] += ddeg
        robot.move_to(*q); time.sleep(WAIT_SEC)
        ep = sense()
        q[j] -= 2*ddeg
        robot.move_to(*q); time.sleep(WAIT_SEC)
        em = sense()
        col = [(ep[i] - em[i]) / (2.0 * drad) for i in range(4)]
        cols.append(col)
        robot.move_to(qA, qB, qC, qD); time.sleep(WAIT_SEC)

    J = tuple(tuple(cols[j][i] for j in range(4)) for i in range(4))
    return J, (qA, qB, qC, qD)

def initial_lift(robot, lift_angle=35.0):
    qA, qB, qC, qD = robot.request_angles()
    qB_new = clamp(qB - lift_angle, *JOINT_LIMITS["B"])
    qC_new = clamp(qC - lift_angle, *JOINT_LIMITS["C"])
    robot.move_to(qA, qB_new, qC_new, qD)
    time.sleep(1.5)

def perform_wave_motion(robot, loops=4):
    qA, qB, qC, qD = robot.request_angles()
    B_target = clamp(qB - 90.0, *JOINT_LIMITS["B"])
    C_target = clamp(qC + 10.0, *JOINT_LIMITS["C"])
    robot.move_to(qA, B_target, C_target, qD)
    time.sleep(1.0)
    D_pos1 = clamp(45.0,  *JOINT_LIMITS["D"])
    D_pos2 = clamp(-45.0, *JOINT_LIMITS["D"])

    for i in range(loops):
        robot.move_to(qA, B_target, C_target, D_pos1)
        time.sleep(0.5)
        robot.move_to(qA, B_target, C_target, D_pos2)
        time.sleep(0.5)
    robot.move_to(qA, qB, qC, qD)
    time.sleep(1.0)

def pick_path_after_convergence(robot, qA, qB, qC, qD, initial_move_q):
    global left
    q_start = (qA, qB, qC, qD)
    robot.move_to(*q_start)
    time.sleep(5.0)
    path = [q_start]

    B_up = clamp(qB - 40.0, *JOINT_LIMITS["B"])
    pose1 = (qA, B_up, qC, qD)
    robot.move_to(*pose1)
    time.sleep(1.0)
    path.append(pose1)

    A_SWING = 25.0
    if left:
        dA = -A_SWING
    else:
        dA = +A_SWING
    A_swing = clamp(qA + dA, *JOINT_LIMITS["A"])
    pose2 = (A_swing, B_up, qC, qD)
    robot.move_to(*pose2)
    time.sleep(1.0)
    path.append(pose2)

    C_up = clamp(qC - 30.0, *JOINT_LIMITS["C"])
    pose3 = (A_swing, B_up, C_up, qD)
    robot.move_to(*pose3)
    time.sleep(1.0)
    path.append(pose3)

    B_down = clamp(B_up + 50.0, *JOINT_LIMITS["B"])
    pose4 = (A_swing, B_down, C_up, qD)
    robot.move_to(*pose4)
    time.sleep(1.0)
    path.append(pose4)

    for q in reversed(path[:-1]):
        robot.move_to(*q)
        time.sleep(0.8)

    if initial_move_q is not None:
        robot.move_to(*initial_move_q)
        time.sleep(1.0)

MODEL_PATH = "mp_gesture.keras"
CLASS_NAMES = ["down", "down_right", "down_left", "wave"]

def decide_left_via_gesture(cam_index=0, min_conf=0.8):
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception:
        return False, None

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        return False, None

    decision = None
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            label = "no hand"
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                lm_vec = []
                for lm in hand.landmark:
                    lm_vec.extend([lm.x, lm.y, lm.z])
                x = np.array(lm_vec, dtype="float32").reshape(1, -1)
                preds = model.predict(x, verbose=0)[0]
                idx = int(np.argmax(preds))
                conf = float(preds[idx])
                gesture_name = CLASS_NAMES[idx]
                label = f"{gesture_name} ({conf*100:.1f}%)"
                if gesture_name in ("down_right", "down_left", "wave") and conf >= min_conf:
                    decision = gesture_name
                    cv2.putText(frame, "LOCKED: " + label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("gesture", frame)
                    cv2.waitKey(500)
                    break
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    cap.release()
    try:
        cv2.destroyWindow("gesture")
    except Exception:
        pass
    if decision == "down_right":
        return True, "down_right"
    elif decision == "down_left":
        return False, "down_left"
    elif decision == "wave":
        return False, "wave"
    else:
        return False, None

def run(host="169.254.56.169", port=9999, cam0=1, cam1=2, width=None, height=None, show=True):
    global left
    robot = TCPRobotClient4(host, port)
    trk   = DualTracker(cam0, cam1, show, width=width, height=height)
    time.sleep(1.0)
    J_last = None
    initial_move_q = None
    running = True

    try:
        while running:
            left, gesture_name = decide_left_via_gesture(cam_index=0)
            if gesture_name == "wave":
                if J_last is None or initial_move_q is None:
                    initial_lift(robot, lift_angle=30.0)
                    J_last, initial_move_q = estimate_J(robot, trk)
                if initial_move_q is not None:
                    robot.move_to(*initial_move_q)
                    time.sleep(1.0)
                perform_wave_motion(robot)
                if initial_move_q is not None:
                    robot.move_to(*initial_move_q)
                    time.sleep(1.0)
                continue

            if J_last is None or initial_move_q is None:
                initial_lift(robot, lift_angle=30.0)
                J_last, initial_move_q = estimate_J(robot, trk)

            qA, qB, qC, qD = initial_move_q
            robot.move_to(qA, qB, qC, qD)
            time.sleep(1.0)
            J = J_last
            miss_count = 0
            prev_en = None
            inc_count = 0
            converged = False

            for k in range(1, 901):
                m = measure_pair(trk, 2, require_both=False)
                if m is None:
                    miss_count += 1
                    if miss_count >= 5:
                        running = False
                        break
                    continue
                else:
                    miss_count = 0
                t0, g0, t1, g1, ok0, ok1, ok0_tip, ok1_tip = m
                ok0_pair = ok0_tip and (trk.ok0_goal or trk.has_goal0_once)
                ok1_pair = ok1_tip and trk.ok1_goal
                if not ok0_pair and not ok1_pair:
                    continue
                e_raw = err4_with_flags(t0, g0, t1, g1, ok0_pair, ok1_pair)
                en = norm_active(e_raw, ok0_pair, ok1_pair)

                if ok1_pair:
                    xy_err = math.sqrt(e_raw[0]*e_raw[0] + e_raw[1]*e_raw[1])
                else:
                    xy_err = float("inf")
                if ok0_pair:
                    z_err = abs(e_raw[3])
                else:
                    z_err = float("inf")

                if ok0_pair and ok1_pair and (xy_err < XY_TOL) and (z_err < Z_TOL):
                    pick_path_after_convergence(robot, qA, qB, qC, qD, initial_move_q)
                    converged = True
                    break

                if prev_en is not None and en > prev_en:
                    inc_count += 1
                else:
                    inc_count = 0
                prev_en = en

                if ok1_pair and not ok0_pair:
                    J_use = (J[0], J[1])
                    e_vec = (e_raw[0], e_raw[1])
                elif ok0_pair and not ok1_pair:
                    J_use = (J[2], J[3])
                    e_vec = (e_raw[2], e_raw[3])
                else:
                    J_use = J
                    e_vec = e_raw

                dth = dls(J_use, e_vec, LAMBDA)
                if dth is None:
                    running = False
                    break

                dq_full = [max(-STEP_CAP, min(STEP_CAP, math.degrees(v))) for v in dth]
                dq_full[0] = -dq_full[0]

                if ok0_pair and not ok1_pair:
                    dq = [0.0, dq_full[1], dq_full[2], 0.0]
                elif ok1_pair and not ok0_pair:
                    dq = [dq_full[0], 0.0, 0.0, dq_full[3]]
                else:
                    dq = dq_full

                q_cmd = (qA + dq[0], qB + dq[1], qC + dq[2], qD + dq[3])
                qA, qB, qC, qD = robot.move_to(*q_cmd)
                time.sleep(WAIT_SEC)

            if not converged and running:
                if initial_move_q is not None:
                    robot.move_to(*initial_move_q)
                    time.sleep(1.0)
            J_last = J
    finally:
        trk.stop()
        robot.shutdown()

if __name__ == "__main__":
    try:
        argc = len(sys.argv)
        if argc >= 7:
            host   = sys.argv[1]
            port   = int(sys.argv[2])
            cam0   = int(sys.argv[3])
            cam1   = int(sys.argv[4])
            width  = int(sys.argv[5])
            height = int(sys.argv[6])
            run(host, port, cam0, cam1, width, height, show=True)
        elif argc >= 5:
            host = sys.argv[1]
            port = int(sys.argv[2])
            cam0 = int(sys.argv[3])
            cam1 = int(sys.argv[4])
            run(host, port, cam0, cam1, show=True)
        elif argc >= 3:
            host = sys.argv[1]
            port = int(sys.argv[2])
            run(host, port, show=True)
        else:
            run(show=True)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass