import os
import random
import time
import math

import cv2
import numpy as np
import pygame

import mediapipe as mp


# ---------------------- 配置 ----------------------

WIN_WIDTH = 1280
WIN_HEIGHT = 720

CAM_W = 320
CAM_H = 240

FPS = 60

CARD_SCALE = 0.5  # 初始牌大小
CENTER = (WIN_WIDTH // 2, WIN_HEIGHT // 2)

# 手势去抖时间（秒）
DEBOUNCE_SEC = 0.4

# 动画时长
SHUFFLE_DURATION = 1.0
PICK_DURATION = 1.2
FLIP_DURATION = 0.8
RESHUFFLE_DURATION = 1.5

# ---------------------- 状态/手势枚举 ----------------------


class GestureType:
    NONE = "NONE"
    FIST = "FIST"
    OPEN_PALM = "OPEN_PALM"
    ONE_FINGER = "ONE_FINGER"
    SWING = "SWING_LEFT_RIGHT"


class StateType:
    STACK = "STACK"           # 中心堆积
    SHUFFLING = "SHUFFLING"   # 全屏洗牌
    PICKING = "PICKING"       # 抽牌中（飞向中心）
    SHOW_BACK = "SHOW_BACK"   # 中心放大牌背
    SHOW_FRONT = "SHOW_FRONT" # 已翻到正面


# ---------------------- 工具函数 ----------------------


def ease_in_out_cubic(t: float) -> float:
    """近似 cubic-bezier(0.77, 0, 0.175, 1) 的缓动（简单用 cubic）"""
    # clamp
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    else:
        f = (2 * t - 2)
        return 0.5 * f * f * f + 1


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_tuple(a, b, t):
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))


# ---------------------- 手势识别 ----------------------


class GestureRecognizer:
    """
    基于 MediaPipe Hands 的手势识别模块。
    对外只暴露高层事件：FIST, OPEN_PALM, ONE_FINGER, SWING_LEFT_RIGHT
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.last_gesture = GestureType.NONE
        self.current_gesture = GestureType.NONE
        self.gesture_start_time = 0.0

        # 用于检测左右摆动
        self.swing_history = []  # 存最近若干帧 index finger x
        self.swing_window = 10  # 帧数窗口
        self.swing_threshold = 0.15  # x 的归一化位移阈值（0~1）

    def process(self, frame_bgr):
        """
        输入 BGR 摄像头帧，返回：
        - landmarks (列表，像素坐标)
        - gesture_event: 确认后的手势类型 / NONE
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        landmarks_px = None
        gesture_event = GestureType.NONE

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            # 转为像素
            landmarks_px = []
            for lm in hand_landmarks.landmark:
                landmarks_px.append((int(lm.x * w), int(lm.y * h)))

            # 计算当前帧的手势
            raw_gesture = self._classify_gesture(hand_landmarks)

            now = time.time()

            # SWING 是在 ONE_FINGER 上额外检测
            if raw_gesture == GestureType.ONE_FINGER:
                self._update_swing(hand_landmarks)
                if self._is_swing():
                    raw_gesture = GestureType.SWING

            # 去抖：需要同一 raw_gesture 持续一段时间才认为确认
            if raw_gesture != self.current_gesture:
                self.current_gesture = raw_gesture
                self.gesture_start_time = now
            else:
                if (
                    self.current_gesture != GestureType.NONE
                    and now - self.gesture_start_time > DEBOUNCE_SEC
                    and self.current_gesture != self.last_gesture
                ):
                    gesture_event = self.current_gesture
                    self.last_gesture = self.current_gesture
        else:
            # 没有手，重置 swing 历史
            self.swing_history.clear()
            self.current_gesture = GestureType.NONE

        return landmarks_px, gesture_event

    def _classify_gesture(self, hand_landmarks):
        """
        根据 21 个关键点判断 简单手势：
        - FIST: 伸直手指数量 <= 1
        - OPEN_PALM: 伸直手指数量 >= 4
        - ONE_FINGER: 只有食指伸直
        参考常见 MediaPipe Finger counting 算法
        """
        lm = hand_landmarks.landmark

        def is_finger_extended(tip, pip, mcp=None):
            # y 越小越靠上，这里用 tip.y < pip.y 作为“伸直”
            return tip.y < pip.y

        # 手指索引：食指(8,6,5), 中(12,10,9), 无(16,14,13), 小(20,18,17)
        fingers = []
        fingers.append(is_finger_extended(lm[8], lm[6], lm[5]))   # index
        fingers.append(is_finger_extended(lm[12], lm[10], lm[9]))  # middle
        fingers.append(is_finger_extended(lm[16], lm[14], lm[13])) # ring
        fingers.append(is_finger_extended(lm[20], lm[18], lm[17])) # pinky

        extended_count = sum(1 for f in fingers if f)

        if extended_count <= 0:
            return GestureType.FIST
        elif extended_count >= 4:
            return GestureType.OPEN_PALM
        elif extended_count == 1 and fingers[0]:
            return GestureType.ONE_FINGER
        else:
            return GestureType.NONE

    def _update_swing(self, hand_landmarks):
        # 记录食指尖 x 坐标（归一化）
        idx_tip = hand_landmarks.landmark[8]
        self.swing_history.append(idx_tip.x)
        if len(self.swing_history) > self.swing_window:
            self.swing_history.pop(0)

    def _is_swing(self):
        if len(self.swing_history) < self.swing_window:
            return False
        diff = max(self.swing_history) - min(self.swing_history)
        return diff > self.swing_threshold


# ---------------------- 牌 & 牌堆 ----------------------


class Card:
    def __init__(self, front_surface, back_surface, pos, scale=1.0):
        self.front = front_surface
        self.back = back_surface
        self.pos = pos  # (x, y)
        self.target_pos = pos
        self.scale = scale
        self.target_scale = scale
        self.show_front = False

        # flip 动画
        self.flipping = False
        self.flip_start_time = 0.0

        # 洗牌时的小随机偏移
        self.offset = (0, 0)

    def start_move(self, target_pos, target_scale):
        self.target_pos = target_pos
        self.target_scale = target_scale

    def start_flip(self):
        self.flipping = True
        self.flip_start_time = time.time()

    def update(self, dt):
        # 平滑移动/缩放
        self.pos = lerp_tuple(self.pos, self.target_pos, ease_in_out_cubic(min(dt * 5, 1)))
        self.scale = lerp(self.scale, self.target_scale, ease_in_out_cubic(min(dt * 5, 1)))

        # 翻转动画
        if self.flipping:
            t = (time.time() - self.flip_start_time) / FLIP_DURATION
            if t >= 1.0:
                self.flipping = False
                self.show_front = True
            # 用 scale_x 模拟翻转
            self.flip_t = t
        else:
            self.flip_t = None

    def draw(self, screen):
        img = self.front if self.show_front else self.back

        # 基本缩放
        w, h = img.get_size()
        s = self.scale
        draw_w = int(w * s)
        draw_h = int(h * s)
        if draw_w <= 0 or draw_h <= 0:
            return

        card_img = pygame.transform.smoothscale(img, (draw_w, draw_h))

        # 翻转效果：scaleX 从 1 → 0 → 1
        if self.flip_t is not None:
            t = max(0.0, min(1.0, self.flip_t))
            if t < 0.5:
                sx = 1 - 2 * t
            else:
                sx = -1 + 2 * (t - 0.5)
                # 在中途换正面
                if not self.show_front and t > 0.5:
                    self.show_front = True
            sx = abs(sx)
            new_w = max(1, int(draw_w * sx))
            card_img = pygame.transform.smoothscale(card_img, (new_w, draw_h))

        x, y = self.pos
        rect = card_img.get_rect(center=(int(x), int(y)))
        screen.blit(card_img, rect)

        # 简单粒子/晕光效果（正面展示时）
        if self.show_front:
            pygame.draw.circle(screen, (255, 255, 255), rect.center, max(5, int(10 * self.scale)), 1)


class Deck:
    def __init__(self, card_front_surfaces, back_surface):
        self.cards = []
        self.back_surface = back_surface

        center_pos = CENTER
        for surf in card_front_surfaces:
            c = Card(surf, back_surface, center_pos, CARD_SCALE)
            # 微小偏移，叠在一起
            jitter = (random.uniform(-3, 3), random.uniform(-3, 3))
            c.pos = (center_pos[0] + jitter[0], center_pos[1] + jitter[1])
            c.target_pos = c.pos
            self.cards.append(c)

        self.active_card = None

    def reset_stack(self):
        # 所有牌回到中心堆叠
        for c in self.cards:
            jitter = (random.uniform(-3, 3), random.uniform(-3, 3))
            c.target_pos = (CENTER[0] + jitter[0], CENTER[1] + jitter[1])
            c.target_scale = CARD_SCALE
            c.show_front = False
            c.flipping = False
        self.active_card = None

    def layout_shuffle(self):
        # 所有牌散开：随机分布在屏幕区域
        for c in self.cards:
            x = random.uniform(WIN_WIDTH * 0.2, WIN_WIDTH * 0.8)
            y = random.uniform(WIN_HEIGHT * 0.2, WIN_HEIGHT * 0.8)
            c.target_pos = (x, y)
            c.target_scale = CARD_SCALE * 0.7
            c.show_front = False
            c.flipping = False

    def pick_random_card(self):
        self.active_card = random.choice(self.cards)
        # 其他牌缩小模糊背景
        for c in self.cards:
            if c is self.active_card:
                c.target_pos = CENTER
                c.target_scale = CARD_SCALE * 1.5
            else:
                # 缩到角落
                c.target_scale = CARD_SCALE * 0.4
        return self.active_card

    def update(self, dt):
        for c in self.cards:
            c.update(dt)

    def draw(self, screen):
        # 背景牌先画
        for c in self.cards:
            if c is not self.active_card:
                c.draw(screen)
        # 高亮牌后画
        if self.active_card:
            self.active_card.draw(screen)
        else:
            # 没有 active card 时，按顺序画（叠堆）
            pass


# ---------------------- 主程序 ----------------------


class TarotApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Feminist Tarot Gesture Prototype")
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        self.clock = pygame.time.Clock()

        # 摄像头
        self.cap = cv2.VideoCapture(0)
        self.camera_ok = self.cap.isOpened()

        self.gesture_recognizer = GestureRecognizer()

        # MediaPipe 绘制用
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # 加载牌面
        back_surf, front_surfs = self.load_cards("./cards")
        self.deck = Deck(front_surfs, back_surf)
        self.deck.reset_stack()

        # 状态机
        self.state = StateType.STACK
        self.state_start_time = time.time()

        # fallback: 鼠标按钮
        self.mouse_mode = not self.camera_ok  # 如果没有摄像头，自动进入鼠标模式
        self.font = pygame.font.SysFont("Arial", 18)

    def load_cards(self, folder):
        files = sorted(os.listdir(folder))
        back_path = os.path.join(folder, "back.png")
        back_img = pygame.image.load(back_path).convert_alpha()

        front_surfs = []
        for f in files:
            if f.lower() == "back.png":
                continue
            if f.lower().endswith(".png") or f.lower().endswith(".jpg"):
                img = pygame.image.load(os.path.join(folder, f)).convert_alpha()
                front_surfs.append(img)

        if not front_surfs:
            raise RuntimeError("No front card images found in ./cards/")

        return back_img, front_surfs

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.mouse_mode = not self.mouse_mode
                elif event.type == pygame.MOUSEBUTTONDOWN and self.mouse_mode:
                    self.handle_mouse_click(event.pos)

            self.update(dt)
            self.draw()

    def handle_mouse_click(self, pos):
        x, y = pos
        # 简单四个按钮区域
        if 20 <= x <= 120 and WIN_HEIGHT - 40 <= y <= WIN_HEIGHT - 10:
            self.trigger_gesture(GestureType.FIST)
        elif 140 <= x <= 260 and WIN_HEIGHT - 40 <= y <= WIN_HEIGHT - 10:
            self.trigger_gesture(GestureType.OPEN_PALM)
        elif 280 <= x <= 380 and WIN_HEIGHT - 40 <= y <= WIN_HEIGHT - 10:
            self.trigger_gesture(GestureType.ONE_FINGER)
        elif 400 <= x <= 500 and WIN_HEIGHT - 40 <= y <= WIN_HEIGHT - 10:
            self.trigger_gesture(GestureType.SWING)

    def trigger_gesture(self, gesture):
        # 用于鼠标模式手动触发
        self.handle_gesture_event(gesture)

    def update(self, dt):
        # 手势识别
        hand_landmarks_px = None
        gesture_event = GestureType.NONE

        frame_bgr = None
        if self.camera_ok:
            ret, frame_bgr = self.cap.read()
            if not ret:
                self.camera_ok = False
            else:
                # 镜像翻转，让体验更符合“镜子”
                frame_bgr = cv2.flip(frame_bgr, 1)
                if not self.mouse_mode:
                    hand_landmarks_px, gesture_event = self.gesture_recognizer.process(
                        frame_bgr
                    )

        # 根据手势事件驱动状态机
        if not self.mouse_mode and gesture_event != GestureType.NONE:
            self.handle_gesture_event(gesture_event)

        # 状态机内部逻辑（时间控制等）
        self.update_state_logic()

        # 更新牌面
        self.deck.update(dt)

        # 保存当前摄像头帧和关键点，用于绘制
        self.last_frame_bgr = frame_bgr
        self.last_hand_landmarks_px = hand_landmarks_px

    def handle_gesture_event(self, gesture):
        # 根据当前状态 & gesture 做状态转移
        if gesture == GestureType.FIST:
            # 回到初始堆叠
            self.deck.reset_stack()
            self.set_state(StateType.STACK)

        elif gesture == GestureType.OPEN_PALM:
            if self.state in [StateType.STACK, StateType.SHOW_FRONT, StateType.SHOW_BACK]:
                self.deck.layout_shuffle()
                self.set_state(StateType.SHUFFLING)

        elif gesture == GestureType.ONE_FINGER:
            if self.state == StateType.SHUFFLING:
                self.deck.pick_random_card()
                self.set_state(StateType.PICKING)

        elif gesture == GestureType.SWING:
            if self.state in [StateType.PICKING, StateType.SHOW_BACK]:
                # 触发翻牌
                if self.deck.active_card:
                    self.deck.active_card.start_flip()
                    self.set_state(StateType.SHOW_FRONT)

    def set_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()

    def update_state_logic(self):
        # 可以在这里精细控制每个状态持续时间/自动过渡
        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == StateType.SHUFFLING:
            # 洗牌持续时间结束后，保持洗牌布局，但不强制跳状态
            pass
        elif self.state == StateType.PICKING:
            if elapsed > PICK_DURATION:
                self.set_state(StateType.SHOW_BACK)
        elif self.state == StateType.SHOW_FRONT:
            # SHOW_FRONT 状态暂时不自动结束，等待 OPEN_PALM 或 FIST
            pass

    def draw_camera(self):
        if self.last_frame_bgr is None:
            return

        frame = self.last_frame_bgr.copy()
        h, w, _ = frame.shape

        # 绘制手部骨架（如果有）
        if self.last_hand_landmarks_px is not None:
            pts = self.last_hand_landmarks_px
            # 简单画点和线（只点，线就不画太复杂了）
            for x, y in pts:
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # 缩放到 320x240
        frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        surf.set_alpha(200)  # 约 80% 透明度
        self.screen.blit(surf, (WIN_WIDTH - CAM_W - 20, 20))

    def draw_ui(self):
        # 手势提示文字
        lines = [
            "Gestures:",
            "Fist = reset stack",
            "Open palm = shuffle",
            "One finger = pick card",
            "Swing finger left/right = flip",
            "Press 'M' to toggle mouse mode",
            f"Mouse mode: {'ON' if self.mouse_mode else 'OFF'}",
        ]
        x, y = 20, 20
        for line in lines:
            text_surf = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(text_surf, (x, y))
            y += 20

        # 鼠标模式按钮
        if self.mouse_mode:
            btns = [
                ("Reset(Fist)", (20, WIN_HEIGHT - 40, 100, 30)),
                ("Shuffle(Open)", (140, WIN_HEIGHT - 40, 120, 30)),
                ("Pick(One)", (280, WIN_HEIGHT - 40, 100, 30)),
                ("Flip(Swing)", (400, WIN_HEIGHT - 40, 100, 30)),
            ]
            for label, rect in btns:
                pygame.draw.rect(self.screen, (50, 50, 50), rect, border_radius=4)
                text_surf = self.font.render(label, True, (220, 220, 220))
                tx = rect[0] + 5
                ty = rect[1] + 5
                self.screen.blit(text_surf, (tx, ty))

        if not self.camera_ok:
            warning = "Camera not available. Using mouse mode fallback."
            text_surf = self.font.render(warning, True, (255, 120, 120))
            self.screen.blit(text_surf, (20, WIN_HEIGHT - 70))

    def draw(self):
        self.screen.fill((10, 5, 20))  # 深色背景

        # 画牌
        self.deck.draw(self.screen)

        # 摄像头画面 + 骨架
        if self.camera_ok:
            self.draw_camera()

        # 文本 UI
        self.draw_ui()

        pygame.display.flip()

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    app = TarotApp()
    app.run()