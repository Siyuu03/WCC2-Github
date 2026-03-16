# Title: Feminist Tarot Gesture Prototype
# Author: Siyu Xu
# Date: 22 January 2026
#
# Description:
# A webcam-based interactive tarot prototype controlled by hand gestures.
# The user shuffles a 22-card Major Arcana deck, picks a card, and flips it
# using real-time hand landmark tracking (a visible skeleton overlay).
# After a successful flip, the system displays a poetic English keyword and
# a Chinese translation for the revealed card.
#
# Instructions (operation manual):
# - Put this .py file next to these folders:
#   ./cards/  (must include back.png and 22 front PNGs named with 0..21 prefix)
#   ./fonts/  (must include NotoSerif-VariableFont_wdth,wght.ttf)
# - Run: python tarot_gesture.py
# - Gestures:
#   * Fist = reset stack
#   * Open palm = shuffle (and return to shuffle from a revealed card)
#   * One finger = pick (only when shuffling)
#   * Swing index finger left/right = flip (only after pick)
# - ESC / Q quits.
#
# Acknowledgements / use of AI tools:
# I acknowledge the use of [1] ChatGPT (https://chat.openai.com/) to [2] generate,
# debug, and refine Python code for webcam capture, MediaPipe hand landmark tracking,
# gesture logic (debounce + swing detection), and Pygame rendering in the drafting of
# this assessment. In January 2026 I entered prompts (paraphrased) such as:
# [3] “Help me build a Python + Pygame interactive tarot prototype controlled by a webcam.
# Use MediaPipe Hands to detect fist/open/one-finger gestures and a swing gesture to flip
# a card. Keep the interface smooth (reduced processing frequency), show a small camera
# preview with a skeleton overlay, and display English/Chinese poetic keywords only after
# a successful flip.” The generated code was then edited, extended, parameter-tuned and
# commented by me, and portions of this AI-assisted code are included in the final sketch.

import os
import re
import time
import math
import random

import cv2
import numpy as np
import pygame

import mediapipe as mp


# ---------------------------- CONFIG ----------------------------
WIN_W, WIN_H = 1920, 1080
FPS = 30  # 你机器卡的话 30 比 60 稳很多

CAM_W, CAM_H = 320, 240
CAM_INDEX = 0

# 手势检测频率（降低会更流畅）
MP_FPS = 15

# 去抖（手势稳定时间）
DEBOUNCE_SEC = 0.35

# SWING：窗口内 X 轨迹的“范围”超过阈值就算甩动（比首尾dx更适合来回晃）
SWING_WINDOW_SEC = 0.28
SWING_THRESHOLD_PX = 45
SWING_COOLDOWN_SEC = 0.45

# 动画时长
PICK_DURATION = 1.0
FLIP_DURATION = 0.8
RESHUFFLE_DURATION = 1.0

# 牌尺寸（保持你这版的设定，不动）
CARD_H_STACK = int(WIN_H * 0.30)   # 中心牌堆显示大小（背面）
CARD_H_SMALL = int(WIN_H * 0.20)   # 洗牌/背景小牌
CARD_H_BIG   = int(WIN_H * 0.40)   # 抽中放大（背面/正面）

BG_COLOR = (10, 8, 18)

CENTER = (WIN_W // 2, WIN_H // 2 - 20)

CARDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards")

# 字体：英文用你下载的 NotoSerif；中文强制加载 macOS 系统字体文件（避免方框）
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_EN_FILE = "NotoSerif-VariableFont_wdth,wght.ttf"

# macOS 常见中文字体文件候选（优先 PingFang）
MAC_CN_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Hiragino Sans GB.ttc",
]
# ---------------------------------------------------------------


# 0-21 大阿尔卡纳标准顺序
ARCANA = [
    "The Fool",
    "The Magician",
    "The High Priestess",
    "The Empress",
    "The Emperor",
    "The Hierophant",
    "The Lovers",
    "The Chariot",
    "Strength",
    "The Hermit",
    "Wheel of Fortune",
    "Justice",
    "The Hanged Man",
    "Death",
    "Temperance",
    "The Devil",
    "The Tower",
    "The Star",
    "The Moon",
    "The Sun",
    "Judgement",
    "The World",
]

# 你给的牌意（英文 keyword + 中文映射）
TAROT_TEXT = {
    "The Fool": ("birth", "降生"),
    "The Magician": ("agency", "能动性"),
    "The High Priestess": ("intuition", "直觉"),
    "The Empress": ("motherhood", "母职"),
    "The Emperor": ("patriarchy", "父q"),
    "The Hierophant": ("norms", "规训"),
    "The Lovers": ("sexual relationship", "x缘关系"),
    "The Chariot": ("escape", "出走"),
    "Strength": ("feminism", "女q主义"),
    "The Hermit": ("solitude", "独居"),
    "Wheel of Fortune": ("marriage", "婚姻"),
    "Justice": ("equality", "平等"),
    "The Hanged Man": ("gaslighting", "煤q灯"),
    "Death": ("abortion", "d胎"),
    "Temperance": ("contraception", "避y"),
    "The Devil": ("libido", "x欲"),
    "The Tower": ("domestic violence", "家b"),
    "The Star": ("bestie", "天才女友"),
    "The Moon": ("self-devouring", "自我吞噬"),
    "The Sun": ("freedom", "自由"),
    "Judgement": ("divorce", "离h冷静期"),
    "The World": ("migration", "迁徙"),
}


# ---------------------------- UTIL ----------------------------
def lerp(a, b, t):
    return a + (b - a) * t


def lerp2(p, q, t):
    return (lerp(p[0], q[0], t), lerp(p[1], q[1], t))


def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def load_png_scaled(path, target_h):
    img = pygame.image.load(path).convert_alpha()
    w, h = img.get_width(), img.get_height()
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return pygame.transform.smoothscale(img, (new_w, target_h))


def cv_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = np.transpose(frame_rgb, (1, 0, 2))  # (w,h,3)
    return pygame.surfarray.make_surface(arr)


# ---------------------------- GESTURE ----------------------------
class Gesture:
    NONE = "NONE"
    FIST = "FIST"
    OPEN = "OPEN"
    ONE  = "ONE"
    SWING = "SWING"


class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.last_mp_time = 0.0
        self.last_result = None

        # debounce
        self.candidate = Gesture.NONE
        self.candidate_start = time.time()
        self.stable = Gesture.NONE

        # swing buffer
        self.swing_points = []  # [(t, x_px)]
        self.last_swing_time = 0.0

        # hand motion for shuffle follow
        self.prev_wrist = None
        self.hand_dxdy = (0.0, 0.0)

    def _finger_extended(self, lm, tip, pip, eps=0.02):
        return lm[tip].y < lm[pip].y - eps

    def _thumb_extended(self, lm, eps=0.03):
        return abs(lm[4].x - lm[3].x) > eps

    def _classify(self, lm):
        idx = self._finger_extended(lm, 8, 6)
        mid = self._finger_extended(lm, 12, 10)
        ring = self._finger_extended(lm, 16, 14)
        pink = self._finger_extended(lm, 20, 18)
        thb = self._thumb_extended(lm)

        if (not idx) and (not mid) and (not ring) and (not pink):
            return Gesture.FIST

        if idx and mid and ring and pink and thb:
            return Gesture.OPEN

        if idx and (not mid) and (not ring) and (not pink):
            return Gesture.ONE

        return Gesture.NONE

    def detect(self, frame_bgr):
        """
        输入 frame_bgr 必须是：已经 resize 到 CAM_WxCAM_H 且 flip(1) 后的画面，
        这样骨架才能和右上角小窗对齐。
        """
        now = time.time()
        gesture_event = Gesture.NONE
        pts_px = None

        if now - self.last_mp_time >= 1.0 / MP_FPS:
            self.last_mp_time = now
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.hands.process(frame_rgb)
            self.last_result = res
        else:
            res = self.last_result

        raw = Gesture.NONE

        if res and res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = hand.landmark

            raw = self._classify(lm)

            pts_px = []
            for p in lm:
                pts_px.append((int(p.x * CAM_W), int(p.y * CAM_H)))

            wrist = pts_px[0]
            if self.prev_wrist is not None:
                self.hand_dxdy = (wrist[0] - self.prev_wrist[0], wrist[1] - self.prev_wrist[1])
            self.prev_wrist = wrist

            # ---------------- SWING (更灵敏、更符合“来回晃一下”) ----------------
            if raw == Gesture.ONE:
                idx_tip = pts_px[8]
                self.swing_points.append((now, idx_tip[0]))
                self.swing_points = [(t, x) for (t, x) in self.swing_points if now - t <= SWING_WINDOW_SEC]

                if (now - self.last_swing_time) >= SWING_COOLDOWN_SEC and len(self.swing_points) >= 3:
                    xs = [x for (_, x) in self.swing_points]
                    if (max(xs) - min(xs)) >= SWING_THRESHOLD_PX:
                        gesture_event = Gesture.SWING
                        self.last_swing_time = now
                        self.swing_points.clear()
            else:
                self.swing_points.clear()
            # ------------------------------------------------------------------

        else:
            self.prev_wrist = None
            self.hand_dxdy = (0.0, 0.0)
            self.swing_points.clear()

        # debounce（对 FIST/OPEN/ONE 稳定触发；SWING 直接返回）
        if gesture_event == Gesture.SWING:
            return gesture_event, pts_px

        if raw != self.candidate:
            self.candidate = raw
            self.candidate_start = now

        stable_now = self.stable
        if (now - self.candidate_start) >= DEBOUNCE_SEC:
            stable_now = self.candidate

        if stable_now != self.stable:
            self.stable = stable_now
            if self.stable in (Gesture.FIST, Gesture.OPEN, Gesture.ONE):
                gesture_event = self.stable

        return gesture_event, pts_px


# ---------------------------- CARD / DECK ----------------------------
class Card:
    def __init__(self, arcana_name, idx, front_big, back_big, back_small, back_stack):
        self.name = arcana_name
        self.idx = idx

        self.front_big = front_big
        self.back_big = back_big
        self.back_small = back_small
        self.back_stack = back_stack

        self.pos = (CENTER[0], CENTER[1])
        self.target = (CENTER[0], CENTER[1])

        self.mode = "SMALL"  # SMALL / STACK / BIG
        self.face = "BACK"   # BACK / FRONT

        self.flipping = False
        self.flip_t = 0.0
        self.flip_from = "BACK"
        self.flip_to = "FRONT"

    def set_layout(self, pos, mode):
        self.target = pos
        self.mode = mode

    def force_back(self):
        self.face = "BACK"
        self.flipping = False
        self.flip_t = 0.0

    def start_flip_to_front(self):
        if self.flipping:
            return
        self.flipping = True
        self.flip_t = 0.0
        self.flip_from = self.face
        self.flip_to = "FRONT"

    def start_flip_to_back(self):
        if self.flipping:
            return
        self.flipping = True
        self.flip_t = 0.0
        self.flip_from = self.face
        self.flip_to = "BACK"

    def update(self, dt):
        self.pos = lerp2(self.pos, self.target, min(1.0, dt * 8.0))

        if self.flipping:
            self.flip_t += dt / FLIP_DURATION
            if self.flip_t >= 1.0:
                self.flip_t = 1.0
                self.flipping = False
                self.face = self.flip_to

    def _base_surface(self):
        if self.mode == "STACK":
            return self.back_stack
        if self.mode == "SMALL":
            return self.back_small
        if self.face == "FRONT":
            return self.front_big
        return self.back_big

    def draw(self, screen):
        if self.flipping and self.mode == "BIG":
            t = ease_in_out(self.flip_t)
            base_w = self.back_big.get_width()
            base_h = self.back_big.get_height()

            show_face = self.flip_from if t < 0.5 else self.flip_to
            surf = self.front_big if show_face == "FRONT" else self.back_big

            w_factor = abs(math.cos(math.pi * t))
            w = max(2, int(base_w * w_factor))
            scaled = pygame.transform.smoothscale(surf, (w, base_h))

            x = int(self.pos[0] - w / 2)
            y = int(self.pos[1] - base_h / 2)
            screen.blit(scaled, (x, y))
            return

        surf = self._base_surface()
        w, h = surf.get_width(), surf.get_height()
        x = int(self.pos[0] - w / 2)
        y = int(self.pos[1] - h / 2)
        screen.blit(surf, (x, y))


class Deck:
    def __init__(self, cards):
        self.cards = cards[:]
        self.order = list(range(len(cards)))
        random.shuffle(self.order)
        self.ptr = 0

        self.active = None
        self.scatter = self._make_scatter_positions()

    def _make_scatter_positions(self):
        positions = []
        margin = 90
        cam_block = pygame.Rect(WIN_W - CAM_W - 40, 40, CAM_W + 20, CAM_H + 20)

        for _ in range(len(self.cards)):
            for _try in range(200):
                x = random.randint(margin, WIN_W - margin)
                y = random.randint(margin, WIN_H - margin)
                r = pygame.Rect(x - 60, y - 80, 120, 160)
                if not r.colliderect(cam_block):
                    positions.append((x, y))
                    break
            else:
                positions.append((random.randint(margin, WIN_W - margin), random.randint(margin, WIN_H - margin)))
        return positions

    def reshuffle_order(self):
        self.order = list(range(len(self.cards)))
        random.shuffle(self.order)
        self.ptr = 0

    def draw_next_unique(self):
        if self.ptr >= len(self.order):
            self.reshuffle_order()
        idx = self.order[self.ptr]
        self.ptr += 1
        return self.cards[idx]

    def reset_stack(self):
        self.active = None
        for c in self.cards:
            c.force_back()
            dx = random.randint(-6, 6)
            dy = random.randint(-6, 6)
            c.set_layout((CENTER[0] + dx, CENTER[1] + dy), "STACK")

    def start_shuffle(self):
        self.active = None
        self.scatter = self._make_scatter_positions()
        for i, c in enumerate(self.cards):
            c.force_back()
            c.set_layout(self.scatter[i], "SMALL")

    def follow_hand_offset(self, dxdy):
        ox = max(-40, min(40, dxdy[0] * 0.25))
        oy = max(-40, min(40, dxdy[1] * 0.25))
        for i, c in enumerate(self.cards):
            base = self.scatter[i]
            c.set_layout((base[0] + ox, base[1] + oy), "SMALL")

    def pick_card_to_center(self):
        self.active = self.draw_next_unique()

        self.active.force_back()
        self.active.set_layout((CENTER[0], CENTER[1]), "BIG")

        for c in self.cards:
            if c is self.active:
                continue
            bx, by = c.pos
            vx = bx - CENTER[0]
            vy = by - CENTER[1]
            c.set_layout((CENTER[0] + vx * 1.05, CENTER[1] + vy * 1.05), "SMALL")

    def flip_active_to_front(self):
        if self.active is None:
            return
        self.active.start_flip_to_front()

    def flip_active_to_back(self):
        if self.active is None:
            return
        if self.active.face == "FRONT" or self.active.flipping:
            self.active.start_flip_to_back()

    def force_all_back(self):
        for c in self.cards:
            c.force_back()

    def update(self, dt):
        for c in self.cards:
            c.update(dt)

    def draw(self, screen):
        for c in self.cards:
            if c is self.active:
                continue
            c.draw(screen)
        if self.active:
            self.active.draw(screen)


# ---------------------------- APP ----------------------------
class State:
    STACK = "STACK"
    SHUFFLING = "SHUFFLING"
    PICKED_BACK = "PICKED_BACK"
    FLIPPING = "FLIPPING"
    SHOW_FRONT = "SHOW_FRONT"
    RESHUFFLING = "RESHUFFLING"


class TarotApp:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        print("Display:", info.current_w, info.current_h)
        pygame.display.set_caption("Feminist Tarot Gesture Prototype")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.FULLSCREEN)
        self.canvas = pygame.Surface((WIN_W, WIN_H))
        self.display_size = self.screen.get_size()  # 真实屏幕尺寸（可能不是1920x1080）

        tablecloth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tablecloth.jpg")
        self.tablecloth = None
        if os.path.exists(tablecloth_path):
            self.tablecloth = pygame.image.load(tablecloth_path).convert()
            self.tablecloth = pygame.transform.smoothscale(self.tablecloth, (WIN_W, WIN_H))
        self.clock = pygame.time.Clock()

        # fonts
        self.font_en = self._load_en_font(72)
        self.font_cn = self._load_cn_font(56)
        self.small_ui = pygame.font.SysFont("Arial", 16)

        # camera
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ok = self.cap.isOpened()

        self.gesture = GestureRecognizer()
        self.deck = self._load_deck()

        self.state = State.STACK
        self.state_time = time.time()

        self.current_text = None  # (en, cn)

        self.deck.reset_stack()

    def _load_en_font(self, size):
        path = os.path.join(FONTS_DIR, FONT_EN_FILE)
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                pass
        return pygame.font.SysFont("Times New Roman", size)

    def _load_cn_font(self, size):
        # 1) 直接加载 macOS 系统字体文件：最稳，不会方框
        for p in MAC_CN_FONT_CANDIDATES:
            if os.path.exists(p):
                try:
                    return pygame.font.Font(p, size)
                except Exception:
                    pass

        # 2) SysFont 兜底（不同机器名字可能不一样）
        for name in ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Arial Unicode MS"]:
            try:
                f = pygame.font.SysFont(name, size)
                if f:
                    return f
            except Exception:
                continue

        # 3) 最后兜底
        return pygame.font.SysFont("Arial", size)

    def _load_deck(self):
        back_path = None
        fronts = []

        if not os.path.isdir(CARDS_DIR):
            raise RuntimeError(f"cards folder not found: {CARDS_DIR}")

        for fn in os.listdir(CARDS_DIR):
            if not fn.lower().endswith(".png"):
                continue
            if fn.lower() == "back.png" or fn.lower() == "back.png.png":
                back_path = os.path.join(CARDS_DIR, fn)
            else:
                fronts.append(os.path.join(CARDS_DIR, fn))

        if back_path is None:
            raise RuntimeError("Missing ./cards/back.png")

        back_stack = load_png_scaled(back_path, CARD_H_STACK)
        back_small = load_png_scaled(back_path, CARD_H_SMALL)
        back_big = load_png_scaled(back_path, CARD_H_BIG)

        idx_to_path = {}
        for p in fronts:
            base = os.path.basename(p)
            m = re.match(r"^(\d+)", base)
            if m:
                idx = int(m.group(1))
                if 0 <= idx <= 21:
                    idx_to_path[idx] = p

        missing = [i for i in range(22) if i not in idx_to_path]
        if missing:
            print("WARNING: Some indices missing in ./cards. Expected files starting with 0..21.")
            print("Missing indices:", missing)
            print("Tip: rename like '0愚人.png' ... '21世界.png'")

        cards = []
        for idx, name in enumerate(ARCANA):
            if idx in idx_to_path:
                front_big = load_png_scaled(idx_to_path[idx], CARD_H_BIG)
            else:
                front_big = back_big.copy()
            cards.append(Card(name, idx, front_big, back_big, back_small, back_stack))

        print(f"Loaded deck: {len(cards)} cards. (front images found: {len(idx_to_path)}/22)")
        return Deck(cards)

    def set_state(self, s):
        self.state = s
        self.state_time = time.time()

    def reset_stack(self):
        self.current_text = None
        self.deck.reset_stack()
        self.set_state(State.STACK)

    def start_shuffle(self):
        self.current_text = None
        self.deck.start_shuffle()
        self.set_state(State.SHUFFLING)

    def pick(self):
        if self.state != State.SHUFFLING:
            return
        self.current_text = None
        self.deck.pick_card_to_center()
        self.set_state(State.PICKED_BACK)

    def flip(self):
        if self.state != State.PICKED_BACK:
            return
        if self.deck.active is None:
            return
        self.deck.flip_active_to_front()
        self.set_state(State.FLIPPING)

    def reshuffle_from_show(self):
        self.current_text = None
        self.deck.force_all_back()
        self.deck.start_shuffle()
        self.set_state(State.SHUFFLING)

    def update_logic(self):
        if self.state == State.FLIPPING:
            c = self.deck.active
            if c and (not c.flipping) and c.face == "FRONT":
                self.current_text = TAROT_TEXT.get(c.name, None)
                self.set_state(State.SHOW_FRONT)

    # ---------------------------- ONLY CHANGED PARTS START ----------------------------
    # 1) draw_ui() -> draw_ui_on(self, surf) and draw to surf instead of self.screen
    def draw_ui_on(self, surf):
        lines = [
            "Gestures:",
            "Fist = reset stack",
            "Open palm = shuffle",
            "One finger = pick (when shuffling)",
            "Swing finger left/right = flip (after pick)",
            "ESC / Q = quit",
            "Debug keys: R reset / S shuffle / P pick / F flip",
        ]
        x, y = 24, 20
        for s in lines:
            txt = self.small_ui.render(s, True, (230, 230, 230))
            surf.blit(txt, (x, y))
            y += 18

        # bottom text (only after SHOW_FRONT) —— 修复：固定两行不重叠
        if self.current_text and self.state == State.SHOW_FRONT:
            en, cn = self.current_text
            en_s = self.font_en.render(en, True, (245, 245, 245))
            cn_s = self.font_cn.render(cn, True, (190, 160, 230))

            bx = WIN_W // 2
            en_y = WIN_H - 150
            cn_y = WIN_H - 80

            surf.blit(en_s, (bx - en_s.get_width() // 2, en_y))
            surf.blit(cn_s, (bx - cn_s.get_width() // 2, cn_y))

    # 2) draw_camera() -> draw_camera_on(self, surf, frame_bgr, pts_px) and draw to surf
    def draw_camera_on(self, surf, frame_bgr, pts_px):
        panel_x = WIN_W - CAM_W - 40
        panel_y = 40
        pygame.draw.rect(
            surf,
            (20, 20, 30),
            (panel_x - 6, panel_y - 6, CAM_W + 12, CAM_H + 12),
            border_radius=6,
        )

        if frame_bgr is None:
            msg = self.small_ui.render("Camera not available", True, (240, 180, 180))
            surf.blit(msg, (panel_x + 20, panel_y + CAM_H // 2))
            return

        cam_surf = cv_to_surface(frame_bgr)
        surf.blit(cam_surf, (panel_x, panel_y))

        if pts_px:
            conns = mp.solutions.hands.HAND_CONNECTIONS
            for (x, y) in pts_px:
                pygame.draw.circle(surf, (60, 200, 255), (panel_x + x, panel_y + y), 3)
            for a, b in conns:
                ax, ay = pts_px[a]
                bx, by = pts_px[b]
                pygame.draw.line(
                    surf,
                    (220, 220, 220),
                    (panel_x + ax, panel_y + ay),
                    (panel_x + bx, panel_y + by),
                    1,
                )
    # ---------------------------- ONLY CHANGED PARTS END ----------------------------

    def run(self):
        running = True
        last_time = time.time()

        while running:
            dt = time.time() - last_time
            last_time = time.time()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    if e.key == pygame.K_r:
                        self.reset_stack()
                    if e.key == pygame.K_s:
                        self.start_shuffle()
                    if e.key == pygame.K_p:
                        self.pick()
                    if e.key == pygame.K_f:
                        self.flip()

            frame = None
            pts = None

            if self.camera_ok:
                ok, f = self.cap.read()
                if ok and f is not None:
                    f = cv2.resize(f, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
                    f = cv2.flip(f, 1)
                    frame = f

                    g_event, pts = self.gesture.detect(frame)

                    if g_event == Gesture.FIST:
                        self.reset_stack()
                    elif g_event == Gesture.OPEN:
                        if self.state in (State.SHOW_FRONT, State.PICKED_BACK, State.FLIPPING):
                            self.reshuffle_from_show()
                        else:
                            self.start_shuffle()
                    elif g_event == Gesture.ONE:
                        self.pick()
                    elif g_event == Gesture.SWING:
                        self.flip()

                    if self.state == State.SHUFFLING:
                        self.deck.follow_hand_offset(self.gesture.hand_dxdy)

            self.deck.update(dt)
            self.update_logic()

            # 1) 先画到 canvas
            if self.tablecloth:
                self.canvas.blit(self.tablecloth, (0, 0))
            else:
                self.canvas.fill(BG_COLOR)

            self.deck.draw(self.canvas)
            self.draw_ui_on(self.canvas)
            self.draw_camera_on(self.canvas, frame, pts)

            # 2) 再把 canvas 缩放到真实屏幕并输出
            scaled = pygame.transform.smoothscale(self.canvas, self.display_size)
            self.screen.blit(scaled, (0, 0))
            pygame.display.flip()
            self.clock.tick(FPS)

        try:
            self.cap.release()
        except Exception:
            pass
        pygame.quit()


if __name__ == "__main__":
    TarotApp().run()