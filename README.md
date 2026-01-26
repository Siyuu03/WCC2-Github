# WCC2 – Creative Coding 2 (2025–26)

Goldsmiths – MA Computational Arts  
APBR – Workshops in Creative Coding 2

---

# Feminist Tarot Gesture Prototype

**Author:** Siyu Xu  
**Date:** 12 Jan 2026  
**Tech:** Python (Pygame + OpenCV + MediaPipe)  
**Input:** Webcam hand gestures  
**Output:** Tarot interaction + hand skeleton overlay + bilingual keywords

A webcam-based interactive tarot prototype controlled by hand gestures.  
The user shuffles a 22-card Major Arcana deck, picks a card, and flips it using real-time hand tracking. After a successful flip, the system displays an English keyword and a Chinese translation for the revealed card.

---

## Files you need (important)

Make sure these files exist next to `tarot_gesture.py`:

- `cards/`
  - `back.png`
  - 22 front PNGs named with **0..21** prefix (e.g. `0愚人.png ... 21世界.png`)
- `fonts/`
  - `NotoSerif-VariableFont_wdth,wght.ttf`

Main file to run:
- `tarot_gesture.py`

---

## How to run

Open Terminal and run:

```bash
cd "Coding_Feminist Tarot Gesture Prototype"
pip3 install opencv-python numpy pygame mediapipe
python3 tarot_gesture.py
Quit:

Press ESC or Q

Gestures

Fist → reset stack

Open palm → shuffle / return to shuffle

One finger (index only) → pick (only when shuffling)

Swing index finger left/right → flip (only after pick)

Keyboard debug (if camera is unavailable):

R reset, S shuffle, P pick, F flip

Notes

Chinese text on macOS: the code loads system fonts (PingFang etc.) automatically.
On Windows, if Chinese shows as squares, you may need to change the font loading part in the code.

Screenshots (add 2–4 images)

Shuffle state:

Picked card:

Flipped card + keywords:

(Optional) Setup photo:

Acknowledgements (AI tools)

I acknowledge the use of ChatGPT (https://chat.openai.com/
) to generate, debug, and refine parts of the Python code (webcam capture, MediaPipe hand tracking, gesture logic, and Pygame rendering). The generated code was then edited, extended, parameter-tuned and commented by me.

References

MediaPipe: https://developers.google.com/mediapipe

Pygame: https://www.pygame.org/

OpenCV: https://opencv.org/

Rider–Waite tarot (Pamela Colman Smith) for the Major Arcana imagery