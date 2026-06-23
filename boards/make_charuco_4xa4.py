from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).parent

# ---------- physical board parameters ----------
DPI = 300

SQUARES_X = 8
SQUARES_Y = 10

SQUARE_MM = 50.0
MARKER_MM = 37.5

BOARD_W_MM = SQUARES_X * SQUARE_MM   # 400 mm
BOARD_H_MM = SQUARES_Y * SQUARE_MM   # 500 mm

# A4 portrait
A4_W_MM = 210.0
A4_H_MM = 297.0

# 2x2 A4 poster canvas
POSTER_W_MM = A4_W_MM * 2            # 420 mm
POSTER_H_MM = A4_H_MM * 2            # 594 mm

DICT_NAME = "DICT_5X5_1000"


def mm_to_px(mm: float) -> int:
    return round(mm / 25.4 * DPI)


def make_charuco_image() -> Image.Image:
    board_w_px = mm_to_px(BOARD_W_MM)
    board_h_px = mm_to_px(BOARD_H_MM)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

    square_m = SQUARE_MM / 1000.0
    marker_m = MARKER_MM / 1000.0

    # OpenCV 4.7+ style
    try:
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            square_m,
            marker_m,
            aruco_dict,
        )
        img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    # Older OpenCV fallback
    except Exception:
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            square_m,
            marker_m,
            aruco_dict,
        )
        img = board.draw((board_w_px, board_h_px), marginSize=0, borderBits=1)

    if img.ndim == 2:
        return Image.fromarray(img).convert("RGB")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")


def save_full_files(board_img: Image.Image) -> None:
    png_path = OUT_DIR / "charuco_8x10_50mm_full.png"
    pdf_path = OUT_DIR / "charuco_8x10_50mm_full.pdf"

    board_img.save(png_path, dpi=(DPI, DPI))
    board_img.save(pdf_path, "PDF", resolution=DPI)

    print("Saved full board:")
    print(" ", png_path)
    print(" ", pdf_path)


def make_a4_tiles(board_img: Image.Image) -> None:
    a4_w_px = mm_to_px(A4_W_MM)
    a4_h_px = mm_to_px(A4_H_MM)

    poster_w_px = a4_w_px * 2
    poster_h_px = a4_h_px * 2

    board_w_px = mm_to_px(BOARD_W_MM)
    board_h_px = mm_to_px(BOARD_H_MM)

    poster = Image.new("RGB", (poster_w_px, poster_h_px), "white")
    draw = ImageDraw.Draw(poster)

    # Center board on 2x2 A4 poster.
    # This gives 10 mm left/right margin and 47 mm top/bottom margin.
    left = (poster_w_px - board_w_px) // 2
    top = (poster_h_px - board_h_px) // 2

    poster.paste(board_img, (left, top))

    # Thin outer border around the board, outside/edge only.
    # It helps align after printing.
    draw.rectangle(
        [left, top, left + board_w_px - 1, top + board_h_px - 1],
        outline=(0, 0, 0),
        width=2,
    )

    pages = []
    labels = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for row in range(2):
        for col in range(2):
            x0 = col * a4_w_px
            y0 = row * a4_h_px
            page = poster.crop((x0, y0, x0 + a4_w_px, y0 + a4_h_px))

            # Page label in the margin; should not overlap board.
            d = ImageDraw.Draw(page)
            d.text((30, 30), labels[row * 2 + col], fill=(120, 120, 120))

            pages.append(page)

            page_path = OUT_DIR / f"charuco_8x10_50mm_A4_{labels[row * 2 + col]}.png"
            page.save(page_path, dpi=(DPI, DPI))
            print("Saved tile:", page_path)

    pdf_path = OUT_DIR / "charuco_8x10_50mm_A4_tiles.pdf"
    pages[0].save(
        pdf_path,
        "PDF",
        resolution=DPI,
        save_all=True,
        append_images=pages[1:],
    )

    print("Saved tiled A4 PDF:")
    print(" ", pdf_path)


def save_metadata() -> None:
    meta_path = OUT_DIR / "charuco_8x10_50mm_metadata.txt"
    meta_path.write_text(
        f"""ChArUco board metadata

dictionary: {DICT_NAME}
squares_x: {SQUARES_X}
squares_y: {SQUARES_Y}
square_length_mm: {SQUARE_MM}
marker_length_mm: {MARKER_MM}
board_width_mm: {BOARD_W_MM}
board_height_mm: {BOARD_H_MM}
dpi: {DPI}

Use these exact values in OpenCV detection/calibration.
After printing, measure the real square size and use the measured value if it differs.
""",
        encoding="utf-8",
    )
    print("Saved metadata:", meta_path)


if __name__ == "__main__":
    board = make_charuco_image()
    save_full_files(board)
    make_a4_tiles(board)
    save_metadata()
