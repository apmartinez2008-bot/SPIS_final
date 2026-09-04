import cv2
import numpy as np
import chess

def fen2board_all(fen_line):
    """Tracks presence of ANY piece on the board."""
    player_bool_position = []
    for row in fen_line.split(' ')[0].split('/'):
        bool_row = []
        for cell in list(row):
            if cell.isnumeric():
                for i in range(int(cell)):
                    bool_row.append(0)
            else:
                bool_row.append(1)
        player_bool_position.append(bool_row)
    return np.array(player_bool_position)

def point_in_quad(mid_point, quad_corners):
    """Returns True if mid_point (x, y) is inside the 4-corner polygon."""
    pts = np.array(quad_corners, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(mid_point[0]), float(mid_point[1])), False) >= 0

def find_current_past_position(img_1, img_2, board_squares, bool_position, FEN_line, chess_board, number_to_position_map, map_position):
    board = chess.Board(FEN_line)
    past_bool_position = fen2board_all(FEN_line)
    diff_position = np.zeros((8, 8), dtype=int)

    # 1. Image Difference Analysis
    image_diff = cv2.absdiff(img_1, img_2)
    image_diff_gray = cv2.cvtColor(image_diff, cv2.COLOR_BGR2GRAY)
    
    _, threshold = cv2.threshold(image_diff_gray, 20, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("Difference Debugger", threshold)
    cv2.waitKey(1)

    required_contours_mid_point = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 90:
            (x, y, w, h) = cv2.boundingRect(c)
            mid_x, mid_y = x + int(w / 2), y + int(h / 2)
            required_contours_mid_point.append([mid_x, mid_y])

    if len(required_contours_mid_point) < 2:
        return " ", img_2, 0

    # 2. Polygon Square Mapping
    flag = np.zeros((8, 8), dtype=int)
    for mid_point in required_contours_mid_point:
        for (r, c), quad_corners in board_squares.items():
            if point_in_quad(mid_point, quad_corners):
                if flag[r][c] == 0:
                    diff_position[r][c] = 1
                    flag[r][c] = 1

    changed_rows, changed_cols = np.where(diff_position == 1)
    
    # Map detected grid coordinates to standard algebraic notation
    detected_squares = [number_to_position_map[r][c] for r, c in zip(changed_rows, changed_cols)]
    
    print(f"[DEBUG] Detected Changed Squares: {detected_squares}")

    draw_img = img_2.copy()

    # 3. Castling Detection (Prioritize King Moves)
    if board.turn == chess.WHITE:
        # Check if White King moved
        if "e1" in detected_squares:
            if "g1" in detected_squares:
                return "e1g1", draw_img, 1  # Kingside Castle
            elif "c1" in detected_squares:
                return "e1c1", draw_img, 1  # Queenside Castle
    else:
        # Check if Black King moved
        if "e8" in detected_squares:
            if "g8" in detected_squares:
                return "e8g8", draw_img, 1  # Kingside Castle
            elif "c8" in detected_squares:
                return "e8c8", draw_img, 1  # Queenside Castle

    # 4. Standard Move Handling (2 Squares) + Noisy/Bumped Moves (3+ Squares)
    #
    # A clean move flips exactly 2 squares. A bumped piece, hand shadow, or
    # camera shake can flip additional squares that aren't part of the real
    # move. Previously any count other than exactly 2 was discarded here
    # (flag=0, "No move detected"), which meant main.py's legal-move scoring
    # never even ran on bumped moves - the noise never made it that far.
    # Instead, report every detected square and let main.py's candidate
    # scoring (fewest unexplained extra squares) pick the real move out of
    # the noise.
    if len(changed_rows) == 2:
        sq1 = (changed_rows[0], changed_cols[0])
        sq2 = (changed_rows[1], changed_cols[1])

        if past_bool_position[sq1[0]][sq1[1]] == 1 and past_bool_position[sq2[0]][sq2[1]] == 0:
            r1, c1 = sq1
            r2, c2 = sq2
        elif past_bool_position[sq2[0]][sq2[1]] == 1 and past_bool_position[sq1[0]][sq1[1]] == 0:
            r1, c1 = sq2
            r2, c2 = sq1
        else:
            r1, c1 = sq1
            r2, c2 = sq2

        notation_from = number_to_position_map[r1][c1]
        notation_to = number_to_position_map[r2][c2]
        move_word = notation_from + notation_to

        pts1 = np.array(board_squares[(r1, c1)], np.int32).reshape((-1, 1, 2))
        pts2 = np.array(board_squares[(r2, c2)], np.int32).reshape((-1, 1, 2))
        cv2.polylines(draw_img, [pts1], True, (0, 0, 255), 2)
        cv2.polylines(draw_img, [pts2], True, (0, 255, 0), 2)

        return move_word, draw_img, 1

    elif len(changed_rows) > 2:
        # Don't try to guess which 2 squares are "the" move here - main.py
        # doesn't need ordered from/to, it only needs the full set of
        # detected squares to score legal-move candidates against. Just
        # report everything that changed.
        move_word = "".join(
            number_to_position_map[r][c] for r, c in zip(changed_rows, changed_cols)
        )

        for r, c in zip(changed_rows, changed_cols):
            pts = np.array(board_squares[(r, c)], np.int32).reshape((-1, 1, 2))
            cv2.polylines(draw_img, [pts], True, (0, 165, 255), 2)  # orange = noisy/extra square

        return move_word, draw_img, 1

    return " ", img_2, 0
