import cv2
import numpy as np

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
    past_bool_position = fen2board_all(FEN_line)
    diff_position = np.zeros((8, 8), dtype=int)

    # 1. Image Difference Analysis
    image_diff = cv2.absdiff(img_1, img_2)
    image_diff_gray = cv2.cvtColor(image_diff, cv2.COLOR_BGR2GRAY)
    
    # Thresholding for difference detection
    _, threshold = cv2.threshold(image_diff_gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Show difference window for debugging
    cv2.imshow("Difference Debugger", threshold)
    cv2.waitKey(1)

    required_contours_mid_point = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 40:  # Lowered sensitivity threshold
            (x, y, w, h) = cv2.boundingRect(c)
            mid_x, mid_y = x + int(w / 2), y + int(h / 2)
            required_contours_mid_point.append([mid_x, mid_y])

    print(f"\n--- DEBUG MOVE ANALYSIS ---")
    print(f"Phase 1: Found {len(required_contours_mid_point)} movement centroids.")

    if len(required_contours_mid_point) < 2:
        print("--> FAIL: Camera saw fewer than 2 movement points. Check lighting, hands in frame, or sensitivity.")
        return " ", img_2, 0

    # 2. Polygon Square Mapping
    flag = np.zeros((8, 8), dtype=int)
    mapped_count = 0
    
    for mid_point in required_contours_mid_point:
        matched = False
        for (r, c), quad_corners in board_squares.items():
            if point_in_quad(mid_point, quad_corners):
                matched = True
                if flag[r][c] == 0:
                    diff_position[r][c] = 1
                    flag[r][c] = 1
                    mapped_count += 1
                    print(f"   -> Point {mid_point} matched to Grid Square ({r}, {c})")
        if not matched:
            print(f"   -> Point {mid_point} fell OUTSIDE all grid polygons!")

    changed_squares = np.where(diff_position == 1)
    unique_squares_found = len(changed_squares[0])
    print(f"Phase 2: Mapped points to {unique_squares_found} unique grid square(s).")

    if unique_squares_found < 2:
        print("--> FAIL: Movements detected, but they didn't land in 2 separate grid squares.")
        return " ", img_2, 0

    # 3. Square Source vs Destination Identification
    sq1 = (changed_squares[0][0], changed_squares[1][0])
    sq2 = (changed_squares[0][1], changed_squares[1][1])

    if past_bool_position[sq1[0]][sq1[1]] == 1 and past_bool_position[sq2[0]][sq2[1]] == 0:
        r1, c1 = sq1
        r2, c2 = sq2
    elif past_bool_position[sq2[0]][sq2[1]] == 1 and past_bool_position[sq1[0]][sq1[1]] == 0:
        r1, c1 = sq2
        r2, c2 = sq1
    else:
        # Fallback if both squares had pieces (captures) or neither did
        r1, c1 = sq1
        r2, c2 = sq2

    raw_from = f"({r1},{c1})"
    raw_to = f"({r2},{c2})"
    notation_from = number_to_position_map[r1][c1]
    notation_to = number_to_position_map[r2][c2]
    move_word = notation_from + notation_to

    print(f"Phase 3: Source {raw_from} [{notation_from}] -> Destination {raw_to} [{notation_to}]")
    print(f"Phase 4: Generated Move Command -> '{move_word}'")

    draw_img = img_2.copy()
    pts1 = np.array(board_squares[(r1, c1)], np.int32).reshape((-1, 1, 2))
    pts2 = np.array(board_squares[(r2, c2)], np.int32).reshape((-1, 1, 2))
    cv2.polylines(draw_img, [pts1], True, (0, 0, 255), 2)
    cv2.polylines(draw_img, [pts2], True, (0, 255, 0), 2)

    return move_word, draw_img, 1
