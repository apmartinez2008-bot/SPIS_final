import cv2
import numpy as np

def fen2board_all(fen_line):
    """Tracks presence of ANY piece (White or Black) on the board."""
    player_bool_position = []
    for row in fen_line.split(' ')[0].split('/'):
        bool_row = []
        for cell in list(row):
            if cell.isnumeric():
                for i in range(int(cell)):
                    bool_row.append(0)
            else:
                bool_row.append(1)  # Track both upper and lower case pieces
        player_bool_position.append(bool_row)
    return np.array(player_bool_position)

def point_in_quad(mid_point, quad_corners):
    """Returns True if mid_point (x, y) is inside the 4-corner polygon."""
    pts = np.array(quad_corners, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(mid_point[0]), float(mid_point[1])), False) >= 0

def find_current_past_position(img_1, img_2, board_squares, bool_position, FEN_line, chess_board, number_to_position_map, map_position):
    past_bool_position = fen2board_all(FEN_line)
    diff_position = np.zeros((8, 8), dtype=int)

    # Calculate absolute difference between snapshots
    image_diff = cv2.absdiff(img_1, img_2)
    image_diff_gray = cv2.cvtColor(image_diff, cv2.COLOR_BGR2GRAY)
    
    # LOWERED THRESHOLD: Increased sensitivity from 20 to 10
    _, threshold = cv2.threshold(image_diff_gray, 15, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Show threshold image for real-time debugging
    cv2.imshow("Difference Debugger", threshold)
    cv2.waitKey(1)

    required_contours_mid_point = []
    for c in cnts:
        # LOWERED CONTOUR AREA: Reduced from 300 to 80 for 640x480 resolution
        if cv2.contourArea(c) > 80:
            (x, y, w, h) = cv2.boundingRect(c)
            required_contours_mid_point.append([x + int(w / 2), y + int(h / 2)])

    if len(required_contours_mid_point) >= 2:
        flag = np.zeros((8, 8), dtype=int)
        
        for (r, c), quad_corners in board_squares.items():
            for mid_point in required_contours_mid_point:
                if point_in_quad(mid_point, quad_corners) and flag[r][c] == 0:
                    diff_position[r][c] = 1
                    flag[r][c] = 1

        # Identify departing square (had piece, now changed) and arrival square
        changed_squares = np.where(diff_position == 1)
        
        if len(changed_squares[0]) < 2:
            print("Debug: Changes were detected, but they didn't fall inside 2 distinct grid squares.")
            return " ", img_2, 0

        # Determine source (r1, c1) vs destination (r2, c2) based on past piece board
        r1, c1, r2, c2 = -1, -1, -1, -1
        
        for idx in range(len(changed_squares[0])):
            r = changed_squares[0][idx]
            c = changed_squares[1][idx]
            if past_bool_position[r][c] == 1 and r1 == -1:
                r1, c1 = r, c
            else:
                r2, c2 = r, c

        if r1 == -1 or r2 == -1:
            return " ", img_2, 0

        move_word = number_to_position_map[r1][c1] + number_to_position_map[r2][c2]

        draw_img = img_2.copy()
        pts1 = np.array(board_squares[(r1, c1)], np.int32).reshape((-1, 1, 2))
        pts2 = np.array(board_squares[(r2, c2)], np.int32).reshape((-1, 1, 2))
        cv2.polylines(draw_img, [pts1], True, (0, 0, 255), 2)
        cv2.polylines(draw_img, [pts2], True, (0, 255, 0), 2)

        return move_word, draw_img, 1
    else:
        print(f"Debug: Found {len(required_contours_mid_point)} valid motion contours. Need at least 2.")
        return " ", img_2, 0
