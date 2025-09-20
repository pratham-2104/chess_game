import os
import sys
import math
import time
import pygame
import chess
import threading

# Try to import user's engine if available
try:
    from chess_engine import select_move as external_select_move
except Exception:
    external_select_move = None

# ---------- Config ----------
FPS = 30
BOARD_SIZE = 640                 # board area in pixels (square)
SIDE_PANEL = 220                 # right-side UI width
WINDOW_W = BOARD_SIZE + SIDE_PANEL
WINDOW_H = BOARD_SIZE
SQUARE_SIZE = BOARD_SIZE // 8
THEMES = ["wood", "marble", "blue"]
DEFAULT_THEME = THEMES[0]
ENGINE_DEPTH = 3                 # default engine search depth if using built-in engine
# ----------------------------

# Piece order mapping to image filenames
PIECE_KEYS = {
    chess.PAWN: "p",
    chess.KNIGHT: "n",
    chess.BISHOP: "b",
    chess.ROOK: "r",
    chess.QUEEN: "q",
    chess.KING: "k"
}

pygame.init()
FONT = pygame.font.SysFont("DejaVuSans", 18)
BIG_FONT = pygame.font.SysFont("DejaVuSans", 22, bold=True)

# Colors
LIGHT_SQUARE_COLORS = {
    "wood": (240, 217, 181),
    "marble": (230, 230, 235),
    "blue": (200, 220, 255)
}
DARK_SQUARE_COLORS = {
    "wood": (181, 136, 99),
    "marble": (130, 130, 140),
    "blue": (60, 120, 200)
}
HIGHLIGHT_COLOR = (50, 205, 50, 120)  # last move highlight
LEGAL_MOVE_COLOR = (255, 215, 0, 160)  # legal move hint (semi transparent)

WINDOW = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Chess GUI — Themes / Engine Demo")
CLOCK = pygame.time.Clock()

# Preload theme piece images (lazy loaded)
def load_theme_images(theme_name):
    """
    Attempts to load images from themes/<theme_name>/*.png
    Images expected: wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk
    Returns dict { 'wp': surface, ... } or empty dict on failure.
    """
    images = {}
    base = os.path.join("themes", theme_name)
    if not os.path.isdir(base):
        return images
    for color in ("w", "b"):
        for code in ("p", "n", "b", "r", "q", "k"):
            name = f"{color}{code}.png"
            path = os.path.join(base, name)
            if os.path.isfile(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    img = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
                    images[color + code] = img
                except Exception as e:
                    print("Failed to load image:", path, e)
    return images

# Global theme images cache
THEME_IMAGES = {t: load_theme_images(t) for t in THEMES}

# ---------- Simple builtin engine (if external not available) ----------
# We include a small alpha-beta engine similar to earlier but optimized for speed.
# It uses a light evaluation (material + pst + mobility) and returns a move.

import math

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Tiny piece-square tables (mirrored)
PST_SIMPLE = {
    chess.PAWN: [0]*64,
    chess.KNIGHT: [0]*64,
    chess.BISHOP: [0]*64,
    chess.ROOK: [0]*64,
    chess.QUEEN: [0]*64,
    chess.KING: [0]*64
}

def evaluate_simple(board: chess.Board) -> int:
    if board.is_checkmate():
        return -999999 if board.turn else 999999
    if board.is_stalemate():
        return 0
    score = 0
    for piece_type in PIECE_VALUES:
        for sq in board.pieces(piece_type, chess.WHITE):
            score += PIECE_VALUES[piece_type]
            score += PST_SIMPLE[piece_type][sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            score -= PIECE_VALUES[piece_type]
            score -= PST_SIMPLE[piece_type][chess.square_mirror(sq)]
    # mobility
    board_copy = board.copy()
    board_copy.turn = chess.WHITE
    wm = len(list(board_copy.legal_moves))
    board_copy.turn = chess.BLACK
    bm = len(list(board_copy.legal_moves))
    score += (wm - bm) * 10
    return score

def alphabeta_simple(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool):
    if depth == 0 or board.is_game_over():
        return evaluate_simple(board), None
    best_move = None
    moves = list(board.legal_moves)
    # prefer captures
    moves.sort(key=lambda m: 0 if board.is_capture(m) else 1)
    if maximizing:
        value = -math.inf
        for m in moves:
            board.push(m)
            val, _ = alphabeta_simple(board, depth-1, alpha, beta, False)
            board.pop()
            if val > value:
                value = val
                best_move = m
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = math.inf
        for m in moves:
            board.push(m)
            val, _ = alphabeta_simple(board, depth-1, alpha, beta, True)
            board.pop()
            if val < value:
                value = val
                best_move = m
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move

def select_move_fallback(board: chess.Board, depth=ENGINE_DEPTH):
    # Use external_select_move if present (user's engine), else use builtin
    if external_select_move:
        try:
            mv, score = external_select_move(board, depth)
            return mv
        except Exception:
            pass
    _, mv = alphabeta_simple(board, depth, -math.inf, math.inf, board.turn == chess.WHITE)
    return mv

# ---------- GUI helpers ----------
def board_to_screen(square):
    """square 0..63 -> x,y on screen in pixels (top-left of square)"""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    # Note: screen y grows downward; rank 7 should be top
    x = file * SQUARE_SIZE
    y = (7 - rank) * SQUARE_SIZE
    return x, y

def screen_to_board(x, y):
    """pixel coords -> square index (0..63) or None if outside"""
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return None
    file = int(x // SQUARE_SIZE)
    rank = 7 - int(y // SQUARE_SIZE)
    return chess.square(file, rank)

def draw_board(surface, board, theme, selected_sq, legal_moves, last_move):
    # draw squares
    light = LIGHT_SQUARE_COLORS.get(theme, LIGHT_SQUARE_COLORS[DEFAULT_THEME])
    dark = DARK_SQUARE_COLORS.get(theme, DARK_SQUARE_COLORS[DEFAULT_THEME])
    for r in range(8):
        for f in range(8):
            square = chess.square(f, 7 - r)  # convert loop to square
            rect = pygame.Rect(f * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            if (r + f) % 2 == 0:
                color = light
            else:
                color = dark
            pygame.draw.rect(surface, color, rect)

    # highlight last move
    if last_move:
        from_sq = last_move.from_square
        to_sq = last_move.to_square
        for sq in (from_sq, to_sq):
            x, y = board_to_screen(sq)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill((50, 205, 50, 100))
            surface.blit(s, (x, y))

    # highlight selected square
    if selected_sq is not None:
        x, y = board_to_screen(selected_sq)
        s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        s.fill((255, 255, 0, 80))
        surface.blit(s, (x, y))

    # draw legal move dots
    for mv in legal_moves:
        # mv is square index
        x, y = board_to_screen(mv)
        center = (x + SQUARE_SIZE//2, y + SQUARE_SIZE//2)
        radius = max(6, SQUARE_SIZE//8)
        s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 215, 0, 200), (SQUARE_SIZE//2, SQUARE_SIZE//2), radius)
        surface.blit(s, (x, y))

    # draw pieces
    theme_images = THEME_IMAGES.get(theme, {})
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        x, y = board_to_screen(sq)
        key = ('w' if piece.color == chess.WHITE else 'b') + PIECE_KEYS[piece.piece_type]
        img = theme_images.get(key)
        if img:
            surface.blit(img, (x, y))
        else:
            # draw fallback piece
            piece_text = piece.symbol()
            # white uppercase, black lowercase
            txt = BIG_FONT.render(piece_text, True, (0, 0, 0))
            # center in square
            rect = txt.get_rect(center=(x + SQUARE_SIZE//2, y + SQUARE_SIZE//2))
            surface.blit(txt, rect)

def draw_side_panel(surface, board, engine_on, theme, engine_depth):
    # panel background
    panel_rect = pygame.Rect(BOARD_SIZE, 0, SIDE_PANEL, WINDOW_H)
    pygame.draw.rect(surface, (245, 245, 245), panel_rect)

    # Title
    surface.blit(BIG_FONT.render("Controls", True, (10, 10, 10)), (BOARD_SIZE+12, 10))
    # Buttons
    btn_w = SIDE_PANEL - 24
    def draw_button(y_offset, text, rect_id):
        rect = pygame.Rect(BOARD_SIZE+12, y_offset, btn_w, 36)
        pygame.draw.rect(surface, (220,220,220), rect, border_radius=6)
        surface.blit(FONT.render(text, True, (10,10,10)), (BOARD_SIZE+20, y_offset+8))
        return rect
    y = 50
    btn_new = draw_button(y, "New Game (N)", "new")
    y += 48
    btn_undo = draw_button(y, "Undo (U)", "undo")
    y += 48
    btn_toggle = draw_button(y, f"Engine: {'On' if engine_on else 'Off'} (E)", "toggle")
    y += 48
    btn_theme = draw_button(y, f"Theme: {theme.capitalize()} (1/2/3)", "theme")
    y += 48
    surface.blit(FONT.render(f"Engine depth: {engine_depth} ( +/- )", True, (40,40,40)), (BOARD_SIZE+16, y+8))
    # small legend
    y += 80
    surface.blit(FONT.render("Tips:", True, (10,10,10)), (BOARD_SIZE+12, y))
    y += 20
    lines = [
        "Click or drag to move",
        "Press 1/2/3 to switch theme",
        "Engine plays Black when ON",
        "You can use your engine file:",
        "  put chess_engine.py in the same folder",
        "  it should expose select_move(board, depth)"
    ]
    for i, ln in enumerate(lines):
        surface.blit(FONT.render(ln, True, (30,30,30)), (BOARD_SIZE+12, y + 18*i))

    # show turn and status
    status_y = WINDOW_H - 140
    turn_txt = "White to move" if board.turn == chess.WHITE else "Black to move"
    surface.blit(BIG_FONT.render(turn_txt, True, (20,20,20)), (BOARD_SIZE+12, status_y))
    # game result if over
    if board.is_game_over():
        res = board.result()
        reason = " (checkmate)" if board.is_checkmate() else ""
        surface.blit(BIG_FONT.render("Game Over: " + res + reason, True, (180,30,30)), (BOARD_SIZE+12, status_y+28))

    return {"new": btn_new, "undo": btn_undo, "toggle": btn_toggle, "theme": btn_theme}

# ---------- Main application ----------
def run_gui():
    board = chess.Board()
    running = True
    selected_sq = None
    legal_moves = []
    held_piece = None
    last_move = None
    theme = DEFAULT_THEME
    engine_on = False
    engine_depth = ENGINE_DEPTH
    drag_offset = (0, 0)
    move_history = []

    # for engine threading
    engine_thread = None
    engine_lock = threading.Lock()
    engine_thinking = False

    def engine_worker():
        nonlocal engine_thinking, last_move, move_history
        with engine_lock:
            engine_thinking = True
        try:
            mv = select_move_fallback(board, depth=engine_depth)
            if mv is not None:
                board.push(mv)
                move_history.append(mv)
                last_move = mv
        except Exception as e:
            print("Engine failed:", e)
        with engine_lock:
            engine_thinking = False

    while running:
        CLOCK.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < BOARD_SIZE:
                    sq = screen_to_board(mx, my)
                    if sq is not None:
                        piece = board.piece_at(sq)
                        if piece and piece.color == board.turn or selected_sq is not None:
                            # pick up piece if player's turn or selecting previously selected piece
                            selected_sq = sq
                            # compute legal moves from this square (target squares)
                            legal_moves = []
                            for mv in board.legal_moves:
                                if mv.from_square == sq:
                                    legal_moves.append(mv.to_square)
                            held_piece = piece
                            sx, sy = board_to_screen(sq)
                            drag_offset = (mx - sx, my - sy)
                        else:
                            # clicking empty or opponent piece -> if there is a selected square attempt move
                            if selected_sq is not None:
                                # attempt move selected_sq -> sq
                                try_mv = chess.Move(selected_sq, sq)
                                # handle promotion auto (to queen) if needed
                                if try_mv not in board.legal_moves:
                                    # try promotions
                                    if chess.square_rank(sq) in (0, 7):
                                        for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                                            m2 = chess.Move(selected_sq, sq, promotion=prom)
                                            if m2 in board.legal_moves:
                                                board.push(m2)
                                                move_history.append(m2)
                                                last_move = m2
                                                break
                                else:
                                    board.push(try_mv)
                                    move_history.append(try_mv)
                                    last_move = try_mv
                                selected_sq = None
                                legal_moves = []
                else:
                    # click in side panel: check buttons
                    btns = draw_side_panel(WINDOW, board, engine_on, theme, engine_depth)  # draw to get rects
                    # convert to events after drawing in main loop; here we just capture coords
                    # We'll handle button clicks after rendering (simple approach)
                    pass

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mx, my = event.pos
                if mx < BOARD_SIZE and selected_sq is not None and held_piece is not None:
                    # drop piece
                    target_sq = screen_to_board(mx, my)
                    if target_sq is not None:
                        mv = chess.Move(selected_sq, target_sq)
                        if mv in board.legal_moves:
                            board.push(mv)
                            move_history.append(mv)
                            last_move = mv
                        else:
                            # try promotions to queen by default
                            if chess.square_rank(target_sq) in (0, 7):
                                for prom in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                                    m2 = chess.Move(selected_sq, target_sq, promotion=prom)
                                    if m2 in board.legal_moves:
                                        board.push(m2)
                                        move_history.append(m2)
                                        last_move = m2
                                        break
                    selected_sq = None
                    held_piece = None
                    legal_moves = []

            elif event.type == pygame.MOUSEMOTION:
                # update drag visual if dragging
                pass

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    theme = THEMES[0]
                elif event.key == pygame.K_2:
                    theme = THEMES[1]
                elif event.key == pygame.K_3:
                    theme = THEMES[2]
                elif event.key == pygame.K_n:
                    board.reset()
                    selected_sq = None
                    legal_moves = []
                    last_move = None
                    move_history = []
                elif event.key == pygame.K_u:
                    # undo last move
                    if len(move_history) > 0:
                        try:
                            board.pop()
                            move_history.pop()
                            last_move = move_history[-1] if move_history else None
                        except Exception:
                            pass
                elif event.key == pygame.K_e:
                    engine_on = not engine_on
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    engine_depth = max(1, engine_depth - 1)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    engine_depth = min(5, engine_depth + 1)

        # if engine_on and it's engine's turn (we let engine play black here), trigger engine
        if engine_on and board.turn == chess.BLACK and not board.is_game_over():
            # start engine thread if not running
            if engine_thread is None or not engine_thread.is_alive():
                engine_thread = threading.Thread(target=engine_worker, daemon=True)
                engine_thread.start()

        # Draw frame
        WINDOW.fill((200, 200, 200))
        draw_board(WINDOW, board, theme, selected_sq, legal_moves, last_move)
        btns = draw_side_panel(WINDOW, board, engine_on, theme, engine_depth)

        # handle side-panel button clicks from last mouse event (checking current mouse state)
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if btns["new"].collidepoint(mx, my):
                # debounce - only on click down handled earlier; simple detection here
                # We'll use mouse button up detection instead for reliability. Skip here.
                pass

        # Draw drag piece on top if any
        if selected_sq is not None and held_piece is not None and pygame.mouse.get_focused():
            mx, my = pygame.mouse.get_pos()
            # render piece following mouse
            theme_images = THEME_IMAGES.get(theme, {})
            key = ('w' if held_piece.color == chess.WHITE else 'b') + PIECE_KEYS[held_piece.piece_type]
            img = theme_images.get(key)
            if img:
                WINDOW.blit(img, (mx - drag_offset[0], my - drag_offset[1]))
            else:
                # fallback circle with letter
                txt = BIG_FONT.render(held_piece.symbol(), True, (0,0,0))
                rect = txt.get_rect(center=(mx, my))
                WINDOW.blit(txt, rect)

        # Check mouse button up events for side panel clicks (handle them now)
        for e in pygame.event.get([pygame.MOUSEBUTTONUP]):
            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                mx, my = e.pos
                if btns["new"].collidepoint(mx, my):
                    board.reset()
                    selected_sq = None
                    legal_moves = []
                    last_move = None
                    move_history = []
                elif btns["undo"].collidepoint(mx, my):
                    if len(move_history) > 0:
                        try:
                            board.pop()
                            move_history.pop()
                            last_move = move_history[-1] if move_history else None
                        except Exception:
                            pass
                elif btns["toggle"].collidepoint(mx, my):
                    engine_on = not engine_on
                elif btns["theme"].collidepoint(mx, my):
                    # cycle theme
                    idx = THEMES.index(theme)
                    theme = THEMES[(idx + 1) % len(THEMES)]

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run_gui()