import chess
import chess.svg
import math
import random
import time

# Piece basic values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-square tables (simple, smaller weights) for middlegame orientation (white perspective).

PST = {
    chess.PAWN: [
         0,  0,  0,   0,   0,  0,  0,  0,
         5, 10, 10, -20, -20, 10, 10,  5,
         5, -5, -10,  0,   0, -10, -5,  5,
         0,  0,   0,  20,  20,   0,  0,  0,
         5,  5,  10,  25,  25,  10,  5,  5,
        10, 10,  20,  30,  30,  20, 10, 10,
        50, 50, 50,  50,  50,  50, 50, 50,
         0,  0,   0,   0,   0,   0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-90,-30,-30,-30,-30,-90,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
         0,  0,  5, 10, 10,  5,  0,  0,
        25, 25, 25, 25, 25, 25, 25, 25,
         0,  0,  5, 10, 10,  5,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

def evaluate_board(board: chess.Board) -> int:
    """
    Positive value means advantage for White, negative for Black.
    Combines material + simple piece-square table + mobility.
    """
    if board.is_checkmate():
        # large positive for white win, negative for black win
        return -999999 if board.turn else 999999
    if board.is_stalemate():
        return 0

    material = 0
    pst_score = 0
    for piece_type in PIECE_VALUES:
        for sq in board.pieces(piece_type, chess.WHITE):
            material += PIECE_VALUES[piece_type]
            pst_score += PST[piece_type][sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            material -= PIECE_VALUES[piece_type]
            # invert for black perspective
            pst_score -= PST[piece_type][chess.square_mirror(sq)]

    # mobility: number of legal moves (small weight)
    mobility = len(list(board.legal_moves))
    # flip turn to measure mobility of current player to move
    # but we want static evaluation: advantage to white
    # compute white moves - black moves
    board_copy = board.copy()
    board_copy.turn = chess.WHITE
    white_moves = len(list(board_copy.legal_moves))
    board_copy.turn = chess.BLACK
    black_moves = len(list(board_copy.legal_moves))
    mobility_score = (white_moves - black_moves) * 10

    score = material + pst_score + mobility_score
    return score

# Minimax with alpha-beta and optional move ordering
def alphabeta(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move = None
    # Move ordering: try captures first (improves pruning)
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: 0 if board.is_capture(m) else 1)

    if maximizing:
        max_eval = -math.inf
        for move in moves:
            board.push(move)
            eval_score, _ = alphabeta(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            board.push(move)
            eval_score, _ = alphabeta(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def select_move(board: chess.Board, depth: int):
    # If few legal moves, just pick the only one
    eval_score, move = alphabeta(board, depth, -math.inf, math.inf, board.turn == chess.WHITE)
    return move, eval_score

def play_console():
    board = chess.Board()
    print("Start game. You play White (or press 'c' to let engine play both sides).")
    human_plays_white = input("Play as white? (y/n) [default y]: ").strip().lower() != 'n'
    engine_depth = int(input("Engine search depth (e.g., 3 or 4): ") or "3")

    while not board.is_game_over():
        print(board)
        if (board.turn == chess.WHITE and human_plays_white) or (board.turn == chess.BLACK and not human_plays_white):
            # Human move
            move_uci = input("Your move in UCI (e.g. e2e4) or 'quit': ").strip()
            if move_uci == 'quit':
                break
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
            except Exception as e:
                print("Invalid format. Example: e2e4, g1f3, e7e8q (promotion). Error:", e)
        else:
            # Engine move
            print("Engine thinking...")
            start = time.time()
            move, score = select_move(board, engine_depth)
            end = time.time()
            if move is None:
                print("No move found.")
                break
            board.push(move)
            print(f"Engine ({'White' if not human_plays_white else 'Black'}) played: {move}  eval={score}  time={end-start:.2f}s")

    print("Game over:", board.result())
    print(board)

if __name__ == '__main__':
    play_console()
