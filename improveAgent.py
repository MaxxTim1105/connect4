import math
import random
from multiprocessing import Pool
import time
import os
import pickle

# Cấu hình bàn cờ
ROWS, COLS = 6, 7
WIN_LENGTH = 4
TOTAL_BITS = (ROWS + 1) * COLS

# File pickle cho Opening Book
OPENING_BOOK_FILE = "opening_book.pkl"

# Hàm load Opening Book từ file, nếu chưa có thì trả về dict rỗng
def load_opening_book():
    if os.path.exists(OPENING_BOOK_FILE):
        with open(OPENING_BOOK_FILE, "rb") as f:
            return pickle.load(f)
    return {}

# Hàm lưu Opening Book xuống file
def save_opening_book(opening_book):
    with open(OPENING_BOOK_FILE, "wb") as f:
        pickle.dump(opening_book, f)

# Khởi tạo Opening Book (global)
OPENING_BOOK = load_opening_book()

# Tra cứu Opening Book dựa trên chuỗi các nước đi đã chơi
def opening_book_lookup(moves_sequence):
    key = tuple(moves_sequence)
    return OPENING_BOOK.get(key, None)

# Cập nhật Opening Book: nếu chuỗi nước đi chưa có, lưu nước đi tốt tìm được
def update_opening_book(moves_sequence, best_move):
    key = tuple(moves_sequence)
    if key not in OPENING_BOOK:
        OPENING_BOOK[key] = best_move
        save_opening_book(OPENING_BOOK)

# Lớp Bitboard
class Bitboard:
    def __init__(self):
        self.player = 0
        self.opponent = 0

    def copy(self):
        new_board = Bitboard()
        new_board.player = self.player
        new_board.opponent = self.opponent
        return new_board

# Tạo bàn cờ ban đầu
def create_board():
    return Bitboard()

# Sao chép Bitboard
def copy_board(bitboard):
    return bitboard.copy()

# Tìm các nước đi hợp lệ
def valid_moves(bitboard):
    moves = []
    for col in range(COLS):
        mask = ((1 << ROWS) - 1) << (col * (ROWS + 1))
        height = bin((bitboard.player | bitboard.opponent) & mask).count('1')
        if height < ROWS:
            moves.append(col)
    return moves

# Thêm quân vào Bitboard
def play_move(bitboard, col, is_player):
    mask = ((1 << ROWS) - 1) << (col * (ROWS + 1))
    height = bin((bitboard.player | bitboard.opponent) & mask).count('1')
    if height < ROWS:
        pos = 1 << (col * (ROWS + 1) + height)
        if is_player:
            bitboard.player |= pos
        else:
            bitboard.opponent |= pos
        return True
    return False

# Xóa quân khỏi Bitboard
def undo_move(bitboard, col, is_player):
    mask = ((1 << (ROWS + 1)) - 1) << (col * (ROWS + 1))
    height = bin((bitboard.player | bitboard.opponent) & mask).count('1') - 1
    if height >= 0:
        pos = 1 << (col * (ROWS + 1) + height)
        if is_player:
            bitboard.player &= ~pos
        else:
            bitboard.opponent &= ~pos

# Kiểm tra chiến thắng
def winning_move(bitboard, player_bits):
    # Hàng dọc
    for col in range(COLS):
        for row in range(ROWS - WIN_LENGTH + 1):
            mask = 0
            for i in range(WIN_LENGTH):
                pos = col * (ROWS + 1) + row + i
                mask |= 1 << pos
            if player_bits & mask == mask:
                return True

    # Hàng ngang
    for row in range(ROWS):
        for col in range(COLS - WIN_LENGTH + 1):
            mask = 0
            for i in range(WIN_LENGTH):
                pos = (col + i) * (ROWS + 1) + row
                mask |= 1 << pos
            if player_bits & mask == mask:
                return True

    # Chéo xuôi (dưới trái -> trên phải)
    for r in range(ROWS - WIN_LENGTH + 1):
        for c in range(COLS - WIN_LENGTH + 1):
            mask = 0
            for i in range(WIN_LENGTH):
                pos = (c + i) * (ROWS + 1) + (r + i)
                mask |= 1 << pos
            if player_bits & mask == mask:
                return True

    # Chéo ngược (trên trái -> dưới phải)
    for r in range(WIN_LENGTH - 1, ROWS):
        for c in range(COLS - WIN_LENGTH + 1):
            mask = 0
            for i in range(WIN_LENGTH):
                row = r - i
                col = c + i
                if row >= 0:
                    pos = col * (ROWS + 1) + row
                    mask |= 1 << pos
            if player_bits & mask == mask:
                return True

    return False

# Kiểm tra trạng thái kết thúc
def is_terminal(bitboard):
    return (winning_move(bitboard, bitboard.player) or 
            winning_move(bitboard, bitboard.opponent) or 
            len(valid_moves(bitboard)) == 0)

# Hàm đánh giá cửa sổ
def evaluate_window(bits, player_bits, opp_bits, window_size=WIN_LENGTH):
    score = 0
    player_count = bin(player_bits & bits).count('1')
    opp_count = bin(opp_bits & bits).count('1')
    empty_count = window_size - player_count - opp_count

    if player_count > 0 and opp_count > 0:
        return 0

    if player_count >= 4:
        return 100000
    elif player_count == 3 and empty_count == 1:
        return 100
    elif player_count == 2 and empty_count == 2:
        return 10
    elif player_count == 1 and empty_count == 3:
        return 1

    if opp_count >= 4:
        return -100000
    elif opp_count == 3 and empty_count == 1:
        return -120
    elif opp_count == 2 and empty_count == 2:
        return -15
    elif opp_count == 1 and empty_count == 3:
        return -2

    return score

# Hàm đánh giá toàn bộ Bitboard
def evaluate_board(bitboard, player):
    score = 0
    player_bits = bitboard.player if player == 1 else bitboard.opponent
    opp_bits = bitboard.opponent if player == 1 else bitboard.player

    # Bonus trung tâm (trọng số tăng lên để ưu tiên nước ở trung tâm)
    center_col = COLS // 2
    mask = sum(1 << (center_col * (ROWS + 1) + r) for r in range(ROWS))
    center_count = bin(player_bits & mask).count('1')
    score += center_count * 50

    # Hàng ngang
    for r in range(ROWS):
        for c in range(COLS - WIN_LENGTH + 1):
            mask = sum(1 << ((c + i) * (ROWS + 1) + r) for i in range(WIN_LENGTH))
            score += evaluate_window(mask, player_bits, opp_bits)

    # Hàng dọc
    for c in range(COLS):
        for r in range(ROWS - WIN_LENGTH + 1):
            mask = sum(1 << (c * (ROWS + 1) + r + i) for i in range(WIN_LENGTH))
            score += evaluate_window(mask, player_bits, opp_bits)

    # Chéo xuôi (trái xuống phải)
    for r in range(ROWS - WIN_LENGTH + 1):
        for c in range(COLS - WIN_LENGTH + 1):
            mask = sum(1 << ((c + i) * (ROWS + 1) + (r + i)) for i in range(WIN_LENGTH))
            score += evaluate_window(mask, player_bits, opp_bits)

    # Chéo ngược (trái lên phải)
    for r in range(WIN_LENGTH - 1, ROWS):
        for c in range(COLS - WIN_LENGTH + 1):
            mask = sum(1 << ((c + i) * (ROWS + 1) + (r - i)) for i in range(WIN_LENGTH))
            score += evaluate_window(mask, player_bits, opp_bits)

    return score

# Class Solver
class Solver:
    INVALID_MOVE = -999

    def __init__(self):
        self.transposition = {}
        self.zobrist_table = [[[random.getrandbits(64) for _ in range(2)]
                               for _ in range(COLS)] for _ in range(ROWS)]

    def board_to_zobrist(self, bitboard):
        key = 0
        for row in range(ROWS):
            for col in range(COLS):
                pos = col * (ROWS + 1) + row
                if bitboard.player & (1 << pos):
                    key ^= self.zobrist_table[row][col][0]
                elif bitboard.opponent & (1 << pos):
                    key ^= self.zobrist_table[row][col][1]
        return key

    def order_moves(self, bitboard, player):
        moves = []
        for col in range(COLS):
            if col in valid_moves(bitboard):
                board_copy = bitboard.copy()
                play_move(board_copy, col, player == 1)
                score = evaluate_board(board_copy, player)
                moves.append((col, score))
        moves.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in moves]

    def negamax(self, bitboard, depth, alpha, beta, player):
        key = self.board_to_zobrist(bitboard)
        if key in self.transposition and self.transposition[key]['depth'] >= depth:
            return self.transposition[key]['value']
        if depth == 0 or is_terminal(bitboard):
            return evaluate_board(bitboard, player)

        best_value = -math.inf
        moves = self.order_moves(bitboard, player)
        if not moves:
            return evaluate_board(bitboard, player)
        for col in moves:
            board_copy = bitboard.copy()
            play_move(board_copy, col, player == 1)
            value = -self.negamax(board_copy, depth - 1, -beta, -alpha, -player)
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        self.transposition[key] = {'value': best_value, 'depth': depth}
        return best_value

    def parallel_search(self, args):
        bitboard, player, depth, alpha, beta, col = args
        board_copy = bitboard.copy()
        play_move(board_copy, col, player == 1)
        value = -self.negamax(board_copy, depth - 1, -beta, -alpha, -player)
        return col, value

    def find_best_move(self, bitboard, player, depth):
        valid_cols = self.order_moves(bitboard, player)
        if not valid_cols:
            return None, 0

        with Pool() as pool:
            args = [(bitboard, player, depth, -math.inf, math.inf, col) for col in valid_cols]
            results = pool.map(self.parallel_search, args)

        best_move = None
        best_value = -math.inf
        for col, value in results:
            if value > best_value:
                best_value = value
                best_move = col
        return best_move, best_value

    def iterative_deepening(self, bitboard, player, max_depth, time_limit):
        # Nếu bàn cờ hiện tại đã có trong Opening Book thì trả về nước khai cuộc đã lưu
        # Ở đây ta dùng chuỗi nước đi là trạng thái ban đầu nếu board trống
        # Trong trường hợp board không rỗng, có thể bạn cần duy trì moves_history riêng
        if bitboard.player == 0 and bitboard.opponent == 0:
            ob_move = OPENING_BOOK.get(bitboard)
            if ob_move is not None:
                return ob_move, 100000

        best_move = None
        best_score = -math.inf
        start_time = time.time()
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
            current_move, current_score = self.find_best_move(bitboard, player, depth)
            if current_move is not None:
                best_move, best_score = current_move, current_score
            if abs(best_score) >= 100000:
                break
        return best_move, best_score

# Chuyển Bitboard sang mảng 2D để in
def bitboard_to_array(bitboard):
    board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    for row in range(ROWS):
        for col in range(COLS):
            pos = col * (ROWS + 1) + row
            if bitboard.player & (1 << pos):
                board[row][col] = 1
            elif bitboard.opponent & (1 << pos):
                board[row][col] = -1
    return board

# In bàn cờ
def print_board(bitboard):
    board = bitboard_to_array(bitboard)
    for row in reversed(board):
        print('|' + '|'.join('X' if cell == 1 else 'O' if cell == -1 else ' ' for cell in row) + '|')
    print(" " + " ".join(str(c) for c in range(COLS)))
    print()

# Chơi game
def main():
    board = create_board()
    moves_history = []  # Lưu lại chuỗi nước đi của ván đấu (dùng cho Opening Book nâng cao)
    print("Initial valid moves:", len(valid_moves(board)))  # Phải là 7
    solver = Solver()
    current_player = -1
    depth = 8
    print("Human vs AI: You are O, AI is X.")
    print_board(board)

    while True:
        if len(valid_moves(board)) == 0:
            print("Game is a draw!")
            break

        if current_player == 1:
            print("AI thinking...")
            start_time = time.time()
            # Nếu chuỗi nước đi đã chơi có trong Opening Book thì sử dụng ngay
            ob_move = opening_book_lookup(moves_history)
            if ob_move is not None:
                best_move = ob_move
                best_score = 100000
                print("Using Opening Book move!")
            else:
                best_move, best_score = solver.iterative_deepening(board, 1, depth, 2.5
                                                                  )
                # Cập nhật Opening Book với chuỗi nước đi hiện tại và nước đi tốt tìm được
                update_opening_book(moves_history, best_move)
            elapsed = time.time() - start_time
            if best_move is None:
                print("AI không tìm được nước đi hợp lệ!")
                break
            play_move(board, best_move, True)
            moves_history.append(best_move)
            print(f"AI move: {best_move} (took {elapsed:.2f}s, score={best_score})")
            print_board(board)
            if winning_move(board, board.player):
                print("AI wins!")
                break
        else:
            valid = valid_moves(board)
            move = None
            while move not in valid:
                try:
                    move = int(input(f"Your move (0-6) {valid}: "))
                    if move not in valid:
                        print("Invalid move! Choose a column from", valid)
                except ValueError:
                    print("Please enter an integer!")
            play_move(board, move, False)
            moves_history.append(move)
            print_board(board)
            if winning_move(board, board.opponent):
                print("Human wins!")
                break

        current_player = -current_player

if __name__ == "__main__":
    main()
