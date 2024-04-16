import chess as c
import matplotlib.pyplot as plt
import time, random

class Chess:
    def __init__(self):
        self.UNICODE_PIECE_SYMBOLS = {
            "r": "♖", "R": "♜",
            "n": "♘", "N": "♞",
            "b": "♗", "B": "♝",
            "q": "♕", "Q": "♛",
            "k": "♔", "K": "♚",
            "p": "♙", "P": "♟",
        }

        self.board = c.Board()


    def print_board(self):
        s = self.board.__str__()
        for k, v in self.UNICODE_PIECE_SYMBOLS.items():
            s = s.replace(k, v)

        s = s.split('\n')
        s = [f"{8 - i} {e}" for i, e in enumerate(s)]
        s.append('  ' + ' '.join([chr(e) for e in range(ord('a'), ord('a') + 8)]))
        print('\n'.join(s))
        print()

    def push(self, move):
        if not move in self.board.legal_moves:
            print("No good")
            return
        self.board.push(move)

    def get_random_move(self):
        if self.board.is_game_over():
            return
        
        moves = [e for e in self.board.legal_moves]
        return random.choice(moves)
    
    def play_random(self):
        self.print_board()

        while not self.board.is_game_over():
            moves = [e for e in self.board.legal_moves]
            self.push(random.choice(moves))
            self.print_board()

            time.sleep(0.5)

        print(self.board.outcome().termination.name)
        self.board.reset()



# board = Chess()

# board.play_random()

# Nf3 = c.Move.from_uci("a2a4")
# board.push(Nf3)
# board.push(Nf3)

# board.print_board()

# print(board.board.legal_moves)
# Nf3 = c.Move.from_uci("a2a4")
# board.push(Nf3)
# print(board.board.legal_moves)

# print()
