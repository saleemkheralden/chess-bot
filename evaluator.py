import chess

class Evaluator:

    def __init__(self):
        # Piece values for evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Precompute center squares bitboard
        self.center_bb = [chess.BB_D4, chess.BB_D5, chess.BB_E4, chess.BB_E5]

        # Initialize evaluation score
        self.development_bonus = 0.05
        self.checkmate_score = 1000000

        # Track repeated positions
        self.position_counts = {}

    def evaluate_position(self, position: chess.Board):
        score = 0

        # Check for checkmate
        if position.is_checkmate():
            return -self.checkmate_score if position.turn == chess.WHITE else self.checkmate_score
        
        # Check for repeated position
        position_hash = position.board_fen()
        self.position_counts[position_hash] = self.position_counts.get(position_hash, 0) + 1
        if self.position_counts[position_hash] >= 3:
            return 0  # Draw due to threefold repetition

        # Evaluate material balance
        for piece_type, value in self.piece_values.items():
            score += (len(position.pieces(piece_type, chess.WHITE)) - len(position.pieces(piece_type, chess.BLACK))) * value

        # Evaluate pawn structure
        score += self.evaluate_pawn_structure(position)

        # Evaluate center control
        score += self.evaluate_center_control(position)

        # Evaluate piece development
        score += self.evaluate_development_bonus(position)

        return score
    
    def evaluate_action(self, position: chess.Board, action: chess.Move):
        # Evaluate the position after making the move

        es = self.evaluate_position(position)
        # temp_board = position.copy(stack=False)
        position.push(action)
        es = self.evaluate_position(position) - es
        position.pop()
        return es

    def evaluate_pawn_structure(self, position: chess.Board):
        score = 0

        white_pawns = position.pieces(chess.PAWN, chess.WHITE)
        black_pawns = position.pieces(chess.PAWN, chess.BLACK)

        # Doubled pawn penalty
        
        white_doubled_pawns = sum(1 for square in white_pawns if self.is_doubled_pawns(position, square, chess.WHITE))
        black_doubled_pawns = sum(1 for square in black_pawns if self.is_doubled_pawns(position, square, chess.BLACK))

        # Double pawn using bitboard

        # white_pawn_bitboard = sum([(1 << e) for e in position.pieces(chess.PAWN, chess.WHITE)])
        # black_pawn_bitboard = sum([(1 << e) for e in position.pieces(chess.PAWN, chess.BLACK)])
        
        # white_doubled_pawns = 0
        # black_doubled_pawns = 0

        
        # mask = 1
        # for _ in range(7):
        #     mask = mask << 8
        #     mask = mask + 1

        # init_mask = mask

        # for _ in range(8):
        #     white_doubled_pawns = white_doubled_pawns + (mask & white_pawn_bitboard).bit_count() - 1
        #     black_doubled_pawns = black_doubled_pawns + (mask & black_pawn_bitboard).bit_count() - 1

        #     mask <<= 1

        
        score -= (white_doubled_pawns * 0.1)
        score += (black_doubled_pawns * 0.1)

        # Isolated pawn penalty
        white_isolated_pawns = sum(1 for square in white_pawns if self.is_isolated_pawn(position, square, chess.WHITE))
        black_isolated_pawns = sum(1 for square in black_pawns if self.is_isolated_pawn(position, square, chess.BLACK))

        # Isolated pawn bitboard
        # white_isolated_pawns = 0
        # black_isolated_pawns = 0

        # mask = init_mask
        # neighbor_mask = init_mask << 1

        # for i in range(8):
        #     if i == 1:
        #         neighbor_mask = neighbor_mask | (neighbor_mask >> 2)
        #     if i == 7:
        #         neighbor_mask = neighbor_mask & (init_mask << 6)
            
        #     white_isolated_pawns = ((mask & white_pawn_bitboard).bit_count() - 1) if ((neighbor_mask & white_pawn_bitboard).bit_count() == 0) else 0
        #     black_isolated_pawns = ((mask & black_pawn_bitboard).bit_count() - 1) if ((neighbor_mask & black_pawn_bitboard).bit_count() == 0) else 0

        #     mask <<= 1
        #     neighbor_mask <<= 1

        score -= (white_isolated_pawns * 0.1)
        score += (black_isolated_pawns * 0.1)

        return score

    def evaluate_center_control(self, position):
        score = 0
        for square in self.center_bb:
            piece = position.piece_at(square.bit_length() - 1)
            if piece is not None:
                if piece.color == chess.WHITE:
                    score += 0.1
                else:
                    score -= 0.1
        return score

    def evaluate_development_bonus(self, position):
        total_minor_pieces = len(position.pieces(chess.KNIGHT, chess.WHITE)) + len(position.pieces(chess.BISHOP, chess.WHITE)) + len(position.pieces(chess.KNIGHT, chess.BLACK)) + len(position.pieces(chess.BISHOP, chess.BLACK))
        if total_minor_pieces > 0:
            white_minor_pieces = len(position.pieces(chess.KNIGHT, chess.WHITE)) + len(position.pieces(chess.BISHOP, chess.WHITE))
            black_minor_pieces = len(position.pieces(chess.KNIGHT, chess.BLACK)) + len(position.pieces(chess.BISHOP, chess.BLACK))
            white_development = white_minor_pieces / total_minor_pieces
            black_development = black_minor_pieces / total_minor_pieces
            return (white_development - black_development) * self.development_bonus
        return 0

    def is_doubled_pawns(self, position, square, color):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        for r in range(1, rank + 1):
            if position.piece_type_at(chess.square(file, rank - r)) == chess.PAWN and \
                    position.color_at(chess.square(file, rank - r)) == color:
                return True

        return False

    def is_isolated_pawn(self, position, square, color):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        adjacent_files = [file - 1, file + 1]
        for f in adjacent_files:
            if 0 <= f <= 7:
                if position.piece_type_at(chess.square(f, rank)) == chess.PAWN and \
                        position.color_at(chess.square(f, rank)) == color:
                    return False

        return True


# class Evaluator:

#     def __init__(self):
#         # Piece values for evaluation
#         self.piece_values = {
#             chess.PAWN: 100,
#             chess.KNIGHT: 320,
#             chess.BISHOP: 330,
#             chess.ROOK: 500,
#             chess.QUEEN: 900,
#             chess.KING: 20000
#         }

#         # Bonus for controlling the center
#         self.center_bonus = 0.1

#         # Bonus for piece development
#         self.development_bonus = 0.05

#         # Penalty for doubled pawns
#         self.doubled_pawn_penalty = 0.1

#         # Penalty for isolated pawns
#         self.isolated_pawn_penalty = 0.1

#         # Bonus for king safety
#         self.king_safety_bonus = 0.05

#         # Checkmate score
#         self.checkmate_score = 1000000

#     def evaluate_position(self, position: chess.Board):
#         # Check for checkmate
#         if position.is_checkmate():
#             if position.turn == chess.WHITE:
#                 return -self.checkmate_score  # Black wins
#             else:
#                 return self.checkmate_score  # White wins

#         # Initialize evaluation score
#         score = 0

#         # Evaluate material balance
#         for piece_type in self.piece_values:
#             score += len(position.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
#             score -= len(position.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]

#         # Evaluate pawn structure
#         white_pawns = position.pieces(chess.PAWN, chess.WHITE)
#         black_pawns = position.pieces(chess.PAWN, chess.BLACK)

#         # Doubled pawn penalty
#         white_doubled_pawns = sum(1 for square in white_pawns if self.is_doubled_pawns(position, square, chess.WHITE))
#         black_doubled_pawns = sum(1 for square in black_pawns if self.is_doubled_pawns(position, square, chess.BLACK))
#         score -= (white_doubled_pawns * self.doubled_pawn_penalty)
#         score += (black_doubled_pawns * self.doubled_pawn_penalty)

#         # Isolated pawn penalty
#         white_isolated_pawns = sum(1 for square in white_pawns if self.is_isolated_pawn(position, square, chess.WHITE))
#         black_isolated_pawns = sum(1 for square in black_pawns if self.is_isolated_pawn(position, square, chess.BLACK))
#         score -= (white_isolated_pawns * self.isolated_pawn_penalty)
#         score += (black_isolated_pawns * self.isolated_pawn_penalty)

#         # Bonus for controlling the center
#         for square in chess.SQUARES:
#             if (1 << square & chess.BB_CENTER) != 0:
#                 piece = position.piece_at(square)
#                 if piece is not None:
#                     if piece.color == chess.WHITE:
#                         score += self.center_bonus
#                     else:
#                         score -= self.center_bonus

#         # Evaluate piece development
#         white_minor_pieces = len(position.pieces(chess.KNIGHT, chess.WHITE)) + len(position.pieces(chess.BISHOP, chess.WHITE))
#         black_minor_pieces = len(position.pieces(chess.KNIGHT, chess.BLACK)) + len(position.pieces(chess.BISHOP, chess.BLACK))
#         total_minor_pieces = white_minor_pieces + black_minor_pieces
#         if total_minor_pieces > 0:
#             white_development = white_minor_pieces / total_minor_pieces
#             black_development = black_minor_pieces / total_minor_pieces
#             score += (white_development - black_development) * self.development_bonus

#         # Evaluate king safety
#         white_king_square = position.king(chess.WHITE)
#         black_king_square = position.king(chess.BLACK)
#         if white_king_square and black_king_square:
#             white_king_safety = sum(1 for square in position.attacks(white_king_square) if position.piece_at(square) and position.piece_at(square).color == chess.BLACK)
#             black_king_safety = sum(1 for square in position.attacks(black_king_square) if position.piece_at(square) and position.piece_at(square).color == chess.WHITE)
#             score += (white_king_safety - black_king_safety) * self.king_safety_bonus

#         return score

#     def evaluate_action(self, position: chess.Board, action: chess.Move):
#         # Evaluate the position after making the move
#         temp_board = position.copy()
#         temp_board.push(action)
#         return self.evaluate_position(temp_board)
    

#     def is_doubled_pawns(self, board: chess.Board, square: chess.Square, color: chess.Color):
#         """
#         Checks if there are doubled pawns of the given color on the same file as the given square.

#         Args:
#         - board: The chess board.
#         - square: The square to check.
#         - color: The color of the pawns to check (chess.WHITE or chess.BLACK).

#         Returns:
#         - True if there are doubled pawns, False otherwise.
#         """
#         file = chess.square_file(square)
#         rank = chess.square_rank(square)

#         for r in range(1, rank + 1):
#             if board.piece_type_at(chess.square(file, rank - r)) == chess.PAWN and \
#                     board.color_at(chess.square(file, rank - r)) == color:
#                 return True

#         return False

#     def is_isolated_pawn(self, board: chess.Board, square: chess.Square, color: chess.Color):
#         """
#         Checks if the pawn on the given square is isolated (i.e., no friendly pawns on adjacent files).

#         Args:
#         - board: The chess board.
#         - square: The square to check.
#         - color: The color of the pawn to check (chess.WHITE or chess.BLACK).

#         Returns:
#         - True if the pawn is isolated, False otherwise.
#         """
#         file = chess.square_file(square)
#         rank = chess.square_rank(square)

#         # Check adjacent files
#         adjacent_files = [file - 1, file + 1]
#         for f in adjacent_files:
#             if 0 <= f <= 7:
#                 if board.piece_type_at(chess.square(f, rank)) == chess.PAWN and \
#                         board.color_at(chess.square(f, rank)) == color:
#                     return False  # Adjacent pawn found, not isolated

#         return True  # No adjacent friendly pawns found, pawn is isolated



