import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer
import sys
from Bot import Bot
from time import sleep

class MainWindow(QWidget):
    def __init__(self, board):
        super().__init__()

        self.setGeometry(100, 100, 400, 450)  # Increased height to accommodate additional labels

        layout = QVBoxLayout(self)
        self.widgetSvg = QSvgWidget(parent=self)
        layout.addWidget(self.widgetSvg)

        # Labels for displaying turn and score
        self.label_turn = QLabel("Turn: White", parent=self)
        layout.addWidget(self.label_turn)

        self.label_score = QLabel("Score: White - 0, Black - 0", parent=self)
        layout.addWidget(self.label_score)

        self.setLayout(layout)

        self.chessboard = board
        self.paintEvent(None)  # Initial painting

    def paintEvent(self, event):
        svg_bytes = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(svg_bytes)

    def set_board(self, board):
        self.chessboard = board
        self.update()  # Trigger repaint

    def update_turn_label(self, turn):
        self.label_turn.setText(f"Turn: {turn}")

    def update_score_label(self, white_score, black_score):
        self.label_score.setText(f"Score: White - {white_score}, Black - {black_score}")


if __name__ == "__main__":
    app = QApplication([])
    board = chess.Board()
    print(sys.argv)
    alg1 = alg2 = "MiniMax"
    depth1 = depth2 = 2
    flag1 = flag2 = False
    if len(sys.argv) > 1:
        if len(sys.argv) == 3:
            alg1, alg2 = *sys.argv[1:],
        elif len(sys.argv) == 4:
            alg1, alg2, depth1 = *sys.argv[1:],
        elif len(sys.argv) == 5:
            alg1, alg2, depth1, depth2 = *sys.argv[1:],
        elif len(sys.argv) == 6:
            alg1, alg2, depth1, depth2, flag1 = *sys.argv[1:],
        elif len(sys.argv) == 7:
            alg1, alg2, depth1, depth2, flag1, flag2 = *sys.argv[1:],
    
    print(alg1, alg2, depth1, depth2, flag1, flag2)

    bot_white = Bot(alg1, depth=int(depth1), multiprocess=bool(flag1), color=chess.WHITE, play_by_book=True)
    bot_black = Bot(alg2, depth=int(depth2), multiprocess=bool(flag2), color=chess.BLACK, play_by_book=True)
    window = MainWindow(board)

    # Show the window
    window.show()

    # Create a QTimer object
    timer = QTimer()

    # Function to perform moves
    def perform_moves():
        # Call your MiniMax algorithm here to get the next move
        if board.turn == chess.WHITE:
            window.update_turn_label("Black (Bot thinking...)")
            next_move = bot_white.get_move(board)
        else:
            window.update_turn_label("White (Bot thinking...)")
            next_move = bot_black.get_move(board)

        # Make the move
        board.push(next_move)
        window.set_board(board)

        # Update scores
        white_score = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.WHITE)) + \
                      len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE)) + \
                      len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.KING, chess.WHITE))

        black_score = len(board.pieces(chess.PAWN, chess.BLACK)) + len(board.pieces(chess.ROOK, chess.BLACK)) + \
                      len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                      len(board.pieces(chess.QUEEN, chess.BLACK)) + len(board.pieces(chess.KING, chess.BLACK))

        window.update_score_label(white_score, black_score)

        # Check if the game is over
        if (next_move is None) or board.is_game_over():
            timer.stop()  # Stop the timer when all moves are done
            print("END")
            print(board.outcome())
            window.set_board(board)
            sleep(2)
            board.reset()
            exit(1)

        # Start the timer for the next move with a delay of 800 milliseconds
        timer.start(1)
        # timer.start(800)

    # Connect the timer to the function to perform moves
    timer.timeout.connect(perform_moves)

    # Start performing moves after an initial delay of 800 milliseconds
    QTimer.singleShot(800, perform_moves)

    # Run the application
    sys.exit(app.exec())
