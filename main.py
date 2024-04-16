import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer
import sys
from Bot import Bot

class MainWindow(QWidget):
    def __init__(self, board):
        super().__init__()

        self.setGeometry(100, 100, 400, 400)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, 400, 400)

        self.chessboard = board
        # self.chessboard = chess.Board()

        self.paintEvent(None)  # Initial painting

    def paintEvent(self, event):
        svg_bytes = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(svg_bytes)

    def set_board(self, board):
        self.chessboard = board
        self.update()  # Trigger repaint


if __name__ == "__main__":
    app = QApplication(sys.argv)
    board = chess.Board()
    # board = chess.Board("8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1")
    bot_white = Bot("MiniMax", depth=5, multiprocess=True)
    bot_black = Bot("MiniMax", depth=4)
    window = MainWindow(board)

    # Show the window
    window.show()

    # Create a QTimer object
    timer = QTimer()

    # Function to perform moves
    def perform_moves():
        # Call your MiniMax algorithm here to get the next move
        if board.turn == chess.WHITE:
            next_move = bot_white.get_move(board)
        else:
            next_move = bot_black.get_move(board)

        # Make the move
        board.push(next_move)
        window.set_board(board)

        # Check if the game is over
        if (next_move is None) or board.is_game_over():
            timer.stop()  # Stop the timer when all moves are done
            print("END")
            print(board.outcome().termination.name)
            board.reset()
            exit(1)

        # Start the timer for the next move with a delay of 800 milliseconds
        timer.start(800)

    # Connect the timer to the function to perform moves
    timer.timeout.connect(perform_moves)

    # Start performing moves after an initial delay of 800 milliseconds
    QTimer.singleShot(800, perform_moves)

    # Run the application
    sys.exit(app.exec())
