import random
import chess
from evaluator import Evaluator
import multiprocessing
from enum import Enum
import time
from typing import List, Tuple, Dict
import math
import chess.pgn as cpgn
import threading

# write chess bot, with the three algorithms in the algos
class flag(Enum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

class Bot:
    def __init__(self, algo="Random", **args) -> None:
        algos = ["MiniMax", "MCTS", "Random"]
        self.model = None
        self.color = args['color']
        self.play_by_book = args.get('play_by_book', False)
        self.opening_book = self.load_opening_book('kasparov-deep-blue-1997.pgn')
        self.book_index = 0
        self.book_thresh = 6

        if algo not in algos:
            print(f"algo param should be one of {algos}")
            exit(0)

        if algo == "Random":
            self.model = self.random_move
        else:
            self.model = eval(algo)(**args)
                
    def random_move(self, board):
        moves = [e for e in board.legal_moves]
        if len(moves) == 0:
            return
        return random.choice(moves)
    
    def get_move(self, board: chess.Board):

        if self.play_by_book and (self.book_index < self.book_thresh):
            # Check if the current position exists in the opening book
            book_move = self.get_book_move(board)
            if book_move is not None:
                self.book_index += 1
                return book_move
        
        # If no book move, use the selected algorithm
        return self.model(board)
    
    def load_opening_book(self, pgn_file):
        opening_book = {}
        with open(pgn_file, 'r') as f:
            while True:
                game = cpgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    fen = board.fen()
                    if fen not in opening_book:
                        opening_book[fen] = []
                    opening_book[fen].append(move)
                    board.push(move)
        return opening_book

    def get_book_move(self, board):
        # Check if the current position exists in the opening book
        fen = board.fen()
        if fen in self.opening_book:
            # If the position is found in the opening book, return one of the book moves
            return random.choice(self.opening_book[fen])
        else:
            # If the position is not in the opening book, return None
            return None

class MiniMax:
    def __init__(self, **args) -> None:
        self.depth = args.get('depth', 3)
        self.num_processes = args.get('num_processes', 4)
        self.multiprocess = args.get('multiprocess', False)

        self.evaluator = Evaluator()

        self.transpositionTableLookup: Dict[str, Entry] = {}

    def __call__(self, board: chess.Board):
        ts = time.time()

        self.cells = 0

        allMoves = [e for e in board.legal_moves]
        bestMove = None
        bestEvaluation = float('-inf')

        if self.multiprocess:
            # Divide the moves into chunks for parallel processing
            move_chunks = [allMoves[i::self.num_processes] for i in range(self.num_processes)]

            # Create a multiprocessing Pool
            pool = multiprocessing.Pool(processes=self.num_processes)

            # Evaluate moves in parallel
            evaluations = pool.map(self.evaluate_move, [(board.copy(), move_chunk) for move_chunk in move_chunks])

            # Close the pool to release resources
            pool.close()
            pool.join()

            # Find the best move among evaluated moves
            print(evaluations)
            for move, evaluation in evaluations:
                if evaluation > bestEvaluation:
                    bestMove = move
                    bestEvaluation = evaluation

            # print(f"search {self.cells} nodes")
            print(f"Choose move {bestMove} with eval {bestEvaluation}")
            print(f"calculation of position took {time.time() - ts} seconds")
            print()
            
            return bestMove

        allMoves = sorted(allMoves, key=lambda x: self.evaluator.evaluate_action(board, x), reverse=True)

        for move in allMoves:
            self.cells += 1

            board.push(move)
            evaluateMove = -self.negamax(board.copy(stack=False), self.depth - 1, float('-inf'), float('inf'), color=1 if board.turn == chess.WHITE else -1)
            board.pop()

            print(move, evaluateMove)

            if evaluateMove > bestEvaluation:
                bestMove = move
                bestEvaluation = evaluateMove
        
        print(f"Choose move {bestMove} with eval {bestEvaluation}")
        print(f"Searched {self.cells} nodes")
        print(f"Calculation of position took {time.time() - ts} seconds")
        print()

        return bestMove
    
    def evaluate_move(self, args):
        board, move_chunk = args
        bestMove = None
        bestEvaluation = float('-inf')

        move_chunk = sorted(move_chunk, key=lambda x: self.evaluator.evaluate_action(board, x), reverse=True)

        for move in move_chunk:
            board.push(move)
            evaluateMove = -self.negamax(board.copy(stack=False), self.depth - 1, float('-inf'), float('inf'), color=1 if board.turn == chess.WHITE else -1)
            board.pop()

            print(move, evaluateMove)

            if evaluateMove > bestEvaluation:
                bestMove = move
                bestEvaluation = evaluateMove

        return bestMove, bestEvaluation
    
    def evaluate_action(self, board: chess.Board, move: chess.Move, time_threshold=1/64):
        ts = time.time()
        move_eval = 0
        board = board.copy(stack=False)

        # for depth in range(self.depth):
        #     if time.time() - ts > time_threshold:
        #         break
        #     move_eval = self.negamax(board, depth, float('-inf'), float('inf'), color=1 if board.turn == chess.WHITE else -1)
        
        # return move_eval
            

        ttEntry = self.transpositionTableLookup.get(board.fen())
        if ttEntry is None:
            es = self.evaluator.evaluate_position(board)
        else:
            es = ttEntry.value

        board.push(move)

        ttEntry = self.transpositionTableLookup.get(board.fen())
        if ttEntry is None:
            es = self.evaluator.evaluate_position(board) - es
        else:
            es = ttEntry.value - es

        print(f"time of move {move}: {time.time() - ts}")
        return es


        
    
    def negamax(self, board: chess.Board, depth, alpha, beta, color):
        alphaOrig = alpha

        ttEntry = self.transpositionTableLookup.get(board.fen())
        if (ttEntry is not None) and ttEntry.depth >= depth:
            if ttEntry.flag == flag.EXACT:
                return ttEntry.value
            elif ttEntry.flag == flag.LOWERBOUND:
                alpha = max(alpha, ttEntry.value)
            elif ttEntry.flag == flag.UPPERBOUND:
                beta = min(beta, ttEntry.value)

            if alpha >= beta:
                return ttEntry.value


        if (depth == 0) or board.is_game_over():
            return color * self.evaluator.evaluate_position(board)
            # return color * board.is_checkmate()
        value = float('-inf')

        # allMoves = sorted(board.legal_moves, key=lambda x: self.evaluator.evaluate_action(board, x), reverse=True)
        allMoves = board.legal_moves
        
        for move in allMoves:
            self.cells += 1

            board.push(move)
            value = max(value, -self.negamax(board, depth - 1, -beta, -alpha, -color))
            alpha = max(alpha, value)

            board.pop()

            if alpha >= beta:
                break

        if ttEntry is None:
            ttEntry = Entry()

        ttEntry.value = value
        if value <= alphaOrig:
            ttEntry.flag = flag.UPPERBOUND
        elif value >= beta:
            ttEntry.flag = flag.LOWERBOUND
        else:
            ttEntry.flag = flag.EXACT
        
        ttEntry.depth = depth
        ttEntry.is_valid = True
        self.transpositionTableLookup[board.fen()] = ttEntry

        return value

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state: chess.Board = state
        self.parent: chess.Board = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children: List[Node] = []
        self.expanded = False

    def all_children_visited(self):
        return all(e.visits > 0 for e in self.children)

    def add_child(self, child_state, move):
        child = Node(child_state, self, move)
        self.children.append(child)
        return child

    def select_child(self):
        return max(self.children, key=lambda child: child.uct_value())

    def expand(self, moves):
        if self.expanded:
            return

        for move in moves:
            node = self.state.copy(stack=False)
            node.push(move)
            self.add_child(node, move)

        self.expanded = True

    def update(self, result: chess.Color):
        self.visits += 1

        if result is None:
            a = 0.5
        else:
            a = 1 if result == self.state.turn else 0

        self.wins += a

    def uct_value(self) -> float:
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)


class MCTS:
    def __init__(self, **args):
        self.nodes: Dict[str, Node] = {}
        self.evaluator = Evaluator()
        self.thread_flag = False

    def select(self, node: Node):
        current_node = node
        while (len(current_node.children) != 0) and (current_node.all_children_visited()):
            current_node = current_node.select_child()
        return current_node
    
    def expand(self, node: Node):
        if node.expanded:
            return
        
        node.expand(node.state.legal_moves)
        
        for e in node.children:
            self.nodes[e.state.fen()] = e

    def simulate(self, node: chess.Board) -> chess.Color:
        """
        Preforms a random simulation
        """
        if node.is_game_over():
            return node.outcome().winner
        
        moves = [e for e in node.legal_moves]

        move = max(moves, key=lambda x: self.evaluator.evaluate_action(node, x))
        node.push(move)

        # node.push(random.choice(moves))  # FUCK light playouts

        return self.simulate(node)

    def backpropagate(self, node: Node, result: chess.Color):
        while node is not None:
            node.update(result)
            node = node.parent

    def mcts(self, root, iterations, thread=False):
        if thread and self.thread_flag:
            return
        
        if thread:
            self.thread_flag = True

        for iteration in range(iterations):
            print(iteration, iterations)
            node = self.select(root)
            if node.state.is_game_over():
                self.backpropagate(node, node.state.outcome().winner)
            else:
                self.expand(node)
                result = self.simulate(node.state.copy(stack=False))
                self.backpropagate(node, result)
        
        self.thread_flag = False

    def __call__(self, state: chess.Board, iterations=10) -> Node:
        if state.is_game_over():
            return None

        root = self.nodes.get(state.fen(), None)
        if root is None:
            self.nodes[state.fen()] = Node(state)
            root = self.nodes[state.fen()]
        
        self.mcts(root, iterations)
        move = max(root.children, key=lambda child: child.uct_value()).move

        # if not self.thread_flag:
        #     thread = threading.Thread(target=self.mcts, args=(root, 1000, True))
        #     thread.start()
        # thread.join()

        return move # No exploration
        # return max(root.children, key=lambda child: child.wins / child.visits).move  # No exploration


class Entry:
    def __init__(self):
        self.value = None
        self.flag = None
        self.depth = None
        self.is_valid = None

