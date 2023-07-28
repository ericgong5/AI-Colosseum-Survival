# Mini Max agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
from copy import deepcopy
from random import randint, choice
from math import log, sqrt
from datetime import datetime, timedelta

@register_agent("mini_max_agent")
class MiniMaxAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MiniMaxAgent, self).__init__()
        self.name = "MiniMaxAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.mini_max = None # initialize
        self.autoplay = True 


    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """


        """
        1) way to represent board and search space



        
        """
        
        self.mini_max = MiniMax(chess_board, my_pos, adv_pos, max_step)
                
                
        begin = datetime.utcnow()
        value,r,c,dir = self.mini_max.max()
        move = (r,c),dir
        end = datetime.utcnow()
        #my_pos = move[0]
        #dir = move[1]
        result = end - begin
        print(move)

        return move #my_pos, dir


        # dummy return
        



class MiniMax:
    """
    Minimax algorithm
    """
    def __init__(self, chess_board, my_pos, adv_pos, max_step, **kwargs):
        """
        Initializing
        """
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        sec = kwargs.get('time', 1.95)
        self.calculation_time = timedelta(seconds=sec)

        self.preprocessing = True 

        self.max_moves = kwargs.get('max_moves', 50)

        self.C = kwargs.get("C", 1.4)

        self.wins = {}
        self.plays = {}
        self.movecount = 0 

    def get_play(self, board, my_pos, adv_pos, max_step):
        """
        Calculate best move for the current game state
        and returns move as ((x,y), dir)
        """
        self.chess_board = board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        player = True
        self.movecount += 1

        legal_moves = self.get_moves(self.chess_board, self.my_pos, self.adv_pos, self.max_step) # get list of legal random moves

        begin = datetime.utcnow()

        if self.preprocessing:
            time = timedelta(seconds = 29.95)
            self.preprocessing = False
        else:
            time = self.calculation_time

        while datetime.utcnow() - begin < time:
            self.run_sim()

        moves_states = [(play, (play, self.movecount)) for play in legal_moves] # moves and states in one list

        move, S = moves_states[0]
        percent = self.wins.get((player, S), 0)/self.plays.get((player, S), 1)

        for p, S in moves_states[1:]:
            if self.wins.get((player, S), 0)/self.plays.get((player, S), 1) > percent:
                move = p
                percent = self.wins.get((player, S), 0)/self.plays.get((player, S), 1)
        
        self.movecount += 1
        return move


    def check_endgame(self, chess_board, my_pos, adv_pos):
            """
            Check if the game ends and compute the current score of the agents.
            Returns
            -------
            is_endgame : bool
                Whether the game ends.
            player_1_score : int
                The score of player 1.
            player_2_score : int
                The score of player 2.
            """
            board_size = chess_board.shape[0]
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

            # Union-Find
            father = dict()
            for r in range(board_size):
                for c in range(board_size):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(board_size):
                for c in range(board_size):
                    for dir, move in enumerate(
                        moves[1:3]
                    ):  # Only check down and right
                        if chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(board_size):
                for c in range(board_size):
                    find((r, c))

            # p0_r is for my_pos and p1_r is for adv_pos
            p0_r = find(tuple(my_pos))
            p1_r = find(tuple(adv_pos))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
          
            if p0_r == p1_r:
                return False, p0_score, p1_score

            return True, p0_score, p1_score
    


    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """

        adv_pos = self.adv_pos

        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        if start_pos[0] == end_pos[0] and start_pos[1] == end_pos[1]:
            return True

        # Get position of the adversary
        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue
                

                next_pos = (cur_pos[0] + move[0],cur_pos[1] + move[1])
                if (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]) or tuple(next_pos) in visited:
                    continue
                if (next_pos[0] == adv_pos[0] and next_pos[1] == adv_pos[1]):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    # Player 'O' is max, in this case AI
    def max(self):

        # Possible values for maxv are:
        # -1 - loss
        # 0  - a tie
        # 1  - win

        # We're initially setting it to -2 as worse than the worst case:
        maxv = -2
        player_r = None
        player_c = None
        player_dir = None
        result = self.check_endgame(self.chess_board,self.my_pos,self.adv_pos)

        # If the game came to an end, the function needs to return
        # the evaluation function of the end. That can be:
        # -1 - loss
        # 0  - a tie
        # 1  - win
        if result[0] == True and result[2] > result[1]:
            return (-1, 0, 0)
        elif result[0] == True and result[2] < result[1]:
            return (1, 0, 0)
        elif result[0] == True and result[2] == result[1]:
            return (0, 0, 0)
        list_of_valid_moves = self.get_moves(self.chess_board,self.my_pos,self.adv_pos,self.max_step)
        print(list_of_valid_moves)
        for i in list_of_valid_moves:
                    # On the empty field player 'O' makes a move and calls Min
                    # That's one branch of the game tree.
                    prev_pos = self.my_pos
                    (r,c),dir = i
                    self.chess_board[r,c,dir] = True
                    self.my_pos = (r,c)
                    (m, min_r, min_c,min_dir) = self.min()
                    # Fixing the maxv value if needed
                    if m >= maxv:
                        maxv = m
                        player_r = r
                        player_c = c
                        player_dir = dir
                    # Setting back the field to empty
                    self.chess_board[r,c,dir] = False
                    self.my_pos = prev_pos
        print("max move is: " + " " + str(maxv) + " " + str(player_r) + " " + str(player_c) + " " + str(player_dir))
        return (maxv, player_r, player_c,player_dir)

    # Player 'X' is min, in this case human
    def min(self):

        # Possible values for minv are:
        # -1 - win
        # 0  - a tie
        # 1  - loss

        # We're initially setting it to 2 as worse than the worst case:
        minv = 2

        player_r = None
        player_c = None
        player_dir = None
        result = self.check_endgame(self.chess_board,self.adv_pos,self.my_pos)

        if result[0] == True and result[2] > result[1]:
            return (-1, 0, 0)
        elif result[0] == True and result[2] < result[1]:
            return (1, 0, 0)
        elif result[0] == True and result[2] == result[1]:
            return (0, 0, 0)
        list_of_valid_moves = self.get_moves(self.chess_board,self.adv_pos,self.my_pos,self.max_step)
        for i in list_of_valid_moves:
                    # On the empty field player 'O' makes a move and calls Min
                    # That's one branch of the game tree.
                    prev_pos = self.adv_pos
                    (r,c),dir = i
                    self.chess_board[r,c,dir] = True
                    self.adv_pos = (r,c)
                    (m, max_r, max_c,max_dir) = self.max()
                    # Fixing the maxv value if needed
                    if m <= minv:
                        minv = m
                        player_r = r
                        player_c = c
                        player_dir = dir
                    # Setting back the field to empty
                    self.chess_board[r,c,dir] = False
                    self.adv_pos = prev_pos
        print("min move is: " + str(minv) + " " + str(player_r) + " " + str(player_c) + " " + str(player_dir))

        return (minv, player_r, player_c,player_dir)

    #this dun work, moves arent being addeds
    def get_moves(self, chess_board, my_pos, adv_pos, max_step):
            """
            returns a list of all possible moves at a certain turn

            """
            r,c = my_pos
            moves = set()
            # set up a k+1 by k+1 square over my_pos and eliminate invalid moves
            for i in range(max_step*2,-1,-1):
                for j in range(max_step*2,-1,-1):
                    r_offset = i - max_step
                    c_offset = j - max_step
                    potential_r = r - r_offset
                    potential_c = c - c_offset
                    # make sure its in the board
                    in_grid = (potential_r >= 0) and (potential_c >= 0) and (potential_r < len(chess_board[0])) and (potential_c < len(chess_board[0]))
                    new_pos = (potential_r,potential_c)
                    if(in_grid):
                        #checking for every barrier direction if move is possible
                        for i in range(4):
                            if(chess_board[potential_r,potential_c,i] == False):
                                if(self.check_valid_step(my_pos, new_pos,i)):
                                    moves.add(((potential_r,potential_c),i))

            return list(moves)
 