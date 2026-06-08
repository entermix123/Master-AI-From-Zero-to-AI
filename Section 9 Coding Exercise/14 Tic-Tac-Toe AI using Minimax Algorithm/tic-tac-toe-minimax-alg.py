# Define a board as a list
## The tic tac toe board is represented as 1D list of nine elements. Index is 0 to 8. Each element starts as an empty space with double quotes or quotes spaced to indicate that the slot is available for a move.
board = [' ' for _ in range(9)] # 3x3 Tic_tac_toe board represented as a 1D list

# Function to print the tic tac toe board.
## This function prints the board in a user friendly format. It divides the 1D list into three rows, each containing three elements, and prints them as rows separated by vertical bars to mimic a tic tac tic tac toe board. So that's why here from I'm going from each element and printing those pipes to separate them as elements.
def print_board(board):
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print('| ' + ' | '.join(row) + ' |')

# Function to check if there is a winner
def check_winner(board, player):
    ## Those are our win conditions
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

# Function to check if the board is full
def is_board_full(board):
    return ' ' not in board

# Function to evaluate the board for minimax algorithm
## This evaluate function assigns a score based on the current board state one is the one. If the I which is zero, wins minus one if the human player which is X wins, and zero if the game is still ongoing and ends in a draw.
def evaluate(board):
    if check_winner(board, 'O'):    # AI is 'O'
        return 1
    elif check_winner(board, 'X'):  # Human is 'X'
        return -1
    else:
        return 0

# Minimax function to calculate the best move for the AI.
def minimax(board, depth, is_maximizing):
    # evaluate the board
    score = evaluate(board)
    # if score is one or score equal to minus one or if score board is full like the board is full then we'll return the score
    if score == 1 or score == -1 or is_board_full(board):
        return score
    
    if is_maximizing:   # AI's turn
        best_score = -float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'  # AI makes the move
                best_score = max(best_score, minimax(board, depth + 1, False))
                board[i] = ' '  # Undo move
        return best_score

    else:       # Human move
        best_score = float('inf') 
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'  # Humam make move
                best_score = min(best_score, minimax(board, depth + 1, True))
                board[i] = ' '  # Undo move
        return best_score

# Function to find the best move for the AI
## This function loops over all the available positions on the board, simulates the I making a move at each position and uses the minimax algorithm to evaluate the move. It chooses the move with the highest value. Best move for the I in this case.
def find_best_move(board):
    best_value = -float('inf')
    best_move = -1

    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'     # AI makes move
            move_value = minimax(board, 0, False)
            board[i] = ' '     # Undo move)
            
            if move_value > best_value:
                best_value = move_value
                best_move = i
        
    return best_move

# Main game loop
## This is the main game loop where the human player enters their move by providing a number between 1 and 9 to correspond with the board positions. If the position is valid that is empty, the player makes their move X and the game checks if the player has won.
def play_game():
    while True:
        print_board(board)

        # Player move
        player_move = int(input("Enter your move (1-9): ")) - 1
        if board[player_move] != ' ':
            print("Invalid move! Try Again.")
        board[player_move] = 'X'

        # Check if player won
        if check_winner(board, 'X'):
            print_board[board]
            print("You win!")
            break

        # Check for a draw
        if is_board_full(board):
            print("Its a draw")
            break

        # AI move
        ai_move = find_best_move(board)
        board[ai_move] = 'O'

        # Check if AI won
        if check_winner(board, 'O'):
            print_board(board)
            print("AI wins!")
            break
            
        # Check for draw
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break

# Start the game
play_game()