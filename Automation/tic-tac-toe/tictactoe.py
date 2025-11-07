import random

class TicTacToe:
    def __init__(self):
        # Initialize the board (0 = empty, 1 = player, 2 = computer)
        self.board = [0] * 9
        self.human = 1
        self.computer = 2
        
    def print_board(self):
        """Display the current game board"""
        print("\n")
        for i in range(3):
            row = []
            for j in range(3):
                cell = self.board[i * 3 + j]
                if cell == 0:
                    row.append(" ")  # Show empty space
                elif cell == 1:
                    row.append("X")
                else:
                    row.append("O")
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("-----------")
        print("\n")
    
    def is_winner(self, player):
        """Check if the specified player has won"""
        # All winning combinations
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]               # Diagonals
        ]
        
        for combo in winning_combos:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    
    def is_board_full(self):
        """Check if the board is full"""
        return all(cell != 0 for cell in self.board)
    
    def get_available_moves(self):
        """Return list of available move positions"""
        return [i for i in range(9) if self.board[i] == 0]
    
    def evaluate(self):
        """Evaluate the board state for minimax"""
        if self.is_winner(self.computer):
            return 10
        elif self.is_winner(self.human):
            return -10
        else:
            return 0
    
    def minimax(self, depth, is_maximizing):
        """Minimax algorithm to find the best move"""
        score = self.evaluate()
        
        # Terminal states
        if score == 10:
            return score - depth
        if score == -10:
            return score + depth
        if self.is_board_full():
            return 0
        
        if is_maximizing:
            best_score = -float('inf')
            for move in self.get_available_moves():
                self.board[move] = self.computer
                score = self.minimax(depth + 1, False)
                self.board[move] = 0
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_available_moves():
                self.board[move] = self.human
                score = self.minimax(depth + 1, True)
                self.board[move] = 0
                best_score = min(score, best_score)
            return best_score
    
    def get_best_move(self):
        """Find the best move for the computer using minimax"""
        best_score = -float('inf')
        best_move = None
        
        for move in self.get_available_moves():
            self.board[move] = self.computer
            score = self.minimax(0, False)
            self.board[move] = 0
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def player_move(self):
        """Get player's move"""
        while True:
            try:
                move = int(input("Enter your move (1-9): ")) - 1
                if 0 <= move <= 8 and self.board[move] == 0:
                    self.board[move] = self.human
                    print(f"You placed X at position {move + 1}")
                    break
                else:
                    print("That position is already taken or invalid. Try again!")
            except ValueError:
                print("Please enter a number between 1 and 9.")
    
    def computer_move(self):
        """Make the computer's move"""
        available = self.get_available_moves()
        
        if not available:
            return
        
        # Use minimax algorithm to get best move
        move = self.get_best_move()
        self.board[move] = self.computer
        print(f"Computer placed O at position {move + 1}")
    
    def play(self):
        """Main game loop"""
        print("=" * 40)
        print("Welcome to Tic Tac Toe!")
        print("=" * 40)
        print("\nYou are X, Computer is O")
        print("Positions are numbered 1-9:")
        print(" 1 | 2 | 3 ")
        print("-----------")
        print(" 4 | 5 | 6 ")
        print("-----------")
        print(" 7 | 8 | 9 ")
        
        # Game loop
        while True:
            self.print_board()
            
            # Player's turn
            self.player_move()
            self.print_board()
            
            if self.is_winner(self.human):
                print("🎉 You won! Congratulations!")
                break
            
            if self.is_board_full():
                print("It's a draw!")
                break
            
            # Computer's turn
            print("Computer is thinking...")
            self.computer_move()
            
            if self.is_winner(self.computer):
                print("💻 Computer won! Better luck next time!")
                break
            
            if self.is_board_full():
                print("It's a draw!")
                break


def main():
    while True:
        game = TicTacToe()
        game.play()
        
        play_again = input("\nDo you want to play again? (yes/no): ").lower().strip()
        if play_again != "yes" and play_again != "y":
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
