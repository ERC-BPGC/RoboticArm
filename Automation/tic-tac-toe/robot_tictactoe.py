import time
from collections import deque
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


def replay_episode(base_path, record_id, episode_idx, port="/dev/ttyACM1"):
    """
    Replay a robot episode from a local dataset.
    
    Args:
        base_path (str): Base path to lerobot datasets directory
        record_id (int): Record number (1-9)
        episode_idx (int): Episode ID (0-3)
        port (str): Serial port for robot connection
    
    Returns:
        bool: True if replay completed successfully, False otherwise
    
    Raises:
        ValueError: If episode_idx or record_id is not in valid range or dataset_path doesn't exist
        Exception: If robot connection or dataset loading fails
    """
    # Validate inputs
    if not isinstance(record_id, int) or record_id < 1 or record_id > 9:
        raise ValueError(f"record_id must be an integer between 1 and 9, got {record_id}")
    
    if not isinstance(episode_idx, int) or episode_idx < 0 or episode_idx > 3:
        raise ValueError(f"episode_idx must be an integer between 0 and 3, got {episode_idx}")
    
    # Construct dataset path
    dataset_path = Path(base_path) / f"record{record_id}"
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"\n📁 Loading dataset from: {dataset_path}")
    print(f"🎬 Record ID: {record_id}, Episode ID: {episode_idx}")
    print()
    
    robot = None
    try:
        # Initialize robot
        robot_config = SO101FollowerConfig(port=port)
        robot = SO101Follower(robot_config)
        robot.connect()
        print("✅ Robot connected successfully")
        
        # Load dataset from local path
        dataset = LeRobotDataset(
            repo_id=str(dataset_path),
            episodes=[episode_idx]
        )
        
        # Get the episode data - filter by the specific episode_index
        episode_data = dataset.hf_dataset.filter(lambda x: x['episode_index'] == episode_idx)
        actions = episode_data.select_columns("action")
        
        num_frames = len(episode_data)
        
        print(f"📊 Total frames in this episode: {num_frames}")
        print(f"⏱️  FPS: {dataset.fps}")
        print()
        
        # Replay episode
        log_say(f"Replaying episode {episode_idx} from record {record_id}")
        print(f"🤖 Starting replay of {num_frames} frames...\n")
        
        for idx in range(num_frames):
            t0 = time.perf_counter()
            
            action = {
                name: float(actions[idx]["action"][i]) 
                for i, name in enumerate(dataset.features["action"]["names"])
            }
            robot.send_action(action)
            
            # Print progress every 10 frames
            if (idx + 1) % 10 == 0:
                print(f"Frame {idx + 1}/{num_frames}")
            
            busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))
        
        print(f"\n✅ Replay completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during replay: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if robot is not None:
            try:
                robot.disconnect()
                print("🔌 Robot disconnected")
            except:
                pass


class RobotTicTacToe:
    def __init__(self, base_path, port="/dev/ttyACM1"):
        """
        Initialize the Robot Tic Tac Toe game.
        
        Args:
            base_path (str): Base path to lerobot datasets directory
            port (str): Serial port for robot connection
        """
        # Initialize the board (0 = empty, 1 = player, 2 = computer)
        self.board = [0] * 9
        self.human = 1
        self.computer = 2
        
        # Episode tracking: queue of available episode IDs [0, 1, 2, 3]
        self.episode_queue = deque([0, 1, 2, 3])
        
        # Robot parameters
        self.base_path = base_path
        self.port = port
        
        # Track which position the computer played (for recording moves)
        self.computer_moves = []
        
        # Validate base path exists
        if not Path(base_path).exists():
            raise ValueError(f"Base path does not exist: {base_path}")
        
    def print_board(self):
        """Display the current game board with position tracking"""
        print("\n")
        for i in range(3):
            row = []
            for j in range(3):
                cell_idx = i * 3 + j
                cell = self.board[cell_idx]
                
                if cell == 0:
                    row.append(" ")  # Empty space
                elif cell == 1:
                    row.append("X")  # Player move
                else:  # cell == 2 (computer)
                    # Show record number (position + 1) for computer moves
                    row.append(str(cell_idx + 1))
            
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("-----------")
        print("\n")
    
    def is_winner(self, player):
        """Check if the specified player has won"""
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
        """Make the computer's move and replay corresponding episode"""
        available = self.get_available_moves()
        
        if not available:
            return False
        
        # Get the next episode ID from the queue
        if not self.episode_queue:
            print("❌ No more episodes available!")
            return False
        
        episode_idx = self.episode_queue.popleft()
        
        # Use minimax algorithm to get best move
        move = self.get_best_move()
        self.board[move] = self.computer
        
        # Record the move
        self.computer_moves.append((move, episode_idx))
        
        # Calculate record number based on position (1-9)
        record_number = move + 1
        
        print(f"\n🤖 Computer placed O at position {move + 1}")
        print(f"📊 Using Episode ID: {episode_idx}, Record: {record_number}")
        print(f"📋 Remaining episodes: {list(self.episode_queue)}")
        
        # Replay the episode
        print("\n🔄 Replaying robot movement...\n")
        success = replay_episode(self.base_path, record_number, episode_idx, self.port)
        
        if not success:
            print("⚠️  Robot replay failed, but game continues...\n")
        
        return True
    
    def play(self):
        """Main game loop"""
        print("=" * 60)
        print("Welcome to Robot Tic Tac Toe!")
        print("=" * 60)
        print("\nYou are X, Computer is O")
        print("Computer O positions show the record number (1-9)")
        print("Episode IDs are used in order: [0, 1, 2, 3]")
        print("\nPositions are numbered 1-9:")
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
            if not self.computer_move():
                print("Game ended: No more episodes available")
                break
            
            self.print_board()
            
            if self.is_winner(self.computer):
                print("💻 Computer won! Better luck next time!")
                break
            
            if self.is_board_full():
                print("It's a draw!")
                break
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of computer moves"""
        print("\n" + "=" * 60)
        print("Game Summary - Computer Moves")
        print("=" * 60)
        for move_num, (position, episode_id) in enumerate(self.computer_moves, 1):
            print(f"Move {move_num}: Position {position + 1} (Record {position + 1}), Episode ID: {episode_id}")
        print("=" * 60 + "\n")


def main():
    print("=" * 60)
    print("Robot Tic Tac Toe Setup")
    print("=" * 60)
    
    base_path = input("Enter base dataset path (default: /home/taksh/lerobot_datasets): ").strip()
    if not base_path:
        base_path = "/home/taksh/lerobot_datasets"
    
    # Validate base path exists
    if not Path(base_path).exists():
        print(f"❌ Error: Base path does not exist: {base_path}")
        return
    
    port = input("Enter robot serial port (default: /dev/ttyACM1): ").strip()
    if not port:
        port = "/dev/ttyACM1"
    
    while True:
        try:
            game = RobotTicTacToe(base_path, port)
            game.play()
        except ValueError as e:
            print(f"❌ Error: {e}")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            break
        
        play_again = input("\nDo you want to play again? (yes/no): ").lower().strip()
        if play_again != "yes" and play_again != "y":
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
