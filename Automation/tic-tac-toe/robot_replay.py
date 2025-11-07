import time
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
    print(f"🎬 Record ID: {record_id}")
    print(f"🎬 Episode ID: {episode_idx}")
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
        log_say(f"Replaying episode {episode_idx}")
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

if __name__ == "__main__":
    print("=" * 50)
    print("SO101 Follower Robot - Episode Replay")
    print("=" * 50)
    
    # Get user inputs
    base_path = input("Enter base dataset path (default: /home/taksh/lerobot_datasets): ").strip()
    if not base_path:
        base_path = "/home/taksh/lerobot_datasets"
    
    while True:
        try:
            record_id = int(input("Enter record ID (1-9): "))
            if 1 <= record_id <= 9:
                break
            else:
                print("Please enter a number between 1 and 9.")
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        try:
            episode_idx = int(input("Enter episode ID (0-3): "))
            if 0 <= episode_idx <= 3:
                break
            else:
                print("Please enter a number between 0 and 3.")
        except ValueError:
            print("Please enter a valid integer.")
    
    port = input("Enter robot port (default: /dev/ttyACM1): ").strip()
    if not port:
        port = "/dev/ttyACM1"
    
    # Call the replay function
    replay_episode(base_path, record_id, episode_idx, port)
