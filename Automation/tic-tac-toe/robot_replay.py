import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


def replay_episode(hf_username, episode_idx, port="/dev/ttyACM1"):
    """
    Replay a robot episode from a HuggingFace dataset.
    
    Args:
        hf_username (str): HuggingFace username for dataset location
        episode_idx (int): Episode ID (0-3)
        port (str): Serial port for robot connection
    
    Returns:
        bool: True if replay completed successfully, False otherwise
    
    Raises:
        ValueError: If episode_idx is not in range 0-3
        Exception: If robot connection or dataset loading fails
    """
    if not isinstance(episode_idx, int) or episode_idx < 0 or episode_idx > 3:
        raise ValueError(f"episode_idx must be an integer between 0 and 3, got {episode_idx}")
    
    # Calculate record number (record1 to record9)
    # If user enters episode 0, use record1; episode 1 uses record2, etc.
    record_number = episode_idx + 1
    dataset_repo_id = f"{hf_username}/record{record_number}"
    
    print(f"\n📁 Loading dataset from: {dataset_repo_id}")
    print(f"🎬 Episode ID: {episode_idx}")
    print()
    
    robot = None
    try:
        # Initialize robot
        robot_config = SO101FollowerConfig(
            port=port
        )
        
        robot = SO101Follower(robot_config)
        robot.connect()
        print("✅ Robot connected successfully")
        
        # Load dataset
        dataset = LeRobotDataset(dataset_repo_id, episodes=[episode_idx])
        actions = dataset.hf_dataset.select_columns("action")
        
        print(f"📊 Total frames: {dataset.num_frames}")
        print(f"⏱️  FPS: {dataset.fps}")
        print()
        
        # Replay episode
        log_say(f"Replaying episode {episode_idx}")
        print(f"🤖 Starting replay of {dataset.num_frames} frames...\n")
        
        for idx in range(dataset.num_frames):
            t0 = time.perf_counter()
            
            action = {
                name: float(actions[idx]["action"][i]) 
                for i, name in enumerate(dataset.features["action"]["names"])
            }
            robot.send_action(action)
            
            # Print progress every 10 frames
            if (idx + 1) % 10 == 0:
                print(f"Frame {idx + 1}/{dataset.num_frames}")
            
            busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))
        
        print(f"\n✅ Replay completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during replay: {e}")
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
    hf_username = input("Enter HuggingFace username: ").strip()
    
    while True:
        try:
            episode_idx = int(input("Enter episode ID (0-3): "))
            if 0 <= episode_idx <= 3:
                break
            else:
                print("Please enter a number between 0 and 3.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Call the replay function
    replay_episode(hf_username, episode_idx)
