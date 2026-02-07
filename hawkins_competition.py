"""
HAWKINS HEIST - COMPETITION RUNNER (SINGLE LEVEL)
==================================================
Live Dashboard: [GAME VIEW 800x800] | [STATS PANEL 400x800]

Usage:
    python hawkins_competition.py agent_submission.py --visualize
    python hawkins_competition.py --all submissions/ --visualize
    python hawkins_competition.py --leaderboard
"""

import sys
import os
import importlib.util
import numpy as np
import argparse
import json
import glob
import time

# Try import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️ Warning: opencv-python not found. Install: pip install opencv-python")

from hawkins_heist import HawkinsHeistEnv


# =============================================================================
# SINGLE DIFFICULTY CONFIGURATION
# =============================================================================

DIFFICULTY = "medium"  # Fixed difficulty for all agents
EPISODES = 500  # Enough episodes for learning


# =============================================================================
# AGENT LOADER
# =============================================================================

def load_participant_agent(filepath):
    """Load participant's agent submission"""
    spec = importlib.util.spec_from_file_location("participant_agent", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# DASHBOARD CREATION
# =============================================================================

def create_dashboard(env, episode, max_episodes, reward, best_reward, detection, success_rate):
    """
    Create 1200x800 dashboard with game view + stats panel
    """
    if not OPENCV_AVAILABLE:
        return None
    
    # 1. Render game view (800x800)
    game_img = np.zeros((800, 800, 3), dtype=np.uint8)
    game_img[:] = (40, 45, 50)  # Dark background
    
    # Draw grid lines
    for i in range(0, 800, 800//30):
        cv2.line(game_img, (i, 0), (i, 800), (60, 65, 70), 1)
        cv2.line(game_img, (0, i), (800, i), (60, 65, 70), 1)
    
    # Draw objects (obstacles and generators)
    for pos in env.obstacle_positions:
        x = int(pos[0] / 30 * 800)
        y = int(pos[1] / 30 * 800)
        cv2.rectangle(game_img, (x-8, y-8), (x+8, y+8), (100, 110, 130), -1)
    
    for pos in env.generator_positions:
        x = int(pos[0] / 30 * 800)
        y = int(pos[1] / 30 * 800)
        cv2.rectangle(game_img, (x-12, y-12), (x+12, y+12), (150, 130, 90), -1)
    
    # Draw vents (hiding spots)
    for pos in env.vent_positions:
        x = int(pos[0] / 30 * 800)
        y = int(pos[1] / 30 * 800)
        cv2.circle(game_img, (x, y), 10, (0, 200, 200), -1)
    
    # Draw exit (target)
    ex = int(env.exit_position[0] / 30 * 800)
    ey = int(env.exit_position[1] / 30 * 800)
    cv2.circle(game_img, (ex, ey), 20, (255, 150, 0), -1)
    cv2.circle(game_img, (ex, ey), 25, (255, 200, 50), 2)
    cv2.putText(game_img, "EXIT", (ex-20, ey+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw guards with vision cones
    for guard in env.guards:
        gx = int(guard.position[0] / 30 * 800)
        gy = int(guard.position[1] / 30 * 800)
        
        # Vision cone
        radius = int(guard.vision_radius / 30 * 800)
        start_angle = int(guard.rotation - guard.vision_angle / 2)
        end_angle = int(guard.rotation + guard.vision_angle / 2)
        
        # Cone color based on detection
        if guard.detection_level > 0.5:
            cone_color = (50, 50, 255)
        elif guard.detection_level > 0.2:
            cone_color = (50, 150, 255)
        else:
            cone_color = (100, 200, 255)
        
        # Draw cone as transparent overlay
        overlay = game_img.copy()
        cv2.ellipse(overlay, (gx, gy), (radius, radius), 
                   -guard.rotation, -guard.vision_angle/2, guard.vision_angle/2,
                   cone_color, -1)
        cv2.addWeighted(overlay, 0.3, game_img, 0.7, 0, game_img)
        
        # Guard body
        color = (200, 50, 50) if guard.detection_level > 0.5 else (150, 150, 150)
        cv2.circle(game_img, (gx, gy), 12, color, -1)
        cv2.circle(game_img, (gx, gy), 12, (100, 100, 100), 2)
        
        # Direction indicator
        end_x = int(gx + 20 * np.cos(np.radians(guard.rotation)))
        end_y = int(gy + 20 * np.sin(np.radians(guard.rotation)))
        cv2.line(game_img, (gx, gy), (end_x, end_y), (255, 255, 255), 2)
    
    # Draw agent
    ax = int(env.agent_pos[0] / 30 * 800)
    ay = int(env.agent_pos[1] / 30 * 800)
    
    # Check if in vent
    in_vent = tuple(env.agent_pos) in env.vent_positions
    if in_vent:
        cv2.circle(game_img, (ax, ay), 15, (100, 255, 255), -1)
        cv2.putText(game_img, "HIDDEN", (ax-25, ay-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    else:
        cv2.circle(game_img, (ax, ay), 15, (0, 255, 100), -1)
        cv2.circle(game_img, (ax, ay), 15, (0, 200, 80), 2)
    
    # Draw path from start to current
    sx = int(env.START_POS[0] / 30 * 800)
    sy = int(env.START_POS[1] / 30 * 800)
    cv2.line(game_img, (sx, sy), (ax, ay), (100, 255, 100), 2)
    
    # 2. Create stats panel (400x800)
    panel = np.zeros((800, 400, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    
    def put_text(img, text, y, size=0.8, color=(200, 200, 200), thickness=2):
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    
    # Header
    cv2.rectangle(panel, (0, 0), (400, 100), (50, 50, 50), -1)
    put_text(panel, "HAWKINS HEIST", 50, 1.2, (0, 255, 255), 3)
    put_text(panel, "RL COMPETITION", 85, 0.7, (150, 150, 150), 1)
    
    # Episode progress
    y = 160
    put_text(panel, "EPISODE:", y, 0.6, (150, 150, 150), 1)
    put_text(panel, f"{episode} / {max_episodes}", y+40, 1.2, (255, 255, 0), 3)
    
    # Progress bar
    bar_width = 340
    progress = episode / max_episodes
    cv2.rectangle(panel, (30, y+60), (30+bar_width, y+70), (50, 50, 50), -1)
    cv2.rectangle(panel, (30, y+60), (30+int(bar_width*progress), y+70), (0, 255, 0), -1)
    
    # Success rate
    y = 280
    put_text(panel, "SUCCESS RATE:", y, 0.6, (150, 150, 150), 1)
    put_text(panel, f"{success_rate:.1f}%", y+40, 1.3, (0, 255, 150), 3)
    
    # Current reward
    y = 380
    put_text(panel, "CURRENT REWARD:", y, 0.6, (150, 150, 150), 1)
    color = (0, 0, 255) if reward < 0 else (0, 255, 0)
    put_text(panel, f"{reward:.1f}", y+40, 1.5, color, 3)
    
    # Best reward
    y = 500
    put_text(panel, "BEST SCORE:", y, 0.6, (150, 150, 150), 1)
    put_text(panel, f"{best_reward:.1f}", y+40, 1.5, (0, 215, 255), 3)
    
    # Detection meter
    y = 620
    put_text(panel, "DETECTION:", y, 0.6, (150, 150, 150), 1)
    
    det_width = 340
    det_fill = int(det_width * min(1.0, detection))
    cv2.rectangle(panel, (30, y+20), (30+det_width, y+40), (50, 50, 50), -1)
    
    if detection > 0.8:
        det_color = (0, 0, 255)
        status = "DETECTED!"
    elif detection > 0.5:
        det_color = (0, 165, 255)
        status = "ALERT"
    elif detection > 0.2:
        det_color = (0, 255, 255)
        status = "SUSPICIOUS"
    else:
        det_color = (0, 255, 150)
        status = "CLEAR"
    
    cv2.rectangle(panel, (30, y+20), (30+det_fill, y+40), det_color, -1)
    put_text(panel, status, y+70, 0.6, det_color, 2)
    
    # Distance to exit
    dist = np.linalg.norm(np.array(env.agent_pos) - np.array(env.exit_position))
    put_text(panel, f"Distance: {dist:.1f}", 750, 0.5, (150, 150, 150), 1)
    
    # Footer
    put_text(panel, "Press ESC to skip", 780, 0.4, (100, 100, 100), 1)
    
    # 3. Combine
    dashboard = np.hstack((game_img, panel))
    return dashboard


# =============================================================================
# TRAINING
# =============================================================================

def train_agent(
    select_action,
    update,
    discretize_func,
    episodes=500,
    visualize=False,
    agent_name="Agent"
):
    """
    Train agent on Hawkins Heist
    """
    
    env = HawkinsHeistEnv(render_mode=None, difficulty=DIFFICULTY)
    
    episode_rewards = []
    best_reward = -float("inf")
    successes = 0
    
    print(f"\n{'='*70}")
    print(f"🎯 Training: {agent_name}")
    print(f"   Difficulty: {DIFFICULTY}")
    print(f"   Episodes: {episodes}")
    print(f"{'='*70}")
    
    window_name = "Hawkins Heist - Competition Dashboard"
    if visualize and OPENCV_AVAILABLE:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
    
    # Which episodes to visualize
    def should_show(ep):
        return (ep <= 5) or (ep % 20 == 0) or (ep > episodes - 5)
    
    for episode in range(1, episodes + 1):
        show_episode = visualize and OPENCV_AVAILABLE and should_show(episode)
        
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        step_counter = 0
        
        while not done:
            step_counter += 1
            
            # Agent action
            state = discretize_func(obs)
            action_idx = select_action(obs)
            
            # Execute
            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            
            # Update agent
            next_state = discretize_func(next_obs)
            update(state, action_idx, reward, next_state)
            
            obs = next_obs
            total_reward += reward
            done = terminated or truncated
            
            # Visualization
            if show_episode and step_counter % 5 == 0:
                max_detection = max([g.detection_level for g in env.guards], default=0)
                success_rate = (successes / episode) * 100 if episode > 0 else 0
                
                dashboard = create_dashboard(
                    env, episode, episodes, 
                    total_reward, best_reward, max_detection, success_rate
                )
                
                if dashboard is not None:
                    cv2.imshow(window_name, dashboard)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        visualize = False
                        cv2.destroyAllWindows()
                        print("\n🛑 Visualization skipped")
        
        # Episode end
        episode_rewards.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
        
        # Check success
        reached_exit = tuple(env.agent_pos) == env.exit_position
        if reached_exit and not info['detected']:
            successes += 1
        
        # Console progress
        if episode % 50 == 0 or episode == episodes:
            success_rate = (successes / episode) * 100
            avg_reward = np.mean(episode_rewards[-50:])
            status = "✓" if reached_exit else "✗"
            print(f"   Ep {episode:3d}/{episodes}: {status} "
                  f"Reward={total_reward:6.1f} | "
                  f"Best={best_reward:6.1f} | "
                  f"Success={success_rate:5.1f}%")
    
    env.close()
    
    final_success_rate = (successes / episodes) * 100
    
    print(f"\n{'='*70}")
    print(f"📊 Training Complete!")
    print(f"   Best Score: {best_reward:.2f}")
    print(f"   Success Rate: {final_success_rate:.1f}% ({successes}/{episodes})")
    print(f"   Avg Last 100: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"{'='*70}")
    
    return best_reward, episode_rewards, final_success_rate


# =============================================================================
# MAIN COMPETITION RUNNER
# =============================================================================

def run_competition(participant_file, episodes=500, visualize=False):
    """
    Run agent through single difficulty level
    """
    
    print("\n" + "="*80)
    print("🕵️‍♂️ HAWKINS HEIST - RL COMPETITION")
    print("="*80)
    print(f"Participant: {participant_file}")
    print("="*80)
    
    try:
        participant = load_participant_agent(participant_file)
        select_action, update, memory = participant.create_agent()
        discretize_func = participant.discretize
        agent_info = participant.get_agent_info()
        
        print("\n📊 Agent Configuration:")
        for k, v in agent_info.items():
            print(f"   {k}: {v}")
    
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Train
    best_score, rewards, success_rate = train_agent(
        select_action,
        update,
        discretize_func,
        episodes=episodes,
        visualize=visualize,
        agent_name=os.path.basename(participant_file)
    )
    
    # Save results
    results = {
        "participant": participant_file,
        "best_score": best_score,
        "success_rate": success_rate,
        "agent_info": agent_info,
        "episodes": episodes
    }
    
    os.makedirs("competition_results", exist_ok=True)
    participant_name = os.path.basename(participant_file).replace(".py", "")
    
    with open(f"competition_results/{participant_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if visualize and OPENCV_AVAILABLE:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*80}")
    print(f"🏆 FINAL SCORE: {best_score:.2f}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"{'='*80}\n")
    
    return results


# =============================================================================
# LEADERBOARD
# =============================================================================

def generate_leaderboard(results_dir="competition_results"):
    """Generate competition leaderboard"""
    
    results_files = glob.glob(f"{results_dir}/*_results.json")
    
    if not results_files:
        print("No results found!")
        return
    
    leaderboard = []
    for filepath in results_files:
        with open(filepath, "r") as f:
            data = json.load(f)
            leaderboard.append({
                "name": os.path.basename(data["participant"]).replace(".py", ""),
                "score": data["best_score"],
                "success_rate": data["success_rate"]
            })
    
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    
    print("\n" + "="*80)
    print("🏆 HAWKINS HEIST - LEADERBOARD")
    print("="*80)
    print(f"{'Rank':<6} {'Agent':<30} {'Best Score':>12} {'Success Rate':>15}")
    print("-"*80)
    
    for rank, entry in enumerate(leaderboard, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:2d}. {entry['name']:<30} "
              f"{entry['score']:>12.2f} {entry['success_rate']:>14.1f}%")
    
    print("="*80)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hawkins Heist Competition")
    parser.add_argument("submission", nargs="?", help="Path to agent submission")
    parser.add_argument("--all", help="Directory with all submissions")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--visualize", action="store_true", help="Live visualization")
    
    args = parser.parse_args()
    
    if args.leaderboard:
        generate_leaderboard()
    elif args.all:
        submissions = glob.glob(f"{args.all}/*.py")
        print(f"Found {len(submissions)} submissions")
        for submission in submissions:
            run_competition(submission, args.episodes, args.visualize)
        generate_leaderboard()
    elif args.submission:
        run_competition(args.submission, args.episodes, args.visualize)
    else:
        print("Usage:")
        print("  python hawkins_competition.py agent.py --visualize")
        print("  python hawkins_competition.py --all submissions/ --episodes 500")
        print("  python hawkins_competition.py --leaderboard")
