"""
HAWKINS HEIST - AGENT SUBMISSION TEMPLATE
==========================================
STEALTH INFILTRATION RL COMPETITION

Your Name/Team: _________________________
Date: _________________________

OBJECTIVE:
Infiltrate Soviet laboratory and reach exit (28,28) from start (1,1) while:
- Avoiding guard detection
- Using cover strategically
- Minimizing exposure time
- Reaching goal efficiently

SCORING:
- Reach exit: +100 points
- Detection penalty: -0.5 per detection level
- High detection (>0.8): -50 points (mission failed)
- Time penalty: -0.01 per step
- Stealth bonus: -(total_detection * 10)

FINAL SCORE = Sum of best scores across all difficulty levels

Good luck, Agent! 🕵️‍♂️
"""

import numpy as np
import random


# =============================================================================
# SECTION 1: STATE REPRESENTATION
# =============================================================================

def discretize(observation):
    """
    Convert continuous observation to discrete state for Q-learning.
    
    Args:
        observation: numpy array with 18 values:
            [0:2]   = agent position (x, y) normalized to 0-1
            [2:17]  = 5 nearest guards (x, y, rotation) each
            [17]    = maximum detection level
    
    Returns:
        A hashable state representation (tuple recommended)
    
    TODO: Implement this function
    HINT: Focus on agent position and nearest guard info
    Example: (int(agent_x*30), int(agent_y*30), int(nearest_guard_dist))
    """
    
    # Extract key information
    agent_x = observation[0] * 30  # Denormalize to 0-30
    agent_y = observation[1] * 30
    
    # Nearest guard info
    guard1_x = observation[2] * 30
    guard1_y = observation[3] * 30
    guard1_rot = observation[4] * 360
    
    detection_level = observation[17]
    
    # ??? YOUR CODE HERE ???
    # Create a tuple representing the state
    # Consider: agent position, distance to nearest guard, detection level
    
    pass  # Remove this when you add your code


# =============================================================================
# SECTION 2: HYPERPARAMETERS  
# =============================================================================

# TODO: Set your hyperparameters for stealth learning

LEARNING_RATE = ???      # Alpha (α): How fast to update Q-values (try 0.1 to 0.5)
DISCOUNT_FACTOR = ???    # Gamma (γ): Future reward importance (try 0.90 to 0.99)
EXPLORATION_RATE = ???   # Epsilon (ε): Random exploration (try 0.1 to 0.3)


# =============================================================================
# SECTION 3: AGENT CREATION
# =============================================================================

def create_agent():
    """
    Create and return your RL agent for stealth navigation.
    
    Returns:
        select_action: function that chooses actions
        update: function that updates agent's knowledge
        memory: agent's memory structure (e.g., Q-table dictionary)
    
    TODO: Implement Q-learning agent optimized for stealth
    """
    
    # Initialize Q-table (state-action value table)
    Q = {}  # Dictionary: (state, action) -> Q-value
    
    # ??? YOUR CODE HERE ???
    # You can add additional data structures for stealth tactics
    
    def select_action(observation):
        """
        Choose an action based on current observation.
        
        Args:
            observation: numpy array [18 values as described above]
        
        Returns:
            action_index: integer from 0 to 5
                0 = STAY
                1 = UP
                2 = RIGHT
                3 = DOWN
                4 = LEFT
                5 = SPRINT (2x speed, 1.5x detection risk)
        
        TODO: Implement epsilon-greedy with stealth awareness
        HINTS:
        - Avoid sprinting when near guards (high detection risk)
        - Consider hiding in vents when detection is high
        - Use exploration carefully in danger zones
        """
        state = discretize(observation)
        
        # ??? YOUR CODE HERE ???
        # Implement smart action selection
        # Consider detection level when choosing actions
        
        pass  # Remove this when you add your code
    
    def update(state, action_index, reward, next_state):
        """
        Update agent's knowledge based on experience.
        
        Args:
            state: current discrete state
            action_index: action that was taken (0-5)
            reward: reward received
            next_state: resulting discrete state
        
        TODO: Implement Q-learning update rule
        FORMULA: Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        """
        
        # ??? YOUR CODE HERE ???
        # Standard Q-learning update
        
        pass  # Remove this when you add your code
    
    return select_action, update, Q


# =============================================================================
# SECTION 4: ADVANCED TACTICS (OPTIONAL)
# =============================================================================

# You can add helper functions here:
# - Guard danger estimation
# - Safe path calculation
# - Vent location memory
# - Cover utilization strategy


# =============================================================================
# DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def get_agent_info():
    """Return information about your agent"""
    return {
        'learning_rate': LEARNING_RATE,
        'discount_factor': DISCOUNT_FACTOR,
        'exploration_rate': EXPLORATION_RATE,
        'agent_type': 'Q-Learning Stealth Agent'
    }


if __name__ == "__main__":
    print("="*70)
    print("HAWKINS HEIST - AGENT SUBMISSION CHECK")
    print("="*70)
    
    print("\n📋 Your Hyperparameters:")
    print(f"   Learning Rate (α):     {LEARNING_RATE}")
    print(f"   Discount Factor (γ):   {DISCOUNT_FACTOR}")
    print(f"   Exploration Rate (ε):  {EXPLORATION_RATE}")
    
    print("\n🔍 Testing your implementation...")
    
    try:
        # Test discretize
        test_obs = np.random.rand(18)
        test_obs[0:2] = [0.5, 0.6]  # Agent at (15, 18)
        state = discretize(test_obs)
        print(f"✓ discretize() works: state = {state}")
        assert isinstance(state, tuple), "State should be a tuple!"
    except Exception as e:
        print(f"✗ discretize() error: {e}")
    
    try:
        # Test agent creation
        select_action, update, Q = create_agent()
        print(f"✓ create_agent() works")
        
        # Test action selection
        test_obs = np.random.rand(18)
        action = select_action(test_obs)
        print(f"✓ select_action() works: chose action {action}")
        assert 0 <= action <= 5, "Action should be 0-5!"
        
        # Test update
        state1 = (15, 18, 0)
        state2 = (15, 19, 0)
        update(state1, 1, -0.5, state2)
        print(f"✓ update() works: Q-table has {len(Q)} entries")
        
    except Exception as e:
        print(f"✗ Agent error: {e}")
    
    print("\n" + "="*70)
    print("If all tests passed, your submission is ready!")
    print("Submit this file to compete in Hawkins Heist!")
    print("="*70)
