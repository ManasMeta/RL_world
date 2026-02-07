# 🕵️‍♂️ HAWKINS HEIST - QUICK START

## 🎯 Single Level Competition

**ONE mission. ONE difficulty. Best score wins!**

---

## 🚀 Run Competition (3 Commands)

### 1. Test System
```bash
python hawkins_competition.py working_solution.py --visualize --episodes 100
```

### 2. Collect Submissions
```bash
mkdir submissions
# Participants submit their agent files to submissions/
```

### 3. Run & Rank
```bash
# Evaluate all agents
python hawkins_competition.py --all submissions/ --episodes 500

# Show leaderboard
python hawkins_competition.py --leaderboard
```

---

## 📊 How It Works

### **Single Mission:**
- **Difficulty:** Medium (5 guards)
- **Start:** (1, 1)
- **Exit:** (28, 28)
- **Episodes:** 500 (agents learn automatically)

### **Scoring:**
```python
Best Score = Highest reward across all 500 episodes

Example:
Episode 234: Reached exit, 0.15 detection → +92.5
Episode 445: Reached exit, 0.08 detection → +95.2 ← BEST!

FINAL SCORE: 95.2
```

### **Leaderboard:**
```
Rank  Agent           Score    Success%
🥇 1. TeamAlpha       95.2     68.4%
🥈 2. TeamBeta        89.7     62.1%
🥉 3. TeamGamma       85.3     58.9%
```

---

## 💡 Why Agents Failed Before (FIXED!)



### ✅ New Solution:
```python
def discretize(obs):
    # Include DIRECTION to exit
    to_exit_x = 28 - agent_x
    to_exit_y = 28 - agent_y
    direction = compute_direction(to_exit_x, to_exit_y)
    
    return (pos_x, pos_y, direction)  # Agent knows where to go!
```

### 🎯 Key Improvements in working_solution.py:
1. **Directional State:** Agent knows which way exit is
2. **Optimistic Init:** Assumes unexplored states are good
3. **Goal-Biased Exploration:** Random actions favor exit direction
4. **Adaptive Epsilon:** Less exploration when close to goal
5. **High Discount (0.95):** Values long-term goal reaching

**Result:** Agents now reach (28,28) reliably! ✅

---

## 📦 File Structure

```
hawkins-competition/
├── hawkins_heist.py              # Environment (share or keep private)
├── hawkins_competition.py        # Competition runner (private)
├── working_solution.py           # Working agent (private - for testing)
├── agent_template.py             # Empty template (share with participants)
├── assets/                       # 12 PNG sprites (share)
└── submissions/                  # Collected participant solutions
```

---

## 🎓 For Participants

### What You Get:
- `agent_template.py` - Fill in the blanks
- `assets/` folder - Game sprites
- This guide

### What You Do:
```python
# 1. Implement discretize()
def discretize(observation):
    # YOUR CODE: Convert 18-dim obs to state tuple
    
# 2. Set hyperparameters
LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 0.2

# 3. Implement select_action() and update()
```

### Submit:
Your completed `agent_submission.py` file

---

## 🏆 For Organizers

### Setup (5 min):
```bash
# 1. Test system
python hawkins_competition.py working_solution.py --visualize --episodes 100

# 2. Distribute to participants
# Give them: agent_template.py + assets/ + this guide
```

### Competition Day:
```bash
# 3. Collect submissions
ls submissions/
# team_alpha.py
# team_beta.py
# team_gamma.py

# 4. Run competition
python hawkins_competition.py --all submissions/ --episodes 500
# Takes ~2-3 hours for all agents

# 5. Show results
python hawkins_competition.py --leaderboard
```

---

## 📈 Expected Performance

### Random Agent (Baseline):
```
Score: -30 to -10
Success: 0%
Never reaches exit
```

### Beginner Agent (100 eps):
```
Score: 20 to 40
Success: 10-20%
Sometimes reaches exit
```

### Working Solution (500 eps):
```
Score: 80 to 95
Success: 60-75%
Consistently reaches exit
```

### Expert Agent (1000+ eps):
```
Score: 90 to 98
Success: 75-85%
Optimal stealth pathing
```

---

## 🔧 Quick Commands

```bash
# Single agent test (fast)
python hawkins_competition.py agent.py --episodes 100

# Single agent with visualization
python hawkins_competition.py agent.py --visualize --episodes 500

# Batch evaluation (no viz, faster)
python hawkins_competition.py --all submissions/ --episodes 500

# Leaderboard only
python hawkins_competition.py --leaderboard

# Clean results
rm -rf competition_results/
```

---

## 🎯 Success Criteria

| Score | Rating | Description |
|-------|--------|-------------|
| < 0 | Failed | Worse than random |
| 0-30 | Beginner | Basic navigation |
| 30-60 | Intermediate | Reaches exit sometimes |
| 60-85 | Advanced | Consistent success |
| 85+ | Expert | Near-optimal play |

---

## 💻 System Requirements

```
Python 3.8+
Dependencies:
  - gymnasium
  - pygame
  - numpy
  - opencv-python (for visualization)
  
Storage: ~50MB
RAM: ~2GB
Time: 500 eps ≈ 30-45 min per agent
```

---

## ❓ FAQ

**Q: Why do agents fail to reach exit?**
A: State representation must include direction to goal! See working_solution.py

**Q: How long to train?**
A: 500 episodes is good balance (~40 min). 1000+ for expert agents.

**Q: Can I use different algorithms?**
A: Yes! Template supports any RL algorithm that fits the API.

**Q: What if visualization doesn't work?**
A: Competition runs fine without it. Install opencv-python to enable.

---

**Simple. Single level. Best agent wins!** 🏆
