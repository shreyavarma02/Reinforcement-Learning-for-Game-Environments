Lunar Lander using Deep Q-Network (DQN)
🧠 Problem Statement
    -Train an agent to land a spacecraft safely using only state inputs like position, velocity, and angle.
    -The environment (LunarLander-v3) provides an 8-dimensional vector as the state and 4 discrete actions (no-op, fire left, fire main, fire right).

🧱 Model Architecture
Feedforward Neural Network with:
    -Input: 8-dimensional state.
    -Hidden Layers: Two fully connected layers with 64 neurons each.
    -Output: Q-values for 4 possible actions.
    -Activation Function: ReLU after each layer.

⚙️ Key RL Techniques Used
    -Deep Q-Learning (DQN): Used to approximate the Q-function.
    -Experience Replay: Stores experiences in a buffer and samples random minibatches to reduce correlation.
    -Target Network: 
        -A separate frozen Q-network is used to calculate stable target Q-values. (Q target = r + γ⋅ max Q target(s′) )
        -The Target Q-Network takes s' as input, runs a forward pass, and outputs Q-values for all actions. 
        -Target Q network weights is updated ,less frequently using soft update (to stabilize the model) 
    -Local Network:
        -The Local Q-Network takes s as input, runs a forward pass, and outputs Q-values for all actions.
        -Q expected is found using epsilon greedy
        -Then compute loss(Q target - Q expected) and update local Q network using back propagation
    -Epsilon-Greedy Policy:
        -Starts with high exploration (ε = 1.0).
        -Gradually decays to ε = 0.01 to favor exploitation as learning progresses.

🔁 Training Loop Highlights
    -For each episode:
        -Choose an action via agent.act(...)
        -Use env.step(action) to interact with the environment.
        -Use agent.step(...) to Store (state, action, reward, next_state, done) into replay buffer, and every 4 steps sample a minibatch and learn from it by updating both Q          networks and minimizing loss
    -Training stops early if the average score over 100 episodes exceeds 200 (task solved).




 Ms. Pac-Man using Deep Q-Network with CNN
🧠 Problem Statement
    -Train an agent to play Ms. Pac-Man using raw pixel input, learning directly from game visuals with no hand-crafted features.

🧱 Model Architecture
    -Convolutional Neural Network (CNN) with:
        -4 convolutional layers (with increasing channels and decreasing kernel and stride).
        -Batch Normalization after each conv layer for faster and more stable learning.
        -Flattened output fed into 3 fully connected layers.
        -Final layer outputs Q-values for all possible discrete actions.
        -Input: Preprocessed game frames resized to 128x128 RGB tensors.

🧰 Preprocessing
    -Each game frame is:
        -Resized from (210x160) to (128x128).
        -Converted to PyTorch tensor and unsqueezed to add batch dimension.
        -This image becomes the state input to the CNN.

⚙️ Key RL Techniques Used
    -Visual Deep Q-Learning:
        -Q-values are learned from raw pixel observations instead of vector states.
    -Epsilon-Greedy Policy for exploration.
    -Experience Replay:
        -A simple deque buffer stores tuples of (state, action, reward, next_state, done).
        -Samples are drawn randomly for training.
   -Target Network: 
        -A separate frozen Q-network is used to calculate stable target Q-values. (Q target = r + γ⋅ max Q target(s′) )
        -The Target Q-Network takes s' as input, runs a forward pass, and outputs Q-values for all actions. 
        -Target Q network weights is updated ,less frequently using soft update (to stabilize the model) 
    -Local Network:
        -The Local Q-Network takes s as input, runs a forward pass, and outputs Q-values for all actions.
        -Q expected is found using epsilon greedy
        -Then compute loss: MSE(Q target - Q expected) and update local Q network using back propagation
   
🔁 Training Loop Highlights
    -For each episode:
        -Reset the environment and receive the raw visual state.
        -Choose an action via agent.act(...)
        -Call env.step(action) to interact with the environment (perform the action to next state).
        -Store and learn from experience using agent.step(...).
    -Training continues until average reward reaches 500 (or desired performance is met).





