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
        -A separate frozen Q-network is used to get target Q-value for next state, thereby calculating stable target Q-value. (Q target = r + γ⋅ max Q target(s′) )
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



<br>
<br>
<br>

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
        -A separate frozen Q-network is used to get target Q-value for next state, thereby calculating stable target Q-value. (Q target = r + γ⋅ max Q target(s′) )
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

<br>
<br>
<br>
🥋 Reinforcement Learning: A3C Agent on Kung Fu Master (Atari 2600)
🧠 Objective

    -This project trains an intelligent agent to play the Atari game Kung Fu Master using the Advantage Actor-Critic (A3C) algorithm.

🏗️ Components

🔸 1. Preprocessing the Environment

    -The Atari environment is wrapped using a custom PreprocessAtari class.
    -The agent receives input as grayscale image frames, resized to 42×42 pixels.
    -Four consecutive frames are stacked to give the agent temporal awareness (e.g., motion of enemies or punches).
    

🔸 2. Neural Network Architecture


Architecture:


    -3 convolutional layers with ReLU activations.
    -A flatten layer followed by fully connected layers.
    -The final two outputs are:
            - Actor Head: Outputs a probability distribution (policy) over 14 possible actions.
            - Critic Head: Outputs a scalar value estimating the state value

🤖 Agent Functionality

🔸 3. Action Selection

    -The agent uses the softmax function to convert logits to a probability distribution over actions.

🔸 4. Learning from Experience

    -Each agent interacts with its environment and learns directly at every step.
    -There’s no separate experience replay buffer or target network.
    -Single NN (no separate local and target NN)
            -The same NN is used to get V (s'), thereby calculating stable V target . (V target(s) = r + max γ⋅V(s′) )
            -The same NN takes s as input, runs a forward pass, and outputs:
                    -actions values
                    -V expected(s)
    -Then compute loss: Actor loss + Critic loss and update NN using back propagation
            - Actor loss: Using entropy and advantage which uses action values
            - Critic loss: MSE(V target, V expected)​



🔸 5. Parallel Environments (EnvBatch)

    -A custom EnvBatch class runs 10 independent game environments simultaneously.
    -Each agent plays its own episode and contributes to a batch update of the shared model.
    -This simulates asynchronous training and improves stability and efficiency.

🏋️ Training Loop

    -Training runs for 3000 iterations.
    -In each iteration:
            -All environments generate their own actions via the agent. (agent.act)
            -All agents take a step (agent.step) and collects rewards, next states, and done signals from each environment.
            -Agents learn from all experiences by updating the shared Network and minimizing the total loss.
            -Every 1000 iterations, the model is evaluated over 10 fresh episodes.

📈 Evaluation

    -The evaluate() function tests the agent on a few episodes and computes average rewards.




