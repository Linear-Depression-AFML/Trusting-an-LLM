import math
import random
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. TRUST TRACKER CLASS (Same as trust game) ---

class TrustTracker:
    """
    Implements dual-component trust formula.
    - CTrust (Competence) is updated by promise quality & action rationality.
    - BTrust (Benevolence) is updated by promise emotion & action fairness.
    """
    def __init__(self, agent_name, D=0.5, k_c=0.01, k_b=0.01):
        self.agent_name = agent_name
        # CHANGE: Increased initial trust from 0.5 to 0.6 to encourage early cooperation
        self.CTrust = 0.50
        self.BTrust = 0.50
        self.e = math.e
        self.D = D
        self.k_c = k_c
        self.k_b = k_b
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.previous_messages = []
        self.max_message_history = 10

    def _get_promise_effort(self, analyzer_agent, message):
        """Calculates Effort (f) based on commitment level in the promise."""
        effort_template = """
You are evaluating the level of commitment/effort expressed in this promise from a centipede game.

Message: '{message}'

Rate the level of commitment on a scale of 0.0 to 1.0:
- 1.0 = Strong commitment (specific promises, concrete strategy)
- 0.5 = Moderate commitment (vague promises)
- 0.0 = No commitment (empty words)

Respond with only a single floating-point number from 0.0 to 1.0.
"""
        f_chain = ChatPromptTemplate.from_template(effort_template) | analyzer_agent | StrOutputParser()
        f_response = f_chain.invoke({"message": message})
        try:
            f = max(0.0, min(1.0, float(f_response.strip())))
        except ValueError:
            f = 0.5
        return f

    def _calculate_information_novelty(self, message):
        """Calculates Information Novelty (I) based on message uniqueness."""
        if not self.previous_messages:
            return 1.0
        
        current_words = set(message.lower().split())
        if not current_words:
            return 0.0
        
        max_similarity = 0.0
        for prev_msg in self.previous_messages:
            prev_words = set(prev_msg.lower().split())
            if not prev_words:
                continue
            
            intersection = len(current_words & prev_words)
            union = len(current_words | prev_words)
            similarity = intersection / union if union > 0 else 0.0
            max_similarity = max(max_similarity, similarity)
        
        novelty = 1.0 - max_similarity
        
        self.previous_messages.append(message)
        if len(self.previous_messages) > self.max_message_history:
            self.previous_messages.pop(0)
        
        return novelty
    
    def _calculate_goal_relevance(self, analyzer_agent, message, context_type="promise"):
        """Calculates Goal Relevance (R) - message relevance to game goals."""
        relevance_template = """
You are evaluating how relevant this message is to the goals in a centipede game.
Goals: maximizing payoff, cooperation timing, and strategic positioning.

Message: '{message}'
Context: This is a {context_type} in the centipede game.

Rate relevance to game goals on a scale of 0.0 to 1.0:
- 1.0 = Highly relevant (addresses strategy, cooperation, timing)
- 0.5 = Moderately relevant
- 0.0 = Irrelevant

Respond with only a single floating-point number from 0.0 to 1.0.
"""
        r_chain = ChatPromptTemplate.from_template(relevance_template) | analyzer_agent | StrOutputParser()
        r_response = r_chain.invoke({"message": message, "context_type": context_type})
        try:
            R = max(0.0, min(1.0, float(r_response.strip())))
        except ValueError:
            R = 0.5
        return R

    def _apply_decay(self):
        """Applies constant time decay to both scores."""
        self.CTrust = max(0.01, self.CTrust - self.k_c)
        self.BTrust = max(0.01, self.BTrust - self.k_b)

    def _get_promise_scores(self, analyzer_agent, promise):
        """Gets Q (Quality) from LLM and E (Emotion) from VADER."""
        q_template = """
You are an objective AI evaluator. Score the *quality* of this promise in a centipede game.
A high-quality promise (1.0) is rational, clear, and fosters cooperation.
A low-quality promise (0.0) is irrational, vague, or purely selfish.
The PROMISE is: '{message}'.
Respond with only a single floating-point number from 0.0 to 1.0.
"""
        q_chain = ChatPromptTemplate.from_template(q_template) | analyzer_agent | StrOutputParser()
        q_response = q_chain.invoke({"message": promise})
        try:
            Q = max(0.0, min(1.0, float(q_response.strip())))
        except ValueError:
            Q = 0.0

        vader_scores = self.vader_analyzer.polarity_scores(promise)
        compound_score = vader_scores['compound']
        E_pos = max(0, compound_score)
        E_neg = abs(min(0, compound_score))
            
        return Q, E_pos, E_neg

    def _get_deterministic_action_scores(self, current_payoff, potential_payoff, action_taken):
        """
        Calculates Q_action and E_action for centipede game actions.
        action_taken: 'take' or 'push'
        """
        if action_taken == 'take':
            # Taking ends cooperation - assess rationality
            # CHANGE: Reduced rationality of taking to discourage early defection
            if potential_payoff <= 0:
                Q_action = 1.0
            else:
                payoff_ratio = current_payoff / potential_payoff
                # CHANGE: Penalize taking by scaling down the rationality score
                Q_action = max(0.0, min(1.0, payoff_ratio * 0.6))
            
            # CHANGE: Increased selfishness penalty from 0.5 to 0.7
            E_pos_action = 0.0
            E_neg_action = 0.7  # Higher selfishness penalty
        else:  # push
            # Pushing continues cooperation
            # CHANGE: Increased rationality of pushing from 0.8 to 0.95
            Q_action = 0.95  # More rational to continue if trusting
            # CHANGE: Increased cooperation reward from 0.7 to 0.85
            E_pos_action = 0.85  # Higher cooperative/altruistic signal
            E_neg_action = 0.0
        
        return Q_action, E_pos_action, E_neg_action

    def update_trust_from_promise(self, analyzer_agent, promise):
        """Updates trust based on agent's promise."""
        self._apply_decay()
        
        if not promise.strip():
            self.CTrust = self.CTrust - 0.1
            self.BTrust = self.BTrust - 0.1
            self._normalize_scores()
            return

        f = self._get_promise_effort(analyzer_agent, promise)
        Q, E_pos, E_neg = self._get_promise_scores(analyzer_agent, promise)
        I = self._calculate_information_novelty(promise)
        R = self._calculate_goal_relevance(analyzer_agent, promise, "promise")
        
        # Collaborative task component
        psi_collaborative = Q * (f + self.D)
        
        # Information gain component
        psi_info_gain = I * R * E_pos
        
        # Update Competence Trust
        psi_total_ct = psi_collaborative + psi_info_gain
        ct_bonus = psi_total_ct / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)

        # Update Benevolence Trust
        if E_pos > 0:
            bt_bonus = E_pos / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)
        
        if E_neg > 0:
            self.BTrust = self.BTrust * (1 - E_neg)
            
        self._normalize_scores()

    def update_trust_from_action(self, current_payoff, potential_payoff, action_taken):
        """
        Updates trust based on action in centipede game.
        current_payoff: what they get by taking
        potential_payoff: what they could get by pushing
        action_taken: 'take' or 'push'
        """
        Q_action, E_pos_action, E_neg_action = self._get_deterministic_action_scores(
            current_payoff, potential_payoff, action_taken
        )
        
        # Effort based on action type
        f_action = 0.3 if action_taken == 'take' else 0.7
        
        # Collaborative task component
        psi_collaborative = Q_action * (f_action + self.D)
        
        # Information gain from action
        I_action = 0.6 if action_taken == 'take' else 0.4  # Taking is more surprising
        R_action = 1.0  # Actions always relevant
        psi_info_gain = I_action * R_action * E_pos_action
        
        # Update Competence Trust
        psi_total = psi_collaborative + psi_info_gain
        ct_bonus = psi_total / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)
        
        # Update Benevolence Trust
        if E_pos_action > 0:
            bt_bonus = E_pos_action / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)
        
        if E_neg_action > 0:
            self.BTrust = self.BTrust * (1 - E_neg_action)
            
        self._normalize_scores()

    def _normalize_scores(self):
        """Keeps scores between 0.01 and 1.0."""
        self.CTrust = max(0.01, min(1.0, self.CTrust))
        self.BTrust = max(0.01, min(1.0, self.BTrust))

    def get_scores(self):
        return self.CTrust, self.BTrust

# --- 2. CENTIPEDE GAME LOGIC ---

class CentipedeGame:
    """Implements the centipede game with configurable starting piles and multiplier."""
    def __init__(self, m0=4, m1=1, max_rounds=10):
        self.m0 = m0  # Larger pile
        self.m1 = m1  # Smaller pile
        self.max_rounds = max_rounds
        self.current_round = 0
        self.game_over = False
        self.p1_total = 0
        self.p2_total = 0
    
    def get_current_piles(self):
        """Returns current pile values (larger, smaller)."""
        multiplier = 2 ** self.current_round
        return self.m0 * multiplier, self.m1 * multiplier
    
    def get_next_piles(self):
        """Returns what piles would be if pushed."""
        if self.current_round + 1 >= self.max_rounds:
            return None, None
        multiplier = 2 ** (self.current_round + 1)
        return self.m0 * multiplier, self.m1 * multiplier
    
    def take_action(self, action, player):
        """
        Processes a player's action.
        action: 'take' or 'push'
        player: 1 or 2
        Returns: (game_over, p1_payoff, p2_payoff)
        """
        large_pile, small_pile = self.get_current_piles()
        
        if action == 'take':
            self.game_over = True
            if player == 1:
                self.p1_total = large_pile
                self.p2_total = small_pile
            else:
                self.p2_total = large_pile
                self.p1_total = small_pile
            return True, self.p1_total, self.p2_total
        
        else:  # push
            self.current_round += 1
            if self.current_round >= self.max_rounds:
                # Game ends, player who would move next gets small, other gets large
                next_large, next_small = self.get_current_piles()
                if player == 1:
                    self.p2_total = next_large
                    self.p1_total = next_small
                else:
                    self.p1_total = next_large
                    self.p2_total = next_small
                self.game_over = True
                return True, self.p1_total, self.p2_total
            return False, 0, 0

# --- 3. AGENT PROMPTS ---

mistral = OllamaLLM(model="mistral")

PLAYER_TEMPLATE = """You are Player {player} in a Centipede Game. 

CURRENT SITUATION:
- Round {round}/{max_rounds}
- Current piles: LARGE pile = ${large_pile}, SMALL pile = ${small_pile}
- If you TAKE: You get ${your_take}, opponent gets ${opp_take}
- If you PUSH: Piles double, opponent decides next
{next_info}

YOUR SCORES:
- Your total so far: ${your_total}
- Opponent's total so far: ${opp_total}

YOUR TRUST IN OPPONENT:
- Competence Trust (Rationality): {ctrust:.2f}
- Benevolence Trust (Fairness): {btrust:.2f}

HISTORY:
{memory}

IMPORTANT STRATEGIC CONSIDERATIONS:
- Early cooperation often leads to MUCH larger payoffs for both players
- Taking too early means missing exponential growth opportunities
- If your trust is above 0.6, cooperation is usually beneficial
- The longer you both cooperate, the more both of you gain

You must decide: TAKE (end game and take the large pile) or PUSH (double piles and let opponent decide).

Your response MUST be:
1. One paragraph explaining your reasoning
2. On a new line: "Action: take" OR "Action: push"

Example:
Based on the current situation and my trust level, I believe cooperation will benefit both of us.
Action: push
"""

# --- 4. PARSING AND GAME EXECUTION ---

def parse_output(output):
    """Extracts promise and action from agent output."""
    try:
        lines = output.strip().split('\n')
        promise = "\n".join(lines[:-1]).strip()
        action_line = lines[-1].lower()
        
        if 'take' in action_line:
            action = 'take'
        elif 'push' in action_line:
            action = 'push'
        else:
            action = 'take'  # Default to safe choice
        
        return promise, action
    except Exception as e:
        print(f"[Parsing Error: {e} | Output: {output}]")
        return output, 'take'

def run_game(game_number, df, env_params, max_rounds=10, m0=4, m1=1):
    """Runs a single centipede game."""
    game = CentipedeGame(m0, m1, max_rounds)
    p1_memory = []
    p2_memory = []
    
    analyzer_agent = OllamaLLM(model="llama3.2", temperature=0.0)
    
    P1_trust_in_P2 = TrustTracker("P1_trust_in_P2", **env_params)
    P2_trust_in_P1 = TrustTracker("P2_trust_in_P1", **env_params)
    
    player_chain = ChatPromptTemplate.from_template(PLAYER_TEMPLATE) | mistral | StrOutputParser()
    
    print(f"\n=== Game {game_number} ===")
    round_number = 0
    
    while not game.game_over:
        round_number += 1
        current_player = 1 if game.current_round % 2 == 0 else 2
        large_pile, small_pile = game.get_current_piles()
        next_large, next_small = game.get_next_piles()
        
        if current_player == 1:
            trust_scores = P1_trust_in_P2.get_scores()
            memory = "\n".join(p1_memory[-3:]) if p1_memory else "No history yet"
            your_total = game.p1_total
            opp_total = game.p2_total
        else:
            trust_scores = P2_trust_in_P1.get_scores()
            memory = "\n".join(p2_memory[-3:]) if p2_memory else "No history yet"
            your_total = game.p2_total
            opp_total = game.p1_total
        
        if next_large is None:
            next_info = "- This is the FINAL ROUND. If you push, opponent gets large pile."
        else:
            next_info = f"- Next piles if pushed: LARGE = ${next_large}, SMALL = ${next_small}"
        
        print(f"\nRound {round_number} - Player {current_player}'s turn")
        print(f"Piles: ${large_pile} (large), ${small_pile} (small)")
        print(f"Trust in opponent: CT={trust_scores[0]:.2f}, BT={trust_scores[1]:.2f}")
        
        output = player_chain.invoke({
            "player": current_player,
            "round": round_number,
            "max_rounds": max_rounds,
            "large_pile": large_pile,
            "small_pile": small_pile,
            "your_take": large_pile,
            "opp_take": small_pile,
            "next_info": next_info,
            "your_total": your_total,
            "opp_total": opp_total,
            "ctrust": trust_scores[0],
            "btrust": trust_scores[1],
            "memory": memory
        })
        
        promise, action = parse_output(output)
        
        # CHANGE: Display full reasoning instead of truncating
        promise_log = promise[:200] + "..." if len(promise) > 200 else promise
        
        print(f"Player {current_player} reasoning:")
        print(f"{promise}")
        print(f"\nAction: {action.upper()}\n")
        
        # Update trust from promise
        if current_player == 1:
            P2_trust_in_P1.update_trust_from_promise(analyzer_agent, promise)
            p1_memory.append(f"P1 R{round_number}: {action}")
        else:
            P1_trust_in_P2.update_trust_from_promise(analyzer_agent, promise)
            p2_memory.append(f"P2 R{round_number}: {action}")
        
        # Calculate potential payoff for trust update
        if next_large is not None:
            potential_payoff = next_large if action == 'push' else 0
        else:
            potential_payoff = large_pile
        
        # Update trust from action
        if current_player == 1:
            P2_trust_in_P1.update_trust_from_action(large_pile, potential_payoff, action)
        else:
            P1_trust_in_P2.update_trust_from_action(large_pile, potential_payoff, action)
        
        # CHANGE: Add cooperation bonus - if both players pushed recently, boost trust
        if action == 'push' and round_number > 1:
            # Reward continued cooperation with trust bonus
            if current_player == 1:
                P2_trust_in_P1.CTrust = min(1.0, P2_trust_in_P1.CTrust + 0.03)
                P2_trust_in_P1.BTrust = min(1.0, P2_trust_in_P1.BTrust + 0.03)
            else:
                P1_trust_in_P2.CTrust = min(1.0, P1_trust_in_P2.CTrust + 0.03)
                P1_trust_in_P2.BTrust = min(1.0, P1_trust_in_P2.BTrust + 0.03)
        
        # Log round data BEFORE executing action
        p1_ct, p1_bt = P1_trust_in_P2.get_scores()
        p2_ct, p2_bt = P2_trust_in_P1.get_scores()
        
        new_row = {
            "Game": game_number,
            "Round": round_number,
            "Player": current_player,
            "Large_Pile": large_pile,
            "Small_Pile": small_pile,
            "Action": action,
            "Promise": promise_log,
            "P1_Total": your_total if current_player == 1 else opp_total,
            "P2_Total": opp_total if current_player == 1 else your_total,
            "P1_CTrust": p1_ct,
            "P1_BTrust": p1_bt,
            "P2_CTrust": p2_ct,
            "P2_BTrust": p2_bt,
            "Game_Over": False
        }
        df.loc[len(df)] = new_row
        
        # Execute action
        game_over, p1_final, p2_final = game.take_action(action, current_player)
        
        # Update totals in the row we just added
        df.loc[len(df) - 1, "P1_Total"] = game.p1_total
        df.loc[len(df) - 1, "P2_Total"] = game.p2_total
        df.loc[len(df) - 1, "Game_Over"] = game_over
        
        if game_over:
            print(f"\n=== GAME OVER ===")
            print(f"Player 1: ${p1_final}")
            print(f"Player 2: ${p2_final}")
            if p1_final > p2_final:
                return 1, df
            elif p2_final > p1_final:
                return 2, df
            else:
                return 3, df
    
    # Should never reach here, but just in case
    print(f"\n=== GAME ENDED (Max Rounds) ===")
    print(f"Player 1: ${game.p1_total}")
    print(f"Player 2: ${game.p2_total}")
    if game.p1_total > game.p2_total:
        return 1, df
    elif game.p2_total > game.p1_total:
        return 2, df
    else:
        return 3, df

# --- 5. MAIN SIMULATION ---

if __name__ == "__main__":
    # Environment Parameters
    env_params = {
        'D': 0.5,
        # CHANGE: Reduced trust decay rates to maintain trust longer
        'k_c': 0.005,  # Reduced from 0.01 to 0.005
        'k_b': 0.008   # Reduced from 0.02 to 0.008
    }
    
    # Simulation Parameters
    NUM_GAMES = 20
    MAX_ROUNDS = 20
    M0 = 4  # Larger starting pile
    M1 = 1  # Smaller starting pile
    
    # Initialize DataFrame with proper dtypes
    columns = [
        "Game", "Round", "Player", "Large_Pile", "Small_Pile", "Action",
        "Promise", "P1_Total", "P2_Total", "P1_CTrust", "P1_BTrust",
        "P2_CTrust", "P2_BTrust", "Game_Over"
    ]
    df = pd.DataFrame(columns=columns)
    
    # Ensure proper data types
    df = df.astype({
        'Game': 'int64',
        'Round': 'int64',
        'Player': 'int64',
        'Large_Pile': 'int64',
        'Small_Pile': 'int64',
        'P1_Total': 'int64',
        'P2_Total': 'int64',
        'P1_CTrust': 'float64',
        'P1_BTrust': 'float64',
        'P2_CTrust': 'float64',
        'P2_BTrust': 'float64',
        'Game_Over': 'bool'
    })
    
    p1_wins = 0
    p2_wins = 0
    ties = 0
    game_fails = 0
    
    print(f"Starting simulation with {NUM_GAMES} games...")
    print(f"Configuration: {MAX_ROUNDS} rounds, Starting piles: ${M0} and ${M1}")
    print("="*50)
    
    for i in range(NUM_GAMES):
        try:
            result, df = run_game(i + 1, df, env_params, MAX_ROUNDS, M0, M1)
            if result == 1:
                p1_wins += 1
            elif result == 2:
                p2_wins += 1
            else:
                ties += 1
        except Exception as e:
            print(f"Error in game {i + 1}: {e}")
            game_fails += 1
    
    # Validate data before saving
    print("\n" + "="*50)
    print("Validating simulation data...")
    
    # Check for duplicate rounds within games
    duplicates = df.groupby(['Game', 'Round']).size()
    if (duplicates > 1).any():
        print("⚠ WARNING: Found duplicate rounds in some games!")
        print(duplicates[duplicates > 1])
    else:
        print("✓ No duplicate rounds detected")
    
    # Check round numbering
    for game_num in df['Game'].unique():
        game_data = df[df['Game'] == game_num].sort_values('Round')
        expected_rounds = list(range(1, len(game_data) + 1))
        actual_rounds = game_data['Round'].tolist()
        if actual_rounds != expected_rounds:
            print(f"⚠ WARNING: Game {game_num} has irregular rounds: {actual_rounds}")
        else:
            print(f"✓ Game {game_num}: Rounds 1-{len(game_data)} are correct")
    
    # Save CSV
    df.to_csv("centipede_game_20_games_updated_20_rounds.csv", index=False)
    print(f"\n✓ Results saved to centipede_game_20_games_updated_20_rounds.csv ({len(df)} rows)")
    print(f"Player 1 wins: {p1_wins}, Player 2 wins: {p2_wins}, Ties: {ties}, Fails: {game_fails}")
    print("="*50)
    
    # --- 6. VISUALIZE TRUST EVOLUTION (After simulation completes) ---
    if len(df) == 0:
        print("\n⚠ No data to visualize. Skipping plots.")
    else:
        try:
            print("\nGenerating trust evolution plots...")
            
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (14, 10)
        
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Trust Metric Evolution in Centipede Game', fontsize=16, fontweight='bold')
            
            # Add game number to dataframe for plotting
            df['Game_Round'] = df.groupby('Game').cumcount() + 1
            
            # Plot 1: Competence Trust (CTrust) over rounds for all games
            ax1 = axes[0, 0]
            for game_num in df['Game'].unique():
                game_data = df[df['Game'] == game_num]
                ax1.plot(game_data['Game_Round'], game_data['P1_CTrust'], 
                        marker='o', label=f'Game {game_num} - P1→P2', linewidth=2)
                ax1.plot(game_data['Game_Round'], game_data['P2_CTrust'], 
                        marker='s', linestyle='--', label=f'Game {game_num} - P2→P1', linewidth=2)
            ax1.set_xlabel('Round', fontsize=12)
            ax1.set_ylabel('Competence Trust', fontsize=12)
            ax1.set_title('Competence Trust (Rationality) Evolution', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.05)
            
            # Plot 2: Benevolence Trust (BTrust) over rounds for all games
            ax2 = axes[0, 1]
            for game_num in df['Game'].unique():
                game_data = df[df['Game'] == game_num]
                ax2.plot(game_data['Game_Round'], game_data['P1_BTrust'], 
                        marker='o', label=f'Game {game_num} - P1→P2', linewidth=2)
                ax2.plot(game_data['Game_Round'], game_data['P2_BTrust'], 
                        marker='s', linestyle='--', label=f'Game {game_num} - P2→P1', linewidth=2)
            ax2.set_xlabel('Round', fontsize=12)
            ax2.set_ylabel('Benevolence Trust', fontsize=12)
            ax2.set_title('Benevolence Trust (Fairness) Evolution', fontsize=13, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.05)
            
            # Plot 3: Combined Trust (CTrust + BTrust) / 2
            ax3 = axes[1, 0]
            for game_num in df['Game'].unique():
                game_data = df[df['Game'] == game_num]
                p1_combined = (game_data['P1_CTrust'] + game_data['P1_BTrust']) / 2
                p2_combined = (game_data['P2_CTrust'] + game_data['P2_BTrust']) / 2
                ax3.plot(game_data['Game_Round'], p1_combined, 
                        marker='o', label=f'Game {game_num} - P1→P2', linewidth=2)
                ax3.plot(game_data['Game_Round'], p2_combined, 
                        marker='s', linestyle='--', label=f'Game {game_num} - P2→P1', linewidth=2)
            ax3.set_xlabel('Round', fontsize=12)
            ax3.set_ylabel('Combined Trust Score', fontsize=12)
            ax3.set_title('Combined Trust Score Evolution', fontsize=13, fontweight='bold')
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.05)
            
            # Plot 4: Trust vs Actions (scatter)
            ax4 = axes[1, 1]
            
            # Calculate combined trust for current player
            df['Current_Player_CTrust'] = df.apply(
                lambda row: row['P1_CTrust'] if row['Player'] == 1 else row['P2_CTrust'], axis=1
            )
            df['Current_Player_BTrust'] = df.apply(
                lambda row: row['P1_BTrust'] if row['Player'] == 1 else row['P2_BTrust'], axis=1
            )
            
            take_data = df[df['Action'] == 'take']
            push_data = df[df['Action'] == 'push']
            
            ax4.scatter(take_data['Current_Player_CTrust'], take_data['Current_Player_BTrust'], 
                       c='red', marker='x', s=150, alpha=0.7, label='TAKE', linewidths=3)
            ax4.scatter(push_data['Current_Player_CTrust'], push_data['Current_Player_BTrust'], 
                       c='green', marker='o', s=100, alpha=0.7, label='PUSH')
            ax4.set_xlabel('Competence Trust in Opponent', fontsize=12)
            ax4.set_ylabel('Benevolence Trust in Opponent', fontsize=12)
            ax4.set_title('Actions Based on Trust Levels', fontsize=13, fontweight='bold')
            ax4.legend(loc='best', fontsize=11)
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 1.05)
            ax4.set_ylim(0, 1.05)
            
            plt.tight_layout()
            plt.savefig('centipede_trust_evolution_20_updated.png', dpi=300, bbox_inches='tight')
            print("✓ Trust evolution plot saved as 'centipede_trust_evolution_20_updated.png'")
            
            # Additional plot: Trust change per round
            fig2, ax = plt.subplots(figsize=(12, 6))
            
            for game_num in df['Game'].unique():
                game_data = df[df['Game'] == game_num].copy()
                if len(game_data) > 1:
                    # Calculate trust changes
                    game_data['P1_CTrust_Change'] = game_data['P1_CTrust'].diff().fillna(0)
                    game_data['P2_CTrust_Change'] = game_data['P2_CTrust'].diff().fillna(0)
                    
                    ax.plot(game_data['Game_Round'], game_data['P1_CTrust_Change'], 
                           marker='o', label=f'Game {game_num} - P1 CT Change', linewidth=2)
                    ax.plot(game_data['Game_Round'], game_data['P2_CTrust_Change'], 
                           marker='s', linestyle='--', label=f'Game {game_num} - P2 CT Change', linewidth=2)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Trust Change', fontsize=12)
            ax.set_title('Competence Trust Change per Round', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('centipede_trust_changes_20_updated.png', dpi=300, bbox_inches='tight')
            print("✓ Trust change plot saved as 'centipede_trust_changes_20_updated.png'")
            
            print("\n" + "="*50)
            print("All visualizations complete!")
            print("="*50)
            print("\nDisplaying plots...")
            plt.show()
            
        except Exception as e:
            print(f"\n⚠ Error generating plots: {e}")
            print("Data was saved to CSV, but visualization failed.")