import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CentipedeVisualizer:
    """Visualizer for Centipede Game results with improved legend handling."""
    
    def __init__(self, csv_path):
        """Initialize with path to CSV file."""
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path("plots_20_games")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set consistent style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
        print(f"✓ Loaded {len(self.df)} records from {csv_path}")
        print(f"  Games: {self.df['Game'].nunique()}, Max Round: {self.df['Round'].max()}")
    
    def plot_trust_evolution_aggregate(self):
        """Plot average trust evolution across all games with confidence intervals."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calculate round-wise statistics
        round_stats = self.df.groupby('Round').agg({
            'P1_CTrust': ['mean', 'std'],
            'P1_BTrust': ['mean', 'std'],
            'P2_CTrust': ['mean', 'std'],
            'P2_BTrust': ['mean', 'std']
        }).reset_index()
        
        rounds = round_stats['Round']
        
        # Plot 1: Competence Trust
        ax1 = axes[0]
        ax1.plot(rounds, round_stats['P1_CTrust']['mean'], 
                marker='o', linewidth=2.5, label='P1→P2', color='#2E86AB')
        ax1.fill_between(rounds, 
                         round_stats['P1_CTrust']['mean'] - round_stats['P1_CTrust']['std'],
                         round_stats['P1_CTrust']['mean'] + round_stats['P1_CTrust']['std'],
                         alpha=0.2, color='#2E86AB')
        
        ax1.plot(rounds, round_stats['P2_CTrust']['mean'], 
                marker='s', linewidth=2.5, label='P2→P1', color='#A23B72', linestyle='--')
        ax1.fill_between(rounds,
                         round_stats['P2_CTrust']['mean'] - round_stats['P2_CTrust']['std'],
                         round_stats['P2_CTrust']['mean'] + round_stats['P2_CTrust']['std'],
                         alpha=0.2, color='#A23B72')
        
        ax1.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Competence Trust Score', fontsize=12, fontweight='bold')
        ax1.set_title('Average Competence Trust Evolution', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9, fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Plot 2: Benevolence Trust
        ax2 = axes[1]
        ax2.plot(rounds, round_stats['P1_BTrust']['mean'], 
                marker='o', linewidth=2.5, label='P1→P2', color='#2E86AB')
        ax2.fill_between(rounds,
                         round_stats['P1_BTrust']['mean'] - round_stats['P1_BTrust']['std'],
                         round_stats['P1_BTrust']['mean'] + round_stats['P1_BTrust']['std'],
                         alpha=0.2, color='#2E86AB')
        
        ax2.plot(rounds, round_stats['P2_BTrust']['mean'], 
                marker='s', linewidth=2.5, label='P2→P1', color='#A23B72', linestyle='--')
        ax2.fill_between(rounds,
                         round_stats['P2_BTrust']['mean'] - round_stats['P2_BTrust']['std'],
                         round_stats['P2_BTrust']['mean'] + round_stats['P2_BTrust']['std'],
                         alpha=0.2, color='#A23B72')
        
        ax2.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Benevolence Trust Score', fontsize=12, fontweight='bold')
        ax2.set_title('Average Benevolence Trust Evolution', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9, fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        output_path = self.output_dir / "trust_evolution_aggregate.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_individual_games(self, max_games=5):
        """Plot trust evolution for individual games (limited to avoid legend clutter)."""
        games = sorted(self.df['Game'].unique())[:max_games]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(games)))
        
        for idx, game_num in enumerate(games):
            game_data = self.df[self.df['Game'] == game_num].sort_values('Round')
            rounds = game_data['Round']
            color = colors[idx]
            
            # Competence Trust
            axes[0].plot(rounds, game_data['P1_CTrust'], 
                        marker='o', linewidth=2, label=f'G{game_num} P1→P2',
                        color=color, alpha=0.8)
            axes[0].plot(rounds, game_data['P2_CTrust'], 
                        marker='s', linewidth=2, linestyle='--', label=f'G{game_num} P2→P1',
                        color=color, alpha=0.6)
            
            # Benevolence Trust
            axes[1].plot(rounds, game_data['P1_BTrust'], 
                        marker='o', linewidth=2, label=f'G{game_num} P1→P2',
                        color=color, alpha=0.8)
            axes[1].plot(rounds, game_data['P2_BTrust'], 
                        marker='s', linewidth=2, linestyle='--', label=f'G{game_num} P2→P1',
                        color=color, alpha=0.6)
        
        axes[0].set_xlabel('Round', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Competence Trust', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Competence Trust - First {max_games} Games', fontsize=13, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.05)
        
        axes[1].set_xlabel('Round', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Benevolence Trust', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Benevolence Trust - First {max_games} Games', fontsize=13, fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        
        plt.tight_layout()
        output_path = self.output_dir / "trust_evolution_individual.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_action_distribution(self):
        """Visualize action distribution and trust relationship."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Calculate trust for current player
        self.df['Current_CTrust'] = self.df.apply(
            lambda row: row['P1_CTrust'] if row['Player'] == 1 else row['P2_CTrust'], axis=1
        )
        self.df['Current_BTrust'] = self.df.apply(
            lambda row: row['P1_BTrust'] if row['Player'] == 1 else row['P2_BTrust'], axis=1
        )
        
        # Plot 1: Action counts by round
        ax1 = axes[0, 0]
        action_by_round = self.df.groupby(['Round', 'Action']).size().unstack(fill_value=0)
        action_by_round.plot(kind='bar', ax=ax1, color=['#E63946', '#2A9D8F'], width=0.7)
        ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Actions', fontsize=12, fontweight='bold')
        ax1.set_title('Action Distribution by Round', fontsize=13, fontweight='bold')
        ax1.legend(title='Action', labels=['Push', 'Take'], framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: Trust vs Action scatter
        ax2 = axes[0, 1]
        take_data = self.df[self.df['Action'] == 'take']
        push_data = self.df[self.df['Action'] == 'push']
        
        ax2.scatter(take_data['Current_CTrust'], take_data['Current_BTrust'], 
                   c='#E63946', marker='x', s=150, alpha=0.7, label='TAKE', linewidths=3)
        ax2.scatter(push_data['Current_CTrust'], push_data['Current_BTrust'], 
                   c='#2A9D8F', marker='o', s=100, alpha=0.7, label='PUSH')
        ax2.set_xlabel('Competence Trust in Opponent', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Benevolence Trust in Opponent', fontsize=12, fontweight='bold')
        ax2.set_title('Actions Based on Trust Levels', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        
        # Plot 3: Trust distribution by action
        ax3 = axes[1, 0]
        trust_by_action = self.df.groupby('Action')[['Current_CTrust', 'Current_BTrust']].mean()
        x = np.arange(len(trust_by_action.index))
        width = 0.35
        ax3.bar(x - width/2, trust_by_action['Current_CTrust'], width, 
               label='Competence Trust', color='#457B9D')
        ax3.bar(x + width/2, trust_by_action['Current_BTrust'], width, 
               label='Benevolence Trust', color='#E76F51')
        ax3.set_ylabel('Average Trust Score', fontsize=12, fontweight='bold')
        ax3.set_title('Average Trust by Action Type', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(trust_by_action.index.str.upper())
        ax3.legend(framealpha=0.9)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1.0)
        
        # Plot 4: Payoff distribution
        ax4 = axes[1, 1]
        final_payoffs = self.df[self.df['Game_Over'] == True][['P1_Total', 'P2_Total']]
        if len(final_payoffs) > 0:
            payoff_data = [final_payoffs['P1_Total'], final_payoffs['P2_Total']]
            bp = ax4.boxplot(payoff_data, labels=['Player 1', 'Player 2'], 
                           patch_artist=True, notch=True)
            for patch, color in zip(bp['boxes'], ['#2E86AB', '#A23B72']):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax4.set_ylabel('Final Payoff ($)', fontsize=12, fontweight='bold')
            ax4.set_title('Final Payoff Distribution', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "action_trust_analysis.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_game_outcomes(self):
        """Visualize game outcomes and durations."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Calculate game durations
        game_lengths = self.df.groupby('Game')['Round'].max()
        final_outcomes = self.df[self.df['Game_Over'] == True].groupby('Game')[['P1_Total', 'P2_Total']].first()
        
        # Plot 1: Game duration histogram
        ax1 = axes[0]
        ax1.hist(game_lengths, bins=range(1, game_lengths.max() + 2), 
                color='#4A90E2', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Number of Rounds', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
        ax1.set_title('Game Duration Distribution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axvline(game_lengths.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {game_lengths.mean():.1f}')
        ax1.legend(framealpha=0.9)
        
        # Plot 2: Winner distribution
        ax2 = axes[1]
        winners = []
        for _, row in final_outcomes.iterrows():
            if row['P1_Total'] > row['P2_Total']:
                winners.append('Player 1')
            elif row['P2_Total'] > row['P1_Total']:
                winners.append('Player 2')
            else:
                winners.append('Tie')
        
        winner_counts = pd.Series(winners).value_counts()
        colors_pie = ['#2E86AB', '#A23B72', '#F4A261']
        ax2.pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Game Winners', fontsize=13, fontweight='bold')
        
        # Plot 3: Total payoff comparison
        ax3 = axes[2]
        x = np.arange(len(final_outcomes))
        width = 0.35
        ax3.bar(x - width/2, final_outcomes['P1_Total'], width, 
               label='Player 1', color='#2E86AB', alpha=0.8)
        ax3.bar(x + width/2, final_outcomes['P2_Total'], width, 
               label='Player 2', color='#A23B72', alpha=0.8)
        ax3.set_xlabel('Game Number', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Final Payoff ($)', fontsize=12, fontweight='bold')
        ax3.set_title('Final Payoffs by Game', fontsize=13, fontweight='bold')
        ax3.legend(framealpha=0.9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "game_outcomes.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_simple_trust_zones(self):
        """Simple plot showing trust zones and what they mean."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Calculate average trust (combine competence and benevolence)
        round_stats = self.df.groupby('Round').agg({
            'P1_CTrust': 'mean',
            'P1_BTrust': 'mean',
            'P2_CTrust': 'mean',
            'P2_BTrust': 'mean'
        }).reset_index()
        
        # Average trust for each player
        round_stats['P1_Avg_Trust'] = (round_stats['P1_CTrust'] + round_stats['P1_BTrust']) / 2
        round_stats['P2_Avg_Trust'] = (round_stats['P2_CTrust'] + round_stats['P2_BTrust']) / 2
        
        rounds = round_stats['Round']
        
        # Add colored zones
        ax.axhspan(0, 0.3, alpha=0.15, color='red', label='Low Trust Zone')
        ax.axhspan(0.3, 0.6, alpha=0.15, color='yellow', label='Medium Trust Zone')
        ax.axhspan(0.6, 1.0, alpha=0.15, color='green', label='High Trust Zone')
        
        # Plot trust lines
        ax.plot(rounds, round_stats['P1_Avg_Trust'], 
               marker='o', linewidth=3, label='Player 1 Trust in P2', 
               color='#2E86AB', markersize=10)
        ax.plot(rounds, round_stats['P2_Avg_Trust'], 
               marker='s', linewidth=3, label='Player 2 Trust in P1', 
               color='#A23B72', markersize=10, linestyle='--')
        
        # Add annotations
        ax.text(0.5, 0.15, 'LOW TRUST\nLikely to TAKE early', 
               ha='center', fontsize=11, fontweight='bold', alpha=0.7)
        ax.text(0.5, 0.45, 'MEDIUM TRUST\nUncertain behavior', 
               ha='center', fontsize=11, fontweight='bold', alpha=0.7)
        ax.text(0.5, 0.8, 'HIGH TRUST\nLikely to PUSH/cooperate', 
               ha='center', fontsize=11, fontweight='bold', alpha=0.7)
        
        ax.set_xlabel('Round Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Trust Score', fontsize=14, fontweight='bold')
        ax.set_title('Trust Evolution with Decision Zones\n(Combines Competence + Benevolence Trust)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='lower right', framealpha=0.95, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(rounds.min() - 0.5, rounds.max() + 0.5)
        
        plt.tight_layout()
        output_path = self.output_dir / "trust_zones_simple.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_game_summary_cards(self):
        """Create a summary card for each game showing key metrics."""
        games = sorted(self.df['Game'].unique())
        
        # Calculate metrics per game
        game_metrics = []
        for game in games:
            game_data = self.df[self.df['Game'] == game]
            final_row = game_data[game_data['Game_Over'] == True].iloc[0] if any(game_data['Game_Over']) else game_data.iloc[-1]
            
            metrics = {
                'game': game,
                'rounds': game_data['Round'].max(),
                'final_p1_trust': (final_row['P1_CTrust'] + final_row['P1_BTrust']) / 2,
                'final_p2_trust': (final_row['P2_CTrust'] + final_row['P2_BTrust']) / 2,
                'p1_payoff': final_row['P1_Total'],
                'p2_payoff': final_row['P2_Total'],
                'cooperation': (game_data['Action'] == 'push').sum() / len(game_data) * 100
            }
            game_metrics.append(metrics)
        
        df_metrics = pd.DataFrame(game_metrics)
        
        # Create grid of subplots
        n_games = len(games)
        cols = 5
        rows = (n_games + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 3.5 * rows))
        axes = axes.flatten() if n_games > 1 else [axes]
        
        for idx, (_, row) in enumerate(df_metrics.iterrows()):
            ax = axes[idx]
            
            # Create a simple card layout
            ax.axis('off')
            
            # Background
            rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2)
            ax.add_patch(rect)
            
            # Title
            ax.text(0.5, 0.85, f'GAME {int(row["game"])}', 
                   ha='center', fontsize=14, fontweight='bold', 
                   transform=ax.transAxes)
            
            # Duration
            ax.text(0.5, 0.72, f'{int(row["rounds"])} rounds', 
                   ha='center', fontsize=11, style='italic',
                   transform=ax.transAxes)
            
            # Trust scores
            trust_color = '#2ecc71' if row['final_p1_trust'] > 0.6 else '#f39c12' if row['final_p1_trust'] > 0.3 else '#e74c3c'
            ax.text(0.5, 0.58, f'Trust: {row["final_p1_trust"]:.2f}', 
                   ha='center', fontsize=12, color=trust_color, fontweight='bold',
                   transform=ax.transAxes)
            
            # Cooperation rate
            ax.text(0.5, 0.48, f'Cooperation: {row["cooperation"]:.0f}%', 
                   ha='center', fontsize=10,
                   transform=ax.transAxes)
            
            # Payoffs
            winner = 'P1' if row['p1_payoff'] > row['p2_payoff'] else 'P2' if row['p2_payoff'] > row['p1_payoff'] else 'TIE'
            winner_color = '#2E86AB' if winner == 'P1' else '#A23B72' if winner == 'P2' else '#95a5a6'
            
            ax.text(0.5, 0.35, f'P1: ${int(row["p1_payoff"])}', 
                   ha='center', fontsize=10,
                   transform=ax.transAxes)
            ax.text(0.5, 0.25, f'P2: ${int(row["p2_payoff"])}', 
                   ha='center', fontsize=10,
                   transform=ax.transAxes)
            ax.text(0.5, 0.12, f'Winner: {winner}', 
                   ha='center', fontsize=11, fontweight='bold', color=winner_color,
                   transform=ax.transAxes)
        
        # Hide extra subplots
        for idx in range(n_games, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Game-by-Game Summary Cards', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        output_path = self.output_dir / "game_summary_cards.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_decision_matrix(self):
        """Show when players decide to PUSH vs TAKE based on trust and round."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate average trust
        self.df['Avg_Trust'] = self.df.apply(
            lambda row: (row['P1_CTrust'] + row['P1_BTrust'])/2 if row['Player'] == 1 
                       else (row['P2_CTrust'] + row['P2_BTrust'])/2,
            axis=1
        )
        
        # Plot 1: Trust vs Round colored by action
        ax1 = axes[0]
        take_data = self.df[self.df['Action'] == 'take']
        push_data = self.df[self.df['Action'] == 'push']
        
        scatter1 = ax1.scatter(push_data['Round'], push_data['Avg_Trust'], 
                              c='#2A9D8F', marker='o', s=120, alpha=0.6, 
                              label='PUSH (Cooperate)', edgecolors='white', linewidths=1.5)
        scatter2 = ax1.scatter(take_data['Round'], take_data['Avg_Trust'], 
                              c='#E63946', marker='X', s=180, alpha=0.8, 
                              label='TAKE (Defect)', edgecolors='white', linewidths=2)
        
        ax1.set_xlabel('Round Number', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Player\'s Trust in Opponent', fontsize=13, fontweight='bold')
        ax1.set_title('Decision Patterns: When Do Players PUSH vs TAKE?', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=12, framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # Add zone indicators
        ax1.axhline(0.6, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax1.text(ax1.get_xlim()[1], 0.62, 'High Trust', 
                va='bottom', ha='right', fontsize=10, color='green', fontweight='bold')
        
        # Plot 2: Action distribution heatmap
        ax2 = axes[1]
        
        # Create bins for trust and round
        self.df['Trust_Bin'] = pd.cut(self.df['Avg_Trust'], bins=[0, 0.3, 0.6, 1.0], 
                                       labels=['Low\n(0-0.3)', 'Medium\n(0.3-0.6)', 'High\n(0.6-1.0)'])
        self.df['Round_Bin'] = pd.cut(self.df['Round'], bins=[0, 3, 6, 10], 
                                       labels=['Early\n(1-3)', 'Mid\n(4-6)', 'Late\n(7-10)'])
        
        # Count actions in each bin
        heatmap_data = pd.crosstab(self.df['Trust_Bin'], self.df['Round_Bin'], 
                                    self.df['Action'], aggfunc='count', normalize='index') * 100
        
        # Plot heatmap for PUSH actions
        if 'push' in heatmap_data.columns.get_level_values(0):
            push_pct = heatmap_data['push'].fillna(0)
            sns.heatmap(push_pct, annot=True, fmt='.0f', cmap='RdYlGn', 
                       ax=ax2, cbar_kws={'label': '% PUSH Actions'}, 
                       vmin=0, vmax=100, linewidths=2, linecolor='white')
            ax2.set_xlabel('Game Stage', fontsize=13, fontweight='bold')
            ax2.set_ylabel('Trust Level', fontsize=13, fontweight='bold')
            ax2.set_title('Cooperation Rate by Trust & Game Stage\n(% of PUSH actions)', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "decision_matrix.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_trust_vs_payoff(self):
        """Show relationship between trust building and final payoffs."""
        # Get final game data
        final_games = []
        for game in self.df['Game'].unique():
            game_data = self.df[self.df['Game'] == game]
            final_row = game_data[game_data['Game_Over'] == True].iloc[0] if any(game_data['Game_Over']) else game_data.iloc[-1]
            
            avg_p1_trust = game_data['P1_CTrust'].mean()
            avg_p2_trust = game_data['P2_CTrust'].mean()
            avg_trust = (avg_p1_trust + avg_p2_trust) / 2
            
            final_games.append({
                'game': game,
                'avg_trust': avg_trust,
                'p1_payoff': final_row['P1_Total'],
                'p2_payoff': final_row['P2_Total'],
                'total_payoff': final_row['P1_Total'] + final_row['P2_Total'],
                'rounds': game_data['Round'].max()
            })
        
        df_final = pd.DataFrame(final_games)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Trust vs Total Payoff
        ax1 = axes[0]
        scatter = ax1.scatter(df_final['avg_trust'], df_final['total_payoff'], 
                             c=df_final['rounds'], cmap='viridis', s=200, 
                             alpha=0.7, edgecolors='black', linewidths=2)
        
        # Add trend line
        z = np.polyfit(df_final['avg_trust'], df_final['total_payoff'], 1)
        p = np.poly1d(z)
        ax1.plot(df_final['avg_trust'], p(df_final['avg_trust']), 
                "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax1.set_xlabel('Average Trust Level in Game', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Total Payoff (Both Players)', fontsize=13, fontweight='bold')
        ax1.set_title('Does Higher Trust Lead to Higher Payoffs?', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Game Duration (Rounds)', fontsize=11, fontweight='bold')
        
        # Plot 2: Game duration vs Trust
        ax2 = axes[1]
        colors_duration = ['#E63946' if r < 4 else '#F4A261' if r < 7 else '#2A9D8F' 
                          for r in df_final['rounds']]
        
        ax2.bar(df_final['game'], df_final['rounds'], color=colors_duration, 
               alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax2.set_xlabel('Game Number', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Number of Rounds', fontsize=13, fontweight='bold')
        ax2.set_title('Game Duration (Color = Early/Mid/Late Defection)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E63946', label='Short (1-3 rounds)'),
            Patch(facecolor='#F4A261', label='Medium (4-6 rounds)'),
            Patch(facecolor='#2A9D8F', label='Long (7+ rounds)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / "trust_vs_payoff.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_interpretation_report(self):
        """Generate text-based interpretation of the results."""
        report = []
        report.append("=" * 70)
        report.append("CENTIPEDE GAME ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Basic statistics
        num_games = self.df['Game'].nunique()
        total_rounds = len(self.df)
        avg_game_length = self.df.groupby('Game')['Round'].max().mean()
        
        report.append("1. BASIC STATISTICS")
        report.append(f"   - Total Games: {num_games}")
        report.append(f"   - Total Rounds: {total_rounds}")
        report.append(f"   - Average Game Length: {avg_game_length:.2f} rounds")
        report.append("")
        
        # Trust evolution
        initial_trust = self.df.groupby('Game').first()[['P1_CTrust', 'P1_BTrust', 'P2_CTrust', 'P2_BTrust']].mean()
        final_trust = self.df.groupby('Game').last()[['P1_CTrust', 'P1_BTrust', 'P2_CTrust', 'P2_BTrust']].mean()
        
        report.append("2. TRUST EVOLUTION")
        report.append(f"   Initial Trust (avg):")
        report.append(f"     - P1 Competence Trust: {initial_trust['P1_CTrust']:.3f}")
        report.append(f"     - P1 Benevolence Trust: {initial_trust['P1_BTrust']:.3f}")
        report.append(f"     - P2 Competence Trust: {initial_trust['P2_CTrust']:.3f}")
        report.append(f"     - P2 Benevolence Trust: {initial_trust['P2_BTrust']:.3f}")
        report.append(f"   Final Trust (avg):")
        report.append(f"     - P1 Competence Trust: {final_trust['P1_CTrust']:.3f}")
        report.append(f"     - P1 Benevolence Trust: {final_trust['P1_BTrust']:.3f}")
        report.append(f"     - P2 Competence Trust: {final_trust['P2_CTrust']:.3f}")
        report.append(f"     - P2 Benevolence Trust: {final_trust['P2_BTrust']:.3f}")
        report.append("")
        
        # Action analysis
        action_counts = self.df['Action'].value_counts()
        report.append("3. ACTION DISTRIBUTION")
        report.append(f"   - PUSH actions: {action_counts.get('push', 0)} ({action_counts.get('push', 0)/len(self.df)*100:.1f}%)")
        report.append(f"   - TAKE actions: {action_counts.get('take', 0)} ({action_counts.get('take', 0)/len(self.df)*100:.1f}%)")
        
        # Trust thresholds for actions
        self.df['Current_CTrust'] = self.df.apply(
            lambda row: row['P1_CTrust'] if row['Player'] == 1 else row['P2_CTrust'], axis=1
        )
        avg_trust_push = self.df[self.df['Action'] == 'push']['Current_CTrust'].mean()
        avg_trust_take = self.df[self.df['Action'] == 'take']['Current_CTrust'].mean()
        
        report.append(f"   - Average trust when PUSH: {avg_trust_push:.3f}")
        report.append(f"   - Average trust when TAKE: {avg_trust_take:.3f}")
        report.append("")
        
        # Outcomes
        final_outcomes = self.df[self.df['Game_Over'] == True].groupby('Game')[['P1_Total', 'P2_Total']].first()
        p1_wins = (final_outcomes['P1_Total'] > final_outcomes['P2_Total']).sum()
        p2_wins = (final_outcomes['P2_Total'] > final_outcomes['P1_Total']).sum()
        ties = (final_outcomes['P1_Total'] == final_outcomes['P2_Total']).sum()
        
        report.append("4. GAME OUTCOMES")
        report.append(f"   - Player 1 Wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
        report.append(f"   - Player 2 Wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
        report.append(f"   - Ties: {ties} ({ties/num_games*100:.1f}%)")
        report.append(f"   - Average P1 Payoff: ${final_outcomes['P1_Total'].mean():.2f}")
        report.append(f"   - Average P2 Payoff: ${final_outcomes['P2_Total'].mean():.2f}")
        report.append("")
        
        # Key insights
        report.append("5. KEY INSIGHTS")
        
        cooperation_rate = action_counts.get('push', 0) / len(self.df) * 100
        if cooperation_rate > 60:
            report.append(f"   ✓ HIGH cooperation rate ({cooperation_rate:.1f}%) suggests effective trust building")
        elif cooperation_rate > 40:
            report.append(f"   → MODERATE cooperation rate ({cooperation_rate:.1f}%) indicates cautious play")
        else:
            report.append(f"   ✗ LOW cooperation rate ({cooperation_rate:.1f}%) suggests early defection")
        
        trust_change = (final_trust.mean() - initial_trust.mean()) / initial_trust.mean() * 100
        if trust_change > 10:
            report.append(f"   ✓ Trust INCREASED by {trust_change:.1f}% over games")
        elif trust_change > -10:
            report.append(f"   → Trust remained STABLE (change: {trust_change:.1f}%)")
        else:
            report.append(f"   ✗ Trust DECREASED by {abs(trust_change):.1f}% over games")
        
        if avg_game_length > 5:
            report.append(f"   ✓ Long average game length ({avg_game_length:.1f}) suggests sustained cooperation")
        elif avg_game_length > 3:
            report.append(f"   → Moderate game length ({avg_game_length:.1f}) indicates mixed strategies")
        else:
            report.append(f"   ✗ Short game length ({avg_game_length:.1f}) suggests early defection")
        
        report.append("")
        report.append("=" * 70)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\n✓ Report saved: {report_path}")
    
    def run_all_visualizations(self):
        """Run all visualization methods."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        print("=== SIMPLE, EASY-TO-UNDERSTAND PLOTS ===")
        print("1. Trust Zones (Simple)...")
        self.plot_simple_trust_zones()
        
        print("2. Game Summary Cards...")
        self.plot_game_summary_cards()
        
        print("3. Decision Matrix (When PUSH vs TAKE)...")
        self.plot_decision_matrix()
        
        print("4. Trust vs Payoff Analysis...")
        self.plot_trust_vs_payoff()
        
        print("\n=== DETAILED ANALYTICAL PLOTS ===")
        print("5. Trust Evolution (Aggregate)...")
        self.plot_trust_evolution_aggregate()
        
        print("6. Trust Evolution (Individual Games)...")
        self.plot_individual_games(max_games=5)
        
        print("7. Action & Trust Analysis...")
        self.plot_action_distribution()
        
        print("8. Game Outcomes...")
        self.plot_game_outcomes()
        
        print("\n9. Generating Interpretation Report...")
        self.generate_interpretation_report()
        
        print("\n" + "="*70)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print(f"✓ All files saved in: {self.output_dir.absolute()}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Find the most recent CSV file
    csv_files = [
        "centipede_game_20_games_updated_20_rounds.csv"#,
        #"centipede_game_10_games_updated.csv",
        #"centipede_game_log.csv"
    ]
    
    csv_path = None
    for file in csv_files:
        if Path(file).exists():
            csv_path = file
            break
    
    if csv_path is None:
        print("❌ No CSV file found! Please run the game simulation first.")
    else:
        print(f"\n📊 Using data from: {csv_path}\n")
        visualizer = CentipedeVisualizer(csv_path)
        visualizer.run_all_visualizations()
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("  1. Check the 'plots/' folder for all generated visualizations")
        print("  2. Review 'analysis_report.txt' for detailed interpretation")
        print("\n  START WITH THESE SIMPLE PLOTS:")
        print("     ⭐ trust_zones_simple.png - Trust over time with colored zones")
        print("     ⭐ game_summary_cards.png - One card per game with key stats")
        print("     ⭐ decision_matrix.png - When players cooperate vs defect")
        print("     ⭐ trust_vs_payoff.png - Does trust lead to better outcomes?")
        print("\n  THEN EXPLORE DETAILED PLOTS:")
        print("     - trust_evolution_aggregate.png (statistical averages)")
        print("     - trust_evolution_individual.png (specific game trajectories)")
        print("     - action_trust_analysis.png (comprehensive analysis)")
        print("     - game_outcomes.png (win rates and distributions)")
        print("="*70 + "\n")