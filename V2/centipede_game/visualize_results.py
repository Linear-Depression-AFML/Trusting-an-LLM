import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class CentipedeVisualizer:
    """
    Visualizer for Centipede Game results.

    Supports the enhanced 4D DCHTM output that includes:
      - P1_CTrust, P1_BTrust, P1_ITrust, P1_PTrust  (and P2_ equivalents)
      - P1_ToM_Order, P2_ToM_Order
      - P1_Deception, P2_Deception
    Falls back gracefully to 2D columns when running against legacy CSVs.
    """

    # Columns introduced by the 4D expansion
    EXTENDED_COLS = ["P1_ITrust", "P1_PTrust", "P2_ITrust", "P2_PTrust",
                     "P1_ToM_Order", "P2_ToM_Order", "P1_Deception", "P2_Deception"]

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path("plots_30x10")
        self.output_dir.mkdir(exist_ok=True)

        # Detect whether the CSV contains the extended 4D columns
        self.extended = all(c in self.df.columns for c in ["P1_ITrust", "P1_PTrust"])

        if not self.extended:
            print("⚠  Extended 4D columns not found — running in 2D-compatibility mode.")
            for col in self.EXTENDED_COLS:
                if col not in self.df.columns:
                    self.df[col] = np.nan if "Trust" in col else 0

        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"]   = 100
        plt.rcParams["savefig.dpi"]  = 300
        plt.rcParams["font.size"]    = 10

        print(f"✓ Loaded {len(self.df)} records from {csv_path}")
        print(f"  Games: {self.df['Game'].nunique()}, Max Round: {self.df['Round'].max()}")
        print(f"  Mode: {'4D extended' if self.extended else '2D legacy'}")

    # ─────────────────────────────────────────────────────────────────────
    # Original plots (now updated to 4 trust dimensions where appropriate)
    # ─────────────────────────────────────────────────────────────────────

    def plot_trust_evolution_aggregate(self):
        """Plot average trust evolution across all games with confidence intervals (CTrust & BTrust)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        round_stats = self.df.groupby("Round").agg({
            "P1_CTrust": ["mean", "std"],
            "P1_BTrust": ["mean", "std"],
            "P2_CTrust": ["mean", "std"],
            "P2_BTrust": ["mean", "std"],
        }).reset_index()

        rounds = round_stats["Round"]

        for ax, (p1_col, p2_col, ylabel, title) in zip(axes, [
            ("P1_CTrust", "P2_CTrust", "Competence Trust Score",   "Average Competence Trust Evolution"),
            ("P1_BTrust", "P2_BTrust", "Benevolence Trust Score",  "Average Benevolence Trust Evolution"),
        ]):
            ax.plot(rounds, round_stats[p1_col]["mean"], marker="o", linewidth=2.5,
                    label="P1→P2", color="#2E86AB")
            ax.fill_between(rounds,
                            round_stats[p1_col]["mean"] - round_stats[p1_col]["std"],
                            round_stats[p1_col]["mean"] + round_stats[p1_col]["std"],
                            alpha=0.2, color="#2E86AB")
            ax.plot(rounds, round_stats[p2_col]["mean"], marker="s", linewidth=2.5,
                    label="P2→P1", color="#A23B72", linestyle="--")
            ax.fill_between(rounds,
                            round_stats[p2_col]["mean"] - round_stats[p2_col]["std"],
                            round_stats[p2_col]["mean"] + round_stats[p2_col]["std"],
                            alpha=0.2, color="#A23B72")
            ax.set_xlabel("Round Number", fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(loc="best", framealpha=0.9, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        output_path = self.output_dir / "trust_evolution_aggregate.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_individual_games(self, max_games=5):
        """Plot trust evolution for individual games."""
        games  = sorted(self.df["Game"].unique())[:max_games]
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(games)))

        for idx, game_num in enumerate(games):
            game_data = self.df[self.df["Game"] == game_num].sort_values("Round")
            rounds = game_data["Round"]
            color  = colors[idx]
            for ax, (p1, p2, ylabel) in zip(axes, [
                ("P1_CTrust", "P2_CTrust", "Competence Trust"),
                ("P1_BTrust", "P2_BTrust", "Benevolence Trust"),
            ]):
                ax.plot(rounds, game_data[p1], marker="o", linewidth=2,
                        label=f"G{game_num} P1→P2", color=color, alpha=0.8)
                ax.plot(rounds, game_data[p2], marker="s", linewidth=2,
                        linestyle="--", label=f"G{game_num} P2→P1", color=color, alpha=0.6)

        for ax, (ylabel, title) in zip(axes, [
            ("Competence Trust",  f"Competence Trust — First {max_games} Games"),
            ("Benevolence Trust", f"Benevolence Trust — First {max_games} Games"),
        ]):
            ax.set_xlabel("Round", fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        output_path = self.output_dir / "trust_evolution_individual.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_action_distribution(self):
        """Visualise action distribution and trust relationship."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        self.df["Current_CTrust"] = self.df.apply(
            lambda r: r["P1_CTrust"] if r["Player"] == 1 else r["P2_CTrust"], axis=1
        )
        self.df["Current_BTrust"] = self.df.apply(
            lambda r: r["P1_BTrust"] if r["Player"] == 1 else r["P2_BTrust"], axis=1
        )

        ax1 = axes[0, 0]
        action_by_round = self.df.groupby(["Round", "Action"]).size().unstack(fill_value=0)
        action_by_round.plot(kind="bar", ax=ax1, color=["#E63946", "#2A9D8F"], width=0.7)
        ax1.set_xlabel("Round", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Actions", fontsize=12, fontweight="bold")
        ax1.set_title("Action Distribution by Round", fontsize=13, fontweight="bold")
        ax1.legend(title="Action", labels=["Push", "Take"], framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        ax2 = axes[0, 1]
        take_data = self.df[self.df["Action"] == "take"]
        push_data = self.df[self.df["Action"] == "push"]
        ax2.scatter(take_data["Current_CTrust"], take_data["Current_BTrust"],
                    c="#E63946", marker="x", s=150, alpha=0.7, label="TAKE", linewidths=3)
        ax2.scatter(push_data["Current_CTrust"], push_data["Current_BTrust"],
                    c="#2A9D8F", marker="o", s=100, alpha=0.7, label="PUSH")
        ax2.set_xlabel("Competence Trust in Opponent", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Benevolence Trust in Opponent", fontsize=12, fontweight="bold")
        ax2.set_title("Actions Based on Trust Levels", fontsize=13, fontweight="bold")
        ax2.legend(loc="best", fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

        ax3 = axes[1, 0]
        trust_by_action = self.df.groupby("Action")[["Current_CTrust", "Current_BTrust"]].mean()
        x = np.arange(len(trust_by_action.index))
        width = 0.35
        ax3.bar(x - width / 2, trust_by_action["Current_CTrust"], width,
                label="Competence Trust", color="#457B9D")
        ax3.bar(x + width / 2, trust_by_action["Current_BTrust"], width,
                label="Benevolence Trust", color="#E76F51")
        ax3.set_ylabel("Average Trust Score", fontsize=12, fontweight="bold")
        ax3.set_title("Average Trust by Action Type", fontsize=13, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(trust_by_action.index.str.upper())
        ax3.legend(framealpha=0.9)
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.set_ylim(0, 1.0)

        ax4 = axes[1, 1]
        final_payoffs = self.df[self.df["Game_Over"] == True][["P1_Total", "P2_Total"]]
        if len(final_payoffs) > 0:
            bp = ax4.boxplot([final_payoffs["P1_Total"], final_payoffs["P2_Total"]],
                             labels=["Player 1", "Player 2"],
                             patch_artist=True, notch=True)
            for patch, color in zip(bp["boxes"], ["#2E86AB", "#A23B72"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax4.set_ylabel("Final Payoff ($)", fontsize=12, fontweight="bold")
            ax4.set_title("Final Payoff Distribution", fontsize=13, fontweight="bold")
            ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = self.output_dir / "action_trust_analysis.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_game_outcomes(self):
        """Visualise game outcomes and durations."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        game_lengths   = self.df.groupby("Game")["Round"].max()
        final_outcomes = (
            self.df[self.df["Game_Over"] == True]
            .groupby("Game")[["P1_Total", "P2_Total"]]
            .first()
        )

        ax1 = axes[0]
        ax1.hist(game_lengths, bins=range(1, game_lengths.max() + 2),
                 color="#4A90E2", edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Number of Rounds", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Games", fontsize=12, fontweight="bold")
        ax1.set_title("Game Duration Distribution", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.axvline(game_lengths.mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {game_lengths.mean():.1f}")
        ax1.legend(framealpha=0.9)

        ax2 = axes[1]
        winners = []
        for _, row in final_outcomes.iterrows():
            if row["P1_Total"] > row["P2_Total"]:
                winners.append("Player 1")
            elif row["P2_Total"] > row["P1_Total"]:
                winners.append("Player 2")
            else:
                winners.append("Tie")
        winner_counts = pd.Series(winners).value_counts()
        ax2.pie(winner_counts.values, labels=winner_counts.index, autopct="%1.1f%%",
                colors=["#2E86AB", "#A23B72", "#F4A261"], startangle=90,
                textprops={"fontsize": 11, "fontweight": "bold"})
        ax2.set_title("Game Winners", fontsize=13, fontweight="bold")

        ax3 = axes[2]
        x = np.arange(len(final_outcomes))
        width = 0.35
        ax3.bar(x - width / 2, final_outcomes["P1_Total"], width,
                label="Player 1", color="#2E86AB", alpha=0.8)
        ax3.bar(x + width / 2, final_outcomes["P2_Total"], width,
                label="Player 2", color="#A23B72", alpha=0.8)
        ax3.set_xlabel("Game Number", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Final Payoff ($)", fontsize=12, fontweight="bold")
        ax3.set_title("Final Payoffs by Game", fontsize=13, fontweight="bold")
        ax3.legend(framealpha=0.9)
        ax3.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = self.output_dir / "game_outcomes.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_simple_trust_zones(self):
        """Simple plot showing trust zones (uses combined 4D average when available)."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        agg_cols = {"P1_CTrust": "mean", "P1_BTrust": "mean",
                    "P2_CTrust": "mean", "P2_BTrust": "mean"}
        if self.extended:
            agg_cols.update({"P1_ITrust": "mean", "P1_PTrust": "mean",
                             "P2_ITrust": "mean", "P2_PTrust": "mean"})

        round_stats = self.df.groupby("Round").agg(agg_cols).reset_index()

        if self.extended:
            round_stats["P1_Avg_Trust"] = round_stats[
                ["P1_CTrust", "P1_BTrust", "P1_ITrust", "P1_PTrust"]].mean(axis=1)
            round_stats["P2_Avg_Trust"] = round_stats[
                ["P2_CTrust", "P2_BTrust", "P2_ITrust", "P2_PTrust"]].mean(axis=1)
            label_suffix = "(C+B+I+P avg)"
        else:
            round_stats["P1_Avg_Trust"] = (round_stats["P1_CTrust"] + round_stats["P1_BTrust"]) / 2
            round_stats["P2_Avg_Trust"] = (round_stats["P2_CTrust"] + round_stats["P2_BTrust"]) / 2
            label_suffix = "(C+B avg)"

        rounds = round_stats["Round"]
        ax.axhspan(0,   0.3, alpha=0.15, color="red",    label="Low Trust Zone")
        ax.axhspan(0.3, 0.6, alpha=0.15, color="yellow", label="Medium Trust Zone")
        ax.axhspan(0.6, 1.0, alpha=0.15, color="green",  label="High Trust Zone")

        ax.plot(rounds, round_stats["P1_Avg_Trust"], marker="o", linewidth=3,
                label=f"Player 1 Trust in P2 {label_suffix}", color="#2E86AB", markersize=10)
        ax.plot(rounds, round_stats["P2_Avg_Trust"], marker="s", linewidth=3,
                label=f"Player 2 Trust in P1 {label_suffix}", color="#A23B72",
                markersize=10, linestyle="--")

        ax.text(0.5, 0.15, "LOW TRUST\nLikely to TAKE early",     ha="center",
                fontsize=11, fontweight="bold", alpha=0.7)
        ax.text(0.5, 0.45, "MEDIUM TRUST\nUncertain behavior",    ha="center",
                fontsize=11, fontweight="bold", alpha=0.7)
        ax.text(0.5, 0.8,  "HIGH TRUST\nLikely to PUSH/cooperate", ha="center",
                fontsize=11, fontweight="bold", alpha=0.7)

        ax.set_xlabel("Round Number", fontsize=14, fontweight="bold")
        ax.set_ylabel("Average Trust Score", fontsize=14, fontweight="bold")
        ax.set_title("Trust Evolution with Decision Zones", fontsize=15, fontweight="bold", pad=20)
        ax.legend(loc="lower right", framealpha=0.95, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(rounds.min() - 0.5, rounds.max() + 0.5)

        plt.tight_layout()
        output_path = self.output_dir / "trust_zones_simple.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_game_summary_cards(self):
        """Summary card per game."""
        games = sorted(self.df["Game"].unique())
        game_metrics = []

        for game in games:
            game_data = self.df[self.df["Game"] == game]
            final_row = (
                game_data[game_data["Game_Over"] == True].iloc[0]
                if any(game_data["Game_Over"])
                else game_data.iloc[-1]
            )
            if self.extended:
                final_p1 = (final_row["P1_CTrust"] + final_row["P1_BTrust"] +
                            final_row["P1_ITrust"] + final_row["P1_PTrust"]) / 4
                final_p2 = (final_row["P2_CTrust"] + final_row["P2_BTrust"] +
                            final_row["P2_ITrust"] + final_row["P2_PTrust"]) / 4
            else:
                final_p1 = (final_row["P1_CTrust"] + final_row["P1_BTrust"]) / 2
                final_p2 = (final_row["P2_CTrust"] + final_row["P2_BTrust"]) / 2

            game_metrics.append({
                "game":         game,
                "rounds":       game_data["Round"].max(),
                "final_p1_trust": final_p1,
                "final_p2_trust": final_p2,
                "p1_payoff":    final_row["P1_Total"],
                "p2_payoff":    final_row["P2_Total"],
                "cooperation":  (game_data["Action"] == "push").sum() / len(game_data) * 100,
                "deception":    game_data["P1_Deception"].any() or game_data["P2_Deception"].any()
                                if "P1_Deception" in game_data.columns else False,
            })

        df_metrics = pd.DataFrame(game_metrics)
        n_games = len(games)
        cols = 5
        rows = (n_games + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 3.8 * rows))
        axes_flat = axes.flatten() if n_games > 1 else [axes]

        for idx, (_, row) in enumerate(df_metrics.iterrows()):
            ax = axes_flat[idx]
            ax.axis("off")
            rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                                  facecolor="#f8f9fa", edgecolor="#dee2e6", linewidth=2)
            ax.add_patch(rect)
            ax.text(0.5, 0.87, f"GAME {int(row['game'])}",
                    ha="center", fontsize=14, fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.74, f"{int(row['rounds'])} rounds",
                    ha="center", fontsize=11, style="italic", transform=ax.transAxes)
            trust_color = ("#2ecc71" if row["final_p1_trust"] > 0.6
                           else "#f39c12" if row["final_p1_trust"] > 0.3 else "#e74c3c")
            ax.text(0.5, 0.60, f"Trust: {row['final_p1_trust']:.2f}",
                    ha="center", fontsize=12, color=trust_color, fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.5, 0.50, f"Cooperation: {row['cooperation']:.0f}%",
                    ha="center", fontsize=10, transform=ax.transAxes)
            winner = ("P1" if row["p1_payoff"] > row["p2_payoff"]
                      else "P2" if row["p2_payoff"] > row["p1_payoff"] else "TIE")
            winner_color = {"P1": "#2E86AB", "P2": "#A23B72", "TIE": "#95a5a6"}[winner]
            ax.text(0.5, 0.38, f"P1: ${int(row['p1_payoff'])}",
                    ha="center", fontsize=10, transform=ax.transAxes)
            ax.text(0.5, 0.28, f"P2: ${int(row['p2_payoff'])}",
                    ha="center", fontsize=10, transform=ax.transAxes)
            ax.text(0.5, 0.17, f"Winner: {winner}",
                    ha="center", fontsize=11, fontweight="bold",
                    color=winner_color, transform=ax.transAxes)
            if row.get("deception", False):
                ax.text(0.5, 0.08, "⚠ Deception detected",
                        ha="center", fontsize=9, color="#e74c3c",
                        fontweight="bold", transform=ax.transAxes)

        for idx in range(n_games, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.suptitle("Game-by-Game Summary Cards", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        output_path = self.output_dir / "game_summary_cards.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_decision_matrix(self):
        """Show when players PUSH vs TAKE based on trust and round."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        if self.extended:
            self.df["Avg_Trust"] = self.df.apply(
                lambda r: (r["P1_CTrust"] + r["P1_BTrust"] + r["P1_ITrust"] + r["P1_PTrust"]) / 4
                          if r["Player"] == 1
                          else (r["P2_CTrust"] + r["P2_BTrust"] + r["P2_ITrust"] + r["P2_PTrust"]) / 4,
                axis=1,
            )
        else:
            self.df["Avg_Trust"] = self.df.apply(
                lambda r: (r["P1_CTrust"] + r["P1_BTrust"]) / 2
                          if r["Player"] == 1
                          else (r["P2_CTrust"] + r["P2_BTrust"]) / 2,
                axis=1,
            )

        ax1 = axes[0]
        take_data = self.df[self.df["Action"] == "take"]
        push_data = self.df[self.df["Action"] == "push"]
        ax1.scatter(push_data["Round"], push_data["Avg_Trust"],
                    c="#2A9D8F", marker="o", s=120, alpha=0.6,
                    label="PUSH (Cooperate)", edgecolors="white", linewidths=1.5)
        ax1.scatter(take_data["Round"], take_data["Avg_Trust"],
                    c="#E63946", marker="X", s=180, alpha=0.8,
                    label="TAKE (Defect)", edgecolors="white", linewidths=2)
        ax1.set_xlabel("Round Number", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Player's Trust in Opponent", fontsize=13, fontweight="bold")
        ax1.set_title("Decision Patterns: When Do Players PUSH vs TAKE?",
                      fontsize=14, fontweight="bold")
        ax1.legend(loc="best", fontsize=12, framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        ax1.axhline(0.6, color="green", linestyle="--", alpha=0.5, linewidth=2)
        ax1.text(ax1.get_xlim()[1], 0.62, "High Trust",
                 va="bottom", ha="right", fontsize=10, color="green", fontweight="bold")

        ax2 = axes[1]
        self.df["Trust_Bin"] = pd.cut(self.df["Avg_Trust"],
                                       bins=[0, 0.3, 0.6, 1.0],
                                       labels=["Low\n(0-0.3)", "Medium\n(0.3-0.6)", "High\n(0.6-1.0)"])
        self.df["Round_Bin"] = pd.cut(self.df["Round"],
                                       bins=[0, 3, 6, 10],
                                       labels=["Early\n(1-3)", "Mid\n(4-6)", "Late\n(7-10)"])
        heatmap_data = pd.crosstab(
            self.df["Trust_Bin"], self.df["Round_Bin"],
            self.df["Action"], aggfunc="count", normalize="index"
        ) * 100
        if "push" in heatmap_data.columns.get_level_values(0):
            push_pct = heatmap_data["push"].fillna(0)
            sns.heatmap(push_pct, annot=True, fmt=".0f", cmap="RdYlGn",
                        ax=ax2, cbar_kws={"label": "% PUSH Actions"},
                        vmin=0, vmax=100, linewidths=2, linecolor="white")
            ax2.set_xlabel("Game Stage", fontsize=13, fontweight="bold")
            ax2.set_ylabel("Trust Level", fontsize=13, fontweight="bold")
            ax2.set_title("Cooperation Rate by Trust & Game Stage\n(% of PUSH actions)",
                          fontsize=14, fontweight="bold")

        plt.tight_layout()
        output_path = self.output_dir / "decision_matrix.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_trust_vs_payoff(self):
        """Relationship between trust building and final payoffs."""
        final_games = []
        for game in self.df["Game"].unique():
            game_data = self.df[self.df["Game"] == game]
            final_row = (
                game_data[game_data["Game_Over"] == True].iloc[0]
                if any(game_data["Game_Over"])
                else game_data.iloc[-1]
            )
            avg_p1 = game_data["P1_CTrust"].mean()
            avg_p2 = game_data["P2_CTrust"].mean()
            final_games.append({
                "game":         game,
                "avg_trust":    (avg_p1 + avg_p2) / 2,
                "p1_payoff":    final_row["P1_Total"],
                "p2_payoff":    final_row["P2_Total"],
                "total_payoff": final_row["P1_Total"] + final_row["P2_Total"],
                "rounds":       game_data["Round"].max(),
            })

        df_final = pd.DataFrame(final_games)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        ax1 = axes[0]
        scatter = ax1.scatter(df_final["avg_trust"], df_final["total_payoff"],
                              c=df_final["rounds"], cmap="viridis", s=200,
                              alpha=0.7, edgecolors="black", linewidths=2)
        z = np.polyfit(df_final["avg_trust"], df_final["total_payoff"], 1)
        p = np.poly1d(z)
        ax1.plot(df_final["avg_trust"], p(df_final["avg_trust"]),
                 "r--", alpha=0.8, linewidth=2, label="Trend")
        ax1.set_xlabel("Average Trust Level in Game", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Total Payoff (Both Players)", fontsize=13, fontweight="bold")
        ax1.set_title("Does Higher Trust Lead to Higher Payoffs?", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        plt.colorbar(scatter, ax=ax1).set_label("Game Duration (Rounds)",
                                                 fontsize=11, fontweight="bold")

        ax2 = axes[1]
        colors_dur = ["#E63946" if r < 4 else "#F4A261" if r < 7 else "#2A9D8F"
                      for r in df_final["rounds"]]
        ax2.bar(df_final["game"], df_final["rounds"], color=colors_dur,
                alpha=0.7, edgecolor="black", linewidth=1.5)
        ax2.set_xlabel("Game Number", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Number of Rounds", fontsize=13, fontweight="bold")
        ax2.set_title("Game Duration (Color = Early/Mid/Late Defection)",
                      fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        from matplotlib.patches import Patch
        ax2.legend(handles=[
            Patch(facecolor="#E63946", label="Short (1-3 rounds)"),
            Patch(facecolor="#F4A261", label="Medium (4-6 rounds)"),
            Patch(facecolor="#2A9D8F", label="Long (7+ rounds)"),
        ], loc="upper right", fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / "trust_vs_payoff.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    # ─────────────────────────────────────────────────────────────────────
    # NEW plots: Integrity, Predictability, Theory of Mind
    # ─────────────────────────────────────────────────────────────────────

    def plot_integrity_predictability_evolution(self):
        """
        NEW — 4D expansion plot.
        Shows evolution of Integrity Trust and Predictability Trust
        alongside the original CTrust and BTrust for direct comparison.
        """
        if not self.extended:
            print("⚠  Skipping integrity/predictability plot (2D-legacy mode).")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("4D Trust Vector Evolution (DCHTM Expansion)",
                     fontsize=15, fontweight="bold")

        dimension_pairs = [
            ("P1_CTrust", "P2_CTrust", "Competence Trust (CTrust)",   "#2E86AB", "#1A5F7A"),
            ("P1_BTrust", "P2_BTrust", "Benevolence Trust (BTrust)",  "#A23B72", "#6D1E4A"),
            ("P1_ITrust", "P2_ITrust", "Integrity Trust (ITrust) ★",  "#2A9D8F", "#1A6B60"),
            ("P1_PTrust", "P2_PTrust", "Predictability Trust (PTrust) ★", "#E9C46A", "#B8860B"),
        ]

        for ax, (p1_col, p2_col, title, c1, c2) in zip(axes.flatten(), dimension_pairs):
            rs = self.df.groupby("Round").agg({p1_col: ["mean", "std"],
                                               p2_col: ["mean", "std"]}).reset_index()
            rounds = rs["Round"]
            ax.plot(rounds, rs[p1_col]["mean"], marker="o", linewidth=2.5,
                    label="P1→P2", color=c1)
            ax.fill_between(rounds,
                            rs[p1_col]["mean"] - rs[p1_col]["std"],
                            rs[p1_col]["mean"] + rs[p1_col]["std"],
                            alpha=0.2, color=c1)
            ax.plot(rounds, rs[p2_col]["mean"], marker="s", linewidth=2.5,
                    label="P2→P1", color=c2, linestyle="--")
            ax.fill_between(rounds,
                            rs[p2_col]["mean"] - rs[p2_col]["std"],
                            rs[p2_col]["mean"] + rs[p2_col]["std"],
                            alpha=0.2, color=c2)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("Trust Score")
            ax.legend(loc="best", fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        output_path = self.output_dir / "trust_4D_evolution.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_tom_analysis(self):
        """
        NEW — Theory of Mind (ToM) analysis plot.
        Shows ToM order over rounds and flags deception events.
        """
        if not self.extended:
            print("⚠  Skipping ToM plot (2D-legacy mode).")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Theory of Mind (ToM) Analysis", fontsize=14, fontweight="bold")

        # ── Plot 1: ToM order over rounds ──────────────────────────────
        ax1 = axes[0]
        rs = self.df.groupby("Round").agg({
            "P1_ToM_Order": ["mean", "std"],
            "P2_ToM_Order": ["mean", "std"],
        }).reset_index()
        rounds = rs["Round"]

        ax1.plot(rounds, rs["P1_ToM_Order"]["mean"], marker="o", linewidth=2.5,
                 label="P1 ToM Order", color="#3A86FF")
        ax1.fill_between(rounds,
                         rs["P1_ToM_Order"]["mean"] - rs["P1_ToM_Order"]["std"],
                         rs["P1_ToM_Order"]["mean"] + rs["P1_ToM_Order"]["std"],
                         alpha=0.2, color="#3A86FF")
        ax1.plot(rounds, rs["P2_ToM_Order"]["mean"], marker="s", linewidth=2.5,
                 label="P2 ToM Order", color="#FF006E", linestyle="--")
        ax1.fill_between(rounds,
                         rs["P2_ToM_Order"]["mean"] - rs["P2_ToM_Order"]["std"],
                         rs["P2_ToM_Order"]["mean"] + rs["P2_ToM_Order"]["std"],
                         alpha=0.2, color="#FF006E")

        ax1.axhline(2, color="orange", linestyle=":", linewidth=2,
                    label="2nd-order threshold (strategic)")
        ax1.axhline(3, color="red", linestyle=":", linewidth=2,
                    label="3rd-order threshold (deception risk)")
        ax1.set_xlabel("Round Number", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Estimated ToM Order", fontsize=12, fontweight="bold")
        ax1.set_title("Theory of Mind Order Evolution", fontsize=13, fontweight="bold")
        ax1.legend(loc="best", fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.5, 3.5)
        ax1.set_yticks([1, 2, 3])
        ax1.set_yticklabels(["1st order\n(basic)", "2nd order\n(strategic)",
                             "3rd order\n(deceptive)"])

        # ── Plot 2: Deception events per game ──────────────────────────
        ax2 = axes[1]
        game_deception = self.df.groupby("Game").agg(
            P1_any=("P1_Deception", "any"),
            P2_any=("P2_Deception", "any"),
        ).reset_index()

        x = np.arange(len(game_deception))
        width = 0.4
        ax2.bar(x - width / 2,
                game_deception["P1_any"].astype(int),
                width, label="P1 Deception Suspected",
                color="#FF6B6B", alpha=0.8, edgecolor="black")
        ax2.bar(x + width / 2,
                game_deception["P2_any"].astype(int),
                width, label="P2 Deception Suspected",
                color="#4ECDC4", alpha=0.8, edgecolor="black")
        ax2.set_xlabel("Game Number", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Deception Flagged (1=Yes, 0=No)", fontsize=12, fontweight="bold")
        ax2.set_title("Deception Events per Game\n(ToM-driven detection)",
                      fontsize=13, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(game_deception["Game"].astype(int), rotation=45)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["No", "Yes"])
        ax2.legend(loc="best", fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = self.output_dir / "tom_analysis.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_integrity_vs_actions(self):
        """
        NEW — Shows how Integrity Trust correlates with actions.
        Constitutional alignment penalty is visible as ITrust drops after
        deceptive TAKE actions.
        """
        if not self.extended:
            print("⚠  Skipping integrity-vs-actions plot (2D-legacy mode).")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Integrity Trust vs. Agent Actions", fontsize=14, fontweight="bold")

        self.df["Current_ITrust"] = self.df.apply(
            lambda r: r["P1_ITrust"] if r["Player"] == 1 else r["P2_ITrust"], axis=1
        )

        ax1 = axes[0]
        take_data = self.df[self.df["Action"] == "take"]
        push_data = self.df[self.df["Action"] == "push"]
        ax1.scatter(take_data["Round"], take_data["Current_ITrust"],
                    c="#E63946", marker="X", s=180, alpha=0.8,
                    label="TAKE (Defect)", edgecolors="white", linewidths=2)
        ax1.scatter(push_data["Round"], push_data["Current_ITrust"],
                    c="#2A9D8F", marker="o", s=120, alpha=0.6,
                    label="PUSH (Cooperate)", edgecolors="white", linewidths=1.5)
        ax1.set_xlabel("Round Number", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Integrity Trust (ITrust)", fontsize=12, fontweight="bold")
        ax1.set_title("ITrust at Time of Action", fontsize=13, fontweight="bold")
        ax1.legend(loc="best", fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        ax1.axhline(0.5, color="orange", linestyle="--", alpha=0.6, linewidth=1.5,
                    label="Baseline threshold")

        ax2 = axes[1]
        integrity_by_action = (
            self.df.groupby("Action")["Current_ITrust"]
            .agg(["mean", "std"])
            .reset_index()
        )
        x = np.arange(len(integrity_by_action))
        bars = ax2.bar(x, integrity_by_action["mean"],
                       color=["#E63946" if a == "take" else "#2A9D8F"
                              for a in integrity_by_action["Action"]],
                       alpha=0.8, edgecolor="black", linewidth=1.5,
                       yerr=integrity_by_action["std"], capsize=6)
        ax2.set_xlabel("Action", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Mean Integrity Trust (ITrust)", fontsize=12, fontweight="bold")
        ax2.set_title("Average ITrust per Action Type\n(Lower ITrust → constitutional violation)",
                      fontsize=13, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(integrity_by_action["Action"].str.upper())
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim(0, 1.0)

        plt.tight_layout()
        output_path = self.output_dir / "integrity_vs_actions.png"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()

    # ─────────────────────────────────────────────────────────────────────
    # Text report
    # ─────────────────────────────────────────────────────────────────────

    def generate_interpretation_report(self):
        """Generate text-based interpretation including 4D metrics."""
        report = []
        report.append("=" * 70)
        report.append("CENTIPEDE GAME ANALYSIS REPORT (4D DCHTM)")
        report.append("=" * 70)
        report.append("")

        num_games      = self.df["Game"].nunique()
        total_rounds   = len(self.df)
        avg_game_len   = self.df.groupby("Game")["Round"].max().mean()

        report.append("1. BASIC STATISTICS")
        report.append(f"   - Total Games:          {num_games}")
        report.append(f"   - Total Rounds:         {total_rounds}")
        report.append(f"   - Average Game Length:  {avg_game_len:.2f} rounds")
        report.append("")

        trust_cols_2d = ["P1_CTrust", "P1_BTrust", "P2_CTrust", "P2_BTrust"]
        trust_cols_4d = trust_cols_2d + ["P1_ITrust", "P1_PTrust", "P2_ITrust", "P2_PTrust"]
        cols_to_use   = trust_cols_4d if self.extended else trust_cols_2d

        initial_trust = self.df.groupby("Game").first()[cols_to_use].mean()
        final_trust   = self.df.groupby("Game").last()[cols_to_use].mean()

        report.append("2. TRUST EVOLUTION (Initial → Final averages)")
        for col in cols_to_use:
            report.append(f"   {col:16s}: {initial_trust[col]:.3f} → {final_trust[col]:.3f}")
        report.append("")

        action_counts = self.df["Action"].value_counts()
        report.append("3. ACTION DISTRIBUTION")
        report.append(
            f"   - PUSH actions: {action_counts.get('push', 0)} "
            f"({action_counts.get('push', 0) / len(self.df) * 100:.1f}%)"
        )
        report.append(
            f"   - TAKE actions: {action_counts.get('take', 0)} "
            f"({action_counts.get('take', 0) / len(self.df) * 100:.1f}%)"
        )
        self.df["Current_CTrust"] = self.df.apply(
            lambda r: r["P1_CTrust"] if r["Player"] == 1 else r["P2_CTrust"], axis=1
        )
        report.append(
            f"   - Avg CTrust on PUSH: {self.df[self.df['Action'] == 'push']['Current_CTrust'].mean():.3f}"
        )
        report.append(
            f"   - Avg CTrust on TAKE: {self.df[self.df['Action'] == 'take']['Current_CTrust'].mean():.3f}"
        )
        report.append("")

        if self.extended:
            report.append("4. THEORY OF MIND SUMMARY")
            tom_max_p1 = self.df["P1_ToM_Order"].max()
            tom_max_p2 = self.df["P2_ToM_Order"].max()
            tom_avg_p1 = self.df["P1_ToM_Order"].mean()
            tom_avg_p2 = self.df["P2_ToM_Order"].mean()
            deception_games = (
                self.df.groupby("Game")
                .agg(d1=("P1_Deception", "any"), d2=("P2_Deception", "any"))
                .apply(lambda r: r["d1"] or r["d2"], axis=1)
                .sum()
            )
            report.append(f"   - P1 ToM Order  avg / max:  {tom_avg_p1:.2f} / {tom_max_p1}")
            report.append(f"   - P2 ToM Order  avg / max:  {tom_avg_p2:.2f} / {tom_max_p2}")
            report.append(f"   - Games with deception flag: {deception_games} / {num_games}")
            report.append("")

        final_outcomes = (
            self.df[self.df["Game_Over"] == True]
            .groupby("Game")[["P1_Total", "P2_Total"]]
            .first()
        )
        p1_wins = (final_outcomes["P1_Total"] > final_outcomes["P2_Total"]).sum()
        p2_wins = (final_outcomes["P2_Total"] > final_outcomes["P1_Total"]).sum()
        ties    = (final_outcomes["P1_Total"] == final_outcomes["P2_Total"]).sum()

        report.append("5. GAME OUTCOMES")
        report.append(f"   - Player 1 Wins: {p1_wins} ({p1_wins / num_games * 100:.1f}%)")
        report.append(f"   - Player 2 Wins: {p2_wins} ({p2_wins / num_games * 100:.1f}%)")
        report.append(f"   - Ties:          {ties}  ({ties / num_games * 100:.1f}%)")
        report.append(f"   - Avg P1 Payoff: ${final_outcomes['P1_Total'].mean():.2f}")
        report.append(f"   - Avg P2 Payoff: ${final_outcomes['P2_Total'].mean():.2f}")
        report.append("")

        coop_rate  = action_counts.get("push", 0) / len(self.df) * 100
        trust_chg  = (final_trust.mean() - initial_trust.mean()) / initial_trust.mean() * 100

        report.append("6. KEY INSIGHTS")
        report.append(
            f"   {'✓' if coop_rate > 60 else '→' if coop_rate > 40 else '✗'} "
            f"Cooperation rate: {coop_rate:.1f}%"
        )
        report.append(
            f"   {'✓' if trust_chg > 10 else '→' if trust_chg > -10 else '✗'} "
            f"Trust change: {trust_chg:+.1f}%"
        )
        report.append(
            f"   {'✓' if avg_game_len > 5 else '→' if avg_game_len > 3 else '✗'} "
            f"Average game length: {avg_game_len:.1f} rounds"
        )
        report.append("")
        report.append("=" * 70)

        report_text = "\n".join(report)
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print("\n" + report_text)
        print(f"\n✓ Report saved: {report_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Master runner
    # ─────────────────────────────────────────────────────────────────────

    def run_all_visualizations(self):
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70 + "\n")

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

        if self.extended:
            print("\n=== NEW 4D EXPANSION PLOTS ===")
            print("9. Integrity & Predictability Evolution...")
            self.plot_integrity_predictability_evolution()

            print("10. Theory of Mind (ToM) Analysis...")
            self.plot_tom_analysis()

            print("11. Integrity Trust vs. Actions...")
            self.plot_integrity_vs_actions()

        print("\n12. Generating Interpretation Report...")
        self.generate_interpretation_report()

        print("\n" + "=" * 70)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print(f"✓ All files saved in: {self.output_dir.absolute()}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    csv_files = [
        "centipede_game_30x10.csv",
        "centipede_game_20x10.csv",
        "centipede_game_10_games_updated.csv",
        "centipede_game_log.csv",
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

        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("  1. Check the 'plots_30x10/' folder for all generated visualisations")
        print("  2. Review 'analysis_report.txt' for detailed interpretation")
        print("\n  START WITH THESE SIMPLE PLOTS:")
        print("     ⭐ trust_zones_simple.png     — Trust over time with coloured zones")
        print("     ⭐ game_summary_cards.png     — One card per game with key stats")
        print("     ⭐ decision_matrix.png        — When players cooperate vs defect")
        print("     ⭐ trust_vs_payoff.png        — Does trust lead to better outcomes?")
        print("\n  THEN EXPLORE DETAILED PLOTS:")
        print("     - trust_evolution_aggregate.png   (statistical averages, C+B)")
        print("     - trust_evolution_individual.png  (specific game trajectories)")
        print("     - action_trust_analysis.png       (comprehensive action analysis)")
        print("     - game_outcomes.png               (win rates and distributions)")
        print("\n  NEW 4D EXPANSION PLOTS (if running enhanced DCHTM):")
        print("     ★ trust_4D_evolution.png      — All 4 dimensions side-by-side")
        print("     ★ tom_analysis.png            — ToM order + deception events")
        print("     ★ integrity_vs_actions.png    — Constitutional alignment vs actions")
        print("=" * 70 + "\n")
