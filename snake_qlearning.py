"""
Q-Learning ile Snake Oyunu Oynayan Akilli Ajan
================================================
Yapay Zeka Dersi Projesi
Algoritma: Q-Learning (Temporal Difference Reinforcement Learning)

Gereksinimler:
    pip install pygame numpy matplotlib
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')



import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import json
import os

# ─────────────────────────────────────────────
# OYUN SABITLERI
# ─────────────────────────────────────────────
GRID_SIZE   = 20          # Izgara boyutu
CELL_SIZE   = 30          # Piksel
WIDTH       = GRID_SIZE * CELL_SIZE
HEIGHT      = GRID_SIZE * CELL_SIZE + 80  # Ust bilgi icin ekstra alan
FPS         = 30

# Renkler
BLACK       = (10, 15, 30)
GREEN       = (0, 220, 100)
DARK_GREEN  = (0, 140, 60)
RED         = (220, 60, 60)
WHITE       = (200, 220, 240)
CYAN        = (0, 220, 200)
GRAY        = (40, 55, 80)
YELLOW      = (255, 200, 0)

# Yonler: 0=Yukari, 1=Sag, 2=Asagi, 3=Sol
DIRECTIONS  = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES   = ["↑", "→", "↓", "←"]


# ─────────────────────────────────────────────
# CEVRE (ENVIRONMENT)
# ─────────────────────────────────────────────
class SnakeEnv:
    """
    Snake oyun ortami.
    Ajan bu ortamla etkileserek ogrenir.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Oyunu sifirla, baslangic durumunu dondur."""
        mid = GRID_SIZE // 2
        self.snake  = [(mid, mid), (mid, mid + 1), (mid, mid + 2)]
        self.dir    = 0   # Yukari
        self.score  = 0
        self.steps  = 0
        self.food   = self._spawn_food()
        self.death_cause = None
        self.death_obstacle_count = None
        return self._get_state()

    def _spawn_food(self):
        """Yilanin uzerinde olmayan rastgele bir konuma yiyecek yerlestir."""
        while True:
            pos = (
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1)
            )
            if pos not in self.snake:
                return pos

    def _get_state(self):
        """
        16-bitlik durum vektoru olustur:
        [tehlike_duz, tehlike_sag, tehlike_sol,         (3 bit - 1 adim)
         tehlike2_duz, tehlike2_sag, tehlike2_sol,      (3 bit - 2 adim lookahead)
         yon_up, yon_down, yon_left, yon_right,         (4 bit)
         yiyecek_up, yiyecek_down, yiyecek_left, yiyecek_right]  (4 bit)

        Yeni eklenen 3 bit: 2 adim ilerisi tehlikeli mi?
        Bu sayede ajan dar koridorlara girmeden once karar verebilir.
        """
        head = self.snake[0]

        # Goreceli yonler
        straight  = DIRECTIONS[self.dir]
        right_dir = DIRECTIONS[(self.dir + 1) % 4]
        left_dir  = DIRECTIONS[(self.dir + 3) % 4]

        def is_danger(d, steps=1):
            nx, ny = head[0] + d[0] * steps, head[1] + d[1] * steps
            if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
                return 1
            
            # Yiyecek yenecekse kuyruk kaymaz
            next_head = (head[0] + straight[0], head[1] + straight[1])
            will_eat = next_head == self.food
            
            if steps < len(self.snake):
                snake_slice = self.snake[:-steps]
            else:
                snake_slice = []
            future_snake = self.snake if will_eat else snake_slice
            if (nx, ny) in future_snake:
                return 1
            return 0
        
        surroundings = sum(
            1 for dx in [-1,0,1] for dy in [-1,0,1]
            if (dx,dy) != (0,0)
            and 0 <= head[0]+dx < GRID_SIZE
            and 0 <= head[1]+dy < GRID_SIZE
            and (head[0]+dx, head[1]+dy) in self.snake
        )
        density = min(surroundings // 2, 3)
        
        state = (
            # Ani tehlike - 1 adim (3 bit)
            is_danger(straight,  1),
            is_danger(right_dir, 1),
            is_danger(left_dir,  1),
            # Lookahead tehlike - 2 adim (3 bit) <-- YENI
            is_danger(straight,  2),
            is_danger(right_dir, 2),
            is_danger(left_dir,  2),
            # Mevcut yon (4 bit)
            int(self.dir == 0),
            int(self.dir == 2),
            int(self.dir == 3),
            int(self.dir == 1),
            # Yiyecek yonu (4 bit)
            int(self.food[1] < head[1]),
            int(self.food[1] > head[1]),
            int(self.food[0] < head[0]),
            int(self.food[0] > head[0]),
            
            int(density & 2 > 0),
            int(density & 1 > 0),
        )
        return state

    def step(self, action):
        """
        Eylemi uygula.
        action: 0=Duz git, 1=Saga don, 2=Sola don

        Returns:
            next_state, reward, done
        """
        # Yon guncelle
        if action == 1:
            self.dir = (self.dir + 1) % 4
        elif action == 2:
            self.dir = (self.dir + 3) % 4

        head = self.snake[0]
        d    = DIRECTIONS[self.dir]
        new_head = (head[0] + d[0], head[1] + d[1])

        # Duvar kontrolu
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
                new_head[1] < 0 or new_head[1] >= GRID_SIZE):
            self.death_cause = "duvar"
            self.death_obstacle_count = None
            return self._get_state(), -1000, True

        # Kuyruk haricinde govde carpismasi
        body_without_tail = self.snake[:-1]
        if new_head in body_without_tail:
            self.death_cause = "isirma"
            # Olum anindaki 3x3 engel sayisi (duvar + govde, kafa haric)
            obstacles = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx2, ny2 = head[0] + dx, head[1] + dy
                    if nx2 < 0 or nx2 >= GRID_SIZE or ny2 < 0 or ny2 >= GRID_SIZE:
                        obstacles += 1
                    elif (nx2, ny2) in self.snake[1:]:
                        obstacles += 1
            self.death_obstacle_count = obstacles
            return self._get_state(), -1000, True

        # Yiyecek mesafesi (odul sekillendirme)
        prev_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        self.snake.insert(0, new_head)
        self.steps += 1

        # Yiyecek yedi mi?
        if new_head == self.food:
            self.score += 1
            self.steps = 0  # her yiyecekten sonra sayac sifirlanir
            self.food = self._spawn_food()
            return self._get_state(), 10 + self.score, False

        self.snake.pop()

        # Mesafe odulu
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        reward   = 1 if new_dist < prev_dist else -1

        # Zaman asimi
        if self.steps > GRID_SIZE * GRID_SIZE * 2:
            self.death_cause = "zaman_asimi"
            self.death_obstacle_count = None
            return self._get_state(), -1000, True

        return self._get_state(), reward, False


# ─────────────────────────────────────────────
# Q-LEARNING AJAN
# ─────────────────────────────────────────────
class QLearningAgent:
    """
    Tablo tabanli Q-Learning ajani.
    Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    """

    def __init__(
        self,
        alpha=0.001,   # Ogrenme hizi (16-bit state icin biraz arttirildi)
        gamma=0.95,    # Indirim faktoru
        epsilon=1.0,   # Baslangic kesif orani
        eps_min=0.001,  # Minimum kesif orani
        eps_decay=0.999,  # Daha yavash decay - daha fazla kesif
    ):
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self.q_table   = {}            # {state: [Q_straight, Q_right, Q_left]}
        self.n_actions = 3

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, state):
        """ε-greedy politikasi."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)   # Kesif
        q = self._get_q(state)
        return q.index(max(q))                              # Somuru

    def update(self, state, action, reward, next_state):
        """Q tablosunu guncelle."""
        q       = self._get_q(state)
        next_q  = self._get_q(next_state)

        # Bellman denklemi
        q[action] += self.alpha * (
            reward + self.gamma * max(next_q) - q[action]
        )

    def decay_epsilon(self):
        """Kesif oranini azalt."""
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay
            self.epsilon = max(self.epsilon, self.eps_min)

    def save(self, filepath="q_table.json", best_score=0):
        serializable = {str(k): v for k, v in self.q_table.items()}
        with open(filepath, "w") as f:
            json.dump({
                "q_table": serializable,
                "epsilon": self.epsilon,
                "best_score": best_score
            }, f)

    def load(self, filepath="q_table.json"):
        """Q tablosunu yukle."""
        if not os.path.exists(filepath):
            return False
        with open(filepath) as f:
            data = json.load(f)
        self.q_table = {eval(k): v for k, v in data["q_table"].items()}
        self.epsilon = data.get("epsilon", self.eps_min)
        print(f"✓ Q-tablosu yuklendi: {len(self.q_table)} durum")
        return True

def load_top3(save_dir="top3"):
    """Top-3 listesini yukle. [(skor, dosya_adi), ...]"""
    index_path = os.path.join(save_dir, "top3_index.json")
    if not os.path.exists(index_path):
        return []
    with open(index_path) as f:
        return json.load(f)  # [(skor, dosya_adi), ...]


def save_top3(agent, score, save_dir="top3"):
    """
    Skoru top-3 listesiyle karsilastir.
    Girdiyse kaydet, en dusugun dosyasini sil.
    Konsola hangi sira oldugunu yazdir.
    """
    os.makedirs(save_dir, exist_ok=True)
    top3 = load_top3(save_dir)  # [(skor, dosya), ...]

    # Listeye giriyor mu?
    if len(top3) >= 4 and score <= top3[-1][0]:
        return  # Girmiyor

    # Yeni dosya adi
    filename = f"q_table_{score}.json"
    filepath = os.path.join(save_dir, filename)
    agent.save(filepath, best_score=score)

    # Listeye ekle, skora gore sirala (buyukten kucuge)
    top3.append([score, filename])
    top3.sort(key=lambda x: x[0], reverse=True)

    # 3'ten fazlaysa en dusugu sil
    if len(top3) > 4:
        removed = top3.pop()
        removed_path = os.path.join(save_dir, removed[1])
        if os.path.exists(removed_path):
            os.remove(removed_path)

    # Index dosyasini guncelle
    index_path = os.path.join(save_dir, "top3_index.json")
    with open(index_path, "w") as f:
        json.dump(top3, f)

    rank = next(i + 1 for i, (s, _) in enumerate(top3) if s == score)
    print(f"  ★ Top-3'e girdi! Sira: {rank}. | Skor: {score} | {filename}")
    print(f"  Guncel Top-3: {[s for s, _ in top3]}")


# ─────────────────────────────────────────────
# EGITIM DONGUSU (pygame olmadan)
# ─────────────────────────────────────────────
def train(n_episodes=20000, visualize_every=100, save_path="q_table.json", finetune=False):
    """
    Ajani egit ve sonuclari gorsellestir.
    
    Args:
        n_episodes:       Toplam egitim episotu sayisi
        visualize_every:  Kac episotta bir istatistik yazdirilsin
        save_path:        Q-tablosu kayit yolu
        finetune:         True ise onceki Q-tablosunu yukle, epsilon=0.05'ten baslat
    """
    env    = SnakeEnv()
    agent  = QLearningAgent()

    # Ince ayar modu: onceki Q-tablosunu yukle
    if finetune and os.path.exists(save_path):
        agent.load(save_path)
        agent.epsilon = 0.05
        print(f"  [Finetune] Q-tablosu yuklendi, epsilon=0.05'ten basliyor")

    scores         = []
    avg_scores     = []
    best_score     = 0
    recent_scores  = deque(maxlen=100)

    # Olum istatistikleri
    death_stats = {
        "duvar": 0,
        "isirma": 0,
        "zaman_asimi": 0,
    }
    # Isirma anindaki engel sayisi dagilimi (0-8)
    isirma_obstacle_dist = {i: 0 for i in range(9)}

    print("=" * 60)
    print("  Q-LEARNING SNAKE AJAN - EGITIM BASLIYOR")
    print("=" * 60)
    print(f"  Episot sayisi  : {n_episodes}")
    print(f"  α (ogrenme)    : {agent.alpha}")
    print(f"  γ (indirim)    : {agent.gamma}")
    print(f"  ε (baslangic)  : {agent.epsilon}")
    print("=" * 60)

    for ep in range(1, n_episodes + 1):
        state  = env.reset()
        done   = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state  = next_state
            total_reward += reward

        agent.decay_epsilon()

        score = env.score
        scores.append(score)
        recent_scores.append(score)
        avg = np.mean(recent_scores)
        avg_scores.append(avg)

        # Olum istatistiklerini kaydet
        if env.death_cause:
            death_stats[env.death_cause] = death_stats.get(env.death_cause, 0) + 1
            if env.death_cause == "isirma" and env.death_obstacle_count is not None:
                isirma_obstacle_dist[env.death_obstacle_count] += 1

        if score > best_score:
            best_score = score
            agent.save(save_path, best_score=best_score)  # q_table.json

        if ep % visualize_every == 0:
            print(
                f"  Ep {ep:5d} | "
                f"Skor: {score:3d} | "
                f"En iyi: {best_score:3d} | "
                f"Ort(100): {avg:5.1f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Q-Tablo: {len(agent.q_table)}"
            )

    print("\n✓ Egitim tamamlandi!")
    print(f"  En yuksek skor : {best_score}")
    print(f"  Son 100 ort.   : {np.mean(list(recent_scores)):.2f}")
    print(f"  Q-tablo boyutu : {len(agent.q_table)} durum")

    # Eğitim bittikten sonra top-3 güncelle
    save_top3(agent, best_score)

    # Olum istatistiklerini yazdir
    total_deaths = sum(death_stats.values())
    print("\n" + "=" * 60)
    print("  OLUM ISTATISTIKLERI")
    print("=" * 60)
    for cause, count in death_stats.items():
        pct = count / total_deaths * 100 if total_deaths > 0 else 0
        label = {"duvar": "Duvara carpma ", "isirma": "Kendini isirma", "zaman_asimi": "Zaman asimi   "}[cause]
        print(f"  {label}: {count:6d}  ({pct:5.1f}%)")
    print(f"  {'Toplam':14s}: {total_deaths:6d}")

    print("\n  Kendini isirma — 3x3 engel dagilimi:")
    print("  Engel  | Adet   | Oran")
    print("  -------+--------+------")
    isirma_total = death_stats["isirma"]
    for obs, count in sorted(isirma_obstacle_dist.items()):
        if count > 0:
            pct = count / isirma_total * 100 if isirma_total > 0 else 0
            print(f"  {obs:6d} | {count:6d} | {pct:5.1f}%")

    _plot_results(scores, avg_scores, best_score, death_stats, isirma_obstacle_dist)
    return agent, scores


def _plot_results(scores, avg_scores, best_score, death_stats=None, isirma_obstacle_dist=None):
    """Egitim sonuclarini ciz."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0a0f1e")

    for row in axes:
        for ax in row:
            ax.set_facecolor("#070f1e")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#1a3a5c")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("cyan")

    # 1) Skor grafigi
    axes[0][0].plot(scores, color="#00ffe1", alpha=0.4, linewidth=0.8, label="Skor")
    axes[0][0].plot(avg_scores, color="#ff6b35", linewidth=2, label="Ort (100)")
    axes[0][0].axhline(best_score, color="#7fff00", linestyle="--", alpha=0.7, label=f"En Iyi: {best_score}")
    axes[0][0].set_title("EGITIM SKORLARI", fontsize=13, fontweight="bold")
    axes[0][0].set_xlabel("Episot")
    axes[0][0].set_ylabel("Skor")
    axes[0][0].legend(facecolor="#0a0f1e", labelcolor="white")
    axes[0][0].grid(alpha=0.15, color="#1a3a5c")

    # 2) Skor dagilim histogrami
    axes[0][1].hist(scores, bins=30, color="#00ffe1", edgecolor="#0a0f1e", alpha=0.8)
    axes[0][1].axvline(np.mean(scores), color="#ff6b35", linestyle="--",
                    label=f"Ortalama: {np.mean(scores):.1f}")
    axes[0][1].axvline(best_score, color="#7fff00", linestyle="--",
                    label=f"Maks: {best_score}")
    axes[0][1].set_title("SKOR DAGILIMI", fontsize=13, fontweight="bold")
    axes[0][1].set_xlabel("Skor")
    axes[0][1].set_ylabel("Frekans")
    axes[0][1].legend(facecolor="#0a0f1e", labelcolor="white")
    axes[0][1].grid(alpha=0.15, color="#1a3a5c")

    # 3) Olum nedeni pasta grafigi
    if death_stats:
        labels = []
        sizes  = []
        colors_pie = ["#ff4444", "#ff9900", "#4444ff"]
        label_map = {"duvar": "Duvara Carpma", "isirma": "Kendini Isirma", "zaman_asimi": "Zaman Asimi"}
        for (cause, count), color in zip(death_stats.items(), colors_pie):
            if count > 0:
                labels.append(f"{label_map[cause]}\n({count:,})")
                sizes.append(count)
        wedges, texts, autotexts = axes[1][0].pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors_pie[:len(sizes)],
            textprops={"color": "white", "fontsize": 9},
            wedgeprops={"edgecolor": "#0a0f1e", "linewidth": 1.5}
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(9)
        axes[1][0].set_title("OLUM NEDENLERI", fontsize=13, fontweight="bold")

    # 4) Isirma anindaki engel dagilimi bar grafigi
    if isirma_obstacle_dist:
        obs_keys   = sorted(isirma_obstacle_dist.keys())
        obs_vals   = [isirma_obstacle_dist[k] for k in obs_keys]
        isirma_tot = death_stats.get("isirma", 1) or 1
        obs_pcts   = [v / isirma_tot * 100 for v in obs_vals]

        bars = axes[1][1].bar(
            [str(k) for k in obs_keys], obs_pcts,
            color="#ff9900", edgecolor="#0a0f1e", alpha=0.85
        )
        # Her barin ustune sayi yaz
        for bar, val in zip(bars, obs_vals):
            if val > 0:
                axes[1][1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:,}", ha="center", va="bottom",
                    color="white", fontsize=8
                )
        axes[1][1].set_title("KENDINI ISIRMA — 3x3 ENGEL DAGILIMI", fontsize=13, fontweight="bold")
        axes[1][1].set_xlabel("Etraftaki Engel Sayisi (duvar + govde)")
        axes[1][1].set_ylabel("Isirma Olumlerinin Yuzdesi (%)")
        axes[1][1].grid(alpha=0.15, color="#1a3a5c", axis="y")

    plt.suptitle("Q-Learning Snake Ajan — Egitim Analizi",
                 color="white", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("✓ Grafik kaydedildi: training_results.png")


# ─────────────────────────────────────────────
# PYGAME GORSELLESTIRME (Egitilmis ajan)
# ─────────────────────────────────────────────
def play_visual(agent=None, n_games=10):
    """
    Egitilmis ajanin oynadigini pygame ile goster.
    agent=None ise q_table.json yuklenmeye calisilir.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Snake — Gorsel Demo")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("Courier New", 16, bold=True)

    if agent is None:
        agent = QLearningAgent()
        if not agent.load():
            print("Once egitim calistirin: train()")
            return

    # Gorsel oyun icin epsilon = 0 (tamamen somuru)
    agent.epsilon = 0.0
    env = SnakeEnv()

    for game in range(n_games):
        state = env.reset()
        done  = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

            action = agent.choose_action(state)
            state, _, done = env.step(action)

            _draw(screen, env, game + 1, n_games, font, agent)
            clock.tick(FPS)

    pygame.quit()


def _draw(screen, env, game_num, total_games, font, agent):
    """Pygame cizim fonksiyonu."""
    screen.fill(BLACK)

    # Izgara
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 80), (x, HEIGHT), 1)
    for y in range(80, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y), 1)

    # Yilan
    for i, (sx, sy) in enumerate(env.snake):
        color = CYAN if i == 0 else (
            GREEN if i < len(env.snake) // 2 else DARK_GREEN
        )
        rect = pygame.Rect(sx * CELL_SIZE + 1, sy * CELL_SIZE + 81,
                           CELL_SIZE - 2, CELL_SIZE - 2)
        pygame.draw.rect(screen, color, rect, border_radius=4)

    # Yiyecek
    fx, fy = env.food
    pygame.draw.circle(
        screen, RED,
        (fx * CELL_SIZE + CELL_SIZE // 2, fy * CELL_SIZE + CELL_SIZE // 2 + 80),
        CELL_SIZE // 2 - 3
    )

    # Ust bilgi
    pygame.draw.rect(screen, (15, 25, 45), (0, 0, WIDTH, 78))
    pygame.draw.line(screen, CYAN, (0, 78), (WIDTH, 78), 2)

    texts = [
        (f"OYUN: {game_num}/{total_games}", WHITE, (10, 10)),
        (f"SKOR: {env.score}", YELLOW, (10, 35)),
        (f"ADIM: {env.steps}", WHITE, (10, 55)),
        (f"ε: {agent.epsilon:.3f}", CYAN, (WIDTH // 2, 10)),
        (f"Q-TABLE: {len(agent.q_table)}", CYAN, (WIDTH // 2, 35)),
        (f"YON: {DIR_NAMES[env.dir]}", WHITE, (WIDTH // 2, 55)),
    ]
    for text, color, pos in texts:
        surf = font.render(text, True, color)
        screen.blit(surf, pos)

    pygame.display.flip()


# ─────────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Secenekler:")
    print("  1. Egitim baslat (20000 episot)")
    print("  2. Egitim + Gorsel demo")
    print("  3. Sadece gorsel demo (kayitli model)")
    print("  4. Ince ayar - onceki modelden devam et (finetune)")

    choice = input("\n  Secim (1/2/3/4): ").strip()

    if choice == "1":
        train(n_episodes=20000)

    elif choice == "2":
        trained_agent, _ = train(n_episodes=20000)
        print("\nGorsel demo baslatiliyor... (ESC ile cikis)")
        play_visual(agent=trained_agent, n_games=5)

    elif choice == "3":
        play_visual(n_games=10)

    elif choice == "4":
        train(n_episodes=20000, finetune=True)

    else:
        print("Gecersiz secim, egitim baslatiliyor...")
        train(n_episodes=1000)
