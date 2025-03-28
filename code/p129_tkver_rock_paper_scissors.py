import tkinter as tk
import random

# 勝ち負けの結果を表示する関数（じゃんけんのルールをここに書く）
def rock_paper_scissors(player_choice):
    # コンピュータがランダムに選ぶ
    choices = ["rock", "paper", "scissors"]
    computer_choice = random.choice(choices)

    # 勝ち負けの判定
    if player_choice == computer_choice:
        result = "引き分け！"
    elif (player_choice == "rock" and computer_choice == "scissors") or \
         (player_choice == "paper" and computer_choice == "rock") or \
         (player_choice == "scissors" and computer_choice == "paper"):
        result = "プレイヤーの勝ち！"
    else:
        result = "コンピュータの勝ち！"

    # ラベルに結果を表示する（printのかわりに画面に出す）
    result_label["text"] = f"あなた: {player_choice}\nコンピュータ: {computer_choice}\n{result}"

# ボタンが押されたときに呼び出す関数
# それぞれ自分が出す手をrock_paper_scissorsに伝える

def rock():
    rock_paper_scissors("rock")

def paper():
    rock_paper_scissors("paper")

def scissors():
    rock_paper_scissors("scissors")

# ウィンドウを作る
root = tk.Tk()
root.title("じゃんけんゲーム")
root.geometry("250x180")  # 画面の大きさ（横×縦）

# 最初に表示するラベル（画面に文字を出す場所）
result_label = tk.Label(root, text="rock, paper, scissors から選んでね")
result_label.pack(pady=10)

# それぞれのボタンを作って、押したときに対応する関数を動かす
tk.Button(root, text="rock", command=rock).pack()
tk.Button(root, text="paper", command=paper).pack()
tk.Button(root, text="scissors", command=scissors).pack()

# 終了ボタン（押すとアプリを閉じる）
tk.Button(root, text="終了", command=root.quit).pack(pady=10)

# アプリをスタート（これを書かないと画面が出ない）
root.mainloop()
