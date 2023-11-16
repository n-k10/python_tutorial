import random

def rock_paper_scissors():
    choices = ["rock", "paper", "scissors"]

    print("じゃんけんゲームを始めます！")
    while True:
        player_choice = input("rock, paper, scissorsのどれかを選んでください（終了する場合は 'quit' を入力）: ").lower()

        if player_choice == "quit":
            print("ゲームを終了します。")
            break

        if player_choice not in choices:
            print("正しい選択肢を入力してください。")
            continue

        computer_choice = random.choice(choices)
        print("コンピュータの選択:", computer_choice)

        if player_choice == computer_choice:
            print("引き分け！")
        elif (player_choice == "rock" and computer_choice == "scissors") or (player_choice == "paper" and computer_choice == "rock") or (player_choice == "scissors" and computer_choice == "paper"):
            print("プレイヤーの勝ち！")
        else:
            print("コンピュータの勝ち！")

rock_paper_scissors()
