import random

def play_game(player_choice, computer_choice):
    print("プレイヤーの選択:", player_choice)
    print("コンピュータの選択:", computer_choice)

    if player_choice == computer_choice:
        print("引き分け！")
    elif (player_choice == "rock" and (computer_choice == "scissors" or computer_choice == "lizard")) or \
         (player_choice == "paper" and (computer_choice == "rock" or computer_choice == "spock")) or \
         (player_choice == "scissors" and (computer_choice == "paper" or computer_choice == "lizard")) or \
         (player_choice == "lizard" and (computer_choice == "spock" or computer_choice == "paper")) or \
         (player_choice == "spock" and (computer_choice == "rock" or computer_choice == "scissors")):
        print("プレイヤーの勝ち！")
    else:
        print("コンピュータの勝ち！")

def american_rock_paper_scissors():
    choices = ["rock", "paper", "scissors", "lizard", "spock"]

    print("アメリカのじゃんけんゲームを始めます！")
    while True:
        player_choice = input("rock, paper, scissors, lizard, spockのどれかを選んでください（終了する場合は 'quit' を入力）: ").lower()

        if player_choice == "quit":
            print("ゲームを終了します。")
            break

        if player_choice not in choices:
            print("正しい選択肢を入力してください。")
            continue

        computer_choice = random.choice(choices)
        play_game(player_choice, computer_choice)

american_rock_paper_scissors()
