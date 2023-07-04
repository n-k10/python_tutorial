hour = 10
if hour < 0 or hour > 24:
    print("0~24時の範囲で入力してね")
else:
    if hour >= 6 and hour < 12 :
        print("朝です")
    elif hour >= 12 and hour < 18:
        print("昼です")
    else:
        print("夜です")