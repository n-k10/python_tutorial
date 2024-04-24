import random
a = random.randint(1,20) #変数aに1～20のランダムな値を入れる
b = random.randint(1,20) #変数bに1～20のランダムな値を入れる
print("支"*a +"文"+"支"*b)
num = int(input("文は、左から何番目(なんばんめ)にありますか？"))
if num == a + 1:
  print("正解！")
else:
  print("不正解・・・",a+1,"番目です。")
