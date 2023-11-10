import random

# ガチャの中身
contents = ('超激レア ∩(´∀｀)∩', '激レア  (^_^)', 'レア (T^T)')

# 累積的な重み(超激レア=10%, 激レア=20%, レア=70%)
cum_weights = (1, 3, 10)

# 通常ガチャ
gacha = random.choices(contents, cum_weights=cum_weights, k=1)

# 10連ガチャ(k引数で回数を指定しています)
gacha10 = random.choices(contents, cum_weights=cum_weights, k=10)

print(gacha)
