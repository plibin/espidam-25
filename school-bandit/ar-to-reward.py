import pandas as pd

# 1. Load your CSV
df = pd.read_csv('brute-force-ar.csv')

# 2. Rename "attack_rate" to "reward"
df = df.rename(columns={'attack_rate': 'reward'})

# 3. Transform its values to 1 – current_value
df['reward'] = 1 - df['reward']

# 4. (Optional) Save back to CSV
df.to_csv('brute-force-reward.csv', index=False)
