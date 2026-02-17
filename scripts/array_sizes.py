import pandas as pd

pows = range(14,20)
vols = [2**i for i in pows]
df = pd.DataFrame({
    "Power": pows,
    "Volume": vols
})
df["KB per array"] = df["Volume"]*8*2*9//1024
df["KB output"] = df["Volume"]*8//1024
df["KB total"] = df["KB per array"]*4 + df["KB output"]
df["KB 8 threads"] = df["KB total"]//8
df["KB 16 threads"] = df["KB total"]//16
df["KB 32 threads"] = df["KB total"]//32
df["KB 64 threads"] = df["KB total"]//64

print(df.head(10))
print(df.drop(columns=["Power"]).to_csv(index=False, sep='&', lineterminator='\\\\ \n'))