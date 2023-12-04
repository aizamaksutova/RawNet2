import gdown

url = "https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx"
output = "checkpoint-epoch17.pth"
gdown.download(url, output, quiet=False)
