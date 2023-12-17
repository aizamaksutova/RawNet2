import gdown

# a checkpoint
id_model = "1Y8VzCW1hyDLBmI41ze0SaZJppC16TgGk"
output_model = "model_rawnet.pth"
gdown.download(id=id_model, output=output_model, quiet=False)
